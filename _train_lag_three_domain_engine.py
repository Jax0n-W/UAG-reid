# -*- coding: utf-8 -*-
"""LAGPeR three-domain Stage 2: A_far <-> A_near <-> Ground.

This is a separate experimental entry.  It does not modify ``train_lag.py``
or the original trainer.  Stage 1 and the LAGPeR evaluation protocol remain
unchanged.  Stage 2 is based on the no-EMA-loss-only ablation: ``loss_ema`` is
excluded, while the EMA encoder update and the second ALL-memory optimization
are retained.
"""

from __future__ import absolute_import, print_function

import argparse
import os
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.cluster import DBSCAN
from torch.backends import cudnn

from ChannelAug import (ChannelAdapGray, ChannelExchange,
                        ChannelRandomErasing)
from clustercontrast.agva_three_domain import (
    build_candidate_map, remap_pseudo_dataset,
    remap_samples_to_enhanced_root, select_epoch_paths,
    select_stratified_epoch_paths)
from clustercontrast.evaluators import extract_features
from clustercontrast.domain_metadata import (
    DOMAIN_AERIAL, DOMAIN_GROUND, SUBDOMAIN_AERIAL_FAR,
    SUBDOMAIN_AERIAL_NEAR, SUBDOMAIN_GROUND,
    carry_domain_flags_by_basename, concatenate_domain_rows,
    domain_flags_for_samples, validate_subdomain_flags)
from clustercontrast.memory_lifecycle import cluster_precomputed_ephemeral
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.three_domain import (
    build_aerial_subdomain_flags, build_bidirectional_bgm,
    generate_cluster_features,
    generate_joint_cluster_features, generate_random_features,
    joint_cross_modal_flags, split_aerial_trainset)
from clustercontrast.trainers_three_domain import \
    ClusterContrastTrainer_PCLMP_ThreeDomain
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import save_checkpoint
from clustercontrast.viewbridge import (VIEW_PATH_MODES, load_view_score_map,
                                        match_view_score_map)


DEFAULT_STAGE2_LOG_NAME = "lag_three_domain_release_s2"


# Keep parser/help and CPU configuration tests usable on machines without
# faiss. Formal training still fails clearly through _require_baseline().
try:
    import train_lag as baseline
    _BASELINE_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    baseline = None
    _BASELINE_IMPORT_ERROR = error


def _require_baseline():
    if baseline is None:
        raise RuntimeError(
            "formal training requires the repository train_lag.py and its dependencies; "
            "current missing dependency: {}".format(_BASELINE_IMPORT_ERROR))
    return baseline


def get_stage2_output_directory(stage2_log_name, trial, logs_root="logs"):
    name = str(stage2_log_name).strip()
    if not name:
        raise ValueError("--stage2-log-name cannot be empty")
    return osp.normpath(osp.join(logs_root, name, str(trial)))


def check_stage2_output_directory(output_dir, allow_overwrite=False):
    if allow_overwrite or not osp.isdir(output_dir):
        return []
    conflicts = []
    for filename in sorted(os.listdir(output_dir)):
        path = osp.join(output_dir, filename)
        if not osp.isfile(path):
            continue
        lower_name = filename.lower()
        if (lower_name in ("checkpoint.pth.tar", "model_best.pth.tar") or
                lower_name.endswith(".log") or lower_name.endswith("log.txt")):
            conflicts.append(filename)
    if conflicts:
        raise FileExistsError(
            "Stage 2 output directory already contains experiment results: {}; "
            "conflicts: {}. Use --allow-log-overwrite only when reuse is "
            "intentional.".format(output_dir, ", ".join(conflicts)))
    return conflicts


def _resolve_project_path(path):
    if not path or osp.isabs(path):
        return path
    return osp.join(osp.dirname(osp.abspath(__file__)), path)


def _resolve_sub_batch_sizes(args):
    if args.far_batch_size is None and args.near_batch_size is None:
        near_batch_size = args.batch_size // 2
        far_batch_size = args.batch_size - near_batch_size
    elif args.far_batch_size is not None and args.near_batch_size is not None:
        far_batch_size = args.far_batch_size
        near_batch_size = args.near_batch_size
    else:
        raise ValueError(
            "--far-batch-size and --near-batch-size must be provided together")
    if far_batch_size + near_batch_size != args.batch_size:
        raise ValueError(
            "far_batch_size + near_batch_size must equal --batch-size so the "
            "total aerial budget matches baseline; got {} + {} != {}".format(
                far_batch_size, near_batch_size, args.batch_size))
    if far_batch_size < args.num_instances or near_batch_size < args.num_instances:
        raise ValueError(
            "Far/Near batch sizes must each be >= --num-instances; got {}/{}/{}".format(
                far_batch_size, near_batch_size, args.num_instances))
    return far_batch_size, near_batch_size


def _validate_args(args):
    if not 0.0 < args.view_near_ratio < 1.0:
        raise ValueError("--view-near-ratio must be in (0, 1)")
    if args.chain_cross_weight < 0.0:
        raise ValueError("--chain-cross-weight must be non-negative")
    if not 0.0 <= args.agva_probability <= 1.0:
        raise ValueError("--agva-probability must be in [0, 1]")
    for name, value in (("--agva-far-probability",
                         args.agva_far_probability),
                        ("--agva-near-probability",
                         args.agva_near_probability)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("{} must be in [0, 1]".format(name))
    if args.use_agva:
        if not args.agva_dir:
            raise ValueError("--agva-dir is required when --use-agva is enabled")
        if not osp.isdir(args.agva_dir):
            raise FileNotFoundError(
                "AGVA root not found: {}".format(args.agva_dir))
    if args.stage2_aerial_root and not osp.isdir(args.stage2_aerial_root):
        raise FileNotFoundError(
            "Stage 2 enhanced aerial root not found: {}".format(
                args.stage2_aerial_root))
    _resolve_sub_batch_sizes(args)


def _load_trusted_checkpoint(path):
    base = _require_baseline()
    path = osp.abspath(osp.expanduser(path))
    if not osp.isfile(path):
        raise FileNotFoundError("Stage 1 checkpoint not found: {}".format(path))
    try:
        checkpoint = base.load_checkpoint(path)
    except Exception as error:
        if "Weights only load failed" not in str(error):
            raise
        print("PyTorch weights_only compatibility retry for trusted Stage 1 checkpoint")
        try:
            checkpoint = torch.load(
                path, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
    print("Stage 1 checkpoint loaded from: {}".format(path))
    return checkpoint


def _prepare_view_groups(args, dataset_ir):
    score_path = _resolve_project_path(args.view_score_file)
    score_map, csv_stats = load_view_score_map(
        score_path, score_column=args.view_score_column,
        near_ratio=args.view_near_ratio, weight_mode="hard_top",
        path_mode=args.view_score_path_mode, relative_root=args.data_dir)
    aerial_train = sorted(dataset_ir.train)
    matched_map, match_stats = match_view_score_map(
        [sample[0] for sample in aerial_train], score_map,
        path_mode=args.view_score_path_mode, relative_root=args.data_dir,
        image_root=dataset_ir.images_dir)
    if match_stats["missing"]:
        raise ValueError(
            "LAGPeR view-score CSV must match every aerial training image; "
            "matched={}, missing={}, first_missing={}".format(
                match_stats["matched"], match_stats["missing"],
                match_stats["missing_paths"][0]))
    far_train, near_train = split_aerial_trainset(aerial_train, matched_map)
    print("========== Three-domain view split ==========")
    print("View score file: {}".format(score_path))
    print("View score column: {}".format(args.view_score_column))
    print("View score path mode: {}".format(args.view_score_path_mode))
    print("Per-camera near ratio: {:.6f}".format(args.view_near_ratio))
    print("CSV rows: {}".format(csv_stats["csv_rows"]))
    print("Matched aerial images: {}".format(match_stats["matched"]))
    print("A_far images: {}".format(len(far_train)))
    print("A_near images: {}".format(len(near_train)))
    print("=============================================")
    return far_train, near_train, matched_map

def _build_agva_candidate_map(aerial_train, args):
    if not args.use_agva:
        return {sample[0]: sample[0] for sample in aerial_train}

    agva_train_dir = osp.join(
        args.agva_dir, "aerial_modify", str(args.trial),
        "bounding_box_train")

    def candidate_builder(fname):
        return osp.join(agva_train_dir, osp.basename(fname))

    candidate_map, missing = build_candidate_map(
        aerial_train, candidate_builder, require_complete=True)
    print("AGVA coverage: {}/{} aerial images".format(
        len(candidate_map) - len(missing), len(candidate_map)))
    return candidate_map


def _cluster_once(clusterer, distances, name):
    """Run fixed-eps DBSCAN, matching the original LAG Stage 2 protocol."""
    labels = clusterer.fit_predict(distances)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if num_clusters == 0:
        raise RuntimeError(
            "Stage2 {} clustering produced zero clusters at fixed eps={:.3f}".format(
                name, clusterer.eps))
    return labels, num_clusters

def _extract_ordered_features(model, dataset, trainset, args, mode):
    loader = baseline.get_test_loader(
        dataset, args.height, args.width, args.test_batch, args.workers,
        testset=trainset)
    feature_map, _ = extract_features(
        model, loader, print_freq=50, mode=mode)
    return torch.cat([
        feature_map[fname].unsqueeze(0) for fname, _, _ in trainset
    ], dim=0)


def _create_hard_memory(model, args, labels, features, ema_features):
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    centers = generate_cluster_features(labels, features)
    samples = generate_random_features(
        labels, ema_features, num_clusters, args.num_instances)
    memory = ClusterMemory(
        model.module.num_features, num_clusters, temp=args.temp,
        momentum=args.momentum, mode="CMhard", smooth=args.smooth,
        num_instances=args.num_instances).cuda()
    memory.features = F.normalize(centers.repeat(2, 1), dim=1).cuda()
    memory.features_ema = F.normalize(samples, dim=1).cuda()
    return memory, centers, num_clusters


def _make_pseudo_dataset(samples, labels):
    output = []
    for (fname, _, cid), raw_label in zip(samples, labels):
        label = int(raw_label)
        if label != -1:
            output.append((fname, label, cid))
    return output


def _build_transforms(args):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ground = T.Compose([
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalizer,
        ChannelRandomErasing(probability=0.5)])
    ground_aug = T.Compose([
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalizer,
        ChannelRandomErasing(probability=0.5), ChannelExchange(gray=2)])
    aerial = T.Compose([
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)), T.RandomHorizontalFlip(),
        T.ToTensor(), normalizer, ChannelRandomErasing(probability=0.5),
        ChannelAdapGray(probability=0.5)])
    return ground, ground_aug, aerial


def _require_nonempty(name, dataset):
    if not dataset:
        raise RuntimeError("{} pseudo-labeled dataset is empty".format(name))


def _evaluate(model_ema, args, data_dir, epoch, best_r1, best_map,
              best_epoch):
    args.test_batch = 64
    args.img_w = args.width
    args.img_h = args.height
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transform_test = T.Compose([
        T.ToPILImage(), T.Resize((args.img_h, args.img_w)),
        T.ToTensor(), normalize])
    query_img, query_label = baseline.process_test_lag(
        data_dir, trial=args.trial, modal="visible")
    gallery_img, gallery_label = baseline.process_test_lag(
        data_dir, trial=args.trial, modal="thermal")
    gallery_set = baseline.TestData(
        gallery_img, gallery_label, transform=transform_test,
        img_size=(args.img_w, args.img_h))
    gallery_loader = data.DataLoader(
        gallery_set, batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers)
    query_set = baseline.TestData(
        query_img, query_label, transform=transform_test,
        img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(
        query_set, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_features = baseline.extract_query_feat(
        model_ema, query_loader, len(query_label))
    gallery_features = baseline.extract_gall_feat(
        model_ema, gallery_loader, len(gallery_label))
    distances = np.matmul(query_features, np.transpose(gallery_features))
    cmc, mean_ap, minp = baseline.eval_regdb(
        -distances, query_label, gallery_label)
    print("Test Trial: {}".format(args.trial))
    print("FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| "
          "Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}".format(
              cmc[0], cmc[4], cmc[9], cmc[19], mean_ap, minp))
    is_best = cmc[0] > best_r1
    if is_best:
        best_r1 = cmc[0]
        best_map = mean_ap
        best_epoch = epoch
    save_checkpoint({
        "state_dict": model_ema.state_dict(), "epoch": epoch + 1,
        "best_mAP": best_map,
    }, is_best, fpath=osp.join(args.logs_dir, "checkpoint.pth.tar"))
    print("\n * Finished epoch {:3d}   model R1: {:5.1%}  model mAP: "
          "{:5.1%}   best R1: {:5.1%}   best mAP: {:5.1%}"
          "(best_epoch:{})\n".format(
              epoch, cmc[0], mean_ap, best_r1, best_map, best_epoch))
    return best_r1, best_map, best_epoch


def main_worker_stage2(args):
    _require_baseline()
    start_time = time.monotonic()
    data_dir = args.data_dir
    args.memorybank = "CMhard"
    args.logs_dir = get_stage2_output_directory(
        args.stage2_log_name, args.trial, logs_root=args.logs_root)
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, "{}log.txt".format(args.trial)))
    print("==========\nArgs:{}\n==========".format(args))

    print("==> Load unlabeled dataset")
    dataset_ir = baseline.get_data(
        "lag_ir", args.data_dir, trial=args.trial)
    dataset_ground = baseline.get_data(
        "lag_rgb", args.data_dir, trial=args.trial)
    source_aerial_train = sorted(dataset_ir.train)
    source_aerial_domain_flags = domain_flags_for_samples(
        dataset_ir, source_aerial_train)
    if args.stage2_aerial_root:
        original_count = len(dataset_ir.train)
        dataset_ir.train = remap_samples_to_enhanced_root(
            dataset_ir.train, args.stage2_aerial_root)
        print("Stage 2 aerial domain: ENHANCED")
        print("Stage 2 enhanced aerial root: {}".format(args.stage2_aerial_root))
        print("Stage 2 enhanced aerial coverage: {}/{}".format(
            len(dataset_ir.train), original_count))
    aerial_train = sorted(dataset_ir.train)
    aerial_domain_flags = carry_domain_flags_by_basename(
        source_aerial_train, source_aerial_domain_flags, aerial_train)
    agva_candidate_map = _build_agva_candidate_map(aerial_train, args)
    ground_train = sorted(dataset_ground.train)
    far_train, near_train, score_map = _prepare_view_groups(args, dataset_ir)
    far_paths = {sample[0] for sample in far_train}
    near_paths = {sample[0] for sample in near_train}
    ground_domain_flags = domain_flags_for_samples(dataset_ground, ground_train)
    aerial_subdomain_flags = build_aerial_subdomain_flags(
        aerial_train, far_paths, near_paths)
    all_rows = concatenate_domain_rows(
        (ground_train, ground_domain_flags),
        (aerial_train, aerial_domain_flags))
    all_train = list(all_rows.samples)
    all_domain_flags = all_rows.flags
    all_subdomain_flags = validate_subdomain_flags(
        ([SUBDOMAIN_GROUND] * len(ground_train)
         + list(aerial_subdomain_flags)),
        all_domain_flags, "three-domain ALL rows")
    far_indexes = torch.tensor([
        index for index, subdomain in enumerate(aerial_subdomain_flags)
        if subdomain == SUBDOMAIN_AERIAL_FAR], dtype=torch.long)
    near_indexes = torch.tensor([
        index for index, subdomain in enumerate(aerial_subdomain_flags)
        if subdomain == SUBDOMAIN_AERIAL_NEAR], dtype=torch.long)

    far_batch_size, near_batch_size = _resolve_sub_batch_sizes(args)
    dataset_far_ratio = len(far_train) / float(len(aerial_train))
    far_weight = (dataset_far_ratio
                  if args.aerial_loss_weight_mode == "dataset" else 0.5)
    print("Far batch size: {}".format(far_batch_size))
    print("Near batch size: {}".format(near_batch_size))
    print("Ground batch size: {}".format(args.batch_size))
    print("Aerial samples/iteration: {} (baseline={})".format(
        far_batch_size + near_batch_size, args.batch_size))
    print("Aerial loss weight mode: {}".format(
        args.aerial_loss_weight_mode))
    print("Far/near loss weights: {:.6f}/{:.6f}".format(
        far_weight, 1.0 - far_weight))

    model, model_ema = baseline.create_model(args)
    checkpoint = _load_trusted_checkpoint(args.stage1_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model_ema.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(
        [{"params": [value]} for _, value in model.named_parameters()
         if value.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.1)
    trainer = ClusterContrastTrainer_PCLMP_ThreeDomain(
        model, model_ema, far_weight=far_weight,
        chain_cross_weight=args.chain_cross_weight)

    cluster_far = DBSCAN(eps=args.eps, min_samples=4,
                         metric="precomputed", n_jobs=-1)
    cluster_near = DBSCAN(eps=args.eps, min_samples=4,
                          metric="precomputed", n_jobs=-1)
    cluster_ground = DBSCAN(eps=args.eps, min_samples=4,
                            metric="precomputed", n_jobs=-1)
    cluster_all = DBSCAN(eps=args.eps, min_samples=4,
                         metric="precomputed", n_jobs=-1)
    print("Far/Near/Ground/ALL clustering initial eps: {:.3f}".format(args.eps))

    best_r1 = 0.0
    best_map = 0.0
    best_epoch = -1
    iters = args.iters if args.iters > 0 else None

    for epoch in range(args.epochs):
        with torch.no_grad():
            print("==> Extract Ground features")
            features_ground_ema = _extract_ordered_features(
                model_ema, dataset_ground, ground_train, args, mode=1)
            features_ground = _extract_ordered_features(
                model, dataset_ground, ground_train, args, mode=1)
            print("==> Extract Aerial features once, then split Far/Near")
            features_aerial_ema = _extract_ordered_features(
                model_ema, dataset_ir, aerial_train, args, mode=2)
            features_aerial = _extract_ordered_features(
                model, dataset_ir, aerial_train, args, mode=2)
            device = features_aerial.device
            epoch_far_indexes = far_indexes.to(device)
            epoch_near_indexes = near_indexes.to(device)
            features_far = features_aerial.index_select(
                0, epoch_far_indexes)
            features_near = features_aerial.index_select(
                0, epoch_near_indexes)
            features_far_ema = features_aerial_ema.index_select(
                0, epoch_far_indexes)
            features_near_ema = features_aerial_ema.index_select(
                0, epoch_near_indexes)

            labels_far, num_far = cluster_precomputed_ephemeral(
                lambda: baseline.compute_jaccard_distance(
                    features_far, k1=args.k1, k2=args.k2, search_option=3),
                cluster_far, "A_far", require_nonempty=True)
            labels_near, num_near = cluster_precomputed_ephemeral(
                lambda: baseline.compute_jaccard_distance(
                    features_near, k1=args.k1, k2=args.k2, search_option=3),
                cluster_near, "A_near", require_nonempty=True)
            labels_ground, num_ground = cluster_precomputed_ephemeral(
                lambda: baseline.compute_jaccard_distance(
                    features_ground, k1=args.k1, k2=args.k2, search_option=3),
                cluster_ground, "Ground", require_nonempty=True)

            features_all = torch.cat(
                (features_ground, features_aerial), dim=0)
            labels_all, num_all = cluster_precomputed_ephemeral(
                lambda: baseline.compute_modal_invariant_jaccard_distance(
                    features_all, k1=40, k2=32, file=all_train,
                    search_option=3, domain_flags=all_domain_flags),
                cluster_all, "ALL", require_nonempty=True)

        memory_far, centers_far, num_far = _create_hard_memory(
            model, args, labels_far, features_far, features_far_ema)
        memory_near, centers_near, num_near = _create_hard_memory(
            model, args, labels_near, features_near, features_near_ema)
        memory_ground, centers_ground, num_ground = _create_hard_memory(
            model, args, labels_ground, features_ground,
            features_ground_ema)
        centers_all = generate_joint_cluster_features(
            labels_all, num_all, features_all, all_domain_flags)
        memory_all = ClusterMemory(
            model.module.num_features, num_all, temp=args.temp,
            momentum=args.momentum, mode="CMhybrid", smooth=args.smooth,
            num_instances=args.num_instances).cuda()
        memory_all.features = F.normalize(
            centers_all.repeat(2, 1), dim=1).cuda()
        trainer.memory_far = memory_far
        trainer.memory_near = memory_near
        trainer.memory_ground = memory_ground
        trainer.memory_all = memory_all
        del features_all, features_aerial, features_ground
        del features_aerial_ema, features_ground_ema
        del features_far, features_near, features_far_ema, features_near_ema

        print("==> Clusters epoch {}: Far={}, Near={}, Ground={}, ALL={}".format(
            epoch, num_far, num_near, num_ground, num_all))
        print("Start chain Bipartite Graph Matching")
        far_to_near, near_to_far, fn_stats = build_bidirectional_bgm(
            centers_far, centers_near)
        near_to_ground, ground_to_near, ng_stats = build_bidirectional_bgm(
            centers_near, centers_ground)
        print("Far<->Near BGM: {}".format(fn_stats))
        print("Near<->Ground BGM: {}".format(ng_stats))
        print("Direct Far<->Ground BGM: disabled")

        if args.use_agva and args.agva_selection_mode == "camera_group":
            selected_agva_paths, epoch_agva = \
                select_stratified_epoch_paths(
                    aerial_train, agva_candidate_map, far_paths, near_paths,
                    args.agva_probability, args.seed + epoch,
                    group_probabilities={
                        "Far": (args.agva_probability if
                                args.agva_far_probability is None else
                                args.agva_far_probability),
                        "Near": (args.agva_probability if
                                 args.agva_near_probability is None else
                                 args.agva_near_probability),
                    })
        else:
            selected_agva_paths, epoch_agva = select_epoch_paths(
                agva_candidate_map,
                args.agva_probability if args.use_agva else 0.0,
                args.seed + epoch)
        if args.stage2_aerial_root:
            print("Epoch aerial source: fixed enhanced domain ({} images)".format(
                len(selected_agva_paths)))
        else:
            print("Epoch AGVA shared selection: {}".format(epoch_agva))
        pseudo_far = _make_pseudo_dataset(far_train, labels_far)
        pseudo_near = _make_pseudo_dataset(near_train, labels_near)
        pseudo_far = remap_pseudo_dataset(pseudo_far, selected_agva_paths)
        pseudo_near = remap_pseudo_dataset(pseudo_near, selected_agva_paths)
        pseudo_ground = [
            (fname, int(label), cid)
            for (fname, _, cid), label in zip(ground_train, labels_ground)
            if int(label) != -1]
        valid_joint_clusters = joint_cross_modal_flags(
            labels_all, all_domain_flags)
        print("ALL cross-modal clusters: {}/{}".format(
            len(valid_joint_clusters), num_all))
        joint_far = []
        joint_near = []
        joint_ground = []
        for ((fname, _, cid), raw_label), subdomain in zip(
                zip(all_train, labels_all), all_subdomain_flags):
            label = int(raw_label)
            if label not in valid_joint_clusters:
                continue
            if subdomain == SUBDOMAIN_AERIAL_FAR:
                joint_far.append((fname, label, cid))
            elif subdomain == SUBDOMAIN_AERIAL_NEAR:
                joint_near.append((fname, label, cid))
            elif subdomain == SUBDOMAIN_GROUND:
                joint_ground.append((fname, label, cid))
            else:
                raise AssertionError("Unhandled subdomain flag: {}".format(subdomain))
        joint_far = remap_pseudo_dataset(joint_far, selected_agva_paths)
        joint_near = remap_pseudo_dataset(joint_near, selected_agva_paths)

        for name, pseudo_set in (
                ("Far", pseudo_far), ("Near", pseudo_near),
                ("Ground", pseudo_ground), ("ALL Far", joint_far),
                ("ALL Near", joint_near),
                ("ALL Ground", joint_ground)):
            _require_nonempty(name, pseudo_set)

        ground_transform, ground_aug_transform, aerial_transform = \
            _build_transforms(args)
        loader_far = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, far_batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_far, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_near = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, near_batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_near, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_ground = baseline.get_train_loader_color(
            args, dataset_ground, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_ground, no_cam=args.no_cam,
            train_transformer=ground_transform,
            train_transformer1=ground_aug_transform)
        loader_all_far = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, far_batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_far, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_all_near = baseline.get_train_loader_ir(
            args, dataset_ir, args.height, args.width, near_batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_near, no_cam=args.no_cam,
            train_transformer=aerial_transform)
        loader_all_ground = baseline.get_train_loader_color(
            args, dataset_ground, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_ground, no_cam=args.no_cam,
            train_transformer=ground_transform,
            train_transformer1=ground_aug_transform)
        loaders = (loader_far, loader_near, loader_ground,
                   loader_all_far, loader_all_near, loader_all_ground)
        for loader in loaders:
            loader.new_epoch()

        trainer.train(
            epoch, *loaders, optimizer,
            print_freq=args.print_freq, train_iters=len(loader_far),
            far_to_near=far_to_near, near_to_far=near_to_far,
            near_to_ground=near_to_ground,
            ground_to_near=ground_to_near)
        best_r1, best_map, best_epoch = _evaluate(
            model_ema, args, data_dir, epoch,
            best_r1, best_map, best_epoch)
        lr_scheduler.step()

    print("Total running time: {}".format(
        timedelta(seconds=time.monotonic() - start_time)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Released LAGPeR three-domain Stage 2 (no EMA loss)")
    parser.add_argument("-d", "--dataset", type=str, default="lag_ir")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--num-instances", type=int, default=16)
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument("--eps-gap", type=float, default=0.02)
    parser.add_argument("--k1", type=int, default=30)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("-a", "--arch", type=str, default="agw")
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("-mb", "--memorybank", type=str, default="CMhard")
    parser.add_argument("--smooth", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--eval-step", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.05)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument("--data-dir", type=str,
                        default=osp.join(working_dir, "data"))
    parser.add_argument("--logs-dir", type=str,
                        default=osp.join(working_dir, "logs"))
    parser.add_argument("--pooling-type", type=str, default="gem")
    parser.add_argument("--use-hard", action="store_true")
    parser.add_argument("--no-cam", action="store_true")
    parser.add_argument("--stage2-only", action="store_true",
                        help="compatibility flag; this entry is Stage 2 only")
    parser.add_argument("--stage1-checkpoint", type=str, required=True)
    parser.add_argument("--stage2-log-name", type=str,
                        default=DEFAULT_STAGE2_LOG_NAME)
    parser.add_argument("--allow-log-overwrite", action="store_true")
    parser.add_argument("--view-score-file", type=str, required=True,
                        help="OOF LAGPeR aerial view-score CSV")
    parser.add_argument("--view-score-column", type=str,
                        default="p_ground_foreground")
    parser.add_argument("--view-score-path-mode", choices=VIEW_PATH_MODES,
                        default="basename")
    parser.add_argument("--view-near-ratio", type=float, default=0.50,
                        help="top ratio selected independently per camera")
    parser.add_argument("--far-batch-size", type=int, default=None)
    parser.add_argument("--near-batch-size", type=int, default=None)
    parser.add_argument("--aerial-loss-weight-mode",
                        choices=("dataset", "balanced"), default="dataset")
    parser.add_argument("--chain-cross-weight", type=float, default=0.25)
    # This release keeps the accepted three-domain experiment isolated:
    # Stage 2 always uses original aerial images and never offline AGVA.
    parser.set_defaults(
        use_agva=False,
        agva_dir='',
        agva_probability=0.0,
        agva_far_probability=None,
        agva_near_probability=None,
        agva_selection_mode='global',
        stage2_aerial_root='')
    return parser


def main(argv=None):
    args = build_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    _validate_args(args)
    args.logs_root = osp.abspath(osp.expanduser(args.logs_dir))
    output_dir = get_stage2_output_directory(
        args.stage2_log_name, args.trial, logs_root=args.logs_root)
    print("Experiment: THREE-DOMAIN STAGE 2 (NO EMA LOSS)")
    print("Stable LAG AGW/GeM: enabled")
    print("Stage 2 AGVA/offline replacement: disabled")
    print("Stage 1: skipped; explicit checkpoint required")
    print("Stage 2 log name: {}".format(args.stage2_log_name))
    print("Stage 2 output directory: {}".format(output_dir))
    check_stage2_output_directory(
        output_dir, allow_overwrite=args.allow_log_overwrite)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker_stage2(args)


if __name__ == "__main__":
    main()
