# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.methods.rahp import (
    RAHPSelectionStats, compute_rahp_reliability, disabled_rahp_diagnostics,
    format_rahp_epoch,
)
from clustercontrast.methods.cesa import (
    CESAState, disabled_cesa_diagnostics, format_cesa_epoch,
)
from clustercontrast.methods.cli import (
    add_method_arguments, experiment_tag, parse_bool, validate_method_args,
)
from clustercontrast.methods.checkpoint import (
    BEST_SELECTION_METRIC, capture_rng_state,
    resolve_stage1_initialization, restore_best_state, restore_rng_state,
    save_best_checkpoint, save_fixed_epoch_checkpoint, select_agreid_best,
    should_evaluate_during_train,
)
from clustercontrast.methods.evaluation import (
    evaluate_agreid_bidirectional, evaluate_agreid_distances,
    extract_agreid_features, load_agreid_split,
)
from clustercontrast.methods.pgm import build_total_pgm_mapping
from clustercontrast.trainers import ClusterContrastTrainer_DCL, ClusterContrastTrainer_PCLMP
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_modal_invariant_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter
def get_data(name, data_dir,trial=0):
    # Both AG-ReID modalities share one root.  The modality and trial are
    # selected inside agreid_ir/agreid_rgb, matching the server data tree.
    dataset = datasets.create(name, data_dir, trial=trial)
    return dataset

def map_to_agva_path(fname, original_root, agva_root):
    """
    将原始空中图片绝对路径映射到 AGVA 数据根目录。

    例如：
    原始：
    /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train/A.jpg

    输出：
    /home/lab338/Jaxon/dataset/AGreid_train_AGVA035/aerial_modify/1/bounding_box_train/A.jpg
    """
    relative_path = osp.relpath(fname, original_root)
    candidate_path = osp.join(agva_root, relative_path)

    if osp.isfile(candidate_path):
        return candidate_path, True

    return fname, False


def maybe_use_agva_path(fname, args, rng=None):
    """
    只针对空中训练图片，根据 agva_ir_prob 决定读取原图或 AGVA 图。

    参数:
        rng: 独立的随机数生成器（推荐），默认为 random 模块全局状态

    返回：
        (selected_path, used_agva, fallback_missing)
    """
    if not args.use_agva_ir:
        return fname, False, False

    # 只允许空中训练图进入 AGVA 路径映射
    if 'aerial_modify' not in fname:
        return fname, False, False

    if rng is None:
        rng = random

    if rng.random() >= args.agva_ir_prob:
        return fname, False, False

    selected_path, found = map_to_agva_path(
        fname,
        args.agva_orig_root,
        args.agva_ir_root,
    )

    if found:
        return selected_path, True, False

    # AGVA 文件不存在时安全回退原图，不能让训练中断
    return fname, False, True





class channel_select(object):
    def __init__(self,channel=0):
        self.channel = channel

    def __call__(self, img):
        if self.channel == 3:
            img_gray = img.convert('L')
            np_img = np.array(img_gray, dtype=np.uint8)
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        else:
            np_img = np.array(img, dtype=np.uint8)
            np_img = np_img[:,:,self.channel]
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        return img_PIL



def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):


    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    if not train_set:
        raise RuntimeError(
            'Cannot build IR training loader from an empty pseudo-labeled set')
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    sampler_size = len(sampler) if sampler is not None else len(train_set)
    drop_last = sampler_size >= batch_size
    if not drop_last:
        print('[LOADER] IR partial batch: sampler_size={} batch_size={}'.format(
            sampler_size, batch_size))
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=drop_last), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    if not train_set:
        raise RuntimeError(
            'Cannot build RGB training loader from an empty pseudo-labeled set')
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    sampler_size = len(sampler) if sampler is not None else len(train_set)
    drop_last = sampler_size >= batch_size
    if not drop_last:
        print('[LOADER] RGB partial batch: sampler_size={} batch_size={}'.format(
            sampler_size, batch_size))
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=drop_last), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=drop_last), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type,
                          pretrained_path=args.pretrained_resnet50)
    model_ema = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type,
                          pretrained_path=args.pretrained_resnet50)
    # use CUDA
    model.cuda()
    model_ema.cuda()
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    return model, model_ema




class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def extract_gall_feat(model,gall_loader,ngall):
    return extract_agreid_features(model, gall_loader, ngall, modal=2)
    
def extract_query_feat(model,query_loader,nquery):
    return extract_agreid_features(model, query_loader, nquery, modal=1)


def process_test_agreid(img_dir, trial=1, modal='ground'):
    return load_agreid_split(img_dir, trial, modal)
def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    return evaluate_agreid_distances(distmat, q_pids, g_pids, max_rank)


def _print_agreid_metrics(stage, epoch, direction, metrics):
    print('[EVAL][{}][Epoch {:03d}][{}]'.format(
        stage, epoch, direction.upper()))
    print('Rank-1={rank1:.4f} Rank-5={rank5:.4f} '
          'Rank-10={rank10:.4f} Rank-20={rank20:.4f} '
          'mAP={mAP:.4f} mINP={mINP:.4f}'.format(**metrics))


def evaluate_agreid_for_training(model, args, data_path, trial):
    """Run canonical bidirectional evaluation without changing training state."""
    rng_state = capture_rng_state()
    was_training = model.training
    try:
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        transform_test = T.Compose([
            T.ToPILImage(),
            T.Resize((args.height, args.width)),
            T.ToTensor(),
            normalize,
        ])
        aerial_images, aerial_labels = process_test_agreid(
            data_path, trial=trial, modal='aerial')
        ground_images, ground_labels = process_test_agreid(
            data_path, trial=trial, modal='ground')
        aerial_loader = data.DataLoader(
            TestData(aerial_images, aerial_labels, transform=transform_test,
                     img_size=(args.width, args.height)),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers)
        ground_loader = data.DataLoader(
            TestData(ground_images, ground_labels, transform=transform_test,
                     img_size=(args.width, args.height)),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers)

        model.eval()
        with torch.no_grad():
            return evaluate_agreid_bidirectional(
                model, aerial_loader, aerial_labels,
                ground_loader, ground_labels)
    finally:
        model.train(was_training)
        restore_rng_state(rng_state)


def associated_analysis_for_all(all_origin, num_ground_samples, log_dir):
    del log_dir
    all_label_set = sorted(label for label in set(all_origin) if label != -1)
    associate = 0
    flag_ir_list = collections.defaultdict(int)
    flag_rgb_list = collections.defaultdict(int)
    for label in all_label_set:
        indexes = np.flatnonzero(all_origin == label)
        has_ground = bool(np.any(indexes < num_ground_samples))
        has_aerial = bool(np.any(indexes >= num_ground_samples))
        flag_rgb_list[int(label)] = int(has_ground)
        flag_ir_list[int(label)] = int(has_aerial)
        if has_ground and has_aerial:
            associate += 1

    associate_rate = associate / len(all_label_set) if all_label_set else 0.0
    print('associate rate', associate_rate)

    return flag_ir_list, flag_rgb_list

def main():
    args = parser.parse_args()
    if args.resume:
        args.stage2_only = True
    validate_method_args(args)
    args.experiment_tag = experiment_tag(args)
    if args.dry_run:
        print('[CONFIG]\ndataset=AG-ReID\nformal_protocol=A-to-G,G-to-A\n'
              'arch={}\nmemorybank={}\neps={}\nstage1_init={}\n'
              'checkpoint=fixed-final+best\n'
              'eval_during_train={}\nrahp={}\ncesa={}'.format(
                  args.arch, args.memorybank, args.eps, args.stage1_init,
                  args.eval_during_train, args.use_rahp, args.use_cesa))
        return
    # ========== AGVA 参数合法性检查 ==========
    if not (0.0 <= args.agva_ir_prob <= 1.0):
        raise ValueError(
            "--agva-ir-prob must be in [0.0, 1.0], got {}".format(args.agva_ir_prob)
        )

    if args.use_agva_ir:
        if not osp.isdir(args.agva_orig_root):
            raise FileNotFoundError(
                "AGVA original root not found: {}".format(args.agva_orig_root)
            )
        if not osp.isdir(args.agva_ir_root):
            raise FileNotFoundError(
                "AGVA IR root not found: {}".format(args.agva_ir_root)
            )

    print("========== AGVA Configuration ==========")
    print("Enabled:", args.use_agva_ir)
    print("Original root:", args.agva_orig_root)
    print("AGVA root:", args.agva_ir_root)
    print("AGVA probability:", args.agva_ir_prob)
    print("========================================")



    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    log_s1_name = 'agreid_s1' + ('_rahp' if args.use_rahp else '')
    log_s2_name = 'agreid_s2' + ('' if args.experiment_tag == 'baseline'
                                else '_' + args.experiment_tag)
    logs_root = args.logs_dir
    if not args.stage2_only:
        main_worker_stage1(args,log_s1_name)
    args.logs_dir = logs_root
    main_worker_stage2(args,log_s1_name,log_s2_name)

def main_worker_stage1(args,log_s1_name):
    logs_dir_root = osp.join(args.logs_dir+'/'+log_s1_name)
    data_dir = args.data_dir
    trial = args.trial
    start_epoch =0
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    # AG-ReID: aerial (C00) -> IR, ground (C03) -> RGB
    dataset_ir = get_data('agreid_ir', args.data_dir,trial=trial)
    dataset_rgb = get_data('agreid_rgb', args.data_dir,trial=trial)


    # ========== AGVA 路径映射自检 ==========
    if args.use_agva_ir and len(dataset_ir.train) > 0:
        first_fname = dataset_ir.train[0][0]
        first_agva, _ = map_to_agva_path(first_fname, args.agva_orig_root, args.agva_ir_root)
        print("[AGVA Self-Check] original: {}".format(first_fname))
        print("[AGVA Self-Check] mapped:   {}".format(first_agva))
        print("[AGVA Self-Check] exists:   {}".format(osp.isfile(first_agva)))
        if not osp.isfile(first_agva):
            raise RuntimeError(
                "AGVA path mapping failed for first image! "
                "Check --agva-orig-root and --agva-ir-root.\n"
                "  original: {}\n  agva:     {}".format(first_fname, first_agva)
            )


    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model
    model, _ = create_model(args)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer_DCL(model)
    best_R1 = float('-inf')
    best_epoch = None

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = args.eps
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.eps
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)


            # Adaptive clustering for IR: if no clusters found, increase eps gradually
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                if num_cluster_ir > 0:
                    break
                cluster_ir.eps += 0.05
                print('IR: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_ir.eps))
                attempt += 1
            if num_cluster_ir == 0:
                print('WARNING: IR clustering failed after {} attempts, assigning all samples to cluster 0'.format(max_attempts))
                pseudo_labels_ir = np.zeros(len(features_ir), dtype=np.int32)
                num_cluster_ir = 1

            # Adaptive clustering for RGB
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)
            cluster_rgb.eps = getattr(cluster_rgb, 'eps', 0.3)
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                if num_cluster_rgb > 0:
                    break
                cluster_rgb.eps += 0.05
                print('RGB: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_rgb.eps))
                attempt += 1
            if num_cluster_rgb == 0:
                print('WARNING: RGB clustering failed after {} attempts, assigning all samples to cluster 0'.format(max_attempts))
                pseudo_labels_rgb = np.zeros(len(features_rgb), dtype=np.int32)
                num_cluster_rgb = 1

            del rerank_dist_rgb
            del rerank_dist_ir

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            if len(centers) == 0:
                return torch.empty(0, features.size(1), device=features.device)

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                                  momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                  num_instances=args.num_instances).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()
        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        elif args.memorybank == 'CMhybrid':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        rahp_diags = []
        rahp_stats = RAHPSelectionStats() if args.use_rahp else None
        q_ir_full = q_rgb_full = None
        if args.use_rahp:
            q_ir_full, diag = compute_rahp_reliability(
                features_ir, pseudo_labels_ir, knn=args.rahp_knn,
                alpha=args.rahp_alpha, return_diagnostics=True)
            rahp_diags.append(diag)
            q_rgb_full, diag = compute_rahp_reliability(
                features_rgb, pseudo_labels_rgb, knn=args.rahp_knn,
                alpha=args.rahp_alpha, return_diagnostics=True)
            rahp_diags.append(diag)
        else:
            rahp_diags.extend((disabled_rahp_diagnostics(pseudo_labels_ir),
                               disabled_rahp_diagnostics(pseudo_labels_rgb)))

        

        # ========== Stage1 AGVA: 空中 loss 训练数据集（仅 loss-loader 使用 AGVA 图片） ==========
        pseudo_labeled_dataset_ir = []
        ir_label=[]

        stage1_agva_count = 0
        stage1_original_count = 0
        stage1_fallback_count = 0
        agva_rng = random.Random(args.seed + epoch)  # 独立 AGVA RNG，不影响 baseline sampler
        _agva_map_printed = 0  # 每个 epoch 最多打印 3 个路径映射示例

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                train_fname, used_agva, fallback_missing = maybe_use_agva_path(
                    fname, args, agva_rng
                )

                if used_agva and _agva_map_printed < 3:
                    print("[AGVA MAP]")
                    print("original: {}".format(fname))
                    print("selected: {}".format(train_fname))
                    _agva_map_printed += 1

                pseudo_labeled_dataset_ir.append(
                    (train_fname, label.item(), cid)
                )
                ir_label.append(label.item())

                if used_agva:
                    stage1_agva_count += 1
                else:
                    stage1_original_count += 1

                if fallback_missing:
                    stage1_fallback_count += 1

        print(
            "[Stage1 AGVA] augmented={}, original={}, missing_fallback={}".format(
                stage1_agva_count,
                stage1_original_count,
                stage1_fallback_count,
            )
        )
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        # DELME1
        pseudo_labeled_dataset_ir = []
        ir_label=[]
        q_ir_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
                if args.use_rahp:
                    q_ir_filtered.append(q_ir_full[i])
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        q_rgb_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
                if args.use_rahp:
                    q_rgb_filtered.append(q_rgb_full[i])

        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ########################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5)
        ])

        transform_thermal = T.Compose( [
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        args.batch_size//2, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir),
                      rahp_ir=q_ir_filtered if args.use_rahp else None,
                      rahp_rgb=q_rgb_filtered if args.use_rahp else None,
                      rahp_beta=args.rahp_beta, rahp_stats=rahp_stats)
        print(format_rahp_epoch(rahp_diags, rahp_stats, args.use_rahp))

        eval_a2g = None
        eval_g2a = None
        best_updated = False
        if should_evaluate_during_train(args, epoch):
            eval_a2g, eval_g2a = evaluate_agreid_for_training(
                model, args, data_dir, trial)
            _print_agreid_metrics('Stage1', epoch + 1, 'a2g', eval_a2g)
            _print_agreid_metrics('Stage1', epoch + 1, 'g2a', eval_g2a)
            best_R1, best_epoch, best_updated = select_agreid_best(
                eval_a2g, eval_g2a, best_R1, best_epoch, epoch + 1)
            print('[BEST][Stage1]\nselection_metric={} current={:.4f} '
                  'best={:.4f} best_epoch={} updated={}'.format(
                      BEST_SELECTION_METRIC, eval_g2a['rank1'], best_R1,
                      best_epoch, best_updated))

        checkpoint_state = {
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_R1': best_R1,
            'best_epoch': best_epoch,
            'selection_metric': BEST_SELECTION_METRIC,
            'eval_g2a': eval_g2a,
            'eval_a2g': eval_a2g,
        }
        if best_updated:
            best_path = save_best_checkpoint(checkpoint_state, args.logs_dir)
            print('[CHECKPOINT][Stage1]\nbest epoch={} metric={} path={}'.format(
                best_epoch, BEST_SELECTION_METRIC, best_path))
        _, final_path = save_fixed_epoch_checkpoint(
            checkpoint_state, args.logs_dir, epoch + 1 == args.epochs)
        print('[CHECKPOINT][Stage1] latest epoch={} policy=fixed-final{}'.format(
            epoch + 1, ' final={}'.format(final_path) if final_path else ''))
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


def main_worker_stage2(args,log_s1_name,log_s2_name):
    logs_root = args.logs_dir
    logs_dir_root = osp.join(logs_root, log_s2_name)
    trial = args.trial
    start_epoch =0
    args.memorybank = 'CMhard'
    data_dir = args.data_dir
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    stage1_directory = osp.join(logs_root, log_s1_name, str(trial))
    stage1_checkpoint = resolve_stage1_initialization(
        stage1_directory, args.stage1_init, stage2_resume=bool(args.resume))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('agreid_ir', args.data_dir,trial=trial)
    dataset_rgb = get_data('agreid_rgb', args.data_dir,trial=trial)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model
    model, model_ema = create_model(args)
    if stage1_checkpoint is not None:
        print('[INIT][Stage2] stage1_source={} path={}'.format(
            args.stage1_init, stage1_checkpoint))
        checkpoint = load_checkpoint(stage1_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        model_ema.load_state_dict(checkpoint['state_dict'])
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # Trainer
    trainer = ClusterContrastTrainer_PCLMP(model, model_ema)
    cesa_state = (CESAState(rho=args.cesa_rho, eta=args.cesa_eta,
                            lineage_threshold=args.cesa_lineage_thr,
                            warmup=args.cesa_warmup) if args.use_cesa else None)
    resume_cluster_eps = None
    best_R1 = float('-inf')
    best_epoch = None

    if args.resume:
        resumed = load_checkpoint(args.resume)
        model.load_state_dict(resumed.get('model_state_dict', resumed['state_dict']))
        model_ema.load_state_dict(resumed['state_dict'])
        if 'optimizer_state_dict' in resumed:
            optimizer.load_state_dict(resumed['optimizer_state_dict'])
        if 'scheduler_state_dict' in resumed:
            lr_scheduler.load_state_dict(resumed['scheduler_state_dict'])
            lr_scheduler.step()  # checkpoint is saved before the end-of-epoch step
        start_epoch = int(resumed['epoch'])
        resume_cluster_eps = resumed.get('dbscan_eps')
        best_R1, best_epoch = restore_best_state(resumed)
        restore_rng_state(resumed.get('rng_state'))
        if cesa_state is not None:
            cesa_state.load_state_dict(resumed.get('cesa_state'))

    for epoch in range(start_epoch, args.epochs):
        with torch.no_grad():
            if epoch == start_epoch:
                # DBSCAN cluster
                ir_eps = args.eps
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.eps
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                all_eps = args.eps
                print('All Clustering criterion: eps: {:.3f}'.format(all_eps))
                cluster_all = DBSCAN(eps=all_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                if resume_cluster_eps is not None:
                    cluster_ir.eps, cluster_rgb.eps, cluster_all.eps = resume_cluster_eps

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb_ema, _ = extract_features(model_ema, cluster_loader_rgb, print_freq=50, mode=1)
            features_rgb_ema = torch.cat([features_rgb_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir_ema, _ = extract_features(model_ema, cluster_loader_ir, print_freq=50, mode=2)
            features_ir_ema = torch.cat([features_ir_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            print('==> Create pseudo labels for unlabeled ALL data')
            features_all = torch.cat([features_rgb, features_ir], dim=0)
            
            # Adaptive clustering for IR
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                if num_cluster_ir > 0:
                    break
                cluster_ir.eps += 0.05
                print('Stage2 IR: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_ir.eps))
                attempt += 1
            if num_cluster_ir == 0:
                print('WARNING: Stage2 IR clustering failed, assigning all samples to cluster 0')
                pseudo_labels_ir = np.zeros(len(features_ir), dtype=np.int32)
                num_cluster_ir = 1

            # Adaptive clustering for RGB
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                if num_cluster_rgb > 0:
                    break
                cluster_rgb.eps += 0.05
                print('Stage2 RGB: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_rgb.eps))
                attempt += 1
            if num_cluster_rgb == 0:
                print('WARNING: Stage2 RGB clustering failed, assigning all samples to cluster 0')
                pseudo_labels_rgb = np.zeros(len(features_rgb), dtype=np.int32)
                num_cluster_rgb = 1

            # Adaptive clustering for ALL
            rerank_dist_all = compute_modal_invariant_jaccard_distance(
                features_all, k1=args.k1, k2=args.k2,
                file=sorted(dataset_rgb.train) + sorted(dataset_ir.train),
                modalities=(['rgb'] * len(features_rgb) +
                            ['ir'] * len(features_ir)),
                search_option=3)
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_all = cluster_all.fit_predict(rerank_dist_all)
                num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)
                if num_cluster_all > 0:
                    break
                cluster_all.eps += 0.05
                print('Stage2 ALL: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_all.eps))
                attempt += 1
            if num_cluster_all == 0:
                print('WARNING: Stage2 ALL clustering failed, assigning all samples to cluster 0')
                pseudo_labels_all = np.zeros(len(features_all), dtype=np.int32)
                num_cluster_all = 1

            del rerank_dist_rgb
            del rerank_dist_ir
            del rerank_dist_all

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            if len(centers) == 0:
                return torch.empty(0, features.size(1), device=features.device)

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers
        
        # generate new dataset and calculate all cluster centers
        @torch.no_grad()
        def generate_modal_invariant_cluster_features(labels, num_cluster_all, features, file):
            centers_IR = collections.defaultdict(list)
            centers_RBG = collections.defaultdict(list)
            centers_IR_mean = collections.defaultdict(list)
            centers_RBG_mean = collections.defaultdict(list)
            # centers_all = collections.defaultdict(list)
            for i, (label, (fname, _, cid)) in enumerate(zip(labels, file)):
                if label == -1:
                    continue
                if 'aerial_modify' in fname:
                    centers_IR[labels[i]].append(features[i])
                elif 'ground_modify' in fname:
                    centers_RBG[labels[i]].append(features[i])
                else:
                    raise AssertionError
            for i in range(num_cluster_all):
                if centers_RBG[i] != []:
                    centers_RBG_mean[i] = torch.stack(centers_RBG[i], dim=0).mean(0)
                if centers_IR[i] != []:
                    centers_IR_mean[i] = torch.stack(centers_IR[i], dim=0).mean(0)
            centers_all = []
            for i in range(num_cluster_all):
                if centers_RBG_mean[i] == []:
                    centers_all.append(centers_IR_mean[i])
                elif centers_IR_mean[i] == []:
                    centers_all.append(centers_RBG_mean[i])
                else:
                    centers_all.append(torch.mean(torch.stack([centers_RBG_mean[i], centers_IR_mean[i]], dim=0), dim=0))
            centers_all = torch.stack(centers_all, dim=0)

            return centers_all

        # generate instances features
        def generate_random_features(labels, features, num_cluster, num_instances):
            indexes = np.zeros(num_cluster * num_instances)
            for i in range(num_cluster):
                index = [i + k * num_cluster for k in range(num_instances)]
                samples = np.random.choice(np.where(labels == i)[0], num_instances, True)
                indexes[index] = samples
            memory_features = features[indexes]
            return memory_features        

        memory_features_ir = generate_random_features(pseudo_labels_ir, features_ir_ema, num_cluster_ir, args.num_instances)
        memory_features_rgb = generate_random_features(pseudo_labels_rgb, features_rgb_ema, num_cluster_rgb, args.num_instances)
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        cluster_features_all = generate_modal_invariant_cluster_features(pseudo_labels_all, num_cluster_all,
                                                                         features_all,
                                                                         sorted(dataset_rgb.train) + sorted(
                                                                             dataset_ir.train))
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                                  momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                  num_instances=args.num_instances).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()
        memory_all = ClusterMemory(model.module.num_features, num_cluster_all, temp=args.temp,
                                    momentum=args.momentum, mode='CMhybrid', smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()
        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
            memory_all.features = F.normalize(cluster_features_all, dim=1).cuda()
        elif args.memorybank == 'CMhybrid':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()
            memory_all.features = F.normalize(cluster_features_all.repeat(2, 1), dim=1).cuda()
        elif args.memorybank == 'CMhard':
            # Cluster proxies
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()
            memory_all.features = F.normalize(cluster_features_all.repeat(2,1), dim=1).cuda()
            # Instance proxies
            memory_ir.features_ema = F.normalize(memory_features_ir, dim=1).cuda()
            memory_rgb.features_ema = F.normalize(memory_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb
        trainer.memory_all = memory_all

        rahp_diags = []
        rahp_stats = RAHPSelectionStats() if args.use_rahp else None
        q_ir_full = q_rgb_full = q_all_full = None
        if args.use_rahp:
            q_ir_full, diag = compute_rahp_reliability(
                features_ir, pseudo_labels_ir, knn=args.rahp_knn,
                alpha=args.rahp_alpha, return_diagnostics=True)
            rahp_diags.append(diag)
            q_rgb_full, diag = compute_rahp_reliability(
                features_rgb, pseudo_labels_rgb, knn=args.rahp_knn,
                alpha=args.rahp_alpha, return_diagnostics=True)
            rahp_diags.append(diag)
            q_all_full, diag = compute_rahp_reliability(
                features_all, pseudo_labels_all, knn=args.rahp_knn,
                alpha=args.rahp_alpha, return_diagnostics=True)
            rahp_diags.append(diag)
        else:
            rahp_diags.extend((disabled_rahp_diagnostics(pseudo_labels_ir),
                               disabled_rahp_diagnostics(pseudo_labels_rgb),
                               disabled_rahp_diagnostics(pseudo_labels_all)))

        pseudo_labeled_dataset_ir = []
        ir_label=[]
        q_ir_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
                if args.use_rahp:
                    q_ir_filtered.append(q_ir_full[i])
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        q_rgb_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
                if args.use_rahp:
                    q_rgb_filtered.append(q_rgb_full[i])

        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ground_count = len(dataset_rgb.train)
        flag_ir_list, flag_rgb_list = associated_analysis_for_all(
            pseudo_labels_all, ground_count, args.logs_dir)
        print('==> Statistics for ALL epoch {}: {} clusters'.format(epoch, num_cluster_all))

        pseudo_labeled_dataset_all_ir = []
        pseudo_labeled_dataset_all_rgb = []
        q_all_ir_filtered = []
        q_all_rgb_filtered = []
        for i, ((fname, _, cid), label) in enumerate(
                zip(sorted(dataset_rgb.train) + sorted(dataset_ir.train), pseudo_labels_all)):
            if label == -1:
                continue
            if (i >= ground_count and flag_ir_list[label] == 1 and
                    flag_rgb_list[label] == 1):
                pseudo_labeled_dataset_all_ir.append((fname, label.item(), cid))
                if args.use_rahp:
                    q_all_ir_filtered.append(q_all_full[i])
            elif (i < ground_count and flag_ir_list[label] == 1 and
                  flag_rgb_list[label] == 1):
                pseudo_labeled_dataset_all_rgb.append((fname, label.item(), cid))
                if args.use_rahp:
                    q_all_rgb_filtered.append(q_all_full[i])

        print('[ALL-LOADER] aerial_samples={} ground_samples={} '
              'aerial_ids={} ground_ids={}'.format(
                  len(pseudo_labeled_dataset_all_ir),
                  len(pseudo_labeled_dataset_all_rgb),
                  len(set(item[1] for item in pseudo_labeled_dataset_all_ir)),
                  len(set(item[1] for item in pseudo_labeled_dataset_all_rgb))))

        ######################## PGM
        print("Start Bipartite Graph Matching")
        i2r = {}
        r2i = {}
        R = []
        pgm_executed = num_cluster_rgb > 0 and num_cluster_ir > 0
        if pgm_executed:
            # clusternorm
            cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
            cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
            # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
            raw_cosine = (torch.mm(cluster_features_rgb, cluster_features_ir.T)) / 1
            if cesa_state is not None:
                calibrated_aerial, cesa_prepared = cesa_state.prepare(
                    raw_cosine.T, pseudo_labels_ir, pseudo_labels_rgb)
                calibrated_score = calibrated_aerial.T
            else:
                calibrated_score = raw_cosine
            similarity = calibrated_score.exp().cpu()
            dis_similarity = (1 / (similarity))
            cost = dis_similarity / 1
            R, r2i, i2r, pgm_diag = build_total_pgm_mapping(cost)
            print('[PGM] ground_clusters={} aerial_clusters={}'.format(
                pgm_diag['ground_clusters'], pgm_diag['aerial_clusters']))
            print('[PGM] core_matches={}'.format(pgm_diag['core_matches']))
            print('[PGM] ground_completion={} aerial_completion={}'.format(
                pgm_diag['ground_completion'],
                pgm_diag['aerial_completion']))
            print('[PGM] total_edges={}'.format(pgm_diag['total_edges']))
            print('[PGM] r2i_coverage={}/{} i2r_coverage={}/{}'.format(
                pgm_diag['r2i_coverage'], pgm_diag['ground_clusters'],
                pgm_diag['i2r_coverage'], pgm_diag['aerial_clusters']))
            del cluster_features_ir, cluster_features_rgb

        print("Finish Bipartite Graph Matching")
        if cesa_state is not None and pgm_executed:
            # PGM R is (ground, aerial); CESA stores (aerial, ground).
            matched_edges = [(int(a), int(g)) for g, a in R]
            cesa_diag = cesa_state.complete(
                cesa_prepared, matched_edges, pseudo_labels_ir, pseudo_labels_rgb)
            print(format_cesa_epoch(cesa_diag))
        elif cesa_state is not None:
            cesa_diag = cesa_state.advance_without_matching(
                pseudo_labels_ir, pseudo_labels_rgb)
            print(format_cesa_epoch(cesa_diag))
        else:
            print(format_cesa_epoch(disabled_cesa_diagnostics(
                len(R), pgm_executed=pgm_executed)))
        ####################################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5)
        ])

        transform_thermal = T.Compose( [
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
        train_loader_all_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                                  args.batch_size, args.workers, args.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_all_ir, no_cam=args.no_cam,
                                                  train_transformer=transform_thermal)

        train_loader_all_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                      args.batch_size, args.workers, args.num_instances, iters,
                                                      trainset=pseudo_labeled_dataset_all_rgb, no_cam=args.no_cam,
                                                      train_transformer=train_transformer_rgb,
                                                      train_transformer1=train_transformer_rgb1)        

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        train_loader_all_ir.new_epoch()
        train_loader_all_rgb.new_epoch()        


        trainer.train(epoch, train_loader_ir, train_loader_rgb, train_loader_all_ir, train_loader_all_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i,
                      rahp_ir=q_ir_filtered if args.use_rahp else None,
                      rahp_rgb=q_rgb_filtered if args.use_rahp else None,
                      rahp_all_ir=q_all_ir_filtered if args.use_rahp else None,
                      rahp_all_rgb=q_all_rgb_filtered if args.use_rahp else None,
                      rahp_beta=args.rahp_beta, rahp_stats=rahp_stats)
        print(format_rahp_epoch(rahp_diags, rahp_stats, args.use_rahp))

        eval_a2g = None
        eval_g2a = None
        best_updated = False
        if should_evaluate_during_train(args, epoch):
            eval_a2g, eval_g2a = evaluate_agreid_for_training(
                model_ema, args, data_dir, trial)
            _print_agreid_metrics('Stage2', epoch + 1, 'a2g', eval_a2g)
            _print_agreid_metrics('Stage2', epoch + 1, 'g2a', eval_g2a)
            best_R1, best_epoch, best_updated = select_agreid_best(
                eval_a2g, eval_g2a, best_R1, best_epoch, epoch + 1)
            print('[BEST][Stage2]\nselection_metric={} current={:.4f} best={:.4f} '
                  'best_epoch={} updated={}'.format(
                      BEST_SELECTION_METRIC, eval_g2a['rank1'], best_R1,
                      best_epoch, best_updated))

        checkpoint_state = {
            'state_dict': model_ema.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'rng_state': capture_rng_state(),
            'dbscan_eps': (cluster_ir.eps, cluster_rgb.eps, cluster_all.eps),
            'cesa_state': None if cesa_state is None else cesa_state.state_dict(),
            'epoch': epoch + 1,
            'best_R1': best_R1,
            'best_epoch': best_epoch,
            'selection_metric': BEST_SELECTION_METRIC,
            'eval_g2a': eval_g2a,
            'eval_a2g': eval_a2g,
        }
        if best_updated:
            best_path = save_best_checkpoint(checkpoint_state, args.logs_dir)
            print('[CHECKPOINT] best epoch={} metric={} path={}'.format(
                best_epoch, BEST_SELECTION_METRIC, best_path))
        _, final_path = save_fixed_epoch_checkpoint(
            checkpoint_state, args.logs_dir, epoch + 1 == args.epochs)
        print('[CHECKPOINT] latest epoch={} policy=fixed-final{}'.format(
            epoch + 1, ' final={}'.format(final_path) if final_path else ''))
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='agreid_ir',
                        choices=['agreid_ir'])
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--test-batch', type=int, default=64, help="test batch size for feature extraction")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height")
    parser.add_argument('--width', type=int, default=144, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='agw',
                        choices=['agw'], help='AGW with the fixed ResNet-50 backbone')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('-mb', '--memorybank', type=str, default='CM',
                    choices=['CM', 'CMhard', 'CMhybrid'])
    parser.add_argument('--smooth', type=float, default=0, help="label smoothing")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--eval-during-train', type=parse_bool, nargs='?',
                        const=True, default=False,
                        help='evaluate each stage and retain its best G-to-A Rank-1 checkpoint')
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pretrained-resnet50', type=str, default=None,
                        help='baseline resnet50-19c8e357.pth; CLI overrides PCLHD_RESNET50_PRETRAINED')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--stage2-only', action='store_true',
                        help='skip Stage 1 and load checkpoint from Stage 1')
    parser.add_argument('--stage1-init', choices=['final', 'best'],
                        default='final',
                        help='Stage1 checkpoint used to initialize fresh Stage2')
    parser.add_argument('--resume', type=str, default='',
                        help='resume Stage 2 from a checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate method flags without loading data or starting training')
    parser.add_argument('--use-agva-ir',
                        action='store_true',
                        help='Use offline AGVA aerial images only in IR loss-training loaders')
    parser.add_argument('--agva-orig-root', type=str,
                        default='/home/lab338/Jaxon/dataset/AGreid_train',
                        help='Original AG-ReID dataset root used for AGVA path mapping')
    parser.add_argument('--agva-ir-root', type=str,
                        default='/home/lab338/Jaxon/dataset/AGreid_train_AGVA035',
                        help='Offline AGVA aerial dataset root')
    parser.add_argument('--agva-ir-prob', type=float, default=1.0,
                        help='Probability of using AGVA image instead of original aerial image during loss training')

    add_method_arguments(parser)



    main()
