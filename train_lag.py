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
from clustercontrast.datasets.lagper_protocol import evaluation_arrays
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
    capture_rng_state, final_checkpoint_path, restore_rng_state,
    save_fixed_epoch_checkpoint, should_evaluate_during_train,
)
from clustercontrast.trainers import ClusterContrastTrainer_DCL, ClusterContrastTrainer_PCLMP
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance, compute_modal_invariant_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange, Gray
from collections import Counter
from scipy.optimize import linear_sum_assignment
def get_data(name, data_dir, trial=0):
    root = data_dir
    dataset = datasets.create(name, root, trial=trial)
    return dataset


class channel_select(object):
    def __init__(self, channel=0):
        self.channel = channel

    def __call__(self, img):
        if self.channel == 3:
            img_gray = img.convert('L')
            np_img = np.array(img_gray, dtype=np.uint8)
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL = Image.fromarray(img_aug, 'RGB')
        else:
            np_img = np.array(img, dtype=np.uint8)
            np_img = np_img[:, :, self.channel]
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL = Image.fromarray(img_aug, 'RGB')
        return img_PIL


def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                         num_instances, iters, trainset=None, no_cam=False, train_transformer=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                            num_instances, iters, trainset=None, no_cam=False,
                            train_transformer=None, train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir,
                                          transform=train_transformer, transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None, test_transformer=None):
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
    model.cuda()
    model_ema.cuda()
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    return model, model_ema


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
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
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_gall_feat(model, gall_loader, ngall):
    pool_dim = 2048
    net = model
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net(input, input, 2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net(flip_input, flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach()) / 2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_fc


def extract_query_feat(model, query_loader, nquery):
    pool_dim = 2048
    net = model
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net(input, input, 1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net(flip_input, flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach()) / 2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_fc


def process_test_lag(img_dir, trial=1, modal='visible'):
    if trial != 1:
        raise ValueError('Official LAGPeR scene split has one fixed trial (trial=1).')
    if modal == 'visible':
        paths, labels, _ = evaluation_arrays(img_dir, 'query', 'ground')
    elif modal == 'thermal':
        paths, labels, _ = evaluation_arrays(img_dir, 'gallery', 'aerial')
    else:
        raise ValueError("modal must be 'visible' or 'thermal'")
    return paths, labels


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP


def associated_analysis_for_all(all_origin, all_pred, modalities_for_all, log_dir):
    label_count_all = -1
    all_label_set = list(set(all_pred))
    all_label_set.sort()
    class_NIRVIS_list_modal_all = []
    associate = 0
    flag_ir_list = collections.defaultdict(list)
    flag_rgb_list = collections.defaultdict(list)
    for idx_, lab_ in enumerate(all_label_set):
        label_count_all += 1
        class_NIRVIS_list_modal = []
        flag_ir = 0
        flag_rgb = 0
        for idx, lab in enumerate(all_pred):
            if lab_ == lab:
                if modalities_for_all[idx] == 'aerial':
                    flag_ir = 1
                    flag_ir_list[idx_] = 1
                elif modalities_for_all[idx] == 'ground':
                    flag_rgb = 1
                    flag_rgb_list[idx_] = 1
        class_NIRVIS_list_modal_all.extend([class_NIRVIS_list_modal])
        if flag_ir == 1 and flag_rgb == 1:
            associate = associate + 1
    print('associate rate', associate / len(all_label_set))
    return flag_ir_list, flag_rgb_list


def main():
    args = parser.parse_args()
    if args.resume:
        args.stage2_only = True
    validate_method_args(args)
    args.experiment_tag = experiment_tag(args)
    if args.dry_run:
        print('[CONFIG]\ndataset=LAGPeR\nprotocol=official-scene-split\n'
              'train_scenes=4\ntest_scenes=3\narch={}\nmemorybank={}\n'
              'checkpoint=fixed-final\neval_during_train={}\nrahp={}\ncesa={}'.format(
                  args.arch, args.memorybank, args.eval_during_train,
                  args.use_rahp, args.use_cesa))
        return
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    log_s1_name = 'lag_s1' + ('_rahp' if args.use_rahp else '')
    log_s2_name = 'lag_s2' + ('' if args.experiment_tag == 'baseline'
                             else '_' + args.experiment_tag)
    logs_root = args.logs_dir
    if not args.stage2_only:
        main_worker_stage1(args, log_s1_name)
    args.logs_dir = logs_root
    main_worker_stage2(args, log_s1_name, log_s2_name)


def main_worker_stage1(args, log_s1_name):
    logs_dir_root = osp.join(args.logs_dir + '/' + log_s1_name)
    data_dir = args.data_dir
    trial = args.trial
    start_epoch = 0
    args.logs_dir = osp.join(logs_dir_root, str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, str(trial) + 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('lag_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('lag_rgb', args.data_dir, trial=trial)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    model, _ = create_model(args)

    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    trainer = ClusterContrastTrainer_DCL(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                ir_eps = args.eps
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.eps
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                  args.batch_size, args.workers,
                                                  testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                  args.batch_size, args.workers,
                                                  testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            del rerank_dist_rgb
            del rerank_dist_ir
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
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

        pseudo_labeled_dataset_ir = []
        ir_label = []
        q_ir_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
                if args.use_rahp:
                    q_ir_filtered.append(q_ir_full[i])
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        q_rgb_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
                if args.use_rahp:
                    q_rgb_filtered.append(q_rgb_full[i])
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        transform_thermal = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters,
                                                trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,
                                                train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                    args.batch_size // 2, args.workers, args.num_instances, iters,
                                                    trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,
                                                    train_transformer=train_transformer_rgb,
                                                    train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir),
                      rahp_ir=q_ir_filtered if args.use_rahp else None,
                      rahp_rgb=q_rgb_filtered if args.use_rahp else None,
                      rahp_beta=args.rahp_beta, rahp_stats=rahp_stats)
        print(format_rahp_epoch(rahp_diags, rahp_stats, args.use_rahp))

        if should_evaluate_during_train(args, epoch):
            args.test_batch = 64
            args.img_w = args.width
            args.img_h = args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h, args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            data_path = data_dir
            query_img, query_label = process_test_lag(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_lag(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model, query_loader, nquery)
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            print('\n * Debug evaluation epoch {:3d}   model R1: {:5.1%}  model mAP: {:5.1%}\n'.format(
                epoch, cmc[0], mAP))
        checkpoint_state = {
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
        }
        _, final_path = save_fixed_epoch_checkpoint(
            checkpoint_state, args.logs_dir, epoch + 1 == args.epochs)
        print('[CHECKPOINT] latest epoch={} policy=fixed-final{}'.format(
            epoch + 1, ' final={}'.format(final_path) if final_path else ''))
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


def main_worker_stage2(args, log_s1_name, log_s2_name):
    logs_root = args.logs_dir
    logs_dir_root = osp.join(logs_root, log_s2_name)
    trial = args.trial
    start_epoch = 0
    args.memorybank = 'CMhard'
    data_dir = args.data_dir
    args.logs_dir = osp.join(logs_dir_root, str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, str(trial) + 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('lag_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('lag_rgb', args.data_dir, trial=trial)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    model, model_ema = create_model(args)
    if not args.resume:
        checkpoint = load_checkpoint(final_checkpoint_path(
            osp.join(logs_root, log_s1_name, str(trial))))
        model.load_state_dict(checkpoint['state_dict'])
        model_ema.load_state_dict(checkpoint['state_dict'])

    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    trainer = ClusterContrastTrainer_PCLMP(model, model_ema)
    cesa_state = (CESAState(rho=args.cesa_rho, eta=args.cesa_eta,
                            lineage_threshold=args.cesa_lineage_thr,
                            warmup=args.cesa_warmup) if args.use_cesa else None)
    resume_cluster_eps = None

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
        restore_rng_state(resumed.get('rng_state'))
        if cesa_state is not None:
            cesa_state.load_state_dict(resumed.get('cesa_state'))

    for epoch in range(start_epoch, args.epochs):
        with torch.no_grad():
            if epoch == start_epoch:
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
                                                  args.batch_size, args.workers,
                                                  testset=sorted(dataset_rgb.train))
            features_rgb_ema, _ = extract_features(model_ema, cluster_loader_rgb, print_freq=50, mode=1)
            features_rgb_ema = torch.cat([features_rgb_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                  args.batch_size, args.workers,
                                                  testset=sorted(dataset_ir.train))
            features_ir_ema, _ = extract_features(model_ema, cluster_loader_ir, print_freq=50, mode=2)
            features_ir_ema = torch.cat([features_ir_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            print('==> Create pseudo labels for unlabeled ALL data')
            features_all = torch.cat([features_rgb, features_ir], dim=0)

            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            rerank_dist_all = compute_modal_invariant_jaccard_distance(features_all, k1=40, k2=32,
                                                                       file=sorted(dataset_rgb.train) + sorted(
                                                                           dataset_ir.train), search_option=3)
            pseudo_labels_all = cluster_all.fit_predict(rerank_dist_all)
            del rerank_dist_rgb
            del rerank_dist_ir
            del rerank_dist_all
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            return centers

        @torch.no_grad()
        def generate_modal_invariant_cluster_features(labels, num_cluster_all, features, file):
            centers_IR = collections.defaultdict(list)
            centers_RBG = collections.defaultdict(list)
            centers_IR_mean = collections.defaultdict(list)
            centers_RBG_mean = collections.defaultdict(list)
            ground_count = len(dataset_rgb.train)
            for i, (label, (fname, _, cid)) in enumerate(zip(labels, file)):
                if label == -1:
                    continue
                if i >= ground_count:
                    centers_IR[labels[i]].append(features[i])
                else:
                    centers_RBG[labels[i]].append(features[i])
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
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()
            memory_all.features = F.normalize(cluster_features_all.repeat(2, 1), dim=1).cuda()
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
        ir_label = []
        q_ir_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
                if args.use_rahp:
                    q_ir_filtered.append(q_ir_full[i])
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        q_rgb_filtered = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
                if args.use_rahp:
                    q_rgb_filtered.append(q_rgb_full[i])
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        all_label = []
        all_file_name = []
        all_modalities = []
        ground_count = len(dataset_rgb.train)
        for i, ((fname, _, cid), label) in enumerate(
                zip(sorted(dataset_rgb.train) + sorted(dataset_ir.train), pseudo_labels_all)):
            if label != -1:
                all_file_name.append(fname)
                all_label.append(label.item())
                all_modalities.append('ground' if i < ground_count else 'aerial')

        flag_ir_list, flag_rgb_list = associated_analysis_for_all(pseudo_labels_all, all_label, all_modalities,
                                                                  args.logs_dir)
        print('==> Statistics for ALL epoch {}: {} clusters'.format(epoch, num_cluster_all))

        all_label = []
        pseudo_labeled_dataset_all_ir = []
        pseudo_labeled_dataset_all_rgb = []
        q_all_ir_filtered = []
        q_all_rgb_filtered = []
        for i, ((fname, _, cid), label) in enumerate(
                zip(sorted(dataset_rgb.train) + sorted(dataset_ir.train), pseudo_labels_all)):
            if label != -1:
                all_file_name.append(fname)
                all_label.append(label.item())
            if (not args.use_rahp or label != -1) and i >= ground_count and flag_ir_list[label] == 1 and flag_rgb_list[label] == 1:
                pseudo_labeled_dataset_all_ir.append((fname, label.item(), cid))
                if args.use_rahp:
                    q_all_ir_filtered.append(q_all_full[i])
            elif (not args.use_rahp or label != -1) and i < ground_count and flag_ir_list[label] == 1 and flag_rgb_list[label] == 1:
                pseudo_labeled_dataset_all_rgb.append((fname, label.item(), cid))
                if args.use_rahp:
                    q_all_rgb_filtered.append(q_all_full[i])

        ######################## PGM
        print("Start Bipartite Graph Matching")
        i2r = {}
        r2i = {}
        R = []
        bgm = False
        pgm_executed = num_cluster_rgb >= num_cluster_ir
        if pgm_executed:
            cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
            cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
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
            tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
            cost = (torch.cat((cost, tmp), 1))
            unmatched_row = []
            row_ind, col_ind = linear_sum_assignment(cost)
            for idx, item in enumerate(row_ind):
                if col_ind[idx] < similarity.shape[1]:
                    R.append((row_ind[idx], col_ind[idx]))
                    r2i[row_ind[idx]] = col_ind[idx]
                    i2r[col_ind[idx]] = row_ind[idx]
                else:
                    unmatched_row.append(row_ind[idx])
            if bgm is False:
                unmatched_cost = cost[unmatched_row][:, :dis_similarity.shape[1]]
                unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
                for idx, item in enumerate(unmatched_row_ind):
                    R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                    r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
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
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        transform_thermal = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((288, 144)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters,
                                                trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,
                                                train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                    args.batch_size, args.workers, args.num_instances, iters,
                                                    trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,
                                                    train_transformer=train_transformer_rgb,
                                                    train_transformer1=train_transformer_rgb1)
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

        if should_evaluate_during_train(args, epoch):
            args.test_batch = 64
            args.img_w = args.width
            args.img_h = args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h, args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            data_path = data_dir
            query_img, query_label = process_test_lag(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_lag(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model_ema, query_loader, nquery)
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model_ema, gall_loader, ngall)
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            print('\n * Debug evaluation epoch {:3d}   model R1: {:5.1%}  model mAP: {:5.1%}\n'.format(
                epoch, cmc[0], mAP))
        checkpoint_state = {
            'state_dict': model_ema.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'rng_state': capture_rng_state(),
            'dbscan_eps': (cluster_ir.eps, cluster_rgb.eps, cluster_all.eps),
            'cesa_state': None if cesa_state is None else cesa_state.state_dict(),
            'epoch': epoch + 1,
        }
        _, final_path = save_fixed_epoch_checkpoint(
            checkpoint_state, args.logs_dir, epoch + 1 == args.epochs)
        print('[CHECKPOINT] latest epoch={} policy=fixed-final{}'.format(
            epoch + 1, ' final={}'.format(final_path) if final_path else ''))
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on LAG (unsupervised re-ID)")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='lag',
                        choices=['lag'])
    parser.add_argument('-b', '--batch-size', type=int, default=64)
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
    parser.add_argument('--pooling-type', type=str, default='gem')
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
                        help='debug-only test evaluation; never selects checkpoints')
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
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")
    parser.add_argument('--resume', type=str, default='',
                        help='resume Stage 2 from a checkpoint')
    parser.add_argument('--stage2-only', action='store_true',
                        help='skip Stage 1 and load its saved checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate method flags without loading data or starting training')

    add_method_arguments(parser)

    main()
