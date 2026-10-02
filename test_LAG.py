# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import time
from PIL import Image

import torch
from torch import nn
from torch.backends import cudnn
import torch.utils.data as data
from torch.autograd import Variable

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.datasets.lagper_protocol import evaluation_arrays
from clustercontrast.methods.evaluation import load_agw_checkpoint_strict
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.serialization import load_checkpoint

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type,
                          pretrained_path=args.pretrained_resnet50)
    model.cuda()
    model = nn.DataParallel(model)
    return model

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        # 懒加载防爆内存
        self.test_img_file = test_img_file
        self.test_label = test_label
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.test_img_file[index]
        target = self.test_label[index]
        img = Image.open(img_path)
        img = img.resize((self.img_size[0], self.img_size[1]), Image.LANCZOS)
        img_array = np.array(img)
        img_tensor = self.transform(img_array)
        return img_tensor, target

    def __len__(self):
        return len(self.test_img_file)

def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_gall_feat(model, gall_loader, ngall):
    """提取 Aerial/IR 特征 (modal=2)"""
    pool_dim = 2048
    net = model
    net.eval()
    print('Extracting Feature (modal=2, Aerial/IR)...')
    start = time.time()
    ptr = 0
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
            gall_feat_fc[ptr:ptr+batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc

def extract_query_feat(model, query_loader, nquery):
    """提取 Ground/RGB 特征 (modal=1)"""
    pool_dim = 2048
    net = model
    net.eval()
    print('Extracting Feature (modal=1, Ground/RGB)...')
    start = time.time()
    ptr = 0
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
            query_feat_fc[ptr:ptr+batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc

def process_test_lagper(img_dir, split, view):
    return evaluation_arrays(img_dir, split, view)


def valid_gallery_mask(q_pid, q_camid, g_pids, g_camids):
    return ~((g_pids == q_pid) & (g_camids == q_camid))

def eval_lagper(distmat, q_pids, g_pids, max_rank=20, q_camids=None, g_camids=None):
    num_q, num_g = distmat.shape
    if num_g < max_rank: max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    
    if q_camids is None or g_camids is None:
        raise ValueError('LAGPeR evaluation requires real q_camids and g_camids.')
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        keep = valid_gallery_mask(q_pid, q_camid, g_pids[order], g_camids[order])

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc): continue

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
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP

def main_worker(args):
    data_path = args.data_dir
    checkpoint_path = args.checkpoint

    model = create_model(args)
    trial = args.trial
    if trial != 1:
        raise ValueError('Official LAGPeR scene split has one fixed trial (trial=1).')
    
    args.test_batch = args.batch_size
    args.img_w = args.width
    args.img_h = args.height
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = T.Compose([
        T.ToPILImage(), T.Resize((args.img_h, args.img_w)), T.ToTensor(), normalize,
    ])
    
    print(f'==> Loading weights from: {checkpoint_path}')
    checkpoint = load_checkpoint(checkpoint_path)
    load_agw_checkpoint_strict(model, checkpoint)
    model.eval()

    # ==========================================
    # 模式 1: Aerial(IR) -> Ground(RGB) [a2g]
    # ==========================================
    print('\n' + '=' * 50)
    print('Testing Mode: Aerial (IR) to Ground (RGB) [a2g]')
    
    query_img, query_label, q_camids = process_test_lagper(
        data_path, 'query', 'aerial')
    gall_img, gall_label, g_camids = process_test_lagper(
        data_path, 'gallery', 'ground')

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    query_feat_fc = extract_gall_feat(model, query_loader, len(query_label))
    gall_feat_fc = extract_query_feat(model, gall_loader, len(gall_label))
    
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    cmc, mAP, mINP = eval_lagper(-distmat, query_label, gall_label, q_camids=q_camids, g_camids=g_camids)

    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    # ==========================================
    # 模式 2: Ground(RGB) -> Aerial(IR) [g2a]
    # ==========================================
    print('\n' + '=' * 50)
    print('Testing Mode: Ground (RGB) to Aerial (IR) [g2a]')
    
    query_img, query_label, q_camids = process_test_lagper(
        data_path, 'query', 'ground')
    gall_img, gall_label, g_camids = process_test_lagper(
        data_path, 'gallery', 'aerial')

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    query_feat_fc = extract_query_feat(model, query_loader, len(query_label))
    gall_feat_fc = extract_gall_feat(model, gall_loader, len(gall_label))
    
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    cmc, mAP, mINP = eval_lagper(-distmat, query_label, gall_label, q_camids=q_camids, g_camids=g_camids)

    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    # ==========================================
    # 模式 3: Ground(RGB) -> Aerial(IR) + Ground(RGB) [g2a+g]
    # ==========================================
    print('\n' + '=' * 50)
    print('Testing Mode: Ground (RGB) to Aerial + Ground [g2a+g]')
    
    query_img, query_label, q_camids = process_test_lagper(
        data_path, 'query', 'ground')
    gall_a_img, gall_a_label, gall_a_camids = process_test_lagper(
        data_path, 'gallery', 'aerial')
    gall_g_img, gall_g_label, gall_g_camids = process_test_lagper(
        data_path, 'gallery', 'ground')
    
    # 提取 Query (RGB, modal=1)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_feat_fc = extract_query_feat(model, query_loader, len(query_label))
    
    # 提取 Gallery Aerial (IR, modal=2)
    gall_a_set = TestData(gall_a_img, gall_a_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_a_loader = data.DataLoader(gall_a_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    gall_a_feat_fc = extract_gall_feat(model, gall_a_loader, len(gall_a_label))
    
    # 提取 Gallery Ground (RGB, modal=1)
    gall_g_set = TestData(gall_g_img, gall_g_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_g_loader = data.DataLoader(gall_g_set, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    gall_g_feat_fc = extract_query_feat(model, gall_g_loader, len(gall_g_label))
    
    # 特征级别拼接
    gall_feat_fc = np.concatenate([gall_a_feat_fc, gall_g_feat_fc], axis=0)
    gall_label = np.concatenate([gall_a_label, gall_g_label])
    
    g_camids = np.concatenate([gall_a_camids, gall_g_camids])
    
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    cmc, mAP, mINP = eval_lagper(-distmat, query_label, gall_label, q_camids=q_camids, g_camids=g_camids)
    
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('=' * 50 + '\n')


def main():
    parser = argparse.ArgumentParser(description="LAG Test")
    parser.add_argument('-a', '--arch', type=str, default='agw', choices=['agw'],
                        help='evaluation architecture (default: agw)')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pretrained-resnet50', type=str, default=None,
                        help='baseline resnet50-19c8e357.pth; CLI overrides PCLHD_RESNET50_PRETRAINED')
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=144)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    main_worker(args)

if __name__ == '__main__':
    main()
