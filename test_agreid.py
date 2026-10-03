# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import random
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.backends import cudnn
import torch.utils.data as data

from clustercontrast import models
from clustercontrast.methods.evaluation import (
    evaluate_agreid_direction, evaluate_agreid_distances,
    extract_agreid_features, load_agreid_split, load_agw_checkpoint_strict,
)
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

def extract_gall_feat(model, gall_loader, ngall):
    return extract_agreid_features(model, gall_loader, ngall, modal=2)

def extract_query_feat(model, query_loader, nquery):
    return extract_agreid_features(model, query_loader, nquery, modal=1)

def process_test_agreid(img_dir, trial=1, modal='ground'):
    return load_agreid_split(img_dir, trial, modal)

def eval_agreid(distmat, q_pids, g_pids, max_rank=20, q_camids=None, g_camids=None):
    return evaluate_agreid_distances(
        distmat, q_pids, g_pids, max_rank, q_camids, g_camids)

def main_worker(args):
    data_path = args.data_dir
    checkpoint_path = args.checkpoint

    model = create_model(args)
    trial = args.trial
    
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
    
    query_img, query_label = process_test_agreid(data_path, trial=trial, modal='aerial')
    gall_img, gall_label = process_test_agreid(data_path, trial=trial, modal='ground')

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    # Query is IR (modal=2), Gallery is RGB (modal=1)
    query_feat_fc = extract_gall_feat(model, query_loader, len(query_label))
    gall_feat_fc = extract_query_feat(model, gall_loader, len(gall_label))
    
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    cmc, mAP, mINP = eval_agreid(-distmat, query_label, gall_label)

    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    # ==========================================
    # 模式 2: Ground(RGB) -> Aerial(IR) [g2a]
    # ==========================================
    print('\n' + '=' * 50)
    print('Testing Mode: Ground (RGB) to Aerial (IR) [g2a]')
    
    query_img, query_label = process_test_agreid(data_path, trial=trial, modal='ground')
    gall_img, gall_label = process_test_agreid(data_path, trial=trial, modal='aerial')

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    # Query is RGB (modal=1), Gallery is IR (modal=2)
    query_feat_fc = extract_query_feat(model, query_loader, len(query_label))
    gall_feat_fc = extract_gall_feat(model, gall_loader, len(gall_label))
    
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    cmc, mAP, mINP = eval_agreid(-distmat, query_label, gall_label)

    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('=' * 50 + '\n')


def main():
    parser = argparse.ArgumentParser(description="AG-ReID Test")
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
