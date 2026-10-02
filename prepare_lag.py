"""
Prepare LAG dataset for USL-VI-ReID training.

This script transforms the raw AG-ReID dataset structure into the LAG format
expected by the training scripts.

Raw AG-ReID structure:
    bounding_box_train/
        P0001T04041A0C0F31.jpg   (A0=aerial, C0=aerial view)
        P0001T04041A0C3F31.jpg   (A0=aerial, C3=ground view)
        ...

Output structure for LAG:
    LAG_modify/
        ir_modify/<trial>/bounding_box_train/     (aerial images -> IR)
            P0001_c01_000000.jpg   -> pid=P0001, camera=c01
        rgb_modify/<trial>/bounding_box_train/    (ground images -> RGB)
            P0001_c11_000000.jpg   -> pid=P0001, camera=c11

Camera naming rules (original -> new):
    A0C0  -> C00   (aerial, camera 0)
    A0C1  -> C01   (aerial, camera 1)
    A0C2  -> C02   (aerial, camera 2)
    A1C0  -> C10   (aerial but different angle, camera 10)
    A1C1  -> C11   (aerial but different angle, camera 11)
    A1C2  -> C12   (aerial but different angle, camera 12)
    A2C0  -> C20   (aerial but different angle, camera 20)
    A2C1  -> C21   (aerial but different angle, camera 21)
    A2C2  -> C22   (aerial but different angle, camera 22)

For cross-view VI-ReID:
    - Aerial (C0x, C1x, C2x) -> treated as different IR cameras in ir_modify
    - Ground (C3x) -> treated as different RGB cameras in rgb_modify

The script also generates train/test idx files for evaluation protocol.
"""

import os
import re
import shutil
import random
import argparse
from collections import defaultdict


def parse_agreid_filename(filename):
    """
    Parse AG-ReID filename to extract PID and camera info.
    
    Format: P{p_id}T{timestamp}A{angle}C{camera}F{frame}.jpg
    
    Camera naming: AnCm -> C{n*10 + m}
    e.g. A0C0 -> C00 (aerial, camera 0)
         A3C0 -> C30 (ground, camera 0)
    
    Returns: (pid, cam_id_str, is_aerial, is_ground)
    """
    pattern = re.compile(r'P(\d+)T\d+A(\d)C(\d)F(\d+)\.jpg')
    match = pattern.match(filename)
    
    if match is None:
        return None
    
    pid = int(match.group(1))  # P0001 -> 1
    angle = int(match.group(2))  # A0, A1, A2, A3
    cam = int(match.group(3))   # C0, C1, C2, C3
    frame = int(match.group(4))  # frame number
    
    # New camera ID: angle * 10 + cam
    new_cam = angle * 10 + cam
    
    # Camera type is determined by C value, not A (angle):
    # C0, C1, C2 -> aerial cameras
    # C3 -> ground cameras
    is_aerial = cam in [0, 1, 2]
    is_ground = cam == 3
    
    return pid, new_cam, frame, is_aerial, is_ground


def prepare_lag_dataset(raw_dir, output_dir, num_trials=10, train_ratio=0.5, val_ratio=0.2, seed=42):
    """
    Prepare LAG dataset from raw AG-ReID structure.
    
    Args:
        raw_dir: Path to raw AG-ReID dataset (containing bounding_box_train/)
        output_dir: Output directory (will contain LAG_modify/)
        num_trials: Number of trials for evaluation
        train_ratio: Ratio of identities used for training
        val_ratio: Ratio of identities used for validation (test set)
        seed: Random seed for reproducibility
    """
    raw_train_dir = os.path.join(raw_dir, 'bounding_box_train')
    
    if not os.path.exists(raw_train_dir):
        raise FileNotFoundError(f"Training directory not found: {raw_train_dir}")
    
    # Parse all images
    all_aerial = []  # (src_path, pid, cam_id, frame)
    all_ground = []  # (src_path, pid, cam_id, frame)
    
    for fname in os.listdir(raw_train_dir):
        if not fname.endswith('.jpg'):
            continue
        
        parsed = parse_agreid_filename(fname)
        if parsed is None:
            print(f"Warning: Could not parse {fname}, skipping")
            continue
        
        pid, new_cam, frame, is_aerial, is_ground = parsed
        src_path = os.path.join(raw_train_dir, fname)
        
        if is_aerial:
            all_aerial.append((src_path, pid, new_cam, frame))
        elif is_ground:
            all_ground.append((src_path, pid, new_cam, frame))
    
    print(f"Found {len(all_aerial)} aerial images, {len(all_ground)} ground images")
    
    # Get all unique person IDs
    all_pids = set()
    for _, pid, _, _ in all_aerial:
        all_pids.add(pid)
    for _, pid, _, _ in all_ground:
        all_pids.add(pid)
    all_pids = sorted(all_pids)
    print(f"Total unique person IDs: {len(all_pids)}")
    
    # Split PIDs by seed-based random shuffle for trials
    random.seed(seed)
    
    # Output directory structure
    # LAG_modify/
    #   ir_modify/{trial}/bounding_box_train/   (aerial training)
    #   rgb_modify/{trial}/bounding_box_train/  (ground training)
    #   idx/test_visible_{trial}.txt
    #   idx/test_thermal_{trial}.txt
    #   idx/train_visible_{trial}.txt
    #   idx/train_thermal_{trial}.txt
    #   query_aerial/       (test query - aerial, for thermal)
    #   query_ground/       (test query - ground, for visible)
    #   bounding_box_test_aerial/   (test gallery - aerial)
    #   bounding_box_test_ground/    (test gallery - ground)
    
    lag_dir = os.path.join(output_dir, 'LAG_modify')
    os.makedirs(lag_dir, exist_ok=True)
    os.makedirs(os.path.join(lag_dir, 'idx'), exist_ok=True)
    os.makedirs(os.path.join(lag_dir, 'query_aerial'), exist_ok=True)
    os.makedirs(os.path.join(lag_dir, 'query_ground'), exist_ok=True)
    os.makedirs(os.path.join(lag_dir, 'bounding_box_test_aerial'), exist_ok=True)
    os.makedirs(os.path.join(lag_dir, 'bounding_box_test_ground'), exist_ok=True)
    
    # For each trial, create different train/test splits
    for trial in range(1, num_trials + 1):
        print(f"\n========== Trial {trial} ==========")
        
        # Shuffle PIDs for this trial
        trial_seed = seed + trial
        random.seed(trial_seed)
        shuffled_pids = all_pids.copy()
        random.shuffle(shuffled_pids)
        
        # Split: train + test
        num_total = len(shuffled_pids)
        num_train = int(num_total * train_ratio)
        
        train_pids = set(shuffled_pids[:num_train])
        test_pids = set(shuffled_pids[num_train:])
        
        print(f"Train IDs: {len(train_pids)}, Test IDs: {len(test_pids)}")
        
        # Create directories for this trial
        ir_train_dir = os.path.join(lag_dir, 'ir_modify', str(trial), 'bounding_box_train')
        rgb_train_dir = os.path.join(lag_dir, 'rgb_modify', str(trial), 'bounding_box_train')
        os.makedirs(ir_train_dir, exist_ok=True)
        os.makedirs(rgb_train_dir, exist_ok=True)
        
        # Copy training images for aerial (IR modality)
        train_aerial_entries = []  # (src_path, pid, cam_id, frame)
        for src_path, pid, cam_id, frame in all_aerial:
            if pid in train_pids:
                new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
                dst_path = os.path.join(ir_train_dir, new_fname)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                train_aerial_entries.append((new_fname, pid))
        
        # Copy training images for ground (RGB modality)
        train_ground_entries = []  # (src_path, pid, cam_id, frame)
        for src_path, pid, cam_id, frame in all_ground:
            if pid in train_pids:
                new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
                dst_path = os.path.join(rgb_train_dir, new_fname)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                train_ground_entries.append((new_fname, pid))
        
        print(f"  IR train: {len(train_aerial_entries)} images, RGB train: {len(train_ground_entries)} images")
        
        # Write train idx files
        with open(os.path.join(lag_dir, 'idx', f'train_thermal_{trial}.txt'), 'w') as f:
            for fname, pid in train_aerial_entries:
                f.write(f"LAG_modify/ir_modify/{trial}/bounding_box_train/{fname} {pid}\n")
        
        with open(os.path.join(lag_dir, 'idx', f'train_visible_{trial}.txt'), 'w') as f:
            for fname, pid in train_ground_entries:
                f.write(f"LAG_modify/rgb_modify/{trial}/bounding_box_train/{fname} {pid}\n")
        
        # --- Test data ---
        # Copy test images (select up to 4 per ID for query, rest for gallery)
        
        # Test aerial images (thermal gallery)
        test_aerial_by_pid = defaultdict(list)
        for src_path, pid, cam_id, frame in all_aerial:
            if pid in test_pids:
                test_aerial_by_pid[pid].append((src_path, pid, cam_id, frame))
        
        test_ground_by_pid = defaultdict(list)
        for src_path, pid, cam_id, frame in all_ground:
            if pid in test_pids:
                test_ground_by_pid[pid].append((src_path, pid, cam_id, frame))
        
        # Query: up to 4 images per ID (from the query-specific modal)
        # Gallery: all remaining test images
        
        # Thermal query: aerial images for thermal modality
        # Visible query: ground images for visible modality
        # Gallery: all aerial (for thermal gallery) and all ground (for visible gallery)
        
        query_aerial_entries = []
        query_ground_entries = []
        gallery_aerial_entries = []
        gallery_ground_entries = []
        
        for pid, entries in test_aerial_by_pid.items():
            random.shuffle(entries)
            # Take up to 4 for query
            query_count = min(4, len(entries))
            for i, (src_path, pid, cam_id, frame) in enumerate(entries):
                if i < query_count:
                    query_aerial_entries.append((src_path, pid, cam_id, frame))
                else:
                    gallery_aerial_entries.append((src_path, pid, cam_id, frame))
        
        for pid, entries in test_ground_by_pid.items():
            random.shuffle(entries)
            query_count = min(4, len(entries))
            for i, (src_path, pid, cam_id, frame) in enumerate(entries):
                if i < query_count:
                    query_ground_entries.append((src_path, pid, cam_id, frame))
                else:
                    gallery_ground_entries.append((src_path, pid, cam_id, frame))
        
        # Copy query aerial images
        for src_path, pid, cam_id, frame in query_aerial_entries:
            new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
            dst_path = os.path.join(lag_dir, 'query_aerial', new_fname)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        
        # Copy query ground images
        for src_path, pid, cam_id, frame in query_ground_entries:
            new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
            dst_path = os.path.join(lag_dir, 'query_ground', new_fname)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        
        # Copy gallery aerial images
        for src_path, pid, cam_id, frame in gallery_aerial_entries:
            new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
            dst_path = os.path.join(lag_dir, 'bounding_box_test_aerial', new_fname)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        
        # Copy gallery ground images
        for src_path, pid, cam_id, frame in gallery_ground_entries:
            new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
            dst_path = os.path.join(lag_dir, 'bounding_box_test_ground', new_fname)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        
        # Write test idx files
        with open(os.path.join(lag_dir, 'idx', f'test_thermal_{trial}.txt'), 'w') as f:
            for src_path, pid, cam_id, frame in query_aerial_entries:
                new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
                f.write(f"LAG_modify/query_aerial/{new_fname} {pid}\n")
            for src_path, pid, cam_id, frame in gallery_aerial_entries:
                new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
                f.write(f"LAG_modify/bounding_box_test_aerial/{new_fname} {pid}\n")
        
        with open(os.path.join(lag_dir, 'idx', f'test_visible_{trial}.txt'), 'w') as f:
            for src_path, pid, cam_id, frame in query_ground_entries:
                new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
                f.write(f"LAG_modify/query_ground/{new_fname} {pid}\n")
            for src_path, pid, cam_id, frame in gallery_ground_entries:
                new_fname = f"{pid:04d}_c{cam_id:02d}_{frame:06d}.jpg"
                f.write(f"LAG_modify/bounding_box_test_ground/{new_fname} {pid}\n")
        print(f"  Query aerial: {len(query_aerial_entries)}, Query ground: {len(query_ground_entries)}")
        print(f"  Gallery aerial: {len(gallery_aerial_entries)}, Gallery ground: {len(gallery_ground_entries)}")
    
    print("\nDone! LAG dataset prepared at:", lag_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare LAG dataset from raw AG-ReID")
    parser.add_argument('--raw-dir', type=str, default=r"D:\Resource\Datasets\raw_datasets\AG-ReID.v1\AG-ReID",
                        help="Path to raw AG-ReID dataset")
    parser.add_argument('--output-dir', type=str, default='.',
                        help="Output directory for prepared dataset")
    parser.add_argument('--num-trials', type=int, default=10,
                        help="Number of evaluation trials")
    parser.add_argument('--train-ratio', type=float, default=0.5,
                        help="Ratio of identities used for training")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    prepare_lag_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        num_trials=args.num_trials,
        train_ratio=args.train_ratio,
        seed=args.seed
    )