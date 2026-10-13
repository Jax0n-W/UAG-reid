# -*- coding: utf-8 -*-
"""Stage 2 ViewBridge 的分数读取、路径匹配和损失计算工具。

这个模块刻意不修改原有 Dataset、Preprocessor 或 ClusterMemory。训练 batch
本来就包含 ``fname``，因此只需在训练开始前把 CSV 分数匹配到这些文件名即可。
"""

from __future__ import absolute_import

import collections
import csv
import math
import os
import os.path as osp
import posixpath
import statistics

import torch
from torch.nn import functional as F


VIEW_WEIGHT_MODES = ("none", "hard_top", "soft_top")
VIEW_PATH_MODES = ("absolute", "basename", "relative")


def _slash_path(path):
    """统一 Windows/Linux 分隔符，保证服务器和本地检查结果一致。"""
    return str(path).strip().replace("\\", "/")


def _basename(path):
    return posixpath.basename(_slash_path(path))


def make_image_key(path, path_mode="basename", relative_root=None, image_root=None):
    """按命令行指定的模式生成可比较的图片键。

    ``path`` 可以是数据集保存的相对路径。absolute 模式下会先与
    ``image_root`` 拼接；relative 模式下，绝对路径会相对
    ``relative_root`` 转换，而已经是相对路径时保持其相对含义。
    """
    if path_mode not in VIEW_PATH_MODES:
        raise ValueError("未知的 view score 路径模式: {}".format(path_mode))
    if path is None or not str(path).strip():
        raise ValueError("图片路径不能为空")

    raw = str(path).strip()
    if path_mode == "basename":
        return _basename(raw)

    if path_mode == "absolute":
        if not osp.isabs(raw) and image_root:
            raw = osp.join(image_root, raw)
        return osp.normcase(osp.normpath(osp.abspath(osp.expanduser(raw))))

    # relative 模式：相对路径直接规范化；绝对路径必须有清晰的参照根目录。
    if osp.isabs(raw):
        if relative_root is None:
            raise ValueError("relative 路径模式处理绝对路径时必须提供 relative_root")
        raw = osp.relpath(raw, relative_root)
    normalized = posixpath.normpath(_slash_path(raw))
    return normalized.lstrip("./")


def compute_bridge_weight(percentile_score, near_ratio=0.30, weight_mode="soft_top"):
    """把摄像头内百分位转换为 [0, 1] 的桥接权重。"""
    if weight_mode not in VIEW_WEIGHT_MODES:
        raise ValueError("未知的 view weight 模式: {}".format(weight_mode))
    if not (0.0 < float(near_ratio) <= 1.0):
        raise ValueError("view_near_ratio 必须位于 (0, 1]，当前为 {}".format(near_ratio))

    score = min(1.0, max(0.0, float(percentile_score)))
    if weight_mode == "none":
        return 0.0

    threshold = 1.0 - float(near_ratio)
    if weight_mode == "hard_top":
        return 1.0 if score >= threshold else 0.0

    if score < threshold:
        return 0.0
    weight = (score - threshold) / float(near_ratio)
    return min(1.0, max(0.0, weight))


def _empty_weight_summary():
    return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}


def _weight_summary(weights):
    if not weights:
        return _empty_weight_summary()
    return {
        "min": min(weights),
        "mean": sum(weights) / float(len(weights)),
        "median": statistics.median(weights),
        "max": max(weights),
    }


def load_view_score_map(csv_path, score_column="p_ground_foreground",
                        near_ratio=0.30, weight_mode="soft_top",
                        path_mode="basename", relative_root=None):
    """读取 CSV，并按 camera 独立生成确定性的百分位排名。

    百分位定义为 ``1 - rank / N``，其中 rank 是按“分数降序、路径升序”
    排列后的 0 基序号。这样最高分严格为 1，同时阈值 ``1-ratio``
    恰好选中每个 camera 的 ``ceil(N * ratio)`` 张图片。

    返回：
      score_map: ``匹配键 -> 图片分数信息``
      stats: CSV 行数、camera 统计、重复 basename 数和权重分布
    """
    if not csv_path:
        raise ValueError("view score 文件路径不能为空")
    if not osp.isfile(csv_path):
        raise FileNotFoundError("view score 文件不存在: {}".format(csv_path))
    if path_mode not in VIEW_PATH_MODES:
        raise ValueError("未知的 view score 路径模式: {}".format(path_mode))
    if weight_mode not in VIEW_WEIGHT_MODES:
        raise ValueError("未知的 view weight 模式: {}".format(weight_mode))
    if not (0.0 < float(near_ratio) <= 1.0):
        raise ValueError("view_near_ratio 必须位于 (0, 1]")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "pid", "camera", score_column, "view_group"}
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required - fieldnames)
        if missing_columns:
            raise ValueError("view score CSV 缺少必要列: {}".format(", ".join(missing_columns)))

        for line_number, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip()
            camera = (row.get("camera") or "").strip()
            if not image_path:
                raise ValueError("CSV 第 {} 行 image_path 为空".format(line_number))
            if not camera:
                raise ValueError("CSV 第 {} 行 camera 为空".format(line_number))
            try:
                raw_score = float(row.get(score_column, ""))
            except (TypeError, ValueError):
                raise ValueError("CSV 第 {} 行 {} 不能转换为 float".format(
                    line_number, score_column))
            if not math.isfinite(raw_score):
                raise ValueError("CSV 第 {} 行 {} 不是有限数值".format(
                    line_number, score_column))
            rows.append({
                "image_path": image_path,
                "camera": camera,
                "raw_score": raw_score,
                "csv_view_group": (row.get("view_group") or "").strip(),
            })

    if not rows:
        raise ValueError("view score CSV 没有数据行: {}".format(csv_path))

    basename_counts = collections.Counter(_basename(row["image_path"]) for row in rows)
    duplicate_basenames = sorted(name for name, count in basename_counts.items() if count > 1)
    if path_mode == "basename" and duplicate_basenames:
        preview = ", ".join(duplicate_basenames[:5])
        raise ValueError("basename 不唯一，不能安全匹配（共 {} 个重复）: {}".format(
            len(duplicate_basenames), preview))

    camera_rows = collections.defaultdict(list)
    for row in rows:
        camera_rows[row["camera"]].append(row)

    camera_stats = {}
    all_weights = []
    for camera in sorted(camera_rows):
        group = camera_rows[camera]
        # Python sort 是稳定排序；显式加入路径作为第二键可消除同分不确定性。
        ordered = sorted(group, key=lambda item: (-item["raw_score"], _slash_path(item["image_path"])))
        count = len(ordered)
        near_count = 0
        camera_weights = []
        for rank, row in enumerate(ordered):
            percentile = 1.0 - (float(rank) / float(count))
            # 当 N*ratio 恰好为整数时，下一名的百分位会刚好等于阈值。
            # 将这个“Top K 之外”的边界值轻微下移，保证数量严格为 ceil(N*ratio)。
            near_limit = int(math.ceil(float(count) * float(near_ratio)))
            if rank >= near_limit and percentile >= 1.0 - float(near_ratio):
                percentile = max(0.0, 1.0 - float(near_ratio) - 1e-12)
            weight = compute_bridge_weight(percentile, near_ratio, weight_mode)
            view_group = "A_near" if percentile >= 1.0 - float(near_ratio) else "A_far"
            row["percentile_score"] = percentile
            row["view_group"] = view_group
            row["bridge_weight"] = weight
            if view_group == "A_near":
                near_count += 1
            camera_weights.append(weight)
            all_weights.append(weight)
        camera_stats[camera] = {
            "count": count,
            "near_count": near_count,
            "mean_weight": sum(camera_weights) / float(count),
        }

    score_map = {}
    for row in rows:
        key = make_image_key(row["image_path"], path_mode, relative_root=relative_root)
        if key in score_map:
            raise ValueError("view score CSV 中匹配键重复: {}".format(key))
        score_map[key] = dict(row)

    stats = {
        "csv_rows": len(rows),
        "duplicate_basename_count": len(duplicate_basenames),
        "duplicate_basenames": duplicate_basenames,
        "camera_stats": camera_stats,
        "near_count": sum(item["near_count"] for item in camera_stats.values()),
        "far_count": len(rows) - sum(item["near_count"] for item in camera_stats.values()),
        "weight_summary": _weight_summary(all_weights),
    }
    return score_map, stats


def match_view_score_map(image_paths, score_map, path_mode="basename",
                         relative_root=None, image_root=None):
    """把 CSV 映射到真实空中训练路径，并检测训练侧匹配键冲突。"""
    matched = {}
    missing = []
    seen_keys = {}
    for path in image_paths:
        key = make_image_key(path, path_mode, relative_root=relative_root, image_root=image_root)
        if key in seen_keys and seen_keys[key] != path:
            raise ValueError("空中训练集的匹配键不唯一: {} 对应 {} 和 {}".format(
                key, seen_keys[key], path))
        seen_keys[key] = path
        entry = score_map.get(key)
        if entry is None:
            missing.append(path)
        else:
            # Trainer 按 batch 中原始 fname 查询，因此这里以原始 fname 为键。
            matched[path] = dict(entry)

    return matched, {
        "training_images": len(image_paths),
        "matched": len(matched),
        "missing": len(missing),
        "missing_paths": missing,
    }


def bridge_is_enabled(lambda_view_bridge, weight_mode, view_score_map):
    """三个关闭条件任一成立时，都严格不调用 Bridge Loss。"""
    return (float(lambda_view_bridge) != 0.0 and
            weight_mode != "none" and
            view_score_map is not None)


def add_view_bridge_loss(original_loss, bridge_loss, lambda_view_bridge,
                         weight_mode="soft_top"):
    """组合总损失；关闭时直接返回原对象，避免引入任何额外计算图。"""
    if float(lambda_view_bridge) == 0.0 or weight_mode == "none":
        return original_loss
    return original_loss + float(lambda_view_bridge) * bridge_loss


def compute_view_bridge_loss(aerial_features, aerial_labels, image_paths,
                             ground_prototypes, bgm_i2r, view_score_map,
                             ground_num_clusters=None, epsilon=1e-12):
    """计算仅作用于空中样本的加权余弦 Bridge Loss。

    ``aerial_labels`` 是当前 epoch 的空中伪标签；``bgm_i2r`` 是真实的
    “空中伪簇 -> 地面伪簇”映射；``ground_prototypes`` 来自
    ``memory_rgb.features``。原型会 detach，Bridge 只更新网络，不改变
    ClusterMemory 自己的动量更新机制。
    """
    if aerial_features.ndim != 2 or ground_prototypes.ndim != 2:
        raise ValueError("aerial_features 和 ground_prototypes 必须是二维张量")
    if aerial_features.size(0) != aerial_labels.numel():
        raise ValueError("空中特征数与伪标签数不一致")
    if aerial_features.size(0) != len(image_paths):
        raise ValueError("空中特征数与图片路径数不一致")
    if ground_num_clusters is None:
        ground_num_clusters = ground_prototypes.size(0)

    stats = {
        "total_samples": len(image_paths),
        "valid_samples": 0,
        "positive_weight_samples": 0,
        "missing_score": 0,
        "missing_bgm": 0,
        "invalid_label": 0,
        "invalid_target": 0,
        "non_finite": 0,
        "weight_sum": 0.0,
        "mean_weight": 0.0,
    }

    valid_indexes = []
    target_indexes = []
    valid_weights = []
    labels_cpu = aerial_labels.detach().view(-1).cpu().tolist()

    for index, (label, path) in enumerate(zip(labels_cpu, image_paths)):
        entry = None if view_score_map is None else view_score_map.get(path)
        if entry is None:
            stats["missing_score"] += 1
            continue
        weight = float(entry.get("bridge_weight", 0.0))
        if not math.isfinite(weight):
            stats["non_finite"] += 1
            continue
        weight = min(1.0, max(0.0, weight))
        if weight <= 0.0:
            continue
        stats["positive_weight_samples"] += 1

        label = int(label)
        if label < 0:
            stats["invalid_label"] += 1
            continue
        if bgm_i2r is None or label not in bgm_i2r:
            stats["missing_bgm"] += 1
            continue
        try:
            target = int(bgm_i2r[label])
        except (TypeError, ValueError):
            stats["invalid_target"] += 1
            continue
        if target < 0 or target >= int(ground_num_clusters) or target >= ground_prototypes.size(0):
            stats["invalid_target"] += 1
            continue

        valid_indexes.append(index)
        target_indexes.append(target)
        valid_weights.append(weight)

    if not valid_indexes:
        # 保留设备和 dtype，并保留一条值为 0 的可反传计算图。
        return torch.where(torch.isfinite(aerial_features), aerial_features, torch.zeros_like(aerial_features)).sum() * 0.0, stats

    index_tensor = torch.tensor(valid_indexes, device=aerial_features.device, dtype=torch.long)
    target_tensor = torch.tensor(target_indexes, device=ground_prototypes.device, dtype=torch.long)
    weight_tensor = aerial_features.new_tensor(valid_weights)

    selected_features = F.normalize(aerial_features.index_select(0, index_tensor), p=2, dim=1, eps=epsilon)
    # normalize 会生成新张量；detach 确保不会给 memory 建立梯度或改变更新方式。
    selected_prototypes = F.normalize(
        ground_prototypes.detach().index_select(0, target_tensor), p=2, dim=1, eps=epsilon)
    if selected_prototypes.device != selected_features.device:
        selected_prototypes = selected_prototypes.to(selected_features.device)

    cosine = (selected_features * selected_prototypes).sum(dim=1).clamp(-1.0, 1.0)
    distances = 1.0 - cosine
    finite_mask = torch.isfinite(distances)
    if not bool(finite_mask.all().item()):
        stats["non_finite"] += int((~finite_mask).sum().item())
        distances = torch.where(finite_mask, distances, torch.zeros_like(distances))
        weight_tensor = torch.where(finite_mask, weight_tensor, torch.zeros_like(weight_tensor))

    denominator = weight_tensor.sum()
    if float(denominator.detach().item()) <= 0.0:
        return torch.where(torch.isfinite(aerial_features), aerial_features, torch.zeros_like(aerial_features)).sum() * 0.0, stats

    loss = (weight_tensor * distances).sum() / (denominator + float(epsilon))
    if not bool(torch.isfinite(loss).item()):
        # 正常有限输入不会到达这里；显式回退为设备上的零，保证绝不输出 NaN。
        stats["non_finite"] += 1
        return torch.where(torch.isfinite(aerial_features), aerial_features, torch.zeros_like(aerial_features)).sum() * 0.0, stats

    stats["valid_samples"] = len(valid_indexes)
    stats["weight_sum"] = float(sum(valid_weights))
    stats["mean_weight"] = stats["weight_sum"] / float(stats["valid_samples"])
    return loss, stats
