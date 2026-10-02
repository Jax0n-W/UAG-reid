"""Strict prepared-data contract for the official LAGPeR scene split."""

import json
import os.path as osp
import re

import numpy as np


PROTOCOL_NAME = 'official-scene-split'
PROTOCOL_FILE = osp.join('meta', 'protocol.json')
MANIFESTS = {
    ('train', 'aerial'): 'train_aerial.txt',
    ('train', 'ground'): 'train_ground.txt',
    ('query', 'aerial'): 'query_aerial.txt',
    ('gallery', 'aerial'): 'gallery_aerial.txt',
    ('query', 'ground'): 'query_ground.txt',
    ('gallery', 'ground'): 'gallery_ground.txt',
}
_FILENAME = re.compile(r'(?P<pid>\d+)_c(?P<camid>\d+)_(?P<frame>\d+)\.[^.]+$', re.IGNORECASE)


def parse_lagper_filename(path):
    match = _FILENAME.search(osp.basename(path))
    if match is None:
        raise ValueError("LAGPeR filename must contain '<pid>_c<camid>_<frame>': {}".format(path))
    return int(match.group('pid')), int(match.group('camid'))


def validate_lagper_contract(root):
    protocol_path = osp.join(root, PROTOCOL_FILE)
    if not osp.isfile(protocol_path):
        raise RuntimeError(
            "Official LAGPeR prepared metadata is missing: {}. "
            "Provide a prepared official scene split; prepare_lag.py cannot create it."
            .format(protocol_path))
    with open(protocol_path, 'r', encoding='utf-8') as handle:
        protocol = json.load(handle)
    expected = {
        'dataset': 'LAGPeR',
        'protocol': PROTOCOL_NAME,
        'train_scenes': 4,
        'test_scenes': 3,
        'train_aerial_cameras': 4,
        'train_ground_cameras': 8,
        'test_aerial_cameras': 3,
        'test_ground_cameras': 6,
    }
    mismatches = {key: (protocol.get(key), value) for key, value in expected.items()
                  if protocol.get(key) != value}
    if mismatches:
        raise RuntimeError("Invalid LAGPeR official protocol metadata: {}".format(mismatches))
    return protocol


def load_lagper_records(root, split, view, require_images=True):
    validate_lagper_contract(root)
    key = (split, view)
    if key not in MANIFESTS:
        raise ValueError("Unsupported LAGPeR split/view: {} / {}".format(split, view))
    manifest_path = osp.join(root, 'meta', MANIFESTS[key])
    if not osp.isfile(manifest_path):
        raise RuntimeError("Required LAGPeR manifest is missing: {}".format(manifest_path))
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("{}:{} must contain: image_path pid camid view".format(
                    manifest_path, line_number))
            relative_path, pid_text, camid_text, record_view = fields
            if record_view != view:
                raise ValueError("{}:{} view '{}' does not match '{}'".format(
                    manifest_path, line_number, record_view, view))
            try:
                pid, camid = int(pid_text), int(camid_text)
            except ValueError as error:
                raise ValueError("{}:{} pid and camid must be integers".format(
                    manifest_path, line_number)) from error
            absolute_path = osp.normpath(osp.join(root, relative_path))
            if osp.commonpath((osp.abspath(root), osp.abspath(absolute_path))) != osp.abspath(root):
                raise ValueError("{}:{} image path escapes dataset root".format(
                    manifest_path, line_number))
            if require_images and not osp.isfile(absolute_path):
                raise RuntimeError("Manifest image is missing: {}".format(absolute_path))
            records.append((osp.normpath(relative_path), pid, camid))
    if not records:
        raise RuntimeError("LAGPeR manifest contains no records: {}".format(manifest_path))
    return records


def evaluation_arrays(root, split, view):
    records = load_lagper_records(root, split, view)
    paths = [osp.join(root, relative) for relative, _, _ in records]
    pids = np.asarray([pid for _, pid, _ in records], dtype=np.int64)
    camids = np.asarray([camid for _, _, camid in records], dtype=np.int64)
    return paths, pids, camids
