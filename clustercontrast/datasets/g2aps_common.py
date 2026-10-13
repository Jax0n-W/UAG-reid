from __future__ import absolute_import, print_function

from dataclasses import dataclass
from pathlib import Path
import re

from ..utils.data import BaseImageDataset


G2APS_FILENAME_PATTERN = re.compile(
    r"^(?P<pid>\d+)_c(?P<camid>[03])v(?P<sequence>\d+)_(?P<frame>\d+)\.jpg$"
)
UNSUPERVISED_TRAIN_PID = -1
PID_5555 = 5555
CAMERA_TO_DOMAIN = {0: "aerial", 3: "ground"}
EXPECTED_TRAIN_COUNTS = {"aerial": 65482, "ground": 35389}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class G2APSFilename:
    pid: int
    camid: int
    sequence: int
    frame: int

    @property
    def domain(self):
        return CAMERA_TO_DOMAIN[self.camid]


@dataclass(frozen=True)
class G2APSDomainMetadata:
    """Public per-row metadata without the ground-truth training identity."""

    relative_path: str
    camid: int
    domain: str


def parse_g2aps_filename(path):
    name = Path(path).name
    match = G2APS_FILENAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            "Invalid G2APS image name {!r}; expected PID_c[03]vSEQ_FRAME.jpg".format(
                name
            )
        )
    parsed = G2APSFilename(
        pid=int(match.group("pid")),
        camid=int(match.group("camid")),
        sequence=int(match.group("sequence")),
        frame=int(match.group("frame")),
    )
    if parsed.pid <= 0:
        raise ValueError("G2APS PID must be positive: {}".format(path))
    return parsed


def _stable_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise RuntimeError("G2APS directory is unavailable: {}".format(directory))
    images = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            if suffix != ".jpg":
                raise ValueError("Unexpected non-JPEG G2APS image: {}".format(path))
            images.append(path)
    return sorted(images, key=lambda item: item.as_posix())


def _relative_posix(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Path escapes the G2APS root: {}".format(path)) from exc


def assert_domain_metadata_aligned(samples, metadata):
    if len(samples) != len(metadata):
        raise AssertionError(
            "Domain metadata length {} does not match feature rows {}".format(
                len(metadata), len(samples)
            )
        )
    for index, ((path, _, camid), item) in enumerate(zip(samples, metadata)):
        if path != item.relative_path or camid != item.camid:
            raise AssertionError(
                "Domain metadata is misaligned at row {}: {} != {}".format(
                    index, (path, camid), (item.relative_path, item.camid)
                )
            )
        parsed = parse_g2aps_filename(path)
        if parsed.camid != item.camid or parsed.domain != item.domain:
            raise AssertionError("Incorrect domain metadata at row {}".format(index))
    return True


def replace_unsupervised_labels(samples, pseudo_labels, drop_noise=True):
    """Replace placeholder labels with DBSCAN labels without exposing true PIDs.

    This mirrors the tuple reconstruction performed by the current PCLHD engines.
    It is intentionally independent from clustering and does not run DBSCAN.
    """

    if len(samples) != len(pseudo_labels):
        raise ValueError(
            "Pseudo-label count {} does not match sample count {}".format(
                len(pseudo_labels), len(samples)
            )
        )
    replaced = []
    for (path, train_pid, camid), value in zip(samples, pseudo_labels):
        if train_pid != UNSUPERVISED_TRAIN_PID:
            raise ValueError(
                "Expected an unsupervised placeholder at {}, got {}".format(
                    path, train_pid
                )
            )
        label = value.item() if hasattr(value, "item") else value
        label = int(label)
        if label < -1:
            raise ValueError("Invalid pseudo-label {} for {}".format(label, path))
        if label == -1 and drop_noise:
            continue
        replaced.append((path, label, camid))
    return replaced


class G2APSTrainDomain(BaseImageDataset):
    """One G2APS training domain with identity-free public training tuples."""

    domain_name = None
    camera_id = None

    def __init__(
        self,
        root,
        trial=0,
        verbose=True,
        expected_train_count=None,
        verify_isolation=True,
        **kwargs
    ):
        super(G2APSTrainDomain, self).__init__()
        del trial, kwargs
        self.dataset_dir = str(Path(root).expanduser().resolve())
        self.train_dir = str(Path(self.dataset_dir) / "bounding_box_train")
        if self.domain_name not in EXPECTED_TRAIN_COUNTS:
            raise RuntimeError("G2APS adapter has no valid domain_name")
        if self.camera_id not in CAMERA_TO_DOMAIN:
            raise RuntimeError("G2APS adapter has no valid camera_id")
        if CAMERA_TO_DOMAIN[self.camera_id] != self.domain_name:
            raise RuntimeError("G2APS adapter camera/domain declaration is inconsistent")
        if expected_train_count is None:
            expected_train_count = EXPECTED_TRAIN_COUNTS[self.domain_name]

        train, metadata, audit = self._load_train(
            expected_train_count=expected_train_count,
            verify_isolation=verify_isolation,
        )
        self.train = train
        self.query = []
        self.gallery = []
        self.domain_metadata = tuple(metadata)
        self.domain_flags = tuple(item.camid for item in metadata)
        self.audit_summary = audit
        assert_domain_metadata_aligned(self.train, self.domain_metadata)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = (
            self.get_imagedata_info(self.train)
        )
        self.num_query_pids = self.num_query_imgs = self.num_query_cams = 0
        self.num_gallery_pids = self.num_gallery_imgs = self.num_gallery_cams = 0
        if verbose:
            print(
                "=> G2APS {} loaded: {} identity-free training images".format(
                    self.domain_name, len(self.train)
                )
            )

    @property
    def images_dir(self):
        return self.dataset_dir

    def _load_train(self, expected_train_count, verify_isolation):
        root = Path(self.dataset_dir)
        train_root = Path(self.train_dir)
        images = _stable_images(train_root)
        if not images:
            raise RuntimeError("G2APS training directory contains no images")

        selected = []
        metadata = []
        train_pids = set()
        relative_paths = set()
        physical_files = {}
        record_keys = set()

        for path in images:
            parsed = parse_g2aps_filename(path)
            parent = path.parent.name
            if not parent.isdigit() or int(parent) != parsed.pid:
                raise ValueError(
                    "G2APS PID directory does not match filename: {}".format(path)
                )
            if parsed.pid == PID_5555:
                raise ValueError("PID 5555 is forbidden in G2APS training: {}".format(path))
            relative = _relative_posix(path, root)
            if relative in relative_paths:
                raise ValueError("Duplicate G2APS training path: {}".format(relative))
            relative_paths.add(relative)

            stat = path.stat()
            inode_key = (stat.st_dev, stat.st_ino)
            if stat.st_ino and inode_key in physical_files:
                raise ValueError(
                    "Duplicate physical G2APS training image: {} and {}".format(
                        physical_files[inode_key], relative
                    )
                )
            physical_files[inode_key] = relative

            record_key = (parsed.pid, parsed.camid, parsed.sequence, parsed.frame)
            if record_key in record_keys:
                raise ValueError("Duplicate G2APS training record: {}".format(path))
            record_keys.add(record_key)
            train_pids.add(parsed.pid)

            if parsed.camid == self.camera_id:
                selected.append((relative, UNSUPERVISED_TRAIN_PID, parsed.camid))
                metadata.append(
                    G2APSDomainMetadata(relative, parsed.camid, parsed.domain)
                )

        if len(selected) != int(expected_train_count):
            raise RuntimeError(
                "Unexpected G2APS {} training count: expected {}, found {}".format(
                    self.domain_name, expected_train_count, len(selected)
                )
            )

        test_pids = set()
        train_resolved = {path.resolve() for path in images}
        train_inodes = set()
        for path in images:
            stat = path.stat()
            if stat.st_ino:
                train_inodes.add((stat.st_dev, stat.st_ino))
        test_resolved = set()
        test_inodes = set()
        if verify_isolation:
            for name in (
                "bounding_box_test_c0",
                "bounding_box_test_c3",
                "query_c0",
                "query_c3",
                "bounding_box_gallery",
            ):
                split_root = root / name
                if not split_root.exists():
                    continue
                for path in _stable_images(split_root):
                    parsed = parse_g2aps_filename(path)
                    test_resolved.add(path.resolve())
                    stat = path.stat()
                    if stat.st_ino:
                        test_inodes.add((stat.st_dev, stat.st_ino))
                    if parsed.pid != PID_5555:
                        test_pids.add(parsed.pid)
            overlap = sorted(train_pids & test_pids)
            if overlap:
                raise ValueError(
                    "G2APS train/test identity overlap detected: {}".format(overlap[:20])
                )
            path_overlap = train_resolved & test_resolved
            if path_overlap:
                raise ValueError(
                    "G2APS train/test path overlap detected: {}".format(
                        sorted(str(path) for path in path_overlap)[:20]
                    )
                )
            inode_overlap = train_inodes & test_inodes
            if inode_overlap:
                raise ValueError(
                    "G2APS training images are physically reused in test/query data"
                )

        selected.sort(key=lambda item: item[0])
        metadata.sort(key=lambda item: item.relative_path)
        audit = {
            "domain": self.domain_name,
            "camera_id": self.camera_id,
            "images": len(selected),
            "training_identity_count_used_for_audit_only": len(train_pids),
            "public_training_label": UNSUPERVISED_TRAIN_PID,
            "train_test_identity_overlap": 0,
            "pid_5555_in_train": 0,
            "lazy_loading": True,
        }
        return selected, metadata, audit
