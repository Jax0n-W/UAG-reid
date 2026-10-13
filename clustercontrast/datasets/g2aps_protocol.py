from __future__ import absolute_import, print_function

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

from ..utils.data import BaseImageDataset
from .g2aps_common import (
    CAMERA_TO_DOMAIN,
    G2APSDomainMetadata,
    PID_5555,
    assert_domain_metadata_aligned,
    parse_g2aps_filename,
)


PROTOCOL_FILES = {
    "query_c0": "query_c0.txt",
    "query_c3": "query_c3.txt",
    "A2G_plus": "exp3_A2G+.txt",
    "G2A_plus": "exp4_G2A+.txt",
}
OFFICIAL_PROTOCOL_COUNTS = {
    "query_c0": 4876,
    "query_c3": 4876,
    "A2G_plus": 42078,
    "G2A_plus": 67667,
}
OFFICIAL_QUERY_PID_COUNT = 1219


@dataclass(frozen=True)
class G2APSProtocolSummary:
    protocol: str
    query_images: int
    gallery_images: int
    query_pids: int
    gallery_labeled_images: int
    gallery_distractors: int
    queries_without_positive: int
    minimum_positive_count: int
    maximum_positive_count: int
    lazy_loading: bool = True


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_protocol_lines(path):
    lines = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                raise ValueError("Empty protocol entry at {}:{}".format(path, number))
            posix = PurePosixPath(value.replace("\\", "/"))
            if posix.is_absolute() or ".." in posix.parts:
                raise ValueError("Unsafe protocol path at {}:{}: {}".format(path, number, value))
            normalized = posix.as_posix()
            if normalized != value.replace("\\", "/"):
                raise ValueError("Non-canonical protocol path at {}:{}".format(path, number))
            lines.append(normalized)
    if len(lines) != len(set(lines)):
        duplicates = [item for item, count in Counter(lines).items() if count > 1]
        raise ValueError(
            "Duplicate paths in protocol {}: {}".format(path, duplicates[:20])
        )
    return lines


def _load_and_validate_protocol_files(root, expected_counts):
    root = Path(root).expanduser().resolve()
    protocols_root = root / "protocols"
    metadata_path = protocols_root / "protocol_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("Missing G2APS protocol metadata: {}".format(metadata_path))
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("G2A_plus_G_defined") is not False:
        raise ValueError("P1 metadata must explicitly state that G2A+G is undefined")
    metadata_counts = metadata.get("counts")
    metadata_hashes = metadata.get("sha256")
    if not isinstance(metadata_counts, dict) or not isinstance(metadata_hashes, dict):
        raise ValueError("protocol_metadata.json lacks counts or sha256 maps")

    lists = {}
    for key, filename in PROTOCOL_FILES.items():
        path = protocols_root / filename
        if not path.is_file():
            raise RuntimeError("Missing G2APS protocol list: {}".format(path))
        actual_hash = _sha256(path)
        if metadata_hashes.get(key) != actual_hash:
            raise ValueError(
                "SHA256 mismatch for {}: metadata={}, actual={}".format(
                    key, metadata_hashes.get(key), actual_hash
                )
            )
        lists[key] = _read_protocol_lines(path)
        actual_count = len(lists[key])
        if metadata_counts.get(key) != actual_count:
            raise ValueError(
                "Metadata count mismatch for {}: metadata={}, actual={}".format(
                    key, metadata_counts.get(key), actual_count
                )
            )
        if expected_counts is not None and actual_count != int(expected_counts[key]):
            raise ValueError(
                "Official count mismatch for {}: expected {}, actual {}".format(
                    key, expected_counts[key], actual_count
                )
            )
    return metadata, lists


class G2APSProtocolDataset(BaseImageDataset):
    """Fixed P1 A->G+ or G->A+ protocol with lazy image loading."""

    def __init__(
        self,
        root,
        protocol,
        verbose=True,
        expected_counts=None,
        expected_query_pid_count=OFFICIAL_QUERY_PID_COUNT,
    ):
        super(G2APSProtocolDataset, self).__init__()
        self.dataset_dir = str(Path(root).expanduser().resolve())
        self.protocol = protocol
        if protocol not in ("A2G+", "G2A+"):
            raise ValueError("Unsupported G2APS protocol: {}".format(protocol))
        if expected_counts is None:
            expected_counts = OFFICIAL_PROTOCOL_COUNTS
        metadata, lists = _load_and_validate_protocol_files(
            self.dataset_dir, expected_counts
        )
        self.protocol_metadata = metadata
        self.train = []
        self.query, self.gallery, query_meta, gallery_meta, summary = (
            self._build_protocol(lists, expected_query_pid_count)
        )
        self.query_domain_metadata = tuple(query_meta)
        self.gallery_domain_metadata = tuple(gallery_meta)
        self.query_domain_flags = tuple(item.camid for item in query_meta)
        self.gallery_domain_flags = tuple(item.camid for item in gallery_meta)
        assert_domain_metadata_aligned(self.query, self.query_domain_metadata)
        assert_domain_metadata_aligned(self.gallery, self.gallery_domain_metadata)
        self.summary = summary

        self.num_train_pids = self.num_train_imgs = self.num_train_cams = 0
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = (
            self.get_imagedata_info(self.query)
        )
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = (
            self.get_imagedata_info(self.gallery)
        )
        if verbose:
            print(
                "=> G2APS {} loaded: query={}, gallery={}, distractors={}".format(
                    protocol,
                    len(self.query),
                    len(self.gallery),
                    summary.gallery_distractors,
                )
            )

    @property
    def images_dir(self):
        return self.dataset_dir

    def _build_protocol(self, lists, expected_query_pid_count):
        if self.protocol == "A2G+":
            query_key, experiment_key = "query_c0", "A2G_plus"
            query_cam, gallery_cam = 0, 3
            query_prefix, gallery_prefix = "query_c0", "bounding_box_test_c3"
        else:
            query_key, experiment_key = "query_c3", "G2A_plus"
            query_cam, gallery_cam = 3, 0
            query_prefix, gallery_prefix = "query_c3", "bounding_box_test_c0"

        query_paths = lists[query_key]
        experiment = lists[experiment_key]
        if experiment[: len(query_paths)] != query_paths:
            raise ValueError(
                "{} must begin with the fixed {} query list".format(
                    experiment_key, query_key
                )
            )
        gallery_paths = experiment[len(query_paths) :]
        overlap = set(query_paths) & set(gallery_paths)
        if overlap:
            raise ValueError(
                "Query/gallery path overlap in {}: {}".format(
                    self.protocol, sorted(overlap)[:20]
                )
            )

        query, query_meta = self._parse_split(
            query_paths,
            role="query",
            expected_camera=query_cam,
            expected_labeled_prefix=query_prefix,
        )
        gallery, gallery_meta = self._parse_split(
            gallery_paths,
            role="gallery",
            expected_camera=gallery_cam,
            expected_labeled_prefix=gallery_prefix,
        )

        query_pids = {pid for _, pid, _ in query}
        if PID_5555 in query_pids:
            raise ValueError("PID 5555 is forbidden in G2APS queries")
        if expected_query_pid_count is not None and len(query_pids) != int(
            expected_query_pid_count
        ):
            raise ValueError(
                "Unexpected {} query PID count: expected {}, actual {}".format(
                    self.protocol, expected_query_pid_count, len(query_pids)
                )
            )

        gallery_pid_counts = Counter(pid for _, pid, _ in gallery if pid != PID_5555)
        positive_counts = [gallery_pid_counts.get(pid, 0) for _, pid, _ in query]
        missing = [query[index] for index, count in enumerate(positive_counts) if count == 0]
        if missing:
            raise ValueError(
                "{} queries have no legal cross-view positive; examples: {}".format(
                    len(missing), missing[:20]
                )
            )
        # All positives survive the standard same-PID/same-camera filter because
        # this adapter enforces opposite cameras for query and gallery.
        summary = G2APSProtocolSummary(
            protocol=self.protocol,
            query_images=len(query),
            gallery_images=len(gallery),
            query_pids=len(query_pids),
            gallery_labeled_images=sum(
                1 for _, pid, _ in gallery if pid != PID_5555
            ),
            gallery_distractors=sum(1 for _, pid, _ in gallery if pid == PID_5555),
            queries_without_positive=0,
            minimum_positive_count=min(positive_counts),
            maximum_positive_count=max(positive_counts),
        )
        return query, gallery, query_meta, gallery_meta, summary

    def _parse_split(self, relative_paths, role, expected_camera, expected_labeled_prefix):
        root = Path(self.dataset_dir).resolve()
        samples = []
        domain_metadata = []
        resolved_paths = set()
        for relative in relative_paths:
            parts = PurePosixPath(relative).parts
            if not parts:
                raise ValueError("Empty G2APS {} path".format(role))
            top = parts[0]
            parsed = parse_g2aps_filename(relative)
            if parsed.camid != expected_camera:
                raise ValueError(
                    "{} camera mismatch in {}: expected c{}, found c{}".format(
                        role, relative, expected_camera, parsed.camid
                    )
                )
            if parsed.pid == PID_5555:
                if role != "gallery" or top != "bounding_box_gallery":
                    raise ValueError(
                        "PID 5555 is allowed only in bounding_box_gallery: {}".format(
                            relative
                        )
                    )
            else:
                if top != expected_labeled_prefix:
                    raise ValueError(
                        "Unexpected labeled {} path prefix in {}".format(role, relative)
                    )
                if len(parts) < 3 or not parts[-2].isdigit() or int(parts[-2]) != parsed.pid:
                    raise ValueError(
                        "PID directory does not match filename in {}".format(relative)
                    )

            full_path = (root / Path(*parts)).resolve()
            try:
                full_path.relative_to(root)
            except ValueError as exc:
                raise ValueError("Protocol path escapes dataset root: {}".format(relative)) from exc
            if not full_path.is_file():
                raise RuntimeError("Protocol image does not exist: {}".format(full_path))
            if full_path in resolved_paths:
                raise ValueError("Duplicate resolved {} path: {}".format(role, full_path))
            resolved_paths.add(full_path)
            normalized = PurePosixPath(relative).as_posix()
            samples.append((normalized, parsed.pid, parsed.camid))
            domain_metadata.append(
                G2APSDomainMetadata(normalized, parsed.camid, CAMERA_TO_DOMAIN[parsed.camid])
            )
        return samples, domain_metadata


def load_g2aps_protocol(root, protocol, **kwargs):
    return G2APSProtocolDataset(root, protocol=protocol, **kwargs)
