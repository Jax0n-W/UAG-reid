from __future__ import absolute_import

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustercontrast import datasets  # noqa: E402
from clustercontrast.datasets.g2aps_common import (  # noqa: E402
    UNSUPERVISED_TRAIN_PID,
    assert_domain_metadata_aligned,
    replace_unsupervised_labels,
)
from clustercontrast.datasets.g2aps_protocol import (  # noqa: E402
    G2APSProtocolDataset,
)
from clustercontrast.utils.data.preprocessor import Preprocessor  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class G2APSFixture:
    counts = {
        "query_c0": 2,
        "query_c3": 2,
        "A2G_plus": 5,
        "G2A_plus": 5,
    }

    def __init__(self, root):
        self.root = Path(root)
        self.protocols = self.root / "protocols"
        self.protocols.mkdir(parents=True)
        self._make_train()
        self._make_test()
        self.write_protocols()

    def image(self, relative, color=(20, 40, 60)):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 12), color).save(path)
        return relative.replace("\\", "/")

    def _make_train(self):
        self.image("bounding_box_train/1/1_c0v1_1.jpg")
        self.image("bounding_box_train/1/1_c0v1_2.jpg")
        self.image("bounding_box_train/2/2_c3v2_1.jpg")
        self.image("bounding_box_train/2/2_c3v2_2.jpg")

    def _make_test(self):
        self.image("query_c0/10/10_c0v1_1.jpg")
        self.image("query_c0/11/11_c0v1_1.jpg")
        self.image("query_c3/10/10_c3v1_1.jpg")
        self.image("query_c3/11/11_c3v1_1.jpg")
        self.image("bounding_box_test_c0/10/10_c0v2_1.jpg")
        self.image("bounding_box_test_c0/11/11_c0v2_1.jpg")
        self.image("bounding_box_test_c3/10/10_c3v2_1.jpg")
        self.image("bounding_box_test_c3/11/11_c3v2_1.jpg")
        self.image("bounding_box_gallery/5555_c0v9_1.jpg")
        self.image("bounding_box_gallery/5555_c3v9_1.jpg")

    def write_protocols(self):
        lists = {
            "query_c0": [
                "query_c0/10/10_c0v1_1.jpg",
                "query_c0/11/11_c0v1_1.jpg",
            ],
            "query_c3": [
                "query_c3/10/10_c3v1_1.jpg",
                "query_c3/11/11_c3v1_1.jpg",
            ],
        }
        lists["A2G_plus"] = lists["query_c0"] + [
            "bounding_box_test_c3/10/10_c3v2_1.jpg",
            "bounding_box_test_c3/11/11_c3v2_1.jpg",
            "bounding_box_gallery/5555_c3v9_1.jpg",
        ]
        lists["G2A_plus"] = lists["query_c3"] + [
            "bounding_box_test_c0/10/10_c0v2_1.jpg",
            "bounding_box_test_c0/11/11_c0v2_1.jpg",
            "bounding_box_gallery/5555_c0v9_1.jpg",
        ]
        filenames = {
            "query_c0": "query_c0.txt",
            "query_c3": "query_c3.txt",
            "A2G_plus": "exp3_A2G+.txt",
            "G2A_plus": "exp4_G2A+.txt",
        }
        paths = {}
        for key, filename in filenames.items():
            path = self.protocols / filename
            path.write_text("\n".join(lists[key]) + "\n", encoding="utf-8")
            paths[key] = path
        metadata = {
            "G2A_plus_G_defined": False,
            "counts": {key: len(value) for key, value in lists.items()},
            "sha256": {key: sha256(path) for key, path in paths.items()},
        }
        (self.protocols / "protocol_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


class TestG2APSTrainAdapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = G2APSFixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def create_aerial(self, **kwargs):
        expected_train_count = kwargs.pop("expected_train_count", 2)
        return datasets.create(
            "g2aps_aerial",
            self.temp.name,
            expected_train_count=expected_train_count,
            verbose=False,
            **kwargs
        )

    def test_registry_preserves_old_adapters_and_registers_g2aps(self):
        registered = set(datasets.names())
        for name in (
            "agreid_ir",
            "agreid_rgb",
            "lag_ir",
            "lag_rgb",
            "g2aps_aerial",
            "g2aps_ground",
        ):
            self.assertIn(name, registered)

    def test_c0_c3_split_is_stable_and_identity_free(self):
        aerial = self.create_aerial()
        ground = datasets.create(
            "g2aps_ground",
            self.temp.name,
            expected_train_count=2,
            verbose=False,
        )
        self.assertEqual([item[2] for item in aerial.train], [0, 0])
        self.assertEqual([item[2] for item in ground.train], [3, 3])
        self.assertTrue(all(item[1] == UNSUPERVISED_TRAIN_PID for item in aerial.train))
        self.assertTrue(all("/1/" in item[0] for item in aerial.train))
        self.assertEqual(aerial.train, sorted(aerial.train))
        self.assertTrue(assert_domain_metadata_aligned(aerial.train, aerial.domain_metadata))
        self.assertEqual(len(aerial.domain_flags), len(aerial.train))

    def test_invalid_filename_fails_loudly(self):
        self.fixture.image("bounding_box_train/1/not_a_g2aps_name.jpg")
        with self.assertRaisesRegex(ValueError, "Invalid G2APS image name"):
            self.create_aerial(expected_train_count=3)

    def test_pid_5555_is_forbidden_in_train(self):
        self.fixture.image("bounding_box_train/5555/5555_c0v1_1.jpg")
        with self.assertRaisesRegex(ValueError, "PID 5555"):
            self.create_aerial(expected_train_count=3)

    def test_train_test_identity_overlap_is_rejected(self):
        self.fixture.image("bounding_box_test_c0/1/1_c0v8_8.jpg")
        with self.assertRaisesRegex(ValueError, "train/test identity overlap"):
            self.create_aerial()

    def test_train_test_physical_file_reuse_is_rejected(self):
        source = Path(self.temp.name) / "bounding_box_train/1/1_c0v1_1.jpg"
        target = Path(self.temp.name) / "bounding_box_test_c0/10/10_c0v7_7.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        os.link(source, target)
        with self.assertRaisesRegex(ValueError, "physically reused"):
            self.create_aerial()

    def test_pseudo_labels_replace_placeholders_before_sampling(self):
        aerial = self.create_aerial()
        pseudo = replace_unsupervised_labels(aerial.train, np.asarray([7, 7]))
        self.assertEqual([item[1] for item in pseudo], [7, 7])
        self.assertTrue(all(pid >= 0 for _, pid, _ in pseudo))
        self.assertEqual(
            [(path, camid) for path, _, camid in pseudo],
            [(path, camid) for path, _, camid in aerial.train],
        )

    def test_preprocessor_reads_lazily(self):
        with mock.patch("PIL.Image.open", side_effect=AssertionError("eager decode")):
            aerial = self.create_aerial()
        preprocessor = Preprocessor(aerial.train, root=aerial.images_dir)
        image, path, pid, camid, index = preprocessor[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual((pid, camid, index), (-1, 0, 0))
        self.assertEqual(path, aerial.train[0][0])


class TestG2APSProtocolAdapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = G2APSFixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, protocol="A2G+"):
        return G2APSProtocolDataset(
            self.temp.name,
            protocol,
            expected_counts=self.fixture.counts,
            expected_query_pid_count=2,
            verbose=False,
        )

    def test_fixed_a2g_and_g2a_protocols(self):
        for protocol, query_cam, gallery_cam in (("A2G+", 0, 3), ("G2A+", 3, 0)):
            dataset = self.load(protocol)
            self.assertEqual(len(dataset.query), 2)
            self.assertEqual(len(dataset.gallery), 3)
            self.assertTrue(all(item[2] == query_cam for item in dataset.query))
            self.assertTrue(all(item[2] == gallery_cam for item in dataset.gallery))
            self.assertEqual(dataset.summary.gallery_distractors, 1)
            self.assertEqual(dataset.summary.queries_without_positive, 0)
            self.assertEqual(dataset.summary.minimum_positive_count, 1)
            self.assertEqual(len(dataset.query_domain_flags), len(dataset.query))
            self.assertEqual(len(dataset.gallery_domain_flags), len(dataset.gallery))
            self.assertEqual(set(dataset.query_domain_flags), {query_cam})
            self.assertEqual(set(dataset.gallery_domain_flags), {gallery_cam})

    def test_pid_5555_is_gallery_only_and_never_a_positive(self):
        dataset = self.load("A2G+")
        self.assertNotIn(5555, {pid for _, pid, _ in dataset.query})
        self.assertEqual(sum(pid == 5555 for _, pid, _ in dataset.gallery), 1)
        query_pids = {pid for _, pid, _ in dataset.query}
        self.assertNotIn(5555, query_pids)

    def test_sha256_mismatch_is_rejected(self):
        path = self.fixture.protocols / "query_c0.txt"
        path.write_text(path.read_text(encoding="utf-8") + "x\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.load()

    def test_missing_referenced_image_is_rejected(self):
        (Path(self.temp.name) / "bounding_box_test_c3/10/10_c3v2_1.jpg").unlink()
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            self.load()

    def test_query_gallery_path_overlap_is_rejected(self):
        a2g_path = self.fixture.protocols / "exp3_A2G+.txt"
        values = a2g_path.read_text(encoding="utf-8").splitlines()
        values[-1] = values[0]
        a2g_path.write_text("\n".join(values) + "\n", encoding="utf-8")
        metadata_path = self.fixture.protocols / "protocol_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["sha256"]["A2G_plus"] = sha256(a2g_path)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate paths|Query/gallery path overlap"):
            self.load()

    def test_protocol_adapter_is_lazy_and_preprocessor_compatible(self):
        with mock.patch("PIL.Image.open", side_effect=AssertionError("eager decode")):
            dataset = self.load("G2A+")
        image, _, pid, camid, _ = Preprocessor(
            dataset.query, root=dataset.images_dir
        )[0]
        self.assertEqual(image.size, (8, 12))
        self.assertIn(pid, (10, 11))
        self.assertEqual(camid, 3)


if __name__ == "__main__":
    unittest.main()
