from __future__ import absolute_import

import gc
from pathlib import Path
import sys
import unittest
import weakref

import faiss
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clustercontrast.datasets.g2aps_common import G2APSDomainMetadata  # noqa: E402
from clustercontrast.domain_metadata import (  # noqa: E402
    DOMAIN_AERIAL,
    DOMAIN_GROUND,
    SUBDOMAIN_AERIAL_FAR,
    SUBDOMAIN_AERIAL_NEAR,
    SUBDOMAIN_GROUND,
    MissingDomainMetadataError,
    carry_domain_flags_by_basename,
    concatenate_domain_rows,
    cross_domain_cluster_ids,
    domain_flags_for_samples,
    filter_domain_rows,
    generate_domain_balanced_cluster_features,
    infer_legacy_domain_flags,
    partition_cross_domain_pseudo_samples,
    reorder_domain_rows,
    validate_domain_flags,
    validate_subdomain_flags,
)
from clustercontrast.memory_lifecycle import (  # noqa: E402
    cluster_precomputed_ephemeral,
)
from clustercontrast.three_domain import (  # noqa: E402
    build_aerial_subdomain_flags,
    build_bidirectional_bgm,
)
from clustercontrast.utils.faiss_rerank import (  # noqa: E402
    compute_modal_invariant_jaccard_distance,
    domain_balanced_query_expansion,
)


class DummyDataset:
    pass


def canonical_partition(labels):
    groups = {}
    for index, raw in enumerate(labels):
        label = int(raw)
        if label != -1:
            groups.setdefault(label, set()).add(index)
    return {frozenset(group) for group in groups.values()}


def legacy_balanced_prototypes(labels, num_clusters, features, paths):
    centers = []
    for cluster_id in range(num_clusters):
        aerial = [
            features[index] for index, label in enumerate(labels)
            if int(label) == cluster_id and "aerial_modify" in paths[index]
        ]
        ground = [
            features[index] for index, label in enumerate(labels)
            if int(label) == cluster_id and "ground_modify" in paths[index]
        ]
        aerial_mean = torch.stack(aerial).mean(0) if aerial else None
        ground_mean = torch.stack(ground).mean(0) if ground else None
        if aerial_mean is None:
            centers.append(ground_mean)
        elif ground_mean is None:
            centers.append(aerial_mean)
        else:
            centers.append(0.5 * (aerial_mean + ground_mean))
    return torch.stack(centers)


class TestDomainMetadataContract(unittest.TestCase):
    def setUp(self):
        self.samples = (
            ("bounding_box_train/1/1_c0v1_1.jpg", -1, 0),
            ("bounding_box_train/2/2_c3v1_1.jpg", -1, 3),
            ("bounding_box_train/3/3_c0v1_2.jpg", -1, 0),
        )
        self.dataset = DummyDataset()
        self.dataset.train = list(self.samples)
        self.dataset.domain_metadata = (
            G2APSDomainMetadata(self.samples[0][0], 0, "aerial"),
            G2APSDomainMetadata(self.samples[1][0], 3, "ground"),
            G2APSDomainMetadata(self.samples[2][0], 0, "aerial"),
        )
        self.dataset.domain_flags = (0, 3, 0)

    def test_c0_c3_mapping_and_reordered_alignment(self):
        flags = domain_flags_for_samples(self.dataset, self.samples)
        self.assertEqual(flags, (0, 3, 0))
        reordered = (self.samples[2], self.samples[0], self.samples[1])
        self.assertEqual(
            domain_flags_for_samples(self.dataset, reordered), (0, 0, 3))

    def test_concat_filter_and_reorder_keep_flags_aligned(self):
        first = (self.samples[:2], (0, 3))
        second = (self.samples[2:], (0,))
        rows = concatenate_domain_rows(first, second)
        self.assertEqual(rows.flags, (0, 3, 0))
        filtered = filter_domain_rows(rows.samples, rows.flags, [True, False, True])
        self.assertEqual(filtered.flags, (0, 0))
        reordered = reorder_domain_rows(rows.samples, rows.flags, [1, 2, 0])
        self.assertEqual(reordered.flags, (3, 0, 0))
        self.assertEqual(reordered.samples[0], self.samples[1])

    def test_path_remap_carries_flags_without_directory_inference(self):
        remapped = (
            ("enhanced/opaque/3_c0v1_2.jpg", -1, 0),
            ("enhanced/opaque/1_c0v1_1.jpg", -1, 0),
            ("enhanced/opaque/2_c3v1_1.jpg", -1, 3),
        )
        self.assertEqual(
            carry_domain_flags_by_basename(
                self.samples, (0, 3, 0), remapped),
            (0, 0, 3))
        with self.assertRaisesRegex(ValueError, "duplicate basename"):
            carry_domain_flags_by_basename(
                self.samples, (0, 3, 0), remapped + (remapped[0],))
        with self.assertRaises(MissingDomainMetadataError):
            carry_domain_flags_by_basename(
                self.samples, (0, 3, 0), remapped[:2])

    def test_missing_invalid_and_boolean_metadata_fail(self):
        with self.assertRaises(MissingDomainMetadataError):
            infer_legacy_domain_flags(self.samples)
        with self.assertRaisesRegex(ValueError, "invalid domain flag"):
            validate_domain_flags([0, 2, 3], 3)
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_domain_flags([False, 3], 2)
        del self.dataset.domain_metadata
        del self.dataset.domain_flags
        with self.assertRaises(MissingDomainMetadataError):
            domain_flags_for_samples(self.dataset, self.samples)

    def test_legacy_ag_lag_and_cargo_paths_remain_supported(self):
        legacy = [
            ("aerial_modify/1/bounding_box_train/a.jpg", -1, 7),
            ("ground_modify/1/bounding_box_train/g.jpg", -1, 9),
            ("ir_modify/train/i.jpg", -1, 1),
            ("rgb_modify/train/r.jpg", -1, 2),
            ("train/Cam3/Cam3_day_1_1.jpg", -1, 2),
            ("train/Cam11/Cam11_night_2_1.jpg", -1, 10),
        ]
        self.assertEqual(infer_legacy_domain_flags(legacy), (0, 3, 0, 3, 0, 3))

    def test_subdomain_is_separate_from_physical_camera(self):
        aerial = (self.samples[0], self.samples[2])
        subdomains = build_aerial_subdomain_flags(
            aerial, {aerial[0][0]}, {aerial[1][0]})
        self.assertEqual(
            subdomains, (SUBDOMAIN_AERIAL_FAR, SUBDOMAIN_AERIAL_NEAR))
        all_subdomains = (SUBDOMAIN_GROUND,) + subdomains
        self.assertEqual(
            validate_subdomain_flags(all_subdomains, (3, 0, 0)),
            all_subdomains)
        with self.assertRaisesRegex(ValueError, "Ground row"):
            validate_subdomain_flags((SUBDOMAIN_AERIAL_FAR,), (3,))


class TestSemanticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(17)
        cls.features = F.normalize(torch.randn(8, 6), dim=1)
        cls.flags = (3, 3, 3, 3, 0, 0, 0, 0)
        cls.legacy_samples = [
            ("ground_modify/1/bounding_box_train/g{}.jpg".format(i), -1, 3)
            if flag == 3 else
            ("aerial_modify/1/bounding_box_train/a{}.jpg".format(i), -1, 0)
            for i, flag in enumerate(cls.flags)
        ]
        cls.g2aps_samples = [
            ("bounding_box_train/{0}/{0}_c{1}v1_{2}.jpg".format(
                i + 1, flag, i), -1, flag)
            for i, flag in enumerate(cls.flags)
        ]

    def test_knn_indices_are_metadata_independent(self):
        array = self.features.numpy().astype(np.float32)
        index = faiss.IndexFlatL2(array.shape[1])
        index.add(array)
        _, old_rank = index.search(array, 4)
        _, explicit_rank = index.search(array, 4)
        np.testing.assert_array_equal(old_rank, explicit_rank)

    def test_query_expansion_and_dense_jaccard_match_legacy_paths(self):
        rng = np.random.RandomState(4)
        V = rng.rand(8, 8).astype(np.float32)
        initial_rank = np.argsort(rng.rand(8, 8), axis=1)
        legacy_flags = infer_legacy_domain_flags(self.legacy_samples)
        old_qe = domain_balanced_query_expansion(V, initial_rank, 3, legacy_flags)
        new_qe = domain_balanced_query_expansion(V, initial_rank, 3, self.flags)
        np.testing.assert_allclose(old_qe, new_qe, rtol=0, atol=0)

        old_distance = compute_modal_invariant_jaccard_distance(
            self.features, file=self.legacy_samples, k1=4, k2=2,
            print_flag=False, search_option=3)
        new_distance = compute_modal_invariant_jaccard_distance(
            self.features, file=self.g2aps_samples, domain_flags=self.flags,
            k1=4, k2=2, print_flag=False, search_option=3)
        np.testing.assert_allclose(old_distance, new_distance, rtol=0, atol=0)

        old_labels = DBSCAN(
            eps=0.8, min_samples=2, metric="precomputed").fit_predict(old_distance)
        new_labels = DBSCAN(
            eps=0.8, min_samples=2, metric="precomputed").fit_predict(new_distance)
        self.assertEqual(canonical_partition(old_labels), canonical_partition(new_labels))
        np.testing.assert_array_equal(old_labels == -1, new_labels == -1)

    def test_prototypes_bgm_and_all_translation_match(self):
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        paths = [sample[0] for sample in self.legacy_samples]
        old_centers = legacy_balanced_prototypes(labels, 2, self.features, paths)
        new_centers = generate_domain_balanced_cluster_features(
            labels, 2, self.features, self.flags)
        torch.testing.assert_close(old_centers, new_centers, rtol=0, atol=0)

        old_map = build_bidirectional_bgm(old_centers[:1], old_centers)
        new_map = build_bidirectional_bgm(new_centers[:1], new_centers)
        self.assertEqual(old_map, new_map)

        samples_with_real_pid = [
            (sample[0], 9000 + index, sample[2])
            for index, sample in enumerate(self.g2aps_samples)
        ]
        aerial, ground, valid = partition_cross_domain_pseudo_samples(
            samples_with_real_pid, labels, self.flags)
        self.assertEqual(valid, {0, 1})
        self.assertTrue(all(pid in (0, 1) for _, pid, _ in aerial + ground))
        self.assertFalse(any(pid >= 9000 for _, pid, _ in aerial + ground))
        self.assertEqual(
            cross_domain_cluster_ids(labels, self.flags), {0, 1})


class RecordingClusterer:
    eps = 0.5

    def fit_predict(self, distances):
        self.shape = distances.shape
        return np.asarray([0, 0, -1], dtype=np.int64)


class TestMemoryLifecycle(unittest.TestCase):
    def test_distance_is_released_before_next_builder(self):
        references = []

        def build():
            gc.collect()
            if references:
                self.assertIsNone(references[-1]())
            matrix = np.zeros((3, 3), dtype=np.float32)
            references.append(weakref.ref(matrix))
            return matrix

        for name in ("Aerial", "Ground", "ALL"):
            labels, count = cluster_precomputed_ephemeral(
                build, RecordingClusterer(), name)
            self.assertEqual(count, 1)
            np.testing.assert_array_equal(labels, [0, 0, -1])
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_base_and_three_domain_engines_use_ephemeral_lifecycle(self):
        base_source = (PROJECT_ROOT / "train_lag.py").read_text(
            encoding="utf-8")
        three_source = (PROJECT_ROOT / "_train_lag_three_domain_engine.py").read_text(
            encoding="utf-8")
        self.assertGreaterEqual(base_source.count("cluster_precomputed_ephemeral("), 3)
        self.assertNotIn("rerank_dist_all =", base_source)
        self.assertGreaterEqual(three_source.count("cluster_precomputed_ephemeral("), 4)
        self.assertNotIn("distance_all =", three_source)
        self.assertNotIn("distance_far =", three_source)


if __name__ == "__main__":
    unittest.main()
