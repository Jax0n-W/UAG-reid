import collections
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from clustercontrast.methods.rahp import (
    _cosine_knn_indices, compute_rahp_reliability,
    gather_batch_reliability, select_rahp_hard,
)
from clustercontrast.models.cm import cm_hybrid


def original_cm_hybrid_update(inputs, labels, initial, momentum):
    """Frozen original CM_Hybrid backward, kept here as the equivalence oracle."""
    memory = initial.clone()
    groups = collections.defaultdict(list)
    for feature, index in zip(inputs, labels.tolist()):
        groups[index].append(feature)
    count = len(memory) // 2
    for index, features in groups.items():
        distances = []
        for feature in features:
            distance = feature.unsqueeze(0).mm(memory[index].unsqueeze(0).t())[0][0]
            distances.append(distance.cpu().numpy())
        mean = torch.stack(features, dim=0).mean(0)
        memory[index] = memory[index] * momentum + (1 - momentum) * mean
        memory[index] /= memory[index].norm()
        hard = np.argmin(np.array(distances))
        memory[index + count] = memory[index + count] * momentum + (1 - momentum) * features[hard]
        memory[index + count] /= memory[index + count].norm()
    return memory


def hybrid_loss(outputs, labels):
    mean, hard = torch.chunk(outputs / 0.05, 2, dim=1)
    return 0.5 * (F.cross_entropy(hard, labels) +
                  torch.relu(F.cross_entropy(mean, labels) - 0.2))


class RAHPTests(unittest.TestCase):
    def test_off_path_matches_original_loss_mean_and_hard_memory(self):
        inputs = F.normalize(torch.tensor([[1., 0.], [0., 1.],
                                           [-1., 0.], [0., -1.]]), dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        initial = F.normalize(torch.tensor([[1., 0.2], [-1., 0.2],
                                            [0.2, 1.], [-0.2, -1.]]), dim=1)
        momentum = 0.2
        expected_outputs = inputs.mm(initial.t())
        expected_loss = hybrid_loss(expected_outputs, labels)
        expected_memory = original_cm_hybrid_update(inputs, labels, initial, momentum)
        memory = initial.clone()
        outputs = cm_hybrid(inputs.clone().requires_grad_(), labels, memory, momentum)
        loss = hybrid_loss(outputs, labels)
        loss.backward()
        self.assertTrue(torch.equal(loss, expected_loss))
        self.assertTrue(torch.equal(memory, expected_memory))

    def test_uses_old_mean_even_when_updated_mean_reverses_hardness(self):
        inputs = F.normalize(torch.tensor([[-0.15714002, 0.98757637],
                                           [0.66330892, -0.74834573]]), dim=1)
        old_mean = torch.tensor([1., 0.])
        new_mean = F.normalize(inputs.mean(0), dim=0)
        self.assertEqual(int((inputs @ old_mean).argmin()), 0)
        self.assertEqual(int((inputs @ new_mean).argmin()), 1)
        memory = torch.stack((old_mean, torch.tensor([0., 1.])))
        outputs = cm_hybrid(inputs.clone().requires_grad_(), torch.zeros(2, dtype=torch.long),
                            memory, momentum=0.0, reliability=torch.tensor([0.5, 0.5]),
                            rahp_beta=0.5)
        outputs.sum().backward()
        self.assertTrue(torch.allclose(memory[1], inputs[0], rtol=0, atol=1e-7))

    def test_candidate_count_and_reliability_choice(self):
        self.assertEqual(select_rahp_hard([0.2], [0.1], 0.25)[1], 1)
        self.assertEqual(select_rahp_hard([0.9, 0.8, 0.2, 0.1],
                                          [0.1, 0.9, 1.0, 1.0], 0.5)[0], 1)

    def test_ties_choose_hardness_then_original_batch_index(self):
        self.assertEqual(select_rahp_hard([0.8, 0.9], [0.5, 0.5], 1.0)[0], 1)
        self.assertEqual(select_rahp_hard([0.9, 0.9], [0.5, 0.5], 1.0,
                                          batch_positions=[8, 3])[0], 1)

    def test_outliers_and_filtered_dual_view_alignment(self):
        features = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [-1., 0.]])
        labels = torch.tensor([0, 0, -1, 1])
        q_full, diag = compute_rahp_reliability(
            features, labels, knn=2, return_diagnostics=True)
        self.assertEqual(q_full.numel(), 4)
        self.assertEqual(float(q_full[2]), 0.0)
        self.assertEqual(diag['num_valid_samples'], 3)
        q_filtered = q_full[labels != -1]
        q_batch = gather_batch_reliability(q_filtered, [2, 0], duplicate_views=True)
        batch_labels = torch.tensor([1, 0, 1, 0])
        batch_features = torch.randn(4, 2)
        self.assertEqual(len(batch_features), len(batch_labels))
        self.assertEqual(len(batch_labels), len(q_batch))
        self.assertTrue(torch.equal(q_batch[:2], q_batch[2:]))

    def test_exact_knn_backends_match_brute_force_cosine(self):
        vectors = F.normalize(torch.tensor([
            [1.0, 0.0, 0.1],
            [0.8, 0.2, 0.0],
            [0.1, 1.0, 0.0],
            [-0.7, 0.1, 0.3],
            [0.0, -0.8, 0.4],
        ]), dim=1)
        scores = vectors.mm(vectors.t())
        scores.fill_diagonal_(-float('inf'))
        expected = scores.topk(2, dim=1).indices
        torch_result, torch_backend = _cosine_knn_indices(
            vectors, 2, backend='torch_chunk', return_backend=True)
        self.assertEqual(torch_backend, 'torch_chunk')
        self.assertTrue(torch.equal(torch_result, expected))
        try:
            faiss_result, faiss_backend = _cosine_knn_indices(
                vectors, 2, backend='cpu_faiss', return_backend=True)
        except RuntimeError:
            self.skipTest('CPU FAISS unavailable')
        self.assertEqual(faiss_backend, 'cpu_faiss')
        for actual, wanted in zip(faiss_result, expected):
            self.assertEqual(set(actual.tolist()), set(wanted.tolist()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_faiss_matches_brute_force_when_available(self):
        vectors = F.normalize(torch.tensor([
            [1.0, 0.0], [0.9, 0.2], [0.0, 1.0], [-0.8, 0.1],
        ]), dim=1)
        expected_scores = vectors.mm(vectors.t())
        expected_scores.fill_diagonal_(-float('inf'))
        expected = expected_scores.topk(2, dim=1).indices
        try:
            actual, backend = _cosine_knn_indices(
                vectors, 2, backend='gpu_faiss', return_backend=True)
        except RuntimeError:
            self.skipTest('GPU FAISS unavailable')
        self.assertEqual(backend, 'gpu_faiss')
        for row, wanted in zip(actual, expected):
            self.assertEqual(set(row.tolist()), set(wanted.tolist()))


if __name__ == '__main__':
    unittest.main()
