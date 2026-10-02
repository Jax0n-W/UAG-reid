import unittest

import torch
import torch.nn.functional as F

from clustercontrast.methods.cesa import CESAState
from clustercontrast.models.cm import cm_hybrid
from tests.test_pgm_integration import run_actual_stage2_pgm


class AblationSmokeTests(unittest.TestCase):
    def test_four_method_combinations_run_shared_memory_and_actual_pgm(self):
        inputs = F.normalize(torch.tensor([[1., 0.], [0., 1.],
                                           [-1., 0.], [0., -1.]]), dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        initial = F.normalize(torch.tensor([[1., 0.2], [-1., 0.2],
                                            [0.2, 1.], [-0.2, -1.]]), dim=1)
        for script in ('train_agreid.py', 'train_lag.py'):
            for rahp in (False, True):
                for cesa in (False, True):
                    with self.subTest(script=script, rahp=rahp, cesa=cesa):
                        memory = initial.clone()
                        reliability = torch.tensor([0.2, 0.9, 0.8, 0.3]) if rahp else None
                        outputs = cm_hybrid(inputs.clone().requires_grad_(), labels,
                                            memory, 0.2, reliability=reliability)
                        outputs.sum().backward()
                        self.assertTrue(torch.isfinite(memory).all())
                        state = CESAState() if cesa else None
                        edges, r2i, i2r = run_actual_stage2_pgm(script, state)
                        self.assertTrue(edges and r2i and i2r)


if __name__ == '__main__':
    unittest.main()
