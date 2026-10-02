import unittest

from clustercontrast.utils.faiss_rerank import _resolve_modalities


class ModalJaccardTests(unittest.TestCase):
    def test_current_and_legacy_directory_names_resolve(self):
        samples = [
            ('/data/ground_modify/1/a.jpg', 0, 1),
            ('/data/aerial_modify/1/b.jpg', 0, 2),
            ('/data/rgb_modify/c.jpg', 0, 3),
            ('/data/ir_modify/d.jpg', 0, 4),
        ]
        self.assertEqual(
            _resolve_modalities(samples, 4),
            ['rgb', 'ir', 'rgb', 'ir'])

    def test_explicit_modalities_do_not_depend_on_directory_names(self):
        samples = [('/arbitrary/a.jpg', 0, 1), ('/other/b.jpg', 0, 2)]
        self.assertEqual(
            _resolve_modalities(samples, 2, modalities=['rgb', 'ir']),
            ['rgb', 'ir'])

    def test_unknown_path_fails_instead_of_creating_empty_means(self):
        with self.assertRaisesRegex(ValueError, 'Cannot infer modality'):
            _resolve_modalities([('/unknown/a.jpg', 0, 1)], 1)


if __name__ == '__main__':
    unittest.main()
