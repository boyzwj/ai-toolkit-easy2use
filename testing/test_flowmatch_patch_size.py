from types import SimpleNamespace
import unittest

from toolkit.flowmatch_utils import resolve_flowmatch_patch_size


class ResolveFlowmatchPatchSizeTest(unittest.TestCase):
    def test_krea2_patch_field_is_supported(self):
        sd = SimpleNamespace(
            is_flux=False,
            arch="krea2",
            unet=SimpleNamespace(config=SimpleNamespace(patch=2)),
        )

        self.assertEqual(resolve_flowmatch_patch_size(sd), 2)

    def test_standard_patch_size_field_is_preferred(self):
        sd = SimpleNamespace(
            is_flux=False,
            arch="example",
            patch_size=8,
            unet=SimpleNamespace(config=SimpleNamespace(patch_size=4, patch=2)),
        )

        self.assertEqual(resolve_flowmatch_patch_size(sd), 4)

    def test_model_wrapper_patch_size_is_a_fallback(self):
        sd = SimpleNamespace(
            is_flux=False,
            arch="example",
            patch_size=2,
            unet=SimpleNamespace(config=SimpleNamespace()),
        )

        self.assertEqual(resolve_flowmatch_patch_size(sd), 2)

    def test_unknown_model_falls_back_to_one(self):
        sd = SimpleNamespace(
            is_flux=False,
            arch="example",
            unet=SimpleNamespace(config=SimpleNamespace()),
        )

        self.assertEqual(resolve_flowmatch_patch_size(sd), 1)

    def test_flux_keeps_effective_patch_size_two(self):
        sd = SimpleNamespace(
            is_flux=True,
            arch="flux",
            unet=SimpleNamespace(config=SimpleNamespace(patch_size=1)),
        )

        self.assertEqual(resolve_flowmatch_patch_size(sd), 2)


if __name__ == "__main__":
    unittest.main()
