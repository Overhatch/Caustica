import struct
import unittest

import numpy as np

from tools import bake_display_lut as baker


class SoftToeLookTest(unittest.TestCase):
    def test_preserves_black_and_neutral_axis(self):
        rgb = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01]], dtype=np.float32)
        baker.apply_soft_toe_lmt(rgb)

        np.testing.assert_array_equal(rgb[0], np.zeros(3, dtype=np.float32))
        self.assertGreater(float(rgb[1, 0]), 0.01)
        self.assertAlmostEqual(float(rgb[1, 0]), float(rgb[1, 1]), places=7)
        self.assertAlmostEqual(float(rgb[1, 1]), float(rgb[1, 2]), places=7)

    def test_reduces_chroma_by_configured_amount(self):
        source = np.array([[0.03, 0.01, 0.005]], dtype=np.float32)
        source_y = source.astype(np.float64) @ baker.ACESCG_LUMA
        toe_scale = 1.0 + baker.SOFT_TOE_GAIN * np.exp(-source_y / baker.SOFT_TOE_PIVOT)
        lifted = source.astype(np.float64) * toe_scale[:, None]
        lifted_y = lifted @ baker.ACESCG_LUMA

        actual = source.copy()
        baker.apply_soft_toe_lmt(actual)
        actual_y = actual.astype(np.float64) @ baker.ACESCG_LUMA
        np.testing.assert_allclose(actual_y, lifted_y, rtol=2e-6)
        np.testing.assert_allclose(
            actual - actual_y[:, None],
            baker.SOFT_TOE_SATURATION * (lifted - lifted_y[:, None]),
            rtol=2e-6,
            atol=1e-9,
        )


class BakedLookResourceTest(unittest.TestCase):
    def test_all_runtime_looks_have_expected_header_and_finite_payload(self):
        for name in ("caustica-soft", "agx-tone", "arri-reveal-tone", "red-tone"):
            path = baker.OUT_DIR / f"look_{name}.bin"
            data = path.read_bytes()
            magic, version, size, lo_stops, hi_stops = struct.unpack_from("<4sIIff", data)
            self.assertEqual(magic, b"CLUT", path)
            self.assertEqual(version, 1, path)
            self.assertEqual(size, baker.LOOK_LUT_SIZE, path)
            self.assertEqual(lo_stops, baker.SHAPER_LO_STOPS, path)
            self.assertEqual(hi_stops, baker.SHAPER_HI_STOPS, path)
            payload = np.frombuffer(data, dtype=np.float16, offset=20)
            self.assertEqual(payload.size, size ** 3 * 4, path)
            self.assertTrue(np.isfinite(payload).all(), path)
            self.assertGreaterEqual(float(payload.min()), 0.0, path)
            self.assertLessEqual(float(payload.max()), 1.0, path)
            rgba = payload.reshape(size, size, size, 4)
            axis = np.linspace(0.0, 1.0, size)
            b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
            identity = np.stack([r, g, b], axis=-1)
            self.assertGreater(float(np.max(np.abs(rgba[..., :3] - identity))), 0.005, path)


if __name__ == "__main__":
    unittest.main()
