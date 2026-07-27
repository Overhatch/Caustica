#!/usr/bin/env python3
"""Bakes the RT renderer's display-transform 3D LUTs from OCIO's built-in ACES 2.0 config.

See docs/DISPLAY_TRANSFORM_PLAN.md. The renderer feeds these LUTs scene-linear BT.2020 (Rec.2020)
radiance, already multiplied by the auto-exposure scalar (RtExposure); each LUT bakes in the whole
remaining pipeline: ACES 2.0 view transform, gamut mapping, tone scale, and the output display's
transfer function. The shader only has to do the log2 shaper encode (must match SHAPER_LO/HI below
bit-for-bit -- see shaperEncode() in shaders/display/display.comp) and a trilinear fetch.

One SDR LUT (BT.709, sRGB OETF) plus one HDR LUT per REC2020 mastering-nits target ACES 2.0 ships
(500/1000/2000/4000 -- see HDR_REC2020_NITS; ACES 2.0 does not parameterize peak luminance
continuously, this fixed set IS the resolution). RtComposite picks the nearest at LUT-load time to
match the renderer's continuous Hdr.PEAK_NITS config value.

Requires: pip install opencolorio numpy  (tested with opencolorio 2.5.2 / numpy 2.5.1, Python 3.14)

Usage:
    python tools/bake_display_lut.py

Regenerate whenever SHAPER_LO/HI, LUT_SIZE, or the OCIO config/view below changes. The baked
.bin files are committed binary resources (src/main/resources/caustica/rt/luts/) -- this script
is the source of truth for reproducing them, not the .bin files themselves.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import PyOpenColorIO as OCIO

# OCIO 2.2+ ships this config compiled into the library -- no external config file/network fetch
# needed. Matches the config used for the Blender A/B evaluation that motivated this plan.
OCIO_BUILTIN_CONFIG = "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
SOURCE_SPACE = "Linear Rec.2020"  # matches the renderer's scene-linear BT.2020 working space

# Log2 shaper range, in stops relative to linear 1.0. Matches LOG_MIN/LOG_MAX in
# shaders/display/exposure_hist.comp and exposure_resolve.comp -- same renderer quantity metered
# in both places, so the same bounds. Input is exposed scene-linear (may exceed 1.0 for
# unclipped-highlight emitters), so headroom above 0 stops matters, not just below.
SHAPER_LO_STOPS = -12.0
SHAPER_HI_STOPS = 12.0

LUT_SIZE = 65  # samples per axis; N^3 total. See docs/DISPLAY_TRANSFORM_PLAN.md S2 sizing note.

OUT_DIR = Path(__file__).resolve().parent.parent / "src/main/resources/caustica/rt/luts"

# HDR REC2020 nits: ACES 2.0 does not parameterize peak luminance continuously -- OCIO's builtin
# registry ships a fixed table of mastering targets (real HDR mastering always worked this way).
# For BT.2020 that table is exactly {500, 1000, 2000, 4000}. This maps onto the renderer's
# continuous Hdr.PEAK_NITS (clamped 80-5000) by picking the nearest at LUT-load time (see
# RtComposite.nearestHdrNits) -- see docs/DISPLAY_TRANSFORM_PLAN.md S6 open question #2 (resolved).
HDR_REC2020_NITS = [500, 1000, 2000, 4000]

LUTS = [
    dict(
        name="sdr_aces2_rec709",
        display_view=("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"),
        note="SDR output, BT.709 display code values (renderer's existing rgba8 gamma-encoded "
             "presentation path expects sRGB-OETF-encoded BT.709, same as the AgX path it replaces).",
    ),
] + [
    dict(
        name=f"hdr_aces2_rec2020_{nits}nit",
        # Composed directly from BuiltinTransform pieces (verified bit-exact against the config's
        # own Display/View path for 1000nit, the only nits value pre-wired as a named View) rather
        # than via getProcessor(display, view, ...): ACES2065-1_to_CIE-XYZ-D65 is the ACES 2 output
        # transform itself; CIE-XYZ-D65_to_REC.2100-PQ is the final display encode, matching
        # VK_COLOR_SPACE_HDR10_ST2084_EXT's container exactly, so no separate gamut step or
        # pqEncode() needed at sample time.
        builtin_chain=[
            ("colorspace", ("Linear Rec.2020", "ACES2065-1")),
            ("builtin", f"ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - HDR-{nits}nit-REC2020_2.0"),
            ("builtin", "DISPLAY - CIE-XYZ-D65_to_REC.2100-PQ"),
        ],
        note=f"HDR output, {nits} nit peak, BT.2020 primaries + ST.2084/PQ-encoded code values.",
    )
    for nits in HDR_REC2020_NITS
]


def shaper_axis(size: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, size, dtype=np.float64)
    stops = SHAPER_LO_STOPS + t * (SHAPER_HI_STOPS - SHAPER_LO_STOPS)
    return np.exp2(stops)


def make_processor(cfg: "OCIO.Config", spec: dict):
    if "display_view" in spec:
        display, view = spec["display_view"]
        return cfg.getProcessor(SOURCE_SPACE, display, view, OCIO.TRANSFORM_DIR_FORWARD).getDefaultCPUProcessor()
    grp = OCIO.GroupTransform()
    for kind, arg in spec["builtin_chain"]:
        if kind == "colorspace":
            src, dst = arg
            grp.appendTransform(OCIO.ColorSpaceTransform(src=src, dst=dst))
        elif kind == "builtin":
            grp.appendTransform(OCIO.BuiltinTransform(style=arg))
        else:
            raise ValueError(f"unknown chain step kind: {kind}")
    return cfg.getProcessor(grp).getDefaultCPUProcessor()


def bake_one(cfg: "OCIO.Config", spec: dict, size: int) -> np.ndarray:
    axis = shaper_axis(size)  # same axis reused for R, G, B -- the shaper is a per-channel diagonal
    # Grid shape (N,N,N,3) with R fastest-varying (x), G next (y), B slowest (z). This matches
    # VkBufferImageCopy's row-major layout for a 3D image of extent (N,N,N): x is the contiguous
    # texel run, so keep that axis == index 0 of the meshgrid arrays below, i.e. last numpy axis
    # before the channel axis. See decodeToneLut()'s texCoord order in display.comp.
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")  # b,g,r all shape (N,N,N)
    grid = np.stack([r, g, b], axis=-1).astype(np.float32)  # (N,N,N,3), fastest axis = r = x

    cpu = make_processor(cfg, spec)

    flat = grid.reshape(-1, 3).copy()
    cpu.applyRGB(flat)
    flat = np.clip(flat, 0.0, 1.0)
    return flat.reshape(size, size, size, 3)


def write_lut(path: Path, size: int, rgb: np.ndarray) -> None:
    # RGBA16F texel data (alpha unused, kept 1.0 for a well-defined value + simpler Vulkan format
    # matching: R16G16B16_SFLOAT support is patchy, RGBA16F is universal). Header is self-describing
    # so the Java loader doesn't need a second source of truth for size/shaper range.
    rgba = np.concatenate([rgb, np.ones((size, size, size, 1), dtype=np.float32)], axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"CLUT")
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", size))
        f.write(struct.pack("<ff", SHAPER_LO_STOPS, SHAPER_HI_STOPS))
        f.write(rgba.astype(np.float16).tobytes())
    print(f"wrote {path} ({path.stat().st_size} bytes, {size}^3 texels)")


def main() -> None:
    cfg = OCIO.Config.CreateFromBuiltinConfig(OCIO_BUILTIN_CONFIG)
    for spec in LUTS:
        print(f"baking {spec['name']}: {spec['note']}")
        rgb = bake_one(cfg, spec, LUT_SIZE)
        write_lut(OUT_DIR / f"{spec['name']}.bin", LUT_SIZE, rgb)


if __name__ == "__main__":
    sys.exit(main())
