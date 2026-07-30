#!/usr/bin/env python3
"""Bakes the RT renderer's ACES look and display-transform 3D LUTs.

See docs/DISPLAY_TRANSFORM_PLAN.md. The renderer feeds these LUTs scene-linear ACEScg (AP1/D60)
radiance, already multiplied by the auto-exposure scalar (RtExposure). A selected scene-referred
ACES-to-ACES Look Transform (historically called an LMT) runs first, followed by the existing ACES
2.0 output transform, gamut mapping, tone scale, and display transfer function.

One SDR LUT (BT.709, sRGB OETF) plus one HDR LUT per REC2020 mastering-nits target ACES 2.0 ships
(500/1000/2000/4000 -- see HDR_REC2020_NITS; ACES 2.0 does not parameterize peak luminance
continuously, this fixed set IS the resolution). RtComposite picks the nearest at LUT-load time to
match the renderer's continuous Hdr.PEAK_NITS config value. Look LUTs are separate log-to-log
scene-referred tables, so adding a look does not duplicate all five output LUTs.

Requires: pip install opencolorio numpy  (tested with opencolorio 2.5.2 / numpy 2.5.1, Python 3.14)

Usage:
    python tools/bake_display_lut.py
    python tools/bake_display_lut.py --public-source-dir path/to/aces-looks/ACES2Looks/CLF

Regenerate whenever SHAPER_LO/HI, LUT_SIZE, or the OCIO config/view below changes. The baked
.bin files are committed binary resources (src/main/resources/caustica/rt/luts/) -- this script
is the source of truth for reproducing them, not the .bin files themselves. Public CLFs are fetched
from a pinned ACESLooks commit and verified by SHA-256 unless --public-source-dir is provided.
"""
import argparse
import hashlib
import re
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import PyOpenColorIO as OCIO

# OCIO 2.2+ ships this config compiled into the library -- no external config file/network fetch
# needed. Matches the config used for the Blender A/B evaluation that motivated this plan.
OCIO_BUILTIN_CONFIG = "cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
SOURCE_SPACE = "ACEScg"  # matches the renderer's scene-linear ACEScg/AP1/D60 working space

# Log2 shaper range, in stops relative to linear 1.0. Matches LOG_MIN/LOG_MAX in
# shaders/display/exposure_hist.comp and exposure_resolve.comp -- same renderer quantity metered
# in both places, so the same bounds. Input is exposed scene-linear (may exceed 1.0 for
# unclipped-highlight emitters), so headroom above 0 stops matters, not just below.
SHAPER_LO_STOPS = -12.0
SHAPER_HI_STOPS = 12.0

LUT_SIZE = 65  # samples per axis; N^3 total. See docs/DISPLAY_TRANSFORM_PLAN.md S2 sizing note.
LOOK_LUT_SIZE = 33  # log-domain scene-to-scene looks; smooth transforms, one extra sample at runtime

OUT_DIR = Path(__file__).resolve().parent.parent / "src/main/resources/caustica/rt/luts"
PUBLIC_CACHE_DIR = Path(__file__).resolve().parent.parent / "build/aces-lmts"

# Public ACES 2.0 CLFs from ACESLooks, BSD-3-Clause. Pin the immutable commit and every payload hash:
# a changed upstream main branch must never change committed renderer output without review.
ACES_LOOKS_COMMIT = "f1b85e40efb64bf5efeecc34e251bf7c617f6306"
ACES_LOOKS_RAW = (
    "https://raw.githubusercontent.com/priikone/aces-looks/"
    f"{ACES_LOOKS_COMMIT}/ACES2Looks/CLF"
)
PUBLIC_LMTS = [
    dict(
        name="agx-tone",
        file="T-AgX_Tone.clf",
        sha256="540e56fdfb2e0eecd84afc9b97c4171130a48af26c97c5a43622ceae79459ee9",
        note="ACESLooks ACES 2.0 AgX tone-curve look",
    ),
    dict(
        name="arri-reveal-tone",
        file="T-ARRI_REVEAL_Tone.clf",
        sha256="b5ab3d001a859bbb639775f3afb191703cf91cc529ed729af0014ef8f61ed47b",
        note="ACESLooks ACES 2.0 ARRI REVEAL tone-curve look",
    ),
    dict(
        name="red-tone",
        file="T-RED_Tone.clf",
        sha256="165d4e6c78e756ca9b1d3fb35abccf44116dab18fe8351fe71c1a76831db0c47",
        note="ACESLooks ACES 2.0 RED IPP2 tone-curve look",
    ),
]

# Project-authored ACES look. It is deliberately restrained: raise only the deep scene-referred toe
# (black remains exactly black and the adjustment decays smoothly into the mids), then reduce AP1
# saturation without moving luminance. Applying this before the Output Transform lets ACES 2.0 retain
# control of the final display rendering and gamut mapping.
SOFT_TOE_GAIN = 0.50
SOFT_TOE_PIVOT = 0.03
SOFT_TOE_SATURATION = 0.92
ACESCG_LUMA = np.array([0.2722287168, 0.6740817658, 0.0536895174], dtype=np.float64)

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
            ("colorspace", ("ACEScg", "ACES2065-1")),
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


def shaper_encode(linear: np.ndarray) -> np.ndarray:
    stops = np.log2(np.maximum(linear, np.exp2(SHAPER_LO_STOPS)))
    return np.clip(
        (stops - SHAPER_LO_STOPS) / (SHAPER_HI_STOPS - SHAPER_LO_STOPS),
        0.0,
        1.0,
    )


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


def make_lmt_processor(cfg: "OCIO.Config", clf_path: Path):
    # CLF LMTs are standardized as ACES2065-1 (AP0) -> ACES2065-1. The renderer is ACEScg (AP1),
    # so wrap the public file in the appropriate conversions. OCIO optimizes the composed processor.
    grp = OCIO.GroupTransform()
    grp.appendTransform(OCIO.ColorSpaceTransform(src=SOURCE_SPACE, dst="ACES2065-1"))
    grp.appendTransform(OCIO.FileTransform(
        src=str(clf_path.resolve()),
        interpolation=OCIO.INTERP_TETRAHEDRAL,
    ))
    grp.appendTransform(OCIO.ColorSpaceTransform(src="ACES2065-1", dst=SOURCE_SPACE))
    return cfg.getProcessor(grp).getDefaultCPUProcessor()


def apply_soft_toe_lmt(rgb: np.ndarray) -> None:
    # Conceptually AP0 -> AP1, this operation, AP1 -> AP0. Since the renderer and baker surround the
    # look with ACEScg, those two matrices cancel. Uniform RGB scaling recovers toe detail without
    # changing chromaticity; the following luma-axis mix makes the modest saturation reduction.
    y = np.maximum(rgb.astype(np.float64) @ ACESCG_LUMA, 0.0)
    toe_scale = 1.0 + SOFT_TOE_GAIN * np.exp(-y / SOFT_TOE_PIVOT)
    lifted = rgb.astype(np.float64) * toe_scale[:, None]
    lifted_y = lifted @ ACESCG_LUMA
    rgb[:] = (
        lifted_y[:, None]
        + SOFT_TOE_SATURATION * (lifted - lifted_y[:, None])
    ).astype(np.float32)


def checked_public_lmt(spec: dict, source_dir: Path | None) -> Path:
    path = (source_dir / spec["file"]) if source_dir is not None else (PUBLIC_CACHE_DIR / spec["file"])
    if not path.exists() and source_dir is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {spec['file']} from pinned ACESLooks commit {ACES_LOOKS_COMMIT}")
        urllib.request.urlretrieve(f"{ACES_LOOKS_RAW}/{spec['file']}", path)
    if not path.is_file():
        raise FileNotFoundError(f"missing public LMT: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != spec["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch: expected {spec['sha256']}, got {actual}")
    return path


def read_shaper_cube(path: Path) -> tuple[int, np.ndarray, str | None]:
    """Read a normalized .cube as a log-shaper-to-log-shaper scene look.

    A .cube file does not carry reliable color-space semantics. Imported creative curves are therefore
    defined over the renderer's normalized -12..+12 EV shaper coordinates, not over linear ACEScg values:
    applying a conventional [0,1] cube directly to scene-linear input would clamp all values above 1.0.
    The file order (R fastest, then G, then B) already matches the 3D Vulkan image layout used by write_lut.
    """
    size = None
    title = None
    domain_min = np.zeros(3, dtype=np.float64)
    domain_max = np.ones(3, dtype=np.float64)
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        directive = fields[0].upper()
        if directive == "TITLE":
            title = line[len(fields[0]):].strip().strip('"')
        elif directive == "LUT_3D_SIZE":
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: LUT_3D_SIZE requires one integer")
            size = int(fields[1])
        elif directive == "LUT_1D_SIZE":
            raise ValueError(f"{path}:{line_number}: 1D LUTs are not supported")
        elif directive in ("DOMAIN_MIN", "DOMAIN_MAX"):
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: {directive} requires three values")
            domain = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
            if directive == "DOMAIN_MIN":
                domain_min = domain
            else:
                domain_max = domain
        else:
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: unknown directive or malformed RGB row")
            try:
                rows.append([float(value) for value in fields])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: unknown directive {fields[0]!r}") from exc

    if size is None or size < 2:
        raise ValueError(f"{path}: missing or invalid LUT_3D_SIZE")
    expected_rows = size ** 3
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} RGB rows for {size}^3, got {len(rows)}")
    if not np.allclose(domain_min, 0.0) or not np.allclose(domain_max, 1.0):
        raise ValueError(
            f"{path}: imported shaper cubes must use DOMAIN_MIN 0 0 0 and DOMAIN_MAX 1 1 1")

    rgb = np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)
    if not np.isfinite(rgb).all():
        raise ValueError(f"{path}: LUT contains non-finite values")
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise ValueError(f"{path}: shaper-domain LUT output must stay within [0,1]")
    return size, rgb, title


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


def bake_look(processor, size: int, custom=None) -> np.ndarray:
    axis = shaper_axis(size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    flat = np.stack([r, g, b], axis=-1).astype(np.float32).reshape(-1, 3)
    if custom is not None:
        custom(flat)
    else:
        processor.applyRGB(flat)
    # Store scene-referred output in the same log2 shaper domain. The shader decodes this sample back
    # to linear ACEScg before feeding the ordinary SDR/HDR ACES 2.0 output-transform LUT.
    return shaper_encode(flat).reshape(size, size, size, 3)


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-source-dir",
        type=Path,
        help="use already-downloaded ACES2Looks/CLF sources instead of fetching the pinned files",
    )
    parser.add_argument(
        "--import-cube",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="import a normalized .cube as look_NAME.bin in the renderer's log2 shaper domain, then exit",
    )
    args = parser.parse_args()

    if args.import_cube is not None:
        name, source = args.import_cube
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) is None:
            parser.error("--import-cube NAME must contain only lowercase ASCII letters, digits, and hyphens")
        source_path = Path(source)
        size, rgb, title = read_shaper_cube(source_path)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        print(
            f"importing look_{name}: {size}^3 normalized log-shaper cube"
            f"{f' ({title})' if title else ''}; source SHA-256={digest}"
        )
        write_lut(OUT_DIR / f"look_{name}.bin", size, rgb)
        return

    cfg = OCIO.Config.CreateFromBuiltinConfig(OCIO_BUILTIN_CONFIG)
    for spec in LUTS:
        print(f"baking {spec['name']}: {spec['note']}")
        rgb = bake_one(cfg, spec, LUT_SIZE)
        write_lut(OUT_DIR / f"{spec['name']}.bin", LUT_SIZE, rgb)

    print(
        "baking look_caustica-soft: project-authored soft toe "
        f"(gain={SOFT_TOE_GAIN}, pivot={SOFT_TOE_PIVOT}, saturation={SOFT_TOE_SATURATION})"
    )
    rgb = bake_look(None, LOOK_LUT_SIZE, custom=apply_soft_toe_lmt)
    write_lut(OUT_DIR / "look_caustica-soft.bin", LOOK_LUT_SIZE, rgb)

    for spec in PUBLIC_LMTS:
        clf_path = checked_public_lmt(spec, args.public_source_dir)
        print(f"baking look_{spec['name']}: {spec['note']} ({clf_path})")
        rgb = bake_look(make_lmt_processor(cfg, clf_path), LOOK_LUT_SIZE)
        write_lut(OUT_DIR / f"look_{spec['name']}.bin", LOOK_LUT_SIZE, rgb)


if __name__ == "__main__":
    sys.exit(main())
