#!/usr/bin/env python3
"""Build a four-point exposure compensation curve from reference-scene EV readings."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    name: str
    measured_ev: float
    relative_render_ev: float


def finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Convert stabilized RT exposure evScene readings into the four-point "
            "exposure.curve used by Caustica."
        )
    )
    result.add_argument("--noon", required=True, type=finite, help="clear-noon exterior evScene")
    result.add_argument("--overcast", required=True, type=finite, help="overcast or dawn evScene")
    result.add_argument("--night", required=True, type=finite, help="clear moonlit-night evScene")
    result.add_argument("--cave", required=True, type=finite, help="torch-lit cave evScene")
    result.add_argument("--overcast-target", type=finite, default=-0.5,
                        help="render EV relative to noon (default: -0.5)")
    result.add_argument("--night-target", type=finite, default=-1.75,
                        help="render EV relative to noon (default: midpoint -1.75)")
    result.add_argument("--cave-target", type=finite, default=-2.25,
                        help="render EV relative to noon (default: midpoint -2.25)")
    result.add_argument("--key", type=finite, default=0.18, help="exposure key (default: 0.18)")
    result.add_argument("--ev-bias", type=finite, default=0.0, help="manual/creative EV bias")
    result.add_argument("--min-ev", type=finite, default=-1.5, help="minimum exposure EV clamp")
    result.add_argument("--max-ev", type=finite, default=4.0, help="maximum exposure EV clamp")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.key <= 0.0:
        raise SystemExit("--key must be greater than zero")

    scenes = sorted(
        [
            Scene("noon", args.noon, 0.0),
            Scene("overcast", args.overcast, args.overcast_target),
            Scene("night", args.night, args.night_target),
            Scene("cave", args.cave, args.cave_target),
        ],
        key=lambda scene: scene.measured_ev,
    )
    for left, right in zip(scenes, scenes[1:]):
        if right.measured_ev - left.measured_ev < 1.0e-4:
            raise SystemExit(
                f"{left.name} and {right.name} have indistinguishable evScene readings; "
                "capture more stable/different reference views"
            )

    curve = ", ".join(
        f"{scene.measured_ev:.3f}:{scene.relative_render_ev:.3f}" for scene in scenes
    )
    lo = min(args.min_ev, args.max_ev)
    hi = max(args.min_ev, args.max_ev)
    log_key = math.log2(args.key)

    print(f'exposure.curve = "{curve}"')
    print()
    print("scene       evScene   compensation   evTarget   clamp")
    for scene in scenes:
        target = log_key - scene.measured_ev + scene.relative_render_ev + args.ev_bias
        clamp = "min" if target < lo else "max" if target > hi else "-"
        print(
            f"{scene.name:10} {scene.measured_ev:8.3f} "
            f"{scene.relative_render_ev:14.3f} {target:10.3f}   {clamp}"
        )

    print()
    print("segment effective slopes (1 - dComp/dScene):")
    for left, right in zip(scenes, scenes[1:]):
        compensation_slope = (
            right.relative_render_ev - left.relative_render_ev
        ) / (right.measured_ev - left.measured_ev)
        print(f"  {left.name:10} -> {right.name:10}: {1.0 - compensation_slope:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
