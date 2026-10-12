"""Command line entry for Wheelbipe V14 Flat PPO training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def train_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Wheelbipe V14 Flat with PPO and MuJoCo.")
    parser.add_argument("--algo", choices=("ppo",), default="ppo")
    parser.add_argument("--task", choices=("wheelbipe_v14_flat",), default="wheelbipe_v14_flat")
    parser.add_argument("--sim", choices=("mujoco",), default="mujoco")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--resume-path")
    parser.add_argument("--log-root")
    args, overrides = parser.parse_known_args(argv)
    for override in overrides:
        if "=" not in override or override.startswith("--"):
            parser.error(f"Expected a Hydra override such as algo.seed=1, received {override!r}")
    for value, field_name in (
        (args.num_envs, "algo.num_envs"),
        (args.max_iterations, "algo.max_iterations"),
        (args.device, "training.device"),
        (args.resume_path, "algo.resume_path"),
        (args.log_root, "training.log_root"),
    ):
        if value is not None:
            if isinstance(value, str):
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                overrides.append(f'{field_name}="{escaped}"')
            else:
                overrides.append(f"{field_name}={value}")

    from unilab.scripts.train_rsl_rl import main

    main(overrides)


def play_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play a trained Wheelbipe V14 Flat PPO policy.")
    parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt checkpoint")
    parser.add_argument("--config", dest="config_path", help="Optional path to config.yaml")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", default="playback.mp4", help="Output MP4 path")
    parser.add_argument("--render-mode", choices=("record", "none"), default="record")
    parser.add_argument("--cam-distance", type=float, help="Camera distance for recorded playback")
    parser.add_argument("--cam-elevation", type=float, help="Camera elevation in degrees")
    parser.add_argument("--cam-azimuth", type=float, help="Camera azimuth in degrees")
    args = parser.parse_args(argv)

    from unilab.scripts.play_rsl_rl import main

    main(
        args.checkpoint,
        config_path=args.config_path,
        device=args.device,
        num_envs=args.num_envs,
        steps=args.steps,
        output=args.output,
        render_mode=args.render_mode,
        cam_distance=args.cam_distance,
        cam_elevation=args.cam_elevation,
        cam_azimuth=args.cam_azimuth,
    )
