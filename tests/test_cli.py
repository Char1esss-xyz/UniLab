import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from unilab import cli
from unilab.scripts import train_rsl_rl


def test_cuda_device_index_and_capacity_overrides_reach_training_config(monkeypatch):
    captured = []
    monkeypatch.setattr(train_rsl_rl, "main", captured.append)
    cli.train_main(
        [
            "--sim",
            "mjwarp",
            "--device",
            "cuda:1",
            "env.mjwarp_nconmax=2048",
            "env.mjwarp_njmax=1024",
        ]
    )
    config_dir = Path(__file__).resolve().parents[1] / "src/unilab/conf/ppo"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=captured[0])
    assert cfg.training.device == "cuda:1"
    assert cfg.training.sim_backend == "mjwarp"
    assert cfg.env.mjwarp_nconmax == 2048
    assert cfg.env.mjwarp_njmax == 1024


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:invalid", "cpu:1", "xpu"])
def test_invalid_device_is_rejected_during_argument_parsing(device):
    with pytest.raises(SystemExit) as exc:
        cli.train_main(["--device", device])
    assert exc.value.code == 2


def test_mjwarp_launcher_enables_dependencies_and_preserves_device_and_capacity(
    tmp_path, monkeypatch
):
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "ARG=%s\\n" "$@"\n')
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setenv("DEVICE", "cuda:1")
    monkeypatch.setenv("MJWARP_NCONMAX", "2048")
    monkeypatch.setenv("MJWARP_NJMAX", "1024")
    script = Path(__file__).resolve().parents[1] / "tools/run_mjwarp_ppo.sh"
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, check=True)
    args = [
        line.removeprefix("ARG=") for line in result.stdout.splitlines() if line.startswith("ARG=")
    ]
    assert args[:4] == ["run", "--extra", "mjwarp", "train"]
    assert args[args.index("--device") + 1] == "cuda:1"
    assert "env.mjwarp_nconmax=2048" in args
    assert "env.mjwarp_njmax=1024" in args
