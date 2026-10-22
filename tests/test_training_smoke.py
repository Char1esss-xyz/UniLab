import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.scripts.play_rsl_rl import play
from unilab.scripts.train_rsl_rl import train
from unilab.utils.sim2sim import CrossBackendIncompatibleError, extract_contract_snapshot


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["mujoco", "mjwarp"])
def test_train_resume_snapshot_and_cross_backend_playback(tmp_path, backend):
    if backend == "mjwarp":
        if not torch.cuda.is_available():
            pytest.skip("MJWarp training requires CUDA")
        pytest.importorskip("mujoco_warp")
    device = "cuda:0" if backend == "mjwarp" else "cpu"
    config_dir = Path(__file__).resolve().parents[1] / "src/unilab/conf/ppo"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task=wheelbipe_v14_flat/{backend}",
                f"training.device={device}",
                f"training.log_root={tmp_path}",
                "algo.num_envs=4",
                "algo.max_iterations=1",
                "algo.num_steps_per_env=2",
                "algo.algorithm.num_learning_epochs=1",
                "algo.algorithm.num_mini_batches=1",
            ],
        )

    log_dir = train(cfg)
    snapshot = json.loads((log_dir / "run_config.json").read_text())
    assert snapshot["contract_snapshot"] == extract_contract_snapshot(cfg)
    checkpoint = next(log_dir.glob("model_*.pt"))
    before = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert before["unilab_env_training_state"]["step_counter"] == 2

    resumed = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    resumed.algo.resume_path = str(checkpoint)
    resumed_log = train(resumed)
    after = torch.load(next(resumed_log.glob("model_*.pt")), weights_only=False, map_location="cpu")
    assert after["unilab_env_training_state"]["step_counter"] == 4
    assert after["unilab_logger_state"]["tot_timesteps"] == 16

    # A GPU-trained policy can use the same input convention on CPU MuJoCo.
    play(checkpoint, device="cpu", sim_backend="mujoco", steps=2, render_mode="none")

    incompatible = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(incompatible, "env.observations.policy.terms.state.scale", 2.0, force_add=True)
    config_path = tmp_path / "incompatible.yaml"
    OmegaConf.save(incompatible, config_path)
    with pytest.raises(CrossBackendIncompatibleError, match="env.observations"):
        play(checkpoint, config_path=config_path, device="cpu", steps=1, render_mode="none")
