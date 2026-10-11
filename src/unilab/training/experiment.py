from __future__ import annotations

from typing import Any


def _copy_plain_training_state(state: Any) -> dict[str, Any]:
    """Validate and detach versioned training progress at a checkpoint boundary.

    This intentionally accepts only plain JSON-shaped values. It rejects tensors
    and other owner objects before they can enter a checkpoint payload.
    """

    def copy_value(value: Any) -> Any:
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("Training state numbers must be finite")
            return value
        if type(value) is not dict and type(value) is not list:
            raise TypeError(
                "Training state must contain only plain JSON dictionaries/lists/scalars"
            )
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError("Training state dictionary keys must be strings")
            return {key: copy_value(item) for key, item in value.items()}
        return [copy_value(item) for item in value]

    if not isinstance(state, dict):
        raise TypeError("Training state export must return a dictionary")
    result = copy_value(state)
    if not isinstance(result, dict):
        raise TypeError("Training state export must return a dictionary")
    return result


def patch_rsl_rl_resume_state() -> None:
    """Persist + restore UniLab progress through rsl-rl checkpoints.

    Without this patch, rsl-rl's ``Logger.__init__`` writes ``tot_time = 0`` and
    ``tot_timesteps = 0`` and ``OnPolicyRunner.load`` never refreshes them, so the
    ``Train/mean_reward/time`` and ``Train/mean_episode_length/time`` TensorBoard
    scalars (which use ``int(self.tot_time)`` as their step) restart from 0 on
    every resumed run and visually overlap the original segment. See issue #441.

    The patch wraps ``OnPolicyRunner.save`` / ``OnPolicyRunner.load`` to round-trip
    plain progress in the saved dict:

    - ``unilab_logger_state``: ``Logger.tot_time`` / ``tot_timesteps``;
    - ``unilab_env_training_state``: the wrapped TorchEnv's versioned training
      state, if it publishes ``export_training_state``.

    Legacy checkpoints (without either key) load unchanged. Environment state is
    plain versioned JSON data; no Torch tensor, backend, environment, or manager
    object is serialized through this path.
    """
    try:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    except Exception:
        return

    if getattr(OnPolicyRunner, "_UNILAB_RESUME_PATCHED", False):
        return

    import torch

    def _patched_save(self: Any, path: str, infos: dict | None = None) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["unilab_logger_state"] = {
            "tot_time": float(getattr(self.logger, "tot_time", 0.0)),
            "tot_timesteps": int(getattr(self.logger, "tot_timesteps", 0)),
        }
        env = getattr(self, "env", None)
        export_env_training_state = getattr(env, "export_training_state", None)
        if callable(export_env_training_state):
            saved_dict["unilab_env_training_state"] = _copy_plain_training_state(
                export_env_training_state()
            )
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def _patched_load(
        self: Any,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> Any:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        state = loaded_dict.get("unilab_logger_state")
        if state is not None:
            self.logger.tot_time = float(state.get("tot_time", 0.0))
            self.logger.tot_timesteps = int(state.get("tot_timesteps", 0))
        env_state = loaded_dict.get("unilab_env_training_state")
        if env_state is not None:
            env = getattr(self, "env", None)
            import_training_state = getattr(env, "import_training_state", None)
            if not callable(import_training_state):
                raise TypeError(
                    "Checkpoint contains unilab_env_training_state, but the active "
                    "environment does not implement import_training_state"
                )
            import_training_state(_copy_plain_training_state(env_state))
        return loaded_dict["infos"]

    OnPolicyRunner.save = _patched_save  # type: ignore[assignment]
    OnPolicyRunner.load = _patched_load  # type: ignore[assignment]
    setattr(OnPolicyRunner, "_UNILAB_RESUME_PATCHED", True)
