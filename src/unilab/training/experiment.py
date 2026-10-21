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

    import time as _time

    import torch
    from rsl_rl.utils.logger import Logger

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

    if not getattr(Logger, "_UNILAB_ETA_PATCHED", False):
        original_log = Logger.log

        def _patched_log(self: Any, *args: Any, **kwargs: Any) -> None:
            """Keep ETA based on this run while preserving cumulative elapsed time."""
            iteration = kwargs.get("it", args[0] if args else None)
            start_it = kwargs.get("start_it", 0)
            total_it = kwargs.get("total_it", None)
            collect_time = kwargs.get("collect_time", 0.0)
            learn_time = kwargs.get("learn_time", 0.0)
            if iteration is None or total_it is None:
                original_log(self, *args, **kwargs)
                return

            iteration_time = float(collect_time) + float(learn_time)
            session_elapsed = float(getattr(self, "_unilab_session_elapsed", 0.0))
            session_elapsed += iteration_time
            self._unilab_session_elapsed = session_elapsed
            completed = max(int(iteration) - int(start_it) + 1, 1)
            remaining = max(int(total_it) - int(iteration) - 1, 0)
            eta = session_elapsed / completed * remaining

            original_strftime = _time.strftime
            calls = 0

            def _duration(seconds: float) -> str:
                total = max(int(seconds), 0)
                hours, remainder = divmod(total, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            def _strftime(format_string: str, value: Any) -> str:
                nonlocal calls
                if format_string == "%H:%M:%S":
                    calls += 1
                    if calls == 2:
                        return _duration(eta)
                return original_strftime(format_string, value)

            setattr(_time, "strftime", _strftime)
            try:
                original_log(self, *args, **kwargs)
            finally:
                setattr(_time, "strftime", original_strftime)

        Logger.log = _patched_log  # type: ignore[assignment]
        setattr(Logger, "_UNILAB_ETA_PATCHED", True)

    OnPolicyRunner.save = _patched_save  # type: ignore[assignment]
    OnPolicyRunner.load = _patched_load  # type: ignore[assignment]
    setattr(OnPolicyRunner, "_UNILAB_RESUME_PATCHED", True)
