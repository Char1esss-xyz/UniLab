from unilab.base import registry
from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

registry.register_env_config("WheelbipeV14Flat", ManagerBasedRlEnvCfg)
registry.register_env("WheelbipeV14Flat", make_manager_based_rl_env, sim_backend="mujoco")
registry.register_env("WheelbipeV14Flat", make_manager_based_rl_env, sim_backend="mjwarp")
