import gymnasium as gym

from . import agents
from .air4a_env import Air4aEnv, Air4aEnvCfg

__all__ = ["Air4aEnv", "Air4aEnvCfg"]

gym.register(
    id="Isaac-Air4a-Direct-v0",
    entry_point="isaaclab_tasks.direct.air4a:Air4aEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.air4a.air4a_env:Air4aEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Air4aPPORunnerCfg",
    },
)
