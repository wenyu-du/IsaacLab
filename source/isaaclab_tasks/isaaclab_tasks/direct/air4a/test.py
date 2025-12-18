import argparse

from isaaclab.app import AppLauncher

from isaaclab.utils import configclass

import gymnasium as gym
# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a custom articulation.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from air4a_env import Air4aEnvCfg

env_cfg = Air4aEnvCfg()
env = gym.make("Isaac-Air4a-Direct-v0", cfg = env_cfg)
import time
time.sleep(180)
# close sim app
simulation_app.close()