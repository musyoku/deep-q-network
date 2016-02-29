# -*- coding: utf-8 -*-
import os, sys, time
from rlglue.agent import AgentLoader
sys.path.append(os.path.split(os.getcwd())[0])
from PIL import Image
from config import config
from dqn import DQN
from agent import Agent

# Override config
config.ale_actions = [0, 1, 3, 4]
config.apply_batchnorm = True
config.ale_screen_channels = 3
config.rl_replay_memory_size = 5 * 10 ** 4
config.ale_screen_size = [210, 160]
config.ale_scaled_screen_size = [110, 94]
config.rl_replay_start_size = 10 ** 1
config.q_conv_hidden_channels = [100, 200, 300]
config.q_conv_strides = [2, 2, 2]
config.q_conv_filter_sizes = [4, 4, 4]

# Eliminate fully connected layers
# config.q_fc_hidden_units = []

# Override agent
class PongAgent(Agent):
	pass

AgentLoader.loadAgent(PongAgent())
