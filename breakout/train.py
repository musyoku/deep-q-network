# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
import scipy.misc as spm
from rlglue.agent import AgentLoader
sys.path.append(os.path.split(os.getcwd())[0])
from PIL import Image
from config import config
from dqn import DQN
from agent import Agent

# Override config
config.apply_batchnorm = True
config.ale_actions = [4, 3, 1, 0]
config.ale_screen_size = [210, 160]
config.ale_scaled_screen_size = [86, 86]
# config.ale_screen_channels = 3
config.rl_replay_memory_size = 5 * 10 ** 4
config.rl_target_network_update_frequency = 10 ** 3 * 2
config.rl_replay_start_size = 10 ** 4
config.rl_final_exploration_frame = 10 ** 5
config.q_conv_hidden_channels = [128, 256, 512]
config.q_conv_strides = [2, 2, 2]
config.q_conv_filter_sizes = [4, 4, 4]
config.q_conv_output_vector_dimension = 2000
config.q_fc_hidden_units = [2000, 1000, 500]

# Eliminate fully connected layers
# config.q_fc_hidden_units = []

# Override agent
class BreakoutAgent(Agent):
	def scale_screen(self, observation):
		screen_width = config.ale_screen_size[0]
		screen_height = config.ale_screen_size[1]
		new_width = config.ale_scaled_screen_size[0]
		new_height = config.ale_scaled_screen_size[1]
		if len(observation.intArray) == 100928: 
			if config.ale_screen_channels == 1:
				raise Exception("You forgot to set config.ale_screen_channels to 3.")
			# RGB
			observation = np.asarray(observation.intArray[128:], dtype=np.uint8).reshape((screen_width, screen_height, 3))
			# Remove the score area from image
			observation = observation[93:,6:-6,:]
			observation = spm.imresize(observation, (new_height, new_width))
			# Clip the pixel value to be between 0 and 1
			observation = observation.transpose(2, 0, 1) / 255.0
		else:
			# Greyscale
			if config.ale_screen_channels == 3:
				raise Exception("You forgot to add --send_rgb option when you run ALE.")
			observation = np.asarray(observation.intArray[128:]).reshape((screen_width, screen_height))
			observation = observation[93:,6:-6]
			observation = spm.imresize(observation, (new_height, new_width))
			# Clip the pixel value to be between 0 and 1
			observation = observation.reshape((1, new_height, new_width)) / 255.0

		return observation

AgentLoader.loadAgent(BreakoutAgent())
