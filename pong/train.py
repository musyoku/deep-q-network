# -*- coding: utf-8 -*-
import os, sys, time
from rlglue.agent import AgentLoader
sys.path.append(os.path.split(os.getcwd())[0])
from config import config
from dqn import DQN
from agent import Agent

# Override config
config.ale_screen_size = [120, 280]
config.ale_scaled_screen_size = [62, 142]
config.rl_replay_start_size = 100
config.rl_replay_memory_size = 10 ** 4

# Override agent
class PongAgent(Agent):
	pass

AgentLoader.loadAgent(PongAgent())
