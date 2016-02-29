# -*- coding: utf-8 -*-
import copy
import scipy.misc as spm
import numpy as np
from rlglue.agent.Agent import Agent as RLGlueAgent
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3
from dqn import DQN
from config import config
from PIL import Image

class Agent(RLGlueAgent):
	def __init__(self):
		self.last_action = Action()
		self.time_step = 0
		self.total_time_step = 0
		self.episode_step = 0
		self.populating_phase = False

		self.model_save_interval = 30

		# Switch learning phase / evaluation phase
		self.policy_frozen = False

		self.dqn = DQN()
		self.state = np.zeros((config.rl_agent_history_length, config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0]), dtype=np.float32)
		self.exploration_rate = self.dqn.exploration_rate

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
			observation = spm.imresize(observation, (new_height, new_width))
			# Clip the pixel value to be between 0 and 1
			observation = observation.transpose(2, 0, 1) / 255.0
		else:
			# Greyscale
			if config.ale_screen_channels == 3:
				raise Exception("You forgot to add --send_rgb option when you run ALE.")
			observation = np.asarray(observation.intArray[128:]).reshape((screen_width, screen_height))
			observation = spm.imresize(observation, (new_height, new_width))
			# Clip the pixel value to be between 0 and 1
			observation = observation.reshape((1, new_height, new_width)) / 255.0

		return observation

	def agent_init(self, taskSpecString):
		pass

	def reshape_state_to_conv_input(self, state):
		return state.reshape((1, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0]))

	def dump_result(self, reward, q=None):
		if self.time_step % 50 == 0:
			print "time_step:", self.time_step,
			print "reward:", reward,
			print "e:", self.dqn.exploration_rate,
			if q is None:
				print ""
			else:
				print "Q:",
				print "max::", np.max(q),
				print "min::", np.min(q)


	def dump_state(self):
		state = self.reshape_state_to_conv_input(self.state)
		for h in xrange(config.rl_agent_history_length):
			start = h * config.ale_screen_channels
			end = start + config.ale_screen_channels
			image = state[0,start:end,:,:]
			if config.ale_screen_channels == 1:
				image = image.reshape((image.shape[1], image.shape[2]))
			elif config.ale_screen_channels == 3:
				image = image.transpose(1, 2, 0)
			image = np.uint8(image * 255.0)
			image = Image.fromarray(image)
			image.save(("state-%d.png" % h))

	def learn(self, reward):
		self.populating_phase = False
		if self.policy_frozen: # Evaluation phase
			self.exploration_rate = 0.05
		else: # Learning phase
			if self.total_time_step <= config.rl_replay_start_size:
				# A uniform random policy is run for 'replay_start_size' frames before learning starts
				# 経験を積むためランダムに動き回るらしい。
				print "Initial exploration before learning starts:", "%d/%d steps" % (self.total_time_step, config.rl_replay_start_size)
				self.populating_phase = True
				if self.total_time_step == config.rl_replay_start_size:
					# Copy batchnorm statistics to target
					self.dqn.update_target()
			else:
				self.dqn.decrease_exploration_rate()
			self.exploration_rate = self.dqn.exploration_rate

		if self.policy_frozen is False:
			self.dqn.store_transition_in_replay_memory(self.reshape_state_to_conv_input(self.last_state), self.last_action.intArray[0], reward, self.reshape_state_to_conv_input(self.state), False)
			if self.populating_phase is False:
				if self.time_step % (config.rl_action_repeat * config.rl_update_frequency) == 0 and self.time_step != 0:
					self.dqn.replay_experience()
				if self.total_time_step % config.rl_target_network_update_frequency == 0 and self.total_time_step != 0:
					print "Target has been updated."
					self.dqn.update_target()

	def agent_start(self, observation):
		print "Episode", self.episode_step
		observed_screen = self.scale_screen(observation)
		self.state[0] = observed_screen

		return_action = Action()
		action, q = self.dqn.e_greedy(self.reshape_state_to_conv_input(self.state), self.exploration_rate, test=self.policy_frozen)
		return_action.intArray = [action]

		self.last_action = copy.deepcopy(return_action)
		self.last_state = self.state.copy()
		self.last_observation = observed_screen

		return return_action

	def agent_step(self, reward, observation):
		observed_screen = self.scale_screen(observation)
		self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], observed_screen], dtype=np.float32)


		########################### DEBUG ###############################
		if self.total_time_step % 500 == 0 and self.total_time_step != 0:
			self.dump_state()

		self.learn(reward)

		return_action = Action()
		if self.time_step % config.rl_action_repeat == 0:
			action, q = self.dqn.e_greedy(self.reshape_state_to_conv_input(self.state), self.exploration_rate, test=self.policy_frozen)
		else:
			action = self.last_action.intArray[0]
			q = None
		return_action.intArray = [action]

		# [Optional]
		## Visualizing the results
		self.dump_result(reward, q)

		self.last_observation = observed_screen

		if self.policy_frozen is False:
			self.last_action = copy.deepcopy(return_action)
			self.last_state = self.state.copy()
			self.time_step += 1
			self.total_time_step += 1

		return return_action

	def agent_end(self, reward):
		self.learn(reward)

		# [Optional]
		## Visualizing the results
		self.dump_result(reward, q=None)

		if self.policy_frozen is False:
			self.time_step = 0
			self.total_time_step += 1
			self.episode_step += 1

	def agent_cleanup(self):
		pass

	def agent_message(self, inMessage):
		if inMessage.startswith("freeze_policy"):
			self.policy_frozen = True
			return "The policy was freezed."

		if inMessage.startswith("unfreeze_policy"):
			self.policy_frozen = False
			return "The policy was unfreezed."

		if inMessage.startswith("save_model"):
			self.dqn.save()
			return "The model was saved."

if __name__ == "__main__":
	AgentLoader.loadAgent(dqn_agent())
