# -*- coding: utf-8 -*-
import scipy.misc as spm
from rlglue.agent.Agent import Agent as RLGlueAgent
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3
from config import config

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

		self.dqn = DQN(config)
		self.state = np.zeros((config.rl_agent_history_length, config.ale_screen_channels, config.ale_scaled_screen_size[0], self.ale_scaled_screen_size[1]), dtype=np.float32)

		self.exploration_rate = self.dqn.exploration_rate

	def scale_screen(self, observation, new_width, new_height):
		if len(observation.intArray) == 100928:
			pass
		else:
			if config.ale_screen_channels == 3:
				raise Exception("You forgot to add --send_rgb option when you run ALE.")
		observation = np.bitwise_and(np.asarray(observation.intArray[128:]).reshape([210, 160]), 0b0001111)
		observation = (spm.imresize(tmp, (110, 84)))[110-84-8:110-8,:]
		return observation

	def agent_init(self, taskSpecString):
		print "Initializing Agent..."
		print "Task Spec:"
		print TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)

	def reshape_state_to_conv_input(self):
		state = self.state.reshape((1, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[0], self.ale_scaled_screen_size[1]))
		return state

	def dump(self, reward, q):
		if self.time_step % 50 == 0:
			print "time_step:", self.time_step,
			print "reward:", reward,
			print "e:", self.dqn.exploration_rate,
			print "Q:",
			print "max::", np.max(q),
			print "min::", np.min(q)

	def learn():
		self.populating_phase = False
		if self.policy_frozen: # Evaluation phase
			self.exploration_rate = 0.05
		else: # Learning phase
			if self.total_time_step < config.rl_replay_start_size:
				# A uniform random policy is run for 'replay_start_size' frames before learning starts
				# 経験を積むためランダムに動き回るらしい。
				print "Initial exploration before learning starts:", "%d/%d steps" % (self.total_time_step, config.rl_replay_start_size)
				self.populating_phase = True
			else:
				self.dqn.decrease_exploration_rate()
			self.exploration_rate = self.dqn.exploration_rate

		if self.policy_frozen is False:
			self.dqn.store_transition_in_replay_memory(self.time_step, self.last_state, self.last_action.intArray[0], reward, self.state, False)
			if self.populating_phase is False:
				self.dqn.replay_experience(self.time_step)
				if self.time_step % config.rl_target_network_update_frequency == 0 and self.time_step != 0:
					print "Target has been updated."
					self.dqn.update_target()
					if self.episode_step != 0 and self.episode_step % self.model_save_interval == 0:
						self.dqn.save()


	def agent_start(self, observation):
		print "Episode", self.episode_step
		observed_screen = self.scale_screen(observation)
		self.state[0] = observed_screen

		return_action = Action()
		action, q = self.dqn.e_greedy(self.reshape_state_to_conv_input(), self.exploration_rate)
		return_action.intArray = [action]

		self.last_action = copy.deepcopy(return_action)
		self.last_state = self.state.copy()
		self.last_observation = obs_array

		return return_action

	def agent_step(self, reward, observation):
		observed_screen = self.scale_screen(observation)
		self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], observed_screen], dtype=np.float32)

		self.learn()

		return_action = Action()
		action, q = self.dqn.e_greedy(self.reshape_state_to_conv_input(), self.exploration_rate)
		return_action.intArray = [action]

		# [Optional]
		## Visualizing the results
		self.dump(reward, q)

		self.last_observation = observed_screen

		if self.policy_frozen is False:
			self.last_action = copy.deepcopy(return_action)
			self.last_state = self.state.copy()
			self.time_step += 1
			self.total_time_step += 1

		return return_action

	def agent_end(self, reward):
		self.learn()

		# [Optional]
		## Visualizing the results
		self.dump(reward, q)

		if self.policy_frozen is False:
			self.time_step = 0
			self.total_time_step += 1
			self.episode_step += 1

	def agent_cleanup(self):
		pass

	def agent_message(self, inMessage):
		if inMessage.startswith("freeze learning"):
			self.policy_frozen = True
			return "message understood, policy frozen"

		if inMessage.startswith("unfreeze learning"):
			self.policy_frozen = False
			return "message understood, policy unfrozen"

if __name__ == "__main__":
	AgentLoader.loadAgent(dqn_agent())
