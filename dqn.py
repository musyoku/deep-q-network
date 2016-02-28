# -*- coding: utf-8 -*-
import numpy as np
import chainer, math, copy
from chainer import cuda, Variable, optimizers, serializers
from chainer import functions as F
from chainer import links as L
from activations import activations
from config import config

class ConvolutionalNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(ConvolutionalNetwork, self).__init__(**layers)
		self.activation_function = "elu"
		self.projection_type = "fully_connection"
		self.n_hidden_layers = 0
		self.top_filter_size = (1, 1)
		self.apply_batchnorm = True
		self.apply_batchnorm_to_input = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden convolutinal layers
		for i in range(self.n_hidden_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input == False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			chain.append(f(u))

		if self.projection_type == "fully_connection":
			chain.append(self.projection_layer(chain[-1]))

		elif self.projection_type == "global_average_pooling":
			batch_size = chain[-1].data.shape[0]
			n_maps = chain[-1].data[0].shape[0]
			chain.append(F.average_pooling_2d(chain[-1], self.top_filter_size))
			chain.append(F.reshape(chain[-1], (batch_size, n_maps)))

		else:
			raise NotImplementedError()

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class FullyConnectedNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(FullyConnectedNetwork, self).__init__(**layers)
		self.n_layers = 0
		self.activation_function = "elu"
		self.apply_batchnorm_to_input = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden layers
		for i in range(self.n_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input == False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)
		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class DQN:
	def __init__(self):
		print "Initializing DQN..."
		self.exploration_rate = config.rl_initial_exploration

		# Q Network
		conv, fc = build_q_network(config)
		self.conv = conv
		self.fc = fc
		self.fcl_eliminated = True if len(config.q_fc_hidden_units) == 0 else False
		self.update_target()

		# Optimizer
		## RMSProp, ADAM, AdaGrad, AdaDelta, ...
		## See http://docs.chainer.org/en/stable/reference/optimizers.html
		self.optimizer_conv = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)

		# Replay Memory
		## (state, action, reward, next_state, episode_ends_or_not)
		shape_state = (config.rl_replay_memory_size, config.rl_agent_history_length, config.ale_screen_channels, config.ale_scaled_screen_size[0], config.ale_scaled_screen_size[1])
		shape_action = (config.rl_replay_memory_size,)
		self.replay_memory = [
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.uint8),
			np.zeros(shape_action, dtype=np.int8),
			np.zeros(shape_state, dtype=np.float32)
			np.zeros(shape_action, dtype=np.bool),
		]

	def e_greedy(self, state):
		state = Variable(state)
		if config.use_gpu:
			state.to_gpu()
		q = self.compute_q_value(state)
		if config.use_gpu:
			q.to_cpu()
		q = q.data
		prop = np.random.uniform()
		if prop < config.exploration_rate:
			# Select a random action
			action_index = np.random.randint(0, len(config.ale_actions))
		else:
			# Select a greedy action
			action_index = xp.argmax(q)

		return self.get_action_with_index(action_index), q

	def store_transition_in_replay_memory(self, time_step, state, action, reward, new_state, episode_ends_or_not):
		self.replay_memory[0]

	def forward_one_step(self, state, action, reward, next_state, episode_ends_or_not):
		pass

	def replay_experience(self, time_step):
		# Sample random minibatch of transitions from replay memory
		if time_step < config.rl_replay_memory_size:
			replay_index = np.random.randint(0, time_step, (config.rl_minibatch_size, 1))
		else:
			replay_index = np.random.randint(0, config.rl_replay_memory_size, (config.rl_minibatch_size, 1))

		shape_state = (config.rl_minibatch_size, config.rl_agent_history_length, config.ale_screen_channels, config.ale_scaled_screen_size[0], config.ale_scaled_screen_size[1])
		shape_action = (config.rl_minibatch_size,)

		state = np.empty(shape_state, dtype=np.float32)
		action = np.empty(shape_action, dtype=np.uint8)
		reward = np.empty(shape_action, dtype=np.int8)
		next_state = np.empty(shape_state, dtype=np.float32)
		epsode_ends_or_not = np.empty(shape_action, dtype=np.uint8)
		for i in xrange(self.rl_minibatch_size):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[0][replay_index[i]]
			reward[i] = self.replay_memory[0][replay_index[i]]
			next_state[i] = self.replay_memory[0][replay_index[i]]
			epsode_ends_or_not[i] = self.replay_memory[0][replay_index[i]]
		pass

	def compute_q_value(self, state):
		output = self.conv(state)
		if self.fcl_eliminated:
			return output
		output = self.fc(output)
		return output

	def compute_target_q_value(self):
		output = self.target_conv(state)
		if self.fcl_eliminated:
			return output
		output = self.target_fc(output)
		return output

	def update_target():
		self.target_conv = copy.deepcopy(conv)
		self.target_fc = copy.deepcopy(fc)

	def get_action_with_index(self, i):
		return self.actions[i]

	def get_index_with_action(self, action):
		return self.actions.index(action)

	def decrease_exploration_rate():
		# Exploration rate is linearly annealed to its final value
		self.exploration_rate -= 1.0 / config.rl_final_exploration_frame
		if self.exploration_rate < config.rl_final_exploration:
			self.exploration_rate = config.rl_final_exploration

	def save():
		pass

def build_q_network(config):
	config.check()
	initial_weight_variance = 0.0001

	# Convolutional part of Q-Network
	conv_attributes = {}
	conv_channels = [(config.ale_screen_channels, config.q_conv_hidden_channels[0])]
	conv_channels += zip(config.q_conv_hidden_channels[:-1], config.q_conv_hidden_channels[1:])

	output_map_width = config.ale_scaled_screen_size[0]
	output_map_height = config.ale_scaled_screen_size[1]
	for n in xrange(len(config.q_conv_hidden_channels)):
		output_map_width = (output_map_width - config.q_conv_filter_sizes[n]) / config.q_conv_strides[n] + 1
		output_map_height = (output_map_height - config.q_conv_filter_sizes[n]) / config.q_conv_strides[n] + 1

	for i, (n_in, n_out) in enumerate(conv_channels):
		conv_attributes["layer_%i" % i] = L.Convolution2D(n_in, n_out, config.q_conv_filter_sizes[i], stride=config.q_conv_strides[i], pad=1, wscale=initial_weight_variance * math.sqrt(n_in * n_out))
		conv_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	if config.q_conv_output_projection_type == "fully_connection":
		conv_attributes["projection_layer"] = L.Linear(output_map_width * output_map_height * config.q_conv_output_vector_dimension, config.q_conv_output_vector_dimension, wscale=initial_weight_variance * math.sqrt(output_map_width * output_map_height * config.q_conv_output_vector_dimension))

	conv = ConvolutionalNetwork(**conv_attributes)
	conv.n_hidden_layers = len(config.q_conv_hidden_channels)
	conv.activation_function = config.q_conv_activation_function
	conv.top_filter_size = (output_map_width, output_map_height)
	conv.projection_type = config.q_conv_output_projection_type
	conv.apply_batchnorm = config.apply_batchnorm
	conv.apply_batchnorm_to_input = config.q_conv_apply_batchnorm_to_input

	# Fully connected part of Q-Network
	fc_attributes = {}
	fc_units = [(config.q_conv_output_vector_dimension, config.q_fc_hidden_units[0])]
	fc_units += zip(config.q_fc_hidden_units[:-1], config.q_fc_hidden_units[1:])
	fc_units += [(config.q_fc_hidden_units[-1], len(config.ale_controllers))]

	for i, (n_in, n_out) in enumerate(fc_units):
		fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=initial_weight_variance * math.sqrt(n_in * n_out))
		fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	fc = FullyConnectedNetwork(**fc_attributes)
	fc.n_layers = len(fc_units)
	fc.activation_function = config.q_fc_activation_function
	fc.apply_batchnorm = config.apply_batchnorm
	fc.apply_dropout = config.q_fc_apply_dropout
	fc.apply_batchnorm_to_input = config.q_fc_apply_batchnorm_to_input

	if config.use_gpu:
		conv.to_gpu()
		fc.to_gpu()
	return conv, fc