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
				if i == 0 and self.apply_batchnorm_to_input is False:
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
		self.n_hidden_layers = 0
		self.activation_function = "elu"
		self.apply_batchnorm_to_input = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden layers
		for i in range(self.n_hidden_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input is False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		# Output
		u = getattr(self, "layer_%i" % self.n_hidden_layers)(chain[-1])
		if self.apply_batchnorm:
			u = getattr(self, "batchnorm_%i" % self.n_hidden_layers)(u, test=test)
		chain.append(f(u))

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

class DQN:
	def __init__(self):
		print "Initializing DQN..."
		self.exploration_rate = config.rl_initial_exploration
		self.fcl_eliminated = True if len(config.q_fc_hidden_units) == 0 else False

		# Q Network
		conv, fc = build_q_network(config)
		self.conv = conv
		if self.fcl_eliminated is False:
			self.fc = fc
		self.update_target()

		# Optimizer
		## RMSProp, ADAM, AdaGrad, AdaDelta, ...
		## See http://docs.chainer.org/en/stable/reference/optimizers.html
		self.optimizer_conv = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_conv.setup(self.conv)
		if self.fcl_eliminated is False:
			self.optimizer_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
			self.optimizer_fc.setup(self.fc)

		# Replay Memory
		## (state, action, reward, next_state, episode_ends)
		shape_state = (config.rl_replay_memory_size, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0])
		shape_action = (config.rl_replay_memory_size,)
		self.replay_memory = [
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.uint8),
			np.zeros(shape_action, dtype=np.int8),
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.bool)
		]

	def e_greedy(self, state, exploration_rate):
		state = Variable(state)
		if config.use_gpu:
			state.to_gpu()
		q = self.compute_q_value_variable(state)
		if config.use_gpu:
			q.to_cpu()
		q = q.data
		prop = np.random.uniform()
		if prop < exploration_rate:
			# Select a random action
			action_index = np.random.randint(0, len(config.ale_actions))
		else:
			# Select a greedy action
			action_index = np.argmax(q)

		return self.get_action_with_index(action_index), q

	def store_transition_in_replay_memory(self, time_step, state, action, reward, new_state, episode_ends):
		index = time_step % config.rl_replay_memory_size
		self.replay_memory[0][index] = state
		self.replay_memory[1][index] = action
		self.replay_memory[2][index] = reward
		if episode_ends is True:
			self.replay_memory[3][index] = new_state
		self.replay_memory[4][index] = episode_ends

	def forward_one_step(self, state, action, reward, next_state, episode_ends):
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state)
		next_state = Variable(next_state)
		if config.use_gpu:
			state.to_gpu()
			next_state.to_gpu()
		q = self.compute_q_value_variable(state)

		# Generate target
		max_target_q_value = self.compute_target_q_value_variable(next_state)
		max_target_q_value = list(map(xp.max, max_target_q_value.data))
		max_target_q_value = xp.asanyarray(max_target_q_value, dtype=xp.float32)

		# 教師信号を現在のQ値で初期化
		target = xp.asanyarray(q.data, dtype=xp.float32)

		for i in xrange(n_batch):
			# Clip all positive rewards at 1 and all negative rewards at -1
			# プラスの報酬はすべて1にし、マイナスの報酬はすべて-1にする
			if episode_ends[i] is True:
				target_value = np.sign(reward[i])
			else:
				target_value = np.sign(reward[i]) + config.rl_discount_factor * max_target_q_value[i]
			action_index = self.get_index_with_action(action[i])

			# 現在選択した行動に対してのみ誤差を伝播する。
			# それ以外の行動を表すユニットの2乗誤差は0となる。（target=qとなるため）
			target[i, action_index] = target_value

		# Compute error
		target = Variable(target)
		loss = target - q
		loss *= loss
		# Clip the error to be between -1 and 1
		loss /= (abs(loss.data) + 1.0)

		zero = Variable(xp.zeros((n_batch, len(config.ale_actions)), dtype=xp.float32))
		loss = F.mean_squared_error(loss, zero)
		return loss, q

	def replay_experience(self, time_step):
		if time_step == 0:
			return
		# Sample random minibatch of transitions from replay memory
		if time_step < config.rl_replay_memory_size:
			replay_index = np.random.randint(0, time_step, (config.rl_minibatch_size, 1))
		else:
			replay_index = np.random.randint(0, config.rl_replay_memory_size, (config.rl_minibatch_size, 1))

		shape_state = (config.rl_minibatch_size, config.rl_agent_history_length * config.ale_screen_channels, config.ale_scaled_screen_size[1], config.ale_scaled_screen_size[0])
		shape_action = (config.rl_minibatch_size,)

		state = np.empty(shape_state, dtype=np.float32)
		action = np.empty(shape_action, dtype=np.uint8)
		reward = np.empty(shape_action, dtype=np.int8)
		next_state = np.empty(shape_state, dtype=np.float32)
		episode_ends = np.empty(shape_action, dtype=np.uint8)
		for i in xrange(config.rl_minibatch_size):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[1][replay_index[i]]
			reward[i] = self.replay_memory[2][replay_index[i]]
			next_state[i] = self.replay_memory[3][replay_index[i]]
			episode_ends[i] = self.replay_memory[4][replay_index[i]]

		self.optimizer_conv.zero_grads()
		if self.fcl_eliminated is False:
			self.optimizer_fc.zero_grads()
		loss, _ = self.forward_one_step(state, action, reward, next_state, episode_ends)
		loss.backward()
		self.optimizer_conv.update()
		if self.fcl_eliminated is False:
			self.optimizer_fc.update()

	def compute_q_value_variable(self, state):
		output = self.conv(state)
		if self.fcl_eliminated:
			return output
		output = self.fc(output)
		return output

	def compute_target_q_value_variable(self, state):
		output = self.target_conv(state)
		if self.fcl_eliminated:
			return output
		output = self.target_fc(output)
		return output

	def update_target(self):
		self.target_conv = copy.deepcopy(self.conv)
		if self.fcl_eliminated is False:
			self.target_fc = copy.deepcopy(self.fc)

	def get_action_with_index(self, i):
		return config.ale_actions[i]

	def get_index_with_action(self, action):
		return config.ale_actions.index(action)

	def decrease_exploration_rate(self):
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
	conv_channels = [(config.rl_agent_history_length * config.ale_screen_channels, config.q_conv_hidden_channels[0])]
	conv_channels += zip(config.q_conv_hidden_channels[:-1], config.q_conv_hidden_channels[1:])

	output_map_width = config.ale_scaled_screen_size[0]
	output_map_height = config.ale_scaled_screen_size[1]
	for n in xrange(len(config.q_conv_hidden_channels)):
		output_map_width = (output_map_width - config.q_conv_filter_sizes[n]) / config.q_conv_strides[n] + 1
		output_map_height = (output_map_height - config.q_conv_filter_sizes[n]) / config.q_conv_strides[n] + 1

	for i, (n_in, n_out) in enumerate(conv_channels):
		conv_attributes["layer_%i" % i] = L.Convolution2D(n_in, n_out, config.q_conv_filter_sizes[i], stride=config.q_conv_strides[i], wscale=initial_weight_variance * math.sqrt(n_in * n_out))
		conv_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

	if config.q_conv_output_projection_type == "fully_connection":
		conv_attributes["projection_layer"] = L.Linear(output_map_width * output_map_height * config.q_conv_hidden_channels[-1], config.q_conv_output_vector_dimension, wscale=initial_weight_variance * math.sqrt(output_map_width * output_map_height * config.q_conv_output_vector_dimension))

	conv = ConvolutionalNetwork(**conv_attributes)
	conv.n_hidden_layers = len(config.q_conv_hidden_channels)
	conv.activation_function = config.q_conv_activation_function
	conv.top_filter_size = (output_map_width, output_map_height)
	conv.projection_type = config.q_conv_output_projection_type
	conv.apply_batchnorm = config.apply_batchnorm
	conv.apply_batchnorm_to_input = config.q_conv_apply_batchnorm_to_input
	if config.use_gpu:
		conv.to_gpu()

	# Fully connected part of Q-Network
	if len(config.q_fc_hidden_units) > 0:
		fc_attributes = {}
		fc_units = [(config.q_conv_output_vector_dimension, config.q_fc_hidden_units[0])]
		fc_units += zip(config.q_fc_hidden_units[:-1], config.q_fc_hidden_units[1:])
		fc_units += [(config.q_fc_hidden_units[-1], len(config.ale_actions))]

		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=initial_weight_variance * math.sqrt(n_in * n_out))
			fc_attributes["batchnorm_%i" % i] = L.BatchNormalization(n_out)

		fc = FullyConnectedNetwork(**fc_attributes)
		fc.n_hidden_layers = len(fc_units) - 1
		fc.activation_function = config.q_fc_activation_function
		fc.apply_batchnorm = config.apply_batchnorm
		fc.apply_dropout = config.q_fc_apply_dropout
		fc.apply_batchnorm_to_input = config.q_fc_apply_batchnorm_to_input
		if config.use_gpu:
			fc.to_gpu()
	else:
		fc = None

	return conv, fc