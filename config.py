# -*- coding: utf-8 -*-
import math
from activations import activations

class Config:
	def check(self):
		# Check activations
		if self.q_conv_activation_function not in activations:
			raise Exception("Invalid activation function for q_conv_activation_function.")
		if self.q_fc_activation_function not in activations:
			raise Exception("Invalid activation function for q_fc_activation_function.")

		# Check convolutional network
		n_conv_hidden_layers = len(self.q_conv_hidden_channels)
		if len(self.q_conv_filter_sizes) != n_conv_hidden_layers:
			raise Exception("Invlaid number of elements for q_conv_filter_sizes")
		if len(self.q_conv_strides) != n_conv_hidden_layers:
			raise Exception("Invlaid number of elements for q_conv_strides")

		q_output_map_width = self.ale_scaled_screen_size[0]
		q_output_map_height = self.ale_scaled_screen_size[1]
		stndrdth = ("st", "nd", "rd")
		for n in xrange(len(self.q_conv_hidden_channels)):
			if (q_output_map_width - self.q_conv_filter_sizes[n]) % self.q_conv_strides[n] != 0:
				print "WARNING"
				print "at", (("%d%s" % (n + 1, stndrdth[n])) if n < 3 else "th"), "conv layer:"
				print "width of input maps:", q_output_map_width
				print "stride:", self.q_conv_strides[n]
				print "filter size:", (self.q_conv_filter_sizes[n], self.q_conv_filter_sizes[n])
				print "width of outout maps:", (q_output_map_width - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]) + 1
				print "The width of output maps MUST be an integer!"
				possible_strides = []
				for _stride in range(1, 11):
					if (q_output_map_width - self.q_conv_filter_sizes[n]) % _stride == 0:
						possible_strides.append(_stride)
				if len(possible_strides) > 0:
					print "I recommend you to"
					print "	use stride of", possible_strides
					print "	or"
				print "	change input image size to",
				new_image_width = int(math.ceil((q_output_map_width - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]))) * self.q_conv_strides[n] + self.q_conv_filter_sizes[n]
				new_image_height = q_output_map_height
				for _n in xrange(n):
					new_image_width = (new_image_width - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
					new_image_height = (new_image_height - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
				print (new_image_width, new_image_height)
				raise Exception()
			if q_output_map_height % self.q_conv_strides[n] != 0:
				print "WARNING"
				print "at", (("%d%s" % (n + 1, stndrdth[n])) if n < 3 else "th"), "conv layer:"
				print "height of input maps:", q_output_map_height
				print "stride:", self.q_conv_strides[n]
				print "filter size:", (self.q_conv_filter_sizes[n], self.q_conv_filter_sizes[n])
				print "height of outout maps:", (q_output_map_height - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]) + 1
				print "The height of output maps MUST be an integer!"
				possible_strides = []
				for _stride in range(1, 11):
					if (q_output_map_height - self.q_conv_filter_sizes[n]) % _stride == 0:
						possible_strides.append(_stride)
				if len(possible_strides) > 0:
					print "I recommend you to"
					print "	use stride of", possible_strides
					print "	or"
				print "	change input image size to",
				new_image_width = q_output_map_width
				new_image_height = int(math.ceil((q_output_map_height - self.q_conv_filter_sizes[n]) / float(self.q_conv_strides[n]))) * self.q_conv_strides[n] + self.q_conv_filter_sizes[n]
				for _n in xrange(n):
					new_image_width = (new_image_width - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
					new_image_height = (new_image_height - 1) * self.q_conv_strides[_n] + self.q_conv_filter_sizes[_n]
				print (new_image_width, new_image_height)
				raise Exception()
			q_output_map_width = (q_output_map_width - self.q_conv_filter_sizes[n]) / self.q_conv_strides[n] + 1
			q_output_map_height = (q_output_map_height - self.q_conv_filter_sizes[n]) / self.q_conv_strides[n] + 1
		if q_output_map_width <= 0 or q_output_map_height <= 0:
			raise Exception("The size of the output feature maps will be 0 in the current settings.")

		# print "The size of the output feature maps is", (q_output_map_width, q_output_map_height)

		if config.q_conv_output_projection_type not in {"fully_connection", "global_average_pooling"}:
			raise Exception("Invalid type of projection for q_conv_output_projection_type.")

config = Config()

# General
config.use_gpu = True
config.apply_batchnorm = True

# ALE
## Raw screen image width and height.
config.ale_screen_size = [120, 280]

## Scaled screen image width and height.
## Input scaled images to convolutional network
config.ale_scaled_screen_size = [62, 142]

## greyscale -> 1
## rgb -> 3
config.ale_screen_channels = 1

## List of actions
## The required actions are written in ale_dir/src/games/supported/the_name_of_the_game_you_want_to_play.cpp,
## The corrensponding integers are defined in ale_dir/src/common/Constants.h 
## ゲームをプレイするのに必要な操作のリスト。
## それぞれのゲームでどの操作が必要かはale_dir/src/games/supported/ゲーム名.cppに書いてあります。
## 各操作の定義はale_dir/src/common/Constants.hで行われているので参照し、数値に変換してください。
config.ale_actions = [0, 3, 4]

# Reinforcment Learning
## These hyperparameters are based on the original paper in Nature.
## For more details see following:
## [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html)

config.rl_minibatch_size = 32

## The number of most recent frames experienced by the agent that are given as input to the convolutional network
config.rl_agent_history_length = 4

config.rl_replay_memory_size = 10 ** 5
config.rl_target_network_update_frequency = 10 ** 4
config.rl_discount_factor = 0.99
config.rl_update_frequency = 4
config.rl_learning_rate = 0.00025

## Gradient momentum used by optimizer of Chainer (RMSProp, ADAM, etc...)
config.rl_gradient_momentum = 0.95

config.rl_initial_exploration = 1.0
config.rl_final_exploration = 0.1
config.rl_final_exploration_frame = 10 ** 6
config.rl_replay_start_size = 5 * 10 ** 4
config.rl_no_op_max = 30


# Q-Network
## The list of the number of channel for each hidden convolutional layer (input side -> output side)
## The number of elements is the number of hidden layers.
## Note:Be careful of adding too many layers.
### We use each convolutional layer as pooling layer (Strided Convolution), therefore the size of the output feature maps will be zero when you add too many convolutional layers. 
### For more details on Strided Convolution, see following papers:
### [All Convolutional Net](http://arxiv.org/abs/1412.6806)
### [DCGAN](http://arxiv.org/abs/1511.06434)
## Q関数を構成する畳み込みネットワークの隠れ層のチャネル数。
## 要素の数がそのままレイヤー数になります。
## 入力側から出力側に向かって設定してください。
## レイヤーを通過するたび出力マップサイズは1/strideに縮小されるので、レイヤー数を増やしすぎると出力マップサイズが0になってしまうことに注意が必要です。
config.q_conv_hidden_channels = [32, 64, 64, 64]

## The list of stride for each hidden convolutional layer (input side -> output side)
config.q_conv_strides = [2, 2, 2, 2]

## The list of filter size of each convolutional layer (input side -> output side)
config.q_conv_filter_sizes = [4, 4, 4, 4]

## See activations.py
config.q_conv_activation_function = "elu"

## Whether or not to apply batch normalization to the input of convolutional network (the raw screen image from ALE)
## This overrides config.apply_batchnorm
## 畳み込み層への入力（つまりゲーム画面の画像データ）にバッチ正規化を適用するかどうか
## config.apply_batchnormの設定によらずこちらが優先されます
config.q_conv_apply_batchnorm_to_input = False

## Single fully connected layer is placed on top of the convolutional network to convert output feature maps to vector.
## This vector is fed into fully connected layers.
## If you eliminate the fully connected layer, set the value of len(config.ale_actions).
## 畳み込み層の最終的な出力マップをベクトルへ変換するときの次元数です。このベクトルは全結合層へ入力されます。
## 全結合層を使わない場合はlen(config.ale_actions)と同じ値にしてください。
config.q_conv_output_vector_dimension = 100

## "global_average_pooling" or "fully_connection"
## Specify how to convert the output feature maps to vector
## For more details on Global Average Pooling, see following papers:
## Network in Network(http://arxiv.org/abs/1312.440)0
config.q_conv_output_projection_type = "fully_connection"

## The number of units for each fully connected layer.
## These are placed on top of the convolutional network.
## set [] if you want to eliminate fully connected layers. It is OK because convolutinal network outputs a vector.
## Note: There is the trend towards eliminating fully connected layers to avoid overfitting.
## 畳み込み層を接続する全結合層のユニット数を入力側から出力側に向かって並べてください。
## []を指定すれば全結合層を削除できます。
## 最近の畳み込みニューラルネットは過学習を避けるために全結合層を使わない傾向があるようです。
config.q_fc_hidden_units = [512, 256, 64]

## See activations.py
config.q_fc_activation_function = "elu"

## Whether or not to apply dropout to all fully connected layers
config.q_fc_apply_dropout = False

## Whether or not to apply batch normalization to the input of fully connected network (the output of convolutional network)
## This overrides config.apply_batchnorm
## 全結合層への入力（つまり畳み込み層の出力）にバッチ正規化を適用するかどうか
## config.apply_batchnormの設定によらずこちらが優先されます
config.q_fc_apply_batchnorm_to_input = False