# Introduction

This package provides a Chainer implementation of Deep Q-Network(DQN) described in the following papers:

- [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

[この記事](http://musyoku.github.io/2016/03/06/human-level-control-through-deep-reinforcement-learning/)で実装したコードです。

## Requirements

- [Arcade Learning Environment（ALE）](http://www.arcadelearningenvironment.org/)
- [RL-Glue](https://code.google.com/archive/p/rl-glue-ext/wikis/RLGlueCore.wiki)
- [PL-Glue Python codec](https://sites.google.com/a/rl-community.org/rl-glue/Home/Extensions/python-codec)
- [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)
- Chainer 1.6+
- Seaborn
- Pandas

環境構築に関しては [DQN-chainerリポジトリを動かすだけ](http://vaaaaaanquish.hatenablog.com/entry/2015/12/11/215417) が参考になります。

## Running

e.g. Atari Breakout

Open 4 terminal windows and run the following commands on each terminal: 

Terminal #1

```
rl_glue
```

Terminal #2

```
cd path_to_deep-q-network
python experiment.py --csv_dir breakout/csv --plot_dir breakout/plot
```

Terminal #3

```
cd path_to_deep-q-network/breakout
python train.py
```

Terminal #4

```
cd /home/your_name/ALE
./ale -game_controller rlglue -use_starting_actions true -random_seed time -display_screen true -frame_skip 4 -send_rgb true /path_to_rom/breakout.bin
```

# Experiments

実験に用いたコンピュータのスペックは以下の通りです。

|               |                  |
|---------------|------------------|
| OS            | Ubuntu 14.04 LTS |
| CPU           | Core i7          |
| RAM           | 16GB             |
| GPU           | GTX 970M 6GB     |

## Atari Breakout

![Breakout](http://musyoku.github.io/images/post/2016-03-06/breakout_result.gif)

### Preprocessing

We extract the luminance from the RGB frame and rescale it to 84x84.

Then we stack the 4 most recent frames to produce the input to the DQN.

e.g. 

![frame-0](http://musyoku.github.io/images/post/2016-03-06/breakout_state0.png)
![frame-1](http://musyoku.github.io/images/post/2016-03-06/breakout_state1.png)
![frame-2](http://musyoku.github.io/images/post/2016-03-06/breakout_state2.png)
![frame-3](http://musyoku.github.io/images/post/2016-03-06/breakout_state3.png)

### Training

We trained DQN for a total of 42 hours (8200 episodes, 93 epochs, 4670K frames).

本当は10000プレイさせたかったのですが突然コンピュータがシャットダウンしてしまったため中途半端な結果になってしましました。

#### Score:

![Breakout episode-score](http://musyoku.github.io/images/post/2016-03-06/breakout_episode_reward.png)

#### Highscore:

![Breakout episode-highscore](http://musyoku.github.io/images/post/2016-03-06/breakout_training_episode_highscore.png)

### Evaluation

#### Average score:

![Breakout episode-average](http://musyoku.github.io/images/post/2016-03-06/breakout_evaluation_episode_average_reward.png)

## Atari Pong

![Pong](http://musyoku.github.io/images/post/2016-03-06/pong_result.gif)

### Preprocessing

![frame-0](http://musyoku.github.io/images/post/2016-03-06/pong_state0.png)
![frame-1](http://musyoku.github.io/images/post/2016-03-06/pong_state1.png)
![frame-2](http://musyoku.github.io/images/post/2016-03-06/pong_state2.png)
![frame-3](http://musyoku.github.io/images/post/2016-03-06/pong_state3.png)

### Training

We trained DQN for a total of 50 hours (1500 episodes, 96 epochs, 4849K frames).

#### Score:

![Pong episode-score](http://musyoku.github.io/images/post/2016-03-06/pong_episode_reward.png)

#### Highscore:

![Pong episode-highscore](http://musyoku.github.io/images/post/2016-03-06/pong_training_episode_highscore.png)

### Evaluation

#### Average score:

![Pong episode-average](http://musyoku.github.io/images/post/2016-03-06/pong_evaluation_episode_average_reward.png)
