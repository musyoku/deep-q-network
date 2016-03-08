# Chainer implementation of Deep Q-Network(DQN) 

Papers:
- [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

## Requirements

- [Arcade Learning Environment（ALE）](http://www.arcadelearningenvironment.org/)
- [RL-Glue](https://code.google.com/archive/p/rl-glue-ext/wikis/RLGlueCore.wiki)
- [PL-Glue Python codec](https://sites.google.com/a/rl-community.org/rl-glue/Home/Extensions/python-codec)
- [Atari 2600 VCS ROM Collection](http://www.arcadelearningenvironment.org/)
- Chainer 1.6+
- Seaborn
- Pandas

環境構築に関しては [DQN-chainerリポジトリを動かすだけ](http://vaaaaaanquish.hatenablog.com/entry/2015/12/11/215417) が参考になります。

## Summery

We 
- replace any pooling layers with strided convolutions.
- support Batch Normalization.
- supoort variable layers and units.

## How to run

e.g. Atari Breakout

Open 4 terminal windows and run the following commands on each terminal: 

```
rl_glue
```

```
cd path_to_deep-q-network
python experiment.py --csv_dir breakout/csv --plot_dir breakout/plot
```

```
cd path_to_deep-q-network/breakout
python train.py
```

```
cd /home/your_name/ALE
./ale -game_controller rlglue -use_starting_actions true -random_seed time -display_screen true -frame_skip 4 -send_rgb true /path_to_rom/breakout.bin
```

# Atari Breakout

## Training

![Breakout episode-score](http://musyoku.github.io/images/post/2016-03-06/breakout_episode_reward.gif)


## Evaluation


Coming Soon!

![Episode-4560](http://musyoku.github.io/images/post/2016-03-06/playing-breakout-ep4560.gif)