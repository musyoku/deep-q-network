# -*- coding: utf-8 -*-
"""
Simple RL glue experiment setup
"""

import numpy as np
import rlglue.RLGlue as RLGlue

max_episode = 10000

total_episode = 0
learned_episode = 0


def run_episode(learning=True):
    global total_episode, learned_episode
    RLGlue.RL_episode(0)
    total_steps = RLGlue.RL_num_steps()
    total_reward = RLGlue.RL_return()

    total_episode += 1

    if learning:
        learned_episode += 1
        print "Episode " + str(learned_episode) + "\t " + str(total_steps) + " steps \t" + str(total_reward) + " total reward\t "
    else:
        print "Evaluation ::\t " + str(total_steps) + " steps \t" + str(total_reward) + " total reward\t "


RLGlue.RL_init()

while learned_episode < max_episode:
    # Evaluate model every 10 episodes
    if np.mod(total_episode, 10) == 0 and total_episode != 0:
        print "Freezing the policy for evaluation..."
        RLGlue.RL_agent_message("freeze_policy")
        run_episode(learning=False)
    else:
        print "DQN is learning..."
        RLGlue.RL_agent_message("unfreeze_policy")
        run_episode(learning=True)

    if np.mod(learned_episode, 100) == 0 and learned_episode != 0:
        print "Saving the model..."
        RLGlue.RL_agent_message("save_model")

RLGlue.RL_cleanup()

print "Experiment has ended at episode", total_episode
