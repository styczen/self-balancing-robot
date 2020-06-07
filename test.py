#!/usr/bin/env python
import torch
import numpy as np
import pickle
import pybullet as p
from collections import deque
from ddpg.ddpg_agent import Agent
from environment.env import SelfBalancingRobotEnv


if __name__ == '__main__':
    # Initialize PyBullet and set camera
    pc = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    # Environment
    env = SelfBalancingRobotEnv(physics_client_id=pc, measurement_noise=False)

    # DDPG agent
    agent = Agent(state_size=env.OBSERVATION_SIZE, action_size=env.ACTION_SIZE, random_seed=0)

    # Load actor's and critic's network weights
    # agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    # agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    agent.actor_local.load_state_dict(torch.load('finished_episode_checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('finished_episode_checkpoint_critic.pth'))

    shutdown = False
    max_t = 1500
    for i_episode in range(1000):
        state = env.reset()
        for t in range(max_t):
            keys = p.getKeyboardEvents(physicsClientId=pc)
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                shutdown = True
                break
            action = agent.act(state, add_noise=False)
            state, reward, done, _ = env.step(action)
            if done:
                break
        if shutdown:
            break

    # Shutdown simulation
    p.disconnect(physicsClientId=pc)
