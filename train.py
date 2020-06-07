#!/usr/bin/env python
import torch
import numpy as np
import datetime
import pickle
import time
import logging
import pybullet as p
from collections import deque
from ddpg.ddpg_agent import Agent
from environment.env import SelfBalancingRobotEnv

if __name__ == '__main__':
    # Set debug level
    logging.basicConfig(level=logging.WARN)

    # Initialize PyBullet and set camera
    pc = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    # Environment
    env = SelfBalancingRobotEnv(physics_client_id=pc, measurement_noise=False)

    # TODO: Create argument parser
    n_episodes = 100000
    max_t = 1500
    print_every = 10

    # DDPG agent
    agent = Agent(state_size=env.OBSERVATION_SIZE, action_size=env.ACTION_SIZE, random_seed=0)

    # Logging
    scores_deque = deque(maxlen=print_every)
    scores = []
    finished_episodes = []
    shutdown = False
    training_start = time.time()
    for i_episode in range(1, n_episodes + 1):
        logging.debug(f'---Episode {i_episode}---')
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            keys = p.getKeyboardEvents(physicsClientId=pc)
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                shutdown = True
                break
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            logging.debug(
                f't: {t}; action: {action}; state: {next_state}; reward: {reward}; accum_reward: {score}, done: {done}')
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        else:
            finished_episodes.append(i_episode)
            torch.save(agent.actor_local.state_dict(), 'finished_episode_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'finished_episode_checkpoint_critic.pth')
        if shutdown:
            break
        scores_deque.append(score)
        scores.append(score)
        # Logging networks' weights, scores and finished episodes
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        pickle.dump(scores, open('scores.pickle', 'wb'))
        pickle.dump(finished_episodes, open('finished_episodes.pickle', 'wb'))
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    elapsed_time = time.time() - training_start
    print(f'Training time: {datetime.timedelta(seconds=elapsed_time)}')

    # Shutdown simulation
    p.disconnect(physicsClientId=pc)
