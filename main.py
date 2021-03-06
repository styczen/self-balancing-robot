#!/usr/bin/env python
import time
import pybullet as p
import numpy as np
import random
from environment.env import SelfBalancingRobotEnv

DT = 1 / 240

if __name__ == '__main__':
    pc = p.connect(p.GUI)

    env = SelfBalancingRobotEnv(physics_client_id=pc)
    obs = env.reset()

    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    start_time = time.time()
    accumulated_reward = 0
    while True:
        keys = p.getKeyboardEvents(physicsClientId=pc)
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break

        action = random.random() * 2 - 1  # [-1, 1)
        observation, reward, done, _ = env.step(np.array([action, action]))
        if done:
            accumulated_reward = 0
            # obs = env.reset()

        time.sleep(DT)

    # for i in range(10):
    #     print(i)
    #     if i == 5:
    #         break
    # else:
    #     print('no break')
    # print('something else')

    p.disconnect(physicsClientId=pc)

    # plt.figure()
    # plt.plot(times, velocities)
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(times, robot_orns)
    # plt.grid()
    # plt.legend()
    # plt.show()
