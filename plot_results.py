#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == '__main__':
    scores = pickle.load(open('scores.pickle', 'rb'))
    finished_episodes = pickle.load(open('finished_episodes.pickle', 'rb'))
    print(f'There are {len(finished_episodes)} finished episodes.')

    plt.figure()
    plt.plot(scores, color='C0', alpha=0.5)
    plt.plot(moving_average(scores, 100), color='C0')
    plt.title('Accumulated rewards')
    plt.xlabel('Episode')
    plt.grid()
    plt.show()
