# requires OpenAI gym; it can be intsalled using pip or conda
#      conda install -c conda-forge gym
import gym
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import EvoAlg
import myNN

env = gym.make("CarRacing-v0")
env.reset()


def find_beginning_of_car(arr):
    for i in range(len(arr)//2, len(arr)):
        for j in range(len(arr[i])):
            if (arr[i][j][0] == 204) & (arr[i][j][1] == 0) & (arr[i][j][2] == 0):
                return i, j
    return -1, -1


def is_track(arr):
    new_arr = np.zeros(shape=(len(arr), len(arr[0])))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if (arr[i][j][0] > 98) & (arr[i][j][0] < 120) & (arr[i][j][1] > 98) & (arr[i][j][1] < 120) & (arr[i][j][2] > 98) & (arr[i][j][2] < 120):
                new_arr[i][j] = 1
            else:
                new_arr[i][j] = 0
    return new_arr


def test_network(net, render=True, episode_length=1000):
    # Each of these is its own game.
    rwrds = []
    for episode in range(1):
        env.reset()

        total_reward = 0
        observation = None
        num_negative_rewards = 0
        for t in range(episode_length):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            if render:
                env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            if t > 0:
                beg_y, beg_x = find_beginning_of_car(observation)
                if beg_y != -1:
                    z = observation[beg_y-6: beg_y-1, beg_x-5:beg_x+7, :]
                    track_array = is_track(z)
                    action = myNN.net_out(net, track_array.reshape(-1))
                    action[0] = action[0] * 2 - 1
                    #if (action[0] > -0.05) and (action[0] < 0.05):
                    #    action[0] = 0
                    #if ((last_action < 0) and (action[0] > 0)) or ((last_action > 0) and (action[0] < 0)):
                    #    total_reward -= 0.02
                else:
                    action = [0, 0.5, 0]
            else:
                action = env.action_space.sample()
            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if reward > 0:
                num_negative_rewards = 0
            else:
                num_negative_rewards += 1
            if num_negative_rewards > 25:
                total_reward -= (episode_length - t) * 0.1
                break
            if done:
                print("Done")
                break
        print('Obtained reward:', total_reward)
    return total_reward


myNN.COEF = 5
ARCH = [60, 50, 3]
COEF = 5


def NetFitnessCartPole(v):
    x = myNN.vec_to_net(v, ARCH, COEF)
    render = False
    if rnd.random() < 0:
        render = True
    return test_network(x, render=render)


def net_to_file(arch, net, filepath):
    with open(filepath, "w") as f:
        f.write("\t".join(map(str, arch)) + "\n")
        for layer in net:
            height, width = layer.shape
            for y in range(height):
                f.write("\t".join([str(layer[y, x]) for x in range(width)]) + "\n")


def file_to_net(filepath):
    global ARCH
    with open(filepath, "r") as f:
        net = []
        arch = list(map(int, f.readline().split("\t")))
        prev = None
        for layerSize in arch:
            if prev is not None:
                height = prev + 1
                width = layerSize
                layer = np.empty([height, width])
                for y in range(height):
                    layer[y, :] = list(map(float, f.readline().split("\t")))
                net.append(layer)
            prev = layerSize
    ARCH = arch
    return net


myNN.ARCH = ARCH
# res = [EvoAlg.HillClimber(NetFitnessCartPole, MaxGen=200, u=0.05) for i in range(2)]
max_fitness = []
best_people = []
for i in range(1):
    x, y = EvoAlg.GenAlg(NetFitnessCartPole, L=myNN.net_size(ARCH), PopSize=15, MaxGen=20, u=0.05)
    max_fitness.append(x)
    best_people.append(y)
    print(i)

x = np.arange(len(max_fitness[0]))
for fr in max_fitness:
    plt.plot(x, fr)
plt.title("Vývoj fitness v 5 behoch Genetic Algoritmu")
plt.xlabel("Generácia")
plt.ylabel("Fitness")
plt.show()
# plt.ylim(0,1)

nej = -1000
nej_pr = 0
for i in range(len(max_fitness)):
    if max_fitness[i][len(max_fitness) - 1] > nej:
        nej_pr = i
        nej = max_fitness[i][len(max_fitness) - 1]
net = myNN.vec_to_net(best_people[nej_pr][len(max_fitness[nej_pr]) - 1])
for i in range(10):
    print(test_network(net, render=True, episode_length=25000))

print(net)
net_to_file(ARCH, net, "net.txt")

print(file_to_net("net2.txt"))