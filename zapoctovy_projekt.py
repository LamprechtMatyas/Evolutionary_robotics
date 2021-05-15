# requires OpenAI gym; it can be intsalled using pip or conda
#      conda install -c conda-forge gym
import gym
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import EvoAlg
import myNN
import CarRacingTools

env = gym.make("CarRacing-v0")

def test_network(net, action_fn, render=True, episode_length=1000, max_negative_rewards_steps=25):
    # Each of these is its own game.
    rwrds = []
    for episode in range(1):
        negative_rewards_steps = 0
        total_reward = 0

        observation = env.reset()
        for t in range(episode_length):
            if render:
                env.render()

            action = action_fn(net, observation)

            observation, reward, done, info = env.step(action)
            total_reward += reward

            if reward > 0:
                negative_rewards_steps = 0
            else:
                negative_rewards_steps += 1

            if negative_rewards_steps > max_negative_rewards_steps:
                total_reward -= (episode_length - t) * 0.1
                done = True

            if done:
                print(f"Done after {t} steps")
                break

        print('Obtained reward:', total_reward)

    return total_reward


def GetActionByCarPicture(net, observation):
    beg_y, beg_x = CarRacingTools.find_beginning_of_car(observation)
    if beg_y != -1:
        z = observation[beg_y-6: beg_y-1, beg_x-5:beg_x+7, :]
        track_array = CarRacingTools.state_to_track(z)
        action = myNN.net_out(net, track_array.reshape(-1), 0.3)
        action[0] = action[0] * 2 - 1
        #if (action[0] > -0.05) and (action[0] < 0.05):
        #    action[0] = 0
        #if ((last_action < 0) and (action[0] > 0)) or ((last_action > 0) and (action[0] < 0)):
        #    total_reward -= 0.02
    else:
        action = [0, 0.5, 0]

    return action

myNN.COEF = 5
ARCH = [12*5, 50, 3]
COEF = 5

def NetFitnessCartPoleByCarPicture(v):
    x = myNN.vec_to_net(v, ARCH, COEF)
    render = (rnd.random() < 0)
    return test_network(x, GetActionByCarPicture, render=render)




myNN.ARCH = ARCH
# res = [EvoAlg.HillClimber(NetFitnessCartPoleByCarPicture, MaxGen=200, u=0.05) for i in range(2)]
max_fitness = []
best_people = []
for i in range(1):
    x, y = EvoAlg.GenAlg(NetFitnessCartPoleByCarPicture, L=myNN.net_size(ARCH), PopSize=15, MaxGen=20, u=0.05)
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
    print(test_network(net, GetActionByCarPicture, render=True, episode_length=25000))

print(net)
myNN.net_to_file(ARCH, net, "net.txt")

print(myNN.file_to_net("net2.txt"))