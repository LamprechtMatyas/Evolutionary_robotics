# requires OpenAI gym; it can be intsalled using pip or conda
#      conda install -c conda-forge gym
import gym
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import EvoAlg
import myNN
import CarRacingTools

env = gym.make("CarRacing-v0", verbose=0)


def test_network(net, agent, render=True, episodes=1, verbose=False, episode_length=1000, max_negative_rewards_steps=None):
    if max_negative_rewards_steps is None:
        max_negative_rewards_steps = episode_length

    rewards = []
    for episode in range(episodes):
        negative_rewards_steps = 0
        total_reward = 0

        observation = env.reset()
        for t in range(episode_length):
            if render:
                env.render()

            action = agent.action_fn(net, observation)

            observation, reward, done, _ = env.step(action)
            total_reward += reward

            if reward > 0:
                negative_rewards_steps = 0
            else:
                negative_rewards_steps += 1

            if negative_rewards_steps > max_negative_rewards_steps:
                total_reward -= (episode_length - t) * 0.1
                done = True

            if done:
                if verbose:
                    print(f"Done after {t} steps")
                break

        if verbose:
            print('Obtained reward:', total_reward)

        rewards.append(total_reward)

    return np.mean(rewards)

def net_fitness(vec, agent):
    x = agent.vec_to_net(vec)
    render = (rnd.random() < 0.01)
    return test_network(x, agent, render=render, max_negative_rewards_steps=100)

def do_experiment(agent, repeat_count=1, verbose=False):
    max_fitness = []
    best_people = []
    for i in range(repeat_count):
        if verbose:
            print(f"Starting experiment {i+1}/{repeat_count}")

        x, y = EvoAlg.GenAlg(lambda vec: net_fitness(vec, agent), L=myNN.net_size(agent.nn_arch), PopSize=25, MaxGen=200, u=0.05, verbose=verbose)
        # x, y = EvoAlg.HillClimber(lambda vec: net_fitness(vec, agent), L=myNN.net_size(agent.nn_arch), MaxGen=200, u=0.05, verbose=verbose)
        max_fitness.append(x)
        best_people.append(y)
    return max_fitness, best_people

def create_agent():
    # input_transformation = CarRacingTools.CarBoxTransformation()
    input_transformation = CarRacingTools.DownsamplingTransformation(4)

    # output_transformation = CarRacingTools.ContinuousActionTransformation()
    output_transformation = CarRacingTools.DiscreteActionTransformation()

    hidden_layer_size = 50
    logsig_lambda = 0.3
    weight_coef = 5

    return CarRacingTools.CarRacingAgentArchitecture(
        input_transformation,
        output_transformation,
        hidden_layer_size,
        weight_coef,
        logsig_lambda
    )

def plot_fitness(max_fitness, show=True):
    x = np.arange(len(max_fitness[0]))
    for fr in max_fitness:
        plt.plot(x, fr)
    plt.title("Vývoj fitness v 5 behoch")
    plt.xlabel("Generácia")
    plt.ylabel("Fitness")
    # plt.ylim(0,1)
    if show:
        plt.show()

def main():
    agent = create_agent()
    max_fitness, best_people = do_experiment(agent, verbose=True)
    plot_fitness(max_fitness, True)
    

    nej = -1000
    nej_pr = 0
    for i in range(len(max_fitness)):
        if max_fitness[i][len(max_fitness) - 1] > nej:
            nej_pr = i
            nej = max_fitness[i][len(max_fitness) - 1]
    net = agent.vec_to_net(best_people[nej_pr][len(max_fitness[nej_pr]) - 1])
    
    print("Final evaluation")
    for _ in range(10):
        reward = test_network(net, agent, render=True, episode_length=25000)
        print(reward)

    print(net)
    myNN.net_to_file(net, agent.nn_arch, "net.txt")

    net, arch = myNN.file_to_net("net.txt")
    print(net, arch)

main()