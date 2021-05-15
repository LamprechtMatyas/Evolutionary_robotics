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


def test_network(net, agent, render=True, episodes=1, verbose=False, episode_length=1000, max_negative_rewards_steps=None, seed=None):
    if max_negative_rewards_steps is None:
        max_negative_rewards_steps = episode_length

    rewards = []
    for episode in range(episodes):
        negative_rewards_steps = 0
        total_reward = 0

        if seed is not None:
            env.seed(seed)
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
    render = (rnd.random() < 0)
    return test_network(x, agent, render=render, max_negative_rewards_steps=100, seed=42)

def gen_alg_callback(gen, max_gen, best_vec, fitness, agent):
    print(f"Generation {gen + 1}/{max_gen}: {fitness}")
    net = agent.vec_to_net(best_vec)
    myNN.net_to_file(net, agent.nn_arch, f"net_{gen}.txt")

def do_experiment(agent, repeat_count=1, verbose=False):
    on_epoch_end_callback = lambda gen, max_gen, best_vec, fitness: gen_alg_callback(gen, max_gen, best_vec, fitness, agent) if verbose else None
    fitness_fn = lambda vec: net_fitness(vec, agent)

    max_fitness = []
    best_people = []
    for i in range(repeat_count):
        if verbose:
            print(f"Starting experiment {i+1}/{repeat_count}")

        x, y = EvoAlg.GenAlg(fitness_fn, L=myNN.net_size(agent.nn_arch), PopSize=51, MaxGen=60, u=0.05, t=0.8, callback=on_epoch_end_callback)
        # x, y = EvoAlg.HillClimber(lambda vec: net_fitness(vec, agent), L=myNN.net_size(agent.nn_arch), MaxGen=200, u=0.05, verbose=verbose)
        max_fitness.append(x)
        best_people.append(y)
    return max_fitness, best_people

def create_agent():
    # input_transformation = CarRacingTools.CarBoxTransformation()
    input_transformation = CarRacingTools.DownsamplingTransformation(stride=8)

    input_transformation = CarRacingTools.TimeTransformationWrapper(input_transformation, steps_count=4)

    # output_transformation = CarRacingTools.ContinuousActionTransformation()
    output_transformation = CarRacingTools.DiscreteActionTransformation()

    hidden_layer_size = 50
    logsig_lambda = 1
    weight_coef = 10

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
    plt.title("Vývoj fitness")
    plt.xlabel("Generácia")
    plt.ylabel("Fitness")
    # plt.ylim(0,1)
    if show:
        plt.show()

def do_training(agent):
    max_fitness, best_people = do_experiment(agent, verbose=True)
    plot_fitness(max_fitness, True)
    

    nej = -1000
    nej_pr = 0
    for i in range(len(max_fitness)):
        if max_fitness[i][len(max_fitness) - 1] > nej:
            nej_pr = i
            nej = max_fitness[i][len(max_fitness) - 1]
    return agent.vec_to_net(best_people[nej_pr][len(max_fitness[nej_pr]) - 1])

def do_final_evaluation(agent, net):
    print("Final evaluation")
    for _ in range(10):
        reward = test_network(net, agent, render=True, episode_length=25000)
        print(reward)

def main(load_path=None):
    agent = create_agent()
    
    if load_path is None:
        net = do_training(agent)
    else:
        net, _ = myNN.file_to_net(load_path) # assuming same agent architecture
    
    do_final_evaluation(agent, net)

# load_path = None 
load_path = "net_1.txt"
main(load_path)
