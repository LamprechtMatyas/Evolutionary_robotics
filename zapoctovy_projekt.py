# requires OpenAI gym; it can be intsalled using pip or conda
#      conda install -c conda-forge gym
import gym
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

import EvoAlg
import myNN
import CarRacingTools
import gc

import car_racing_environment
env = gym.make("CarRacingSoftFS1-v0")
env.seed(420) # Vyhodnocení


def test_network(net, agent, render=True, episodes=1, verbose=False, episode_length=1000, max_negative_rewards_steps=None, seed=None):
    if max_negative_rewards_steps is None:
        max_negative_rewards_steps = episode_length

    rewards = []
    for episode in range(episodes):
        negative_rewards_steps = 0
        total_reward = 0

        if seed is not None:
            env.seed(seed)

        agent.reset()
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
    render = (rnd.random() < 0.01) or True
    return test_network(x, agent, render=render, max_negative_rewards_steps=100, seed=42)

def gen_alg_callback(gen, max_gen, best_vec, fitness, file, agent):
    print(f"Generation {gen + 1}/{max_gen}: {fitness}")
    net = agent.vec_to_net(best_vec)
    myNN.net_to_file(net, agent.nn_arch, file.format(gen))

def do_experiment(agent, repeat_count=1, max_gen=20, verbose=False, save_path_prefix="nets/net_"):
    fitness_fn = lambda vec: net_fitness(vec, agent)

    max_fitness = []
    best_people = []
    for i in range(repeat_count):
        on_epoch_end_callback = lambda gen, max_gen, best_vec, fitness: gen_alg_callback(gen, max_gen, best_vec, fitness, save_path_prefix + str(i+1) + "_{}.txt", agent) if verbose else None
        if verbose:
            print(f"Starting experiment {i+1}/{repeat_count}")

        x, y = EvoAlg.GenAlg(fitness_fn, L=myNN.net_size(agent.nn_arch), PopSize=15, MaxGen=max_gen, u=0.05, t=0.8, callback=on_epoch_end_callback)
        # x, y = EvoAlg.HillClimber(lambda vec: net_fitness(vec, agent), L=myNN.net_size(agent.nn_arch), MaxGen=200, u=0.05, verbose=verbose)
        max_fitness.append(x)
        best_people.append(y)
    return max_fitness, best_people

def create_agent(input_format, output_format, input_timesteps=None, downsampling_stride=None):
    assert input_format in ["carbox", "downsampling", "sensors"]
    assert input_format != "downsampling" or (type(downsampling_stride) is int and downsampling_stride >= 1)
    assert input_timesteps is None or (type(input_timesteps) is int and input_timesteps >= 1)
    assert output_format in ["continuous", "discrete"]

    if input_format == "carbox":
        input_transformation = CarRacingTools.CarBoxTransformation()
    elif input_format == "downsampling":
        input_transformation = CarRacingTools.DownsamplingTransformation(stride=downsampling_stride)
    elif input_format == "sensors":
        input_transformation = CarRacingTools.SensorDistancesTransformation()

    if input_timesteps is not None and input_timesteps > 1:
        input_transformation = CarRacingTools.TimeTransformationWrapper(input_transformation, steps_count=input_timesteps)

    if output_format == "continuous":
        output_transformation = CarRacingTools.ContinuousActionTransformation()
    elif output_format == "discrete":
        output_transformation = CarRacingTools.DiscreteActionTransformation()

    hidden_layer_size = 50
    logsig_lambda = 1
    weight_coef = 1

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

def do_training(agent, save_path_prefix):
    max_fitness, best_people = do_experiment(agent, repeat_count=1, max_gen=20, verbose=True, save_path_prefix=save_path_prefix)
    plot_fitness(max_fitness, True)
    
    nej = -1000
    nej_pr = 0
    for i in range(len(max_fitness)):
        if max_fitness[i][len(max_fitness) - 1] > nej:
            nej_pr = i
            nej = max_fitness[i][len(max_fitness) - 1]
    return agent.vec_to_net(best_people[nej_pr][len(max_fitness[nej_pr]) - 1])

def do_final_evaluation(agent, net, seed):
    print("Final evaluation")
    for _ in range(10):
        reward = test_network(net, agent, render=True, episode_length=25000, seed=seed)
        print(reward)

def main(load_path=None, input_timesteps=None, seed=None):
    input_format = "sensors"
    output_format = "continuous"
    downsampling_stride = None

    save_path_prefix = f"nets/net_{input_format}_{output_format}_{input_timesteps}_{downsampling_stride}_e_"

    agent = create_agent(
        input_format,
        output_format,
        input_timesteps,
        downsampling_stride
    )
    
    if load_path is None:
        net = do_training(agent, save_path_prefix)
    else:
        net, _ = myNN.file_to_net(load_path) # assuming same agent architecture

    do_final_evaluation(agent, net, seed=seed)

load_path = None 
if False:
    load_path = "nets/net_sensors_continuous_20_None_e_1_19.txt"
    seed = 42 if False else None
    main(load_path, 20, seed)
else:
    load_path = "nets/net_sensors_continuous_4_None_e_1_19.txt"
    seed = 42 if False else None
    main(load_path, 4, seed)
