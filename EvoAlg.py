import numpy as np
import random


def MatrixCreate(a, b):
    return 2 * np.random.random_sample((a, b)) - 1


def Fitness1(v):
    return v.mean()


def Fitness2(v):
    z = np.sqrt(np.sum(v ** 2))
    return (2 - z) * (np.cos(10*z))


def MatrixPerturb(v, u):
    new_v = v.copy()
    for i in range(len(v)):
        for j in range(len(v[i])):
            if random.uniform(0, 1) <= u:
                new_v[i][j] = random.uniform(-1, 1)
    return new_v


def HillClimber(FitFnc, MaxGen=5000, u=0.05, L=50, verbose=False):
    # FitFnc is the maximazed fitness function
    # gradient optimizer;
    # MaxGen is the number of generation
    # u is the probability if mutation
    # L is the length of a vector
    fitnesses = []
    Genes = []
    parent = MatrixCreate(1,L)
    parentFitness = FitFnc(parent)
    for currentGeneration in range(MaxGen):
        fitnesses.append(parentFitness)
        Genes.append(parent[0])
        child = MatrixPerturb(parent,u)
        childFitness = FitFnc(child)
        if (childFitness > parentFitness):
            parent = child
            parentFitness = childFitness
            if verbose:
                print(f"Generation\t{currentGeneration+1}\tNew best\t{parentFitness}")
    return fitnesses, Genes


def ComputeFittness(FitFnc, Pop):
    fitness = []
    for jedinec in Pop:
        fitness.append(FitFnc(jedinec))
    return np.array(fitness).T


def TourSel(Pop, F, t):
    new_pop = []
    for i in range(len(Pop) - 1):
        samples = random.sample(range(len(Pop)), 2)
        if (F[samples[0]] > F[samples[1]]) == (random.uniform(0, 1) <= t):
            new_pop.append(Pop[samples[0]])
        else:
            new_pop.append(Pop[samples[1]])
    return new_pop


def Mutate(v, u, m):
    for i in range(len(v)):
        for j in range(len(v[i])):
            if random.uniform(0, 1) <= u:
                v[i][j] += m*np.random.standard_normal()
    return v


def Crossover(i2, c):
    if random.uniform(0, 1) <= c:
        rez = random.randint(0, len(i2[0]) - 1)
        o1 = np.append(i2[0][:rez], i2[1][rez:])
        o2 = np.append(i2[1][:rez], i2[0][rez:])
        return np.array([o1, o2])
    return i2


def GenAlg(FitFnc, MaxGen=333, PopSize=15, L=50, t=0.98, u=0.1, m=0.9, c=0.7, verbose=False):
    pop = MatrixCreate(PopSize, L)
    max_fitness = []   # sem si budu ukládat hodnoty nejlepší fitness
    Genes = []         # a sem nejlepší jedince
    for gen in range(MaxGen):
        if verbose:
            print(f"Starting generation {gen + 1}/{MaxGen}")
        fitness = ComputeFittness(FitFnc, pop)
        best_jedinec = pop[np.argmax(fitness)]
        max_fit = fitness[np.argmax(fitness)]
        if verbose:
            print(f"\tMax fitness: {max_fit}")
        max_fitness.append(max_fit)
        Genes.append(best_jedinec)
        pop = TourSel(pop, fitness, t)
        new_pop = []
        for i in range(len(pop) // 2):
            i2 = pop[(2*i):(2*i+2)]
            new_pop.extend(Crossover(i2,c))
        new_pop = np.array(new_pop)
        pop = Mutate(new_pop, u, m)
        pop = np.vstack([best_jedinec, pop])
        pop = np.clip(pop, -1, 1)               # dostanu hodnoty do rozmezí <-1,1>
    return max_fitness, Genes




