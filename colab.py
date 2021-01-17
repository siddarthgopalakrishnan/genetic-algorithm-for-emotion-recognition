import model as M
import numpy as np
import pandas as pd
import random
import time

import tensorflow as tf

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


path = "./dataset.csv"
df = pd.read_csv(path).sample(frac=1)
df.drop(df.iloc[:, 696:714], inplace=True, axis=1)
df.drop(df.iloc[:, 0:5], inplace=True, axis=1)
N = len(df.columns)
label = df["Label"]
df.drop(columns=['Label'], inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20, random_state=0)

inital_num_features = len(df.columns)

# CSV FILE
# ng, ps, ff, sf, cf, k, mf, mr, uf, beforeGA, After GA
# 5, 50, lr, roulette, k_point, 1, bit_flip, 0.01, gen, 0.88, 0.92
# 5, 50, lr, roulette, k_point, 1, bit_flip, 0.01, weak_p, 0.86, 0.89

# NPY FILE
# Best Chromosome

fitness_functions = ["svm", "lr", "nn"]

num_generations = [10, 20, 30]
pop_sizes = [50, 100, 200, 250]

selection_functions = ["roulette", "rank", "tournament"]

crossover_functions = ["k_point", "uniform"]
k_values = np.arange(1, 11)

mutations_functions = ["bit_flip", "bit_swap"]
mutations_factors = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]

update_functions = ["gen", "weak_p"]

accuracy_before_GA = M.classify_before_GA_NN(X_train, X_test, y_train, y_test)

index = 0

# 3*4*3*2*1*2*1*2 = 284 possible combinations per person

# Grid search on the parameters
for pop_size in pop_sizes:
    initial_population = initialize_population(pop_size, inital_num_features)
    for num in num_generations:
        for ff in fitness_functions:
            for sf in selection_functions:
                for cf in crossover_functions:
                    for mf in mutations_functions:
                        for uf in update_functions:
                            for i in range(num):
                                fitness_scores, initial_population = compute_fitness_score(
                                    initial_population, X_train, X_test, y_train, y_test)

                                if sf == "roulette":
                                    improved_population = roulette_wheel_selection(
                                        initial_population, fitness_scores)
                                elif sf == "rank":
                                    improved_population = rank_selection(initial_population, fitness_scores, len(
                                        initial_population)//2)
                                elif sf == "tournament":
                                    improved_population = tournament_selection(initial_population, fitness_scores, num_parents=len(
                                        initial_population), tournament_size=4)

                                if cf == "k_point":
                                    k_point_crossover(
                                        improved_population, 1)
                                elif cf == "uniform":
                                    k_point_crossover(
                                        improved_population)

                                if mf == "bit_flip":
                                    bit_flip_mutation(
                                        improved_population, 0.1)
                                elif mf == "bit_swap":
                                    bit_flip_mutation(
                                        improved_population, 0.1)

                                if uf == "gen":
                                    new_generation = generational_update(
                                        initial_population, improved_population)
                                elif uf == "weak_p":
                                    new_generation = weak_parents_update(
                                        initial_population, fitness_scores, improved_population)

                                initial_population = new_generation

                                fitness_scores, initial_population = ga.compute_fitness_score(
                                    initial_population, X_train, X_test, y_train, y_test)
                                indices = np.argsort(fitness_scores)
                                updated_pop = [
                                    initial_population[i] for i in indices][::-1]

                                accuracy_after_GA = M.fitness_score_NN(
                                    updated_pop[0], X_train, X_test, y_train, y_test)
