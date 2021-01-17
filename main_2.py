"""
This file has the driver code to run the GA for feature selection and feed it to ML Model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

import genetic_algo as ga
import model as M
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Pickling the model
def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    # print(result)
    return loaded_model

"""
happy_path = "./happy.csv"
not_happy_path = "./nothappy.csv"

happydf = pd.read_csv(happy_path).sample(frac = 1)
happydf.drop(happydf.iloc[:, 696:714], inplace=True, axis=1)
happydf.drop(happydf.iloc[:, 0:5], inplace=True, axis=1)
happydf_y = happydf["Label"]
happydf.drop(columns=['Label'], inplace=True, axis=1)

nothappydf = pd.read_csv(not_happy_path).sample(frac = 1)
nothappydf.drop(nothappydf.iloc[:, 696:714], inplace=True, axis=1)
nothappydf.drop(nothappydf.iloc[:, 0:5], inplace=True, axis=1)
N = len(nothappydf.columns)
nothappydf_y = nothappydf["Label"]
nothappydf.drop(columns=['Label'], inplace=True, axis=1)

happytempdf = pd.read_csv("./happy_temp.csv").sample(frac = 1)
happytempdf.drop(happytempdf.iloc[:, 696:714], inplace=True, axis=1)
happytempdf.drop(happytempdf.iloc[:, 0:5], inplace=True, axis=1)
N = len(happytempdf.columns)
happy_temp_y = happytempdf["Label"]
happytempdf.drop(columns=['Label'], inplace=True, axis=1)
"""

path = "dataset.csv"
df = pd.read_csv(path).sample(frac=1)
df.drop(df.iloc[:, 696:714], inplace=True, axis=1)
df.drop(df.iloc[:, 0:5], inplace=True, axis=1)
N = len(df.columns)
label = df["Label"]
df.drop(columns=['Label'], inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20, random_state=0)

print("X_train =", str(X_train.shape))
print("X_test =", str(X_test.shape))
print("y_train =", str(y_train.shape))
print("y_test =", str(y_test.shape))

# Genetic Algorithm Hyperparameters:
NUM_GENERATIONS = 15
INIT_POP_SIZE = 25  # higher the better
BEST_K_CHROM = INIT_POP_SIZE // 2
CROSSOVER_RATIO = 1
MUTATION_FACTOR = 0.1
NUM_FEATURES = len(df.columns)
THRESHOLD = 0.05

# Neural Network Hyperparameters
EPOCHS = 250

if __name__ == "__main__":
    
    """
    try:
        fit_chrom = np.load("./svm_chrom.npy")
        classifier = load_model("./finalized_svm.sav")
        print("Happy images\n")
        temp = np.where(fit_chrom, True, False)
        result = classifier.score(happytempdf.iloc[:, temp], happy_temp_y)
        print(result)

        y_pred = classifier.predict(happydf.iloc[:, temp])
        print(y_pred)
        y_pred = (y_pred > 0.5)
        print(y_pred)

        print("Un-Happy images\n")
        y_pred = classifier.predict(nothappydf.iloc[:, temp])
        print(y_pred)
        y_pred = (y_pred > 0.5)
        print(y_pred)

    except:
        print(e)
    """
    
    accuracy_before_GA = M.classify_before_GA_SVM(
        X_train, X_test, y_train, y_test)
    print("ACCURACY FOR MODEL WITHOUT GA =", accuracy_before_GA)

    initial_population = ga.initialize_population(INIT_POP_SIZE, NUM_FEATURES)

    for i in range(NUM_GENERATIONS):
        print("----GENERATION #" + str(i+1) + "----")
        fitness_scores, initial_population = ga.compute_fitness_score(
            initial_population, X_train, X_test, y_train, y_test)
        improved_population = ga.rank_selection(
            initial_population, fitness_scores, len(initial_population)//2)
        # improved_population = ga.tournament_selection(initial_population, fitness_scores, num_parents = 20, tournament_size = 4)
        # improved_population = ga.rank_selection(initial_population, fitness_scores, BEST_K_CHROM)
        # cross_overed_pop = ga.uniform_crossover(improved_population)
        cross_overed_pop = ga.k_point_crossover(
            improved_population, CROSSOVER_RATIO)
        mutated_pop = ga.bit_swap_mutation(cross_overed_pop, MUTATION_FACTOR)
        # new_generation = ga.weak_parents_update(
        #     initial_population, fitness_scores, mutated_pop)
        new_generation = ga.generational_update(
            initial_population, mutated_pop, fitness_scores)
        initial_population = new_generation

    print("----FINAL CALCULATIONS----")
    fitness_scores, initial_population = ga.compute_fitness_score(
        initial_population, X_train, X_test, y_train, y_test)
    indices = np.argsort(fitness_scores)
    updated_pop = [initial_population[i] for i in indices][::-1]

    classifier = SVC(C=10, gamma=0.01, kernel="rbf")
    temp = np.where(updated_pop[0], True, False)
    classifier.fit(X_train.iloc[:, temp], y_train)
    y_pred = classifier.predict(X_test.iloc[:, temp])
    print("ACCURACY OF MODEL AFTER GA =", accuracy_score(y_true=y_test, y_pred=y_pred))

    """
    accuracy_after_GA = M.fitness_score_SVM(
        updated_pop[0], X_train, X_test, y_train, y_test, save = False)
    print("ACCURACY OF MODEL AFTER GA =", accuracy_after_GA)

    fit_chrom = np.load("./svm_chrom.npy")
    classifier = load_model("./finalized_svm.sav")
    
    print("Happy images\n")
    temp = np.where(updated_pop[0], True, False)
    # result = classifier.score(happytempdf.iloc[:, temp], happy_temp_y)
    # print(result)

    y_pred = classifier.predict(happydf.iloc[:, temp])
    print(y_pred)
    y_pred = (y_pred > 0.5)
    print(y_pred)
    
    print("Un-Happy images\n")
    y_pred = classifier.predict(nothappydf.iloc[:, temp])
    print(y_pred)
    y_pred = (y_pred > 0.5)
    print(y_pred)

    np.save("./svm_chrom.npy", updated_pop[0])
    """