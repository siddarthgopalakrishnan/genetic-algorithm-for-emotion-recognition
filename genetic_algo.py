"""
This file has all the different functions used in the Genetic Algorithm.
"""

"""
Hyperparameters:
	Selection : Roulette, Rank based, Steady State
	Crossover: Single point, k = {2,4,6} point, Uniform Crossover
	Mutation: Bit Flip, Bit Swap(swap 2 random positions)
"""

import numpy as np
import pandas as pd
import random
import model as M

def initialize_population(population_size, inital_num_features):
	"""
	Initailizes a generation of `population_size` number of chromosomes 
	with each chromosome being `inital_num_features` long.

	Parameters:
	population_size: int - Size of the inital population
	inital_num_features: int - Number of random features to be considered.

	Returns:
	init_population: list of chromosomes
	"""
	print("Starting init pop process")
	init_population = []
	for _i in range(population_size):
		feature_choice = np.random.randint(2, size=inital_num_features)
		init_population.append(feature_choice)
		# if i==0:
			# print("At iteration 0:", str(init_population))
	return init_population

def compute_fitness_score(init_pop, X_train, X_test, y_train, y_test):
	"""
	Compute the fitness score of current generation using a model.

	Parameters:
	init_pop: list of chromosomes which denotes the population
	X_train: train dataset of openface features as a numpy array
	X_test: test dataset of openface features as a numpy array
	y_train: train labels of emotions as numpy array
	y_test: test labels of emotions as numpy array
	Returns:

	"""
	# print("Starting compute fitness process")
	fitness_scores = np.random.random(len(init_pop))
	# fitness_scores = np.random.random(len(init_pop))
	for index, chromosome in enumerate(init_pop):
		# print("Chromosome shape =", str(chromosome.shape))
		# print("Chromosome going in =", str(chromosome))
		score = M.fitness_score_SVM(
			chromosome, X_train, X_test, y_train, y_test, save = False)
		print("For Chromosome #" + str(index+1) + ", accuracy = " + str(score))
		fitness_scores[index] = score
	return fitness_scores, init_pop


def roulette_wheel_selection(current_pop, fitness_scores):
	"""
	Roulette Wheel Selection chooses individuals based on a probability value
	which is computed as the proportion of the fitness score of the individual
	in the sum of the fitness scores of all the individuals.

	Parameters:
	current_pop: list of chromosomes which denotes the current population
	fitness_scores: list of chromosomes which denotes the next generation population
	threshold: probability threshold value for selecting an individual

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	print("Starting compute roulette wheel process")
	fitness_scores_sum = np.sum(fitness_scores, axis=0)
	individual_probabilities = fitness_scores * (1 / fitness_scores_sum)
	probs_start = np.zeros(len(individual_probabilities), dtype=np.float) # An array holding the start values of the ranges of probabilities.
	probs_end = np.zeros(len(individual_probabilities), dtype=np.float) # An array holding the end values of the ranges of probabilities.
	curr = 0.0

	for _i in range(len(individual_probabilities)):
		min_probs_idx = np.where(individual_probabilities == np.min(individual_probabilities))[0][0]
		probs_start[min_probs_idx] = curr
		curr = curr + individual_probabilities[min_probs_idx]
		probs_end[min_probs_idx] = curr
		individual_probabilities[min_probs_idx] = 999999999

	#improved_pop = np.empty((len(current_pop), ))
	improved_pop = [None]*len(current_pop)
	for parent_num in range(len(current_pop)):
		rand_prob = np.random.rand()
		for idx in range(len(individual_probabilities)):
			if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
				improved_pop[parent_num] = current_pop[idx]
				break
	print(str(improved_pop))
	return improved_pop


	# mean = np.mean(individual_probabilities)
	# # print("ind prob type =", str(type(individual_probabilities)))
	# print(individual_probabilities)
	# print("Mean: ", str(mean))
	# threshold = min(0.90, mean)
	# improved_pop = [current_pop[i] for i in range(individual_probabilities.shape[0]) if individual_probabilities[i] >= threshold]
	# # improved_pop = current_pop[np.where(individual_probabilities>threshold)]
	# return improved_pop


def rank_selection(current_pop, fitness_scores, k):
	"""
	Roulette Wheel Selection chooses individuals based on a probability value
	which is computed as the proportion of the fitness score of the individual
	in the sum of the fitness scores of all the individuals.

	Parameters:
	current_pop: list of chromosomes which denotes the current population
	fitness_scores: list of chromosomes which denotes the next generation population
	k : The number which denotes the top k values to be selected.

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	pop_filter = np.arange(k, dtype=np.int32)
	indices = np.argsort(fitness_scores)
	indices = indices.tolist()
	improved_pop = [current_pop[i] for i in indices]
	improved_pop = improved_pop[::-1]
	improved_pop = [improved_pop[i] for i in pop_filter]
	return improved_pop


def tournament_selection(current_pop, fitness_score, num_parents=10, tournament_size=8):
	"""
	Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.

	Parameters:
	current_pop: list of chromosomes which denotes the current population
	fitness_scores: The fitness values of the solutions in the current population.
	Returns:
	sel_parents: It returns an array of the selected parents.
	"""
	try:
		sel_parents = list()
		fitness_score_temp = fitness_score
		for _i in range(num_parents):
			rand_ind = random.sample(
				range(0, len(current_pop)), tournament_size)
			fit_list = fitness_score_temp[rand_ind]
			selected_parent_ind = np.where(fit_list == np.max(fit_list))[0][0]
			sel_parents.append(current_pop[rand_ind[selected_parent_ind]])
			fitness_score_temp[selected_parent_ind] = -1
		return sel_parents
	except:
		return current_pop


def k_point_crossover(current_pop, k=1):
	"""
	Parameters:
	current_pop: list of chromosomes which denotes the current population
	k: number of crossover points

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	try:
		cross_points = random.sample(range(1, len(current_pop[0])), k)
		cross_points = sorted(cross_points)
		mask = np.ones(len(current_pop[0]), dtype=np.bool)
		for i in range(0, len(cross_points), 2):
			if (i+1) < len(cross_points):
				mask[cross_points[i]:cross_points[i+1]] = False
			else:
				mask[cross_points[i]:len(current_pop[0])] = False
		improved_pop = list()
		for i in range(0, len(current_pop), 2):
			child1 = np.where(mask, current_pop[i], current_pop[i+1])
			child2 = np.where(mask, current_pop[i+1], current_pop[i])
			improved_pop.append(child1)
			improved_pop.append(child2)
		return improved_pop
	except:
		return current_pop


def uniform_crossover(current_pop):
	"""
	Parameters:
	current_pop: list of chromosomes which denotes the current population

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	try:
		improved_pop = list()
		n_feat = len(current_pop[0])
		for i in range(0, len(current_pop), 2):
			mask = np.ones(n_feat, dtype=np.bool)
			mask[:int(0.5*len(current_pop))] = False
			np.random.shuffle(mask)
			child1 = np.where(mask, current_pop[i], current_pop[i+1])
			child2 = np.where(mask, current_pop[i+1], current_pop[i])
			improved_pop.append(child1)
			improved_pop.append(child2)
		return improved_pop
	except:
		return current_pop


def bit_swap_mutation(current_pop, mutation_factor):
	"""
	Eg: 001`1`010`0`10 -> 001`0`010`1`10
	Swap bits at random locations. Flipping based on mutation factor.

	Parameters:
	current_pop: list of chromosomes which denotes the current population

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	try:
		swap_bits = random.sample(range(1, len(current_pop[0])), int(
			mutation_factor*len(current_pop[0])))
		lswap = len(swap_bits)
		for i in range(len(current_pop)):
			for j in range(int(lswap/2)):
				current_pop[i][j], current_pop[i][lswap-j] = current_pop[i][lswap - j], current_pop[i][j]
		return current_pop
	except:
		return current_pop


def bit_flip_mutation(current_pop, mutation_factor):
	"""
	WORKING
	Eg: 001`1`010010 -> 001`0`010010
	Flip bits at random locations. Flipping based on mutation factor.

	Parameters:
	current_pop: list of chromosomes which denotes the current population

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	try:
		for i in range(len(current_pop)):
			mask = np.ones(len(current_pop[0]), dtype=np.bool)
			mask[:int(mutation_factor*len(current_pop))] = False
			np.random.shuffle(mask)
			current_pop[i] = np.where(mask, current_pop[i], 1-current_pop[i])
		return current_pop
	except:
		return current_pop


def generational_update(current_pop, improved_pop, fitness_scores):
	"""
	WORKING
	Parameters:
	current_pop: list of chromosomes which denotes the current population
	improved_pop: list of chromosomes which denotes the improved population

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	if(len(improved_pop)>=len(current_pop)):
		return improved_pop
	pop_filter = np.arange(len(current_pop) - len(improved_pop), dtype=np.int32)
	indices = np.argsort(fitness_scores)
	indices = indices.tolist()
	improved_pop_new = [current_pop[i] for i in indices]
	improved_pop_new = improved_pop_new[::-1]
	improved_pop_new = [improved_pop_new[i] for i in pop_filter]
	improved_pop_new.extend(improved_pop)
	return improved_pop_new


def weak_parents_update(current_pop, fitness_scores, improved_pop):
	"""
	WORKING
	Replaces weakest parents of the population with the improved population

	Parameters:
	current_pop: list of chromosomes which denotes the current population
	fitness_scores: list of chromosomes which denotes the next generation population
	improved_pop: list of chromosomes which denotes the improved population

	Returns:
	improved_pop: list of chromosomes which denotes the improved population
	"""
	pop_filter = np.arange(len(current_pop)//2)
	indices = np.argsort(fitness_scores)
	updated_pop = [current_pop[i] for i in indices]
	updated_pop = updated_pop[::-1]
	updated_pop = [updated_pop[i] for i in pop_filter]
	if len(improved_pop) <= len(current_pop)//2:
		updated_pop.extend(improved_pop)
		return updated_pop
	for _i in range(len(current_pop)//2):
		#rand_int = np.random.randint(0, len(improved_pop))
		updated_pop.append(improved_pop[_i])
	return updated_pop
