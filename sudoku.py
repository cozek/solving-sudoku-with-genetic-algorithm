#!/usr/bin/env python3

import numpy as np
import random
from pprint import pprint
import gc
from collections import deque
import time
from tqdm import tqdm

_problem = np.array(
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
)

_solution = np.array(
    [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]
)


class SudokuSolver:
    """Sudoku Solver Class"""

    def __init__(self):
        """Constructor"""
        pass

    def set_problem(self, problem):
        """Defines the problem to be solved"""

        assert problem.shape == (9, 9)
        self.problem = problem

    def print_problem(self):
        """ Prints the problem under consideration"""

        print(self.problem)

    def check(self, solution, i_, j_, flag_matrix):

        conflicts = 0
        # flag_matrix = np.full((9, 9), False)

        for p in range(9):
            # check row
            if p == j_:
                continue
            if solution[i_, p] == solution[i_, j_] and flag_matrix[i_, p] != True:
                # print(i_,p, 'pivot',i_,j_)
                conflicts += 1
                flag_matrix[i_, p] = True

        for p in range(9):
            if p == i_:
                continue
            if (
                solution[p, j_] == solution[i_, j_]
                and flag_matrix[p, j_] != True
                and p != i_
            ):
                # print(p,j_,'pivot',i_,j_)
                conflicts += 1
                flag_matrix[p, j_] = True

        i_min = int(i_ / 3) * 3
        j_min = int(j_ / 3) * 3

        # check subspace
        for p in range(i_min, i_min + 3):
            for q in range(j_min, j_min + 3):
                if flag_matrix[p, q] == True:
                    continue
                if (p, q) == (i_, j_):
                    continue

                if solution[p, q] == solution[i_, j_]:
                    conflicts += 1
                    # print(p,q,'pivot',i_,j_)
                    flag_matrix[p, q] = True

        return conflicts

    def check_solution_v2(self, solution):
        """Checks the solution for conflicts"""

        flag_matrix = np.full((9, 9), False)

        conflicts = 0

        for i in range(9):
            for j in range(9):
                if solution[i, j] != 0:
                    conflicts += self.check(solution, i, j, flag_matrix)

        # print(conflicts)

        return conflicts

    def gen_rand_solution(self, problem):
        new_solution = problem.copy()
        possible_values = {}
        for i in range(9):
            for j in range(9):
                if problem[i, j] != 0:
                    continue
                sols = []
                for num in range(1, 10):
                    prob = problem.copy()
                    prob[i, j] = num
                    flag_matrix = np.full((9, 9), False)
                    ch = self.check(prob, i, j, flag_matrix)

                    if ch == 0:
                        sols.append(num)
                possible_values[str(i) + str(j)] = sols

                new_solution[i, j] = random.sample(sols, 1)[0]

                del prob
                del flag_matrix

        gc.collect()
        return new_solution

    def gen_rand_sol_tsp(self, problem):
        new_solution = problem.copy()
        possible_values = {}
        nums = {i for i in range(1, 10)}
        for i in range(9):
            row = set(problem[i, :])
            allowed = list(nums - row)
            random.shuffle(allowed)
            idxes = np.where(problem[i, :] == 0)[0]
            # print(allowed, idxes, problem[i,:])
            for idx in idxes:
                new_solution[i, idx] = allowed.pop()
        gc.collect()
        return new_solution

    def gen_random_solutions(self, gen_size=None):
        """Creates n random solutions"""
        if gen_size == None:
            gen_size = 100

        self.population = deque()

        for i in range(gen_size):
            self.population.append(self.gen_rand_sol_tsp(self.problem))

    def gen_pop_fitness(self):
        """Calculates the fitness of the population"""

        self.pop_fitness = deque()

        for sol in self.population:
            self.pop_fitness.append(self.check_solution_v2(sol))

    def softmax(self, vector):

        # Calculate e^x for each x in your vector where e is Euler's
        # number (approximately 2.718)
        vector = np.array(vector) * -1
        exponentVector = np.exp(vector)
        # print(exponentVector)
        # Add up the all the exponentials
        sumOfExponents = np.sum(exponentVector)

        # Divide every exponent by the sum of all exponents
        softmax_vector = exponentVector / sumOfExponents

        # print(np.sum(softmax_vector))
        return softmax_vector

    def selection_decreasing_choice(self, num_selection=None):
        """selection pool decreases"""

        if num_selection == None:
            num_selection = 10

        softmax_fitness = self.softmax(self.pop_fitness)

        mating_pairs = deque()

        temp_fitness_vals = self.pop_fitness.copy()
        temp_fitness_indexes = list(range(len(temp_fitness_vals)))

        while len(mating_pairs) != num_selection:
            pool_of_two = deque()

            while len(pool_of_two) != 2:

                current_pop = [temp_fitness_vals[i] for i in temp_fitness_indexes]
                current_fitness_probs = self.softmax(current_pop)

                choice = np.random.choice(temp_fitness_indexes, p=current_fitness_probs)
                # print(choice)
                temp_fitness_indexes.pop(temp_fitness_indexes.index(choice))

                pool_of_two.append(choice)

            mating_pairs.append(pool_of_two)

        self.mating_pairs = mating_pairs
        # print(mating_pairs)

    def selection_contant_choice(self, num_selection=None):
        """selection pool does not decrease"""
        if num_selection == None:
            num_selection = 10

        softmax_fitness = self.softmax(self.pop_fitness)

        mating_pairs = deque()

        while len(mating_pairs) != num_selection:
            pool_of_two = []
            while len(pool_of_two) != 2:

                choice = np.random.choice(
                    range(len(self.population)), p=self.softmax(self.pop_fitness)
                )
                if choice not in pool_of_two:
                    pool_of_two.append(choice)
                else:
                    continue
            mating_pairs.append(pool_of_two)
        print(mating_pairs)

    def rand_rowswap_crossover(self, num_swaps):
        """randomly swaps the rows between two solutions"""

        new_population = deque()

        for pair in self.mating_pairs:
            parent_one = self.population[pair[0]].copy()
            parent_two = self.population[pair[1]].copy()

            child_one = self.population[pair[0]].copy()
            child_two = self.population[pair[1]].copy()

            rand_rows = random.sample(range(9), num_swaps)

            for row in rand_rows:
                child_one[row, :] = parent_two[row, :]
                child_two[row, :] = parent_one[row, :]

            child_one = self.mutate(child_one)
            child_two = self.mutate(child_two)

            if self.check_solution_v2(child_one) <= self.pop_fitness[pair[0]]:
                new_population.append(child_one)
            else:
                new_population.append(parent_one)

            if self.check_solution_v2(child_two) <= self.pop_fitness[pair[1]]:
                new_population.append(child_two)
            else:
                new_population.append(parent_two)

        self.population = new_population

        gc.collect()

    def mutate(self,solution, chance=None):
        """Generate a random number less than 10 from a pool of 99 elements"""
        if chance == None:
            chance = 10;

        rand = np.random.randint(0,high=100)
        if rand < chance:
            # row = random.sample(range(9),1)
            safe_idxes = np.where(self.problem == 0)
            _idx = random.sample(range(len(safe_idxes[0])),1)
            safe_row_idx = safe_idxes[0][_idx]
            safe_col_idx = safe_idxes[1][_idx]
            solution[safe_row_idx,safe_col_idx] = random.sample(range(1,10),1)

        return solution



def main():

    doku = SudokuSolver()
    doku.set_problem(_problem)
    doku.gen_random_solutions(gen_size=1000)
    max_iter = 200
    _swaps = 8
    num_swaps = [0, 1, 4, 5, 6, 7]
    change_at = [0, 10, 20, 40, 80, 100]

    doku.gen_pop_fitness()

    print("Initial fitness vals: ", sorted(doku.pop_fitness)[:10])
    start_time = time.time()
    for iter in tqdm(range(max_iter)):
        doku.gen_pop_fitness()

        if 0 in doku.pop_fitness:
            print('Found!')
            break

        if iter in change_at:
            _swaps = num_swaps[change_at.index(iter)]
            print(sorted(doku.pop_fitness)[:10])

        doku.selection_decreasing_choice(num_selection=500)
        doku.rand_rowswap_crossover(num_swaps=_swaps)

    print("Time taken: ", time.time() - start_time)

    print("Final fitness vals: ", sorted(doku.pop_fitness)[:10])

    best = doku.population[doku.pop_fitness.index(min(doku.pop_fitness))]
    print('Best Solution',best)
    print('The Actual Solution',_solution)

if __name__ == "__main__":
    main()


