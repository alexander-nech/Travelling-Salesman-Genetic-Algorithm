import random
import math
import numpy
import matplotlib.pyplot as plt


def nameParser(file):
    names = []

    # open the corresponding file that is taken from the parameter and grab all of the lines
    with open(file) as f:
        lines = f.readlines()

    # iterate over each line and store each line (i.e. city name) in the names array and then return the names array
    for i in range(1, len(lines)):
        line = lines[i].strip()
        names.append(line)
    return names

def coordinatesParser(file):
    coordinates = []

    # open the corresponding file that is taken from the parameter and grab all of the lines
    with open(file) as f:
        lines = f.readlines()

    # iterate over each line and for each line convert each value into a float and store the 3 float
    # values in an array. We then append that array into our main array that is later returned
    for i in range(1, len(lines)):
        line = lines[i].strip()

        if line:
            row_data = [float(item.strip()) for item in line.split(" ")]
            coordinates.append(row_data)
    return coordinates


def cityMapping(city_coordiantes, city_names):
    city_data = {}

    # iterate through each city name and assign the corresponding coordinates for the city and then return the
    # dictionary.
    for i in range(len(city_names)):
        city_data[city_names[i]] = city_coordiantes[i]
    return city_data

'''
this function generates a random float between 0 and 1 in a uniform fashion and checks if it is below 0.85 with a 
boolean result. Hence, mimics the probability of 85 percent of a True being returned vs. False.

Note that this function is used to calculate the 85 percent chance of a crossover between parents occurring.
'''
def crossoverProbability():
    return random.random() < 0.85

def mutationProbability():
    return random.random() < 0.30

def uniformCrossoveProbability():
    return random.random() < 0.50


def initialization(start_location, population_size, city_names, starting_bool):
    population = []

    # Loop through the size of the desired population and then generate a path for each
    # chromosome which is randomized. If the start location is taken into consideration,
    # then the start and and city will always be the same
    for i in range(population_size):
        # if starting, then filer out the start location from randomized shuffle
        if starting_bool:
            chromosome = list(filter(lambda city: city != start_location, city_names))
        else:
            chromosome = city_names
        random.shuffle(chromosome)

        # always have start and end the same city
        if starting_bool:
            chromosome.insert(0, start_location)
            chromosome.append(start_location)

        population.append(chromosome)

    return population


def fitnessEvaluation(path, city_data):
    distance = 0

    for i in range(len(path) - 1):
        # get the x, y and z coordinates of the first city
        x_coord_1 = city_data[path[i]][0]
        y_coord_1 = city_data[path[i]][1]
        z_coord_1 = city_data[path[i]][2]

        # get the x, y and z coordinates of the second city
        x_coord_2 = city_data[path[i + 1]][0]
        y_coord_2 = city_data[path[i + 1]][1]
        z_coord_2 = city_data[path[i + 1]][2]

        # determine the euclidean distance between the two cities
        distance += math.sqrt((x_coord_1 - x_coord_2)**2 + (y_coord_1 - y_coord_2)**2 + (z_coord_1 - z_coord_2)**2)

    return 1 / distance


def parentSelectionTournament(population, mating_pool_size, elitism_size, tournament_size):
    selected_to_mate = []
    fps = []

    current_member = 1
    fit_sum = 0

    # loop through the entire population and calculate the sum of all the fitnesses
    for pop in population:
        fit_sum += pop[1]

    for pop in population:
        fps.append(pop[1] / fit_sum)

    for i in range(elitism_size):
        selected_to_mate.append(population[i])
        fps[i] = 0

    indices = []
    while current_member <= mating_pool_size - elitism_size:
        parents = random.sample(list(filter(lambda prob: prob != 0, fps)), k=tournament_size)
        parents.sort(reverse=True)

        parent_idx = fps.index(parents[0])
        indices.append(parent_idx)
        fps[parent_idx] = 0  # set to zero of selected parent so that it doesn't get picked again

        current_member += 1

    # taking the generated list of indices of parents we selected, we then append the individuals to our final array
    # and the return it
    for idx in indices:
        selected_to_mate.append(population[idx])

    return selected_to_mate

def parentSelection(population, mating_pool_size, elitism_size):
    selected_to_mate = []
    fps = []
    cpd = []

    # create random number for selection, where we get the difference of the
    # mating pool size and the elitism size, since we only want to consider the
    # remainder of the selection pool size once the elite individuals are selected
    current_member = 1
    random_sel = numpy.random.uniform(0, 1 / (mating_pool_size - elitism_size))
    fit_sum = 0

    # calculates the sum of all fitnesses of the population
    for pop in population:
        fit_sum += pop[1]

    # Calculate the FPS of the population by dividing each individual fitness by the sum fitness
    for pop in population:
        fps.append(pop[1] / fit_sum)

    # calculate the cumulative probability distribution, where the final item always sums to 1.0
    for idx in range(len(fps)):
        if idx == 0:
            cpd.append(fps[idx])
        else:
            cpd.append(fps[idx] + cpd[-1])

    # utilizing the elitism mechanic, we select the top individuals right away
    for i in range(elitism_size):
        selected_to_mate.append(population[i])

    i = elitism_size
    indices = []
    while current_member <= mating_pool_size - elitism_size:
        while random_sel <= cpd[i]:
            indices.append(i)
            random_sel += (1 / (mating_pool_size - elitism_size))
            current_member += 1
        i += 1

    for idx in indices:
        selected_to_mate.append(population[idx])

    return selected_to_mate


def recombinationGeneral(parent1, parent2):
    # take two crossover points between the range of the two parents lengths, given that this is a
    # two point crossover
    crossover_point1 = random.choice(range(0, len(parent1)))
    crossover_point2 = random.choice(range(0, len(parent1)))

    # we then determine which point is the greater one and smaller one
    min_crossover = min(crossover_point1, crossover_point2)
    max_crossover = max(crossover_point1, crossover_point2)

    # next, between the two points we append the opposing parents elements found between the two
    # crossover points
    offspring1 = []
    offspring2 = []
    for i in range(min_crossover,  max_crossover):
        offspring1.append(parent1[i])
        offspring2.append(parent2[i])

    filtered_list_1 = list(filter(lambda elm: (elm not in offspring1), parent2))
    filtered_list_2 = list(filter(lambda elm: (elm not in offspring2), parent1))
    offspring1 += filtered_list_1
    offspring2 += filtered_list_2

    return offspring1, offspring2


def recombination(parent1, parent2):
    # remove the starting city from first and last elements of both parents
    starting_city = parent1[0]
    parent1.remove(starting_city)
    parent2.remove(starting_city)
    del parent1[-1]
    del parent2[-1]

    crossover_point1 = random.choice(range(0, len(parent1)))
    crossover_point2 = random.choice(range(0, len(parent1)))

    min_crossover = min(crossover_point1, crossover_point2)
    max_crossover = max(crossover_point1, crossover_point2)

    offspring1 = []
    offspring2 = []
    for i in range(min_crossover,  max_crossover):
        offspring1.append(parent1[i])
        offspring2.append(parent2[i])


    filtered_list_1 = list(filter(lambda elm: (elm not in offspring1), parent2))
    filtered_list_2 = list(filter(lambda elm: (elm not in offspring2), parent1))
    offspring1 += filtered_list_1
    offspring2 += filtered_list_2

    # add back the starting city for first and last element to maintain cohesion and return the offspring
    offspring1.append(starting_city)
    offspring1.insert(0, starting_city)
    offspring2.append(starting_city)
    offspring2.insert(0, starting_city)

    return offspring1, offspring2


def uniformCrossover(parent1, parent2, start_bool):
    offspring1 = []
    offspring2 = []

    # check if start location is considered
    if not start_bool:
        # loop through parents and based on the 50 percent chance make the swap
        for i in range(len(parent1)):
            if uniformCrossoveProbability():
                offspring1.append(parent1[i])
                offspring2.append(parent2[i])
            else:
                offspring1.append(parent1[i])
                offspring2.append(parent2[i])

            # filter out any duplicate keys from the offspring
            offspring1 = list(dict.fromkeys(offspring1))
            offspring2 = list(dict.fromkeys(offspring2))

            # and then append the missing cities to the offspring
            filtered_list_1 = list(filter(lambda elm: (elm not in offspring1), parent2))
            filtered_list_2 = list(filter(lambda elm: (elm not in offspring2), parent1))
            offspring1 += filtered_list_1
            offspring2 += filtered_list_2
    else:
        # same as above, but with the start location taken into consideration
        start_location = parent1[0]
        for i in range(1, len(parent1) - 1):
            if uniformCrossoveProbability():
                offspring1.append(parent1[i])
                offspring2.append(parent2[i])
            else:
                offspring1.append(parent2[i])
                offspring2.append(parent1[i])

        offspring1 = list(dict.fromkeys(offspring1))
        offspring2 = list(dict.fromkeys(offspring2))

        filtered_list_1 = list(filter(lambda elm: (elm not in offspring1), parent2))
        filtered_list_2 = list(filter(lambda elm: (elm not in offspring2), parent1))
        offspring1 += filtered_list_1
        offspring2 += filtered_list_2

        offspring1.insert(0, start_location)
        offspring2.insert(0, start_location)
        offspring1.append(start_location)
        offspring2.append(start_location)

    return offspring1, offspring2


def mutation(offspring, start_bool):
    if start_bool:
        swap = random.sample(range(1, len(offspring) - 1), 2)
    else:
        swap = random.sample(range(len(offspring)), 2)

    # take the two cities from randomized indices
    city1 = offspring[swap[0]]
    city2 = offspring[swap[1]]

    # and then swap the city values based on the indices
    offspring[swap[0]] = city2
    offspring[swap[1]] = city1

    return offspring


def mutationScramble(offspring, start_bool):
    # get the two indices we want from a random selection within the defined range
    if start_bool:
        indices = random.sample(range(1, len(offspring) - 1), 2)
    else:
        indices = random.sample(range(len(offspring)), 2)

    # get the min and max index
    min_index = min(indices[0], indices[1])
    max_index = max(indices[0], indices[1])

    # loop between the min and max indices and append the values to the sub list
    sub_list = []
    for i in range(min_index, max_index):
        sub_list.append(offspring[i])

    # we shuffle this list and then return the values to the original offspring array, except in the scrambled
    # format
    random.shuffle(sub_list)
    for idx in range(min_index, max_index):
        offspring[idx] = sub_list[idx - min_index]

    return offspring


def muAndLambdaSelection(population, offspring):
    next_population = []
    mu_lambda_pop_indexing = []

    for i in range(len(population) + len(offspring)):
        if i < len(population):
            pop_index_tuple = (population[i], i)
        else:
            pop_index_tuple = (offspring[i - len(population)], i)

        mu_lambda_pop_indexing.append(pop_index_tuple)

    # We sort the array from highest fitness scores to lowest
    mu_lambda_pop_indexing = sorted(mu_lambda_pop_indexing, key=lambda tuple: tuple[0][1], reverse=True)

    # iterate over the population length and then if the individual is a paren or a child, we append it.
    # the last remaining individuals are not inlcuded in the final next_population array
    for i in range(len(population)):
        pop, pop_index = mu_lambda_pop_indexing[i]

        # figure out of parent or offspring and append
        if pop_index < len(population):
            next_population.append(population[pop_index])
        else:
            next_population.append(offspring[pop_index - len(population)])

    return next_population

def replacementSelection(population, offspring):
    next_population = []
    pop_indexing = []

    # Index the original population
    for i in range(len(population)):
        pop_indexing.append((population[i], i))

    # we sort the original population from greatest to lowest fitness, given that we have the indexing
    # we know where to find the pop individual in the original array
    pop_indexing = sorted(pop_indexing, key=lambda tuple: tuple[0][1], reverse=True)
    # delete the lowest fitness parents that will be replaced by the offspring
    del pop_indexing[len(population) - len(offspring): len(population)]

    # iterate over the length of the original population length
    for j in range(len(pop_indexing) + len(offspring)):
        # we determine if the individual is the parent or offspring in order to
        # append accurately
        if j < len(pop_indexing):
            pop, pop_index = pop_indexing[j]
            next_population.append(population[pop_index])
        else:
            next_population.append(offspring[j - len(pop_indexing)])

    return next_population

'''
This function plots out the map in 3D of all of the cities and it also shows the path taken to travel the globe
of each marked city in the most efficient way possible that the algorithm was able to determine
'''
def plotMap(solution, city_data, start_boolean):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    fig.suptitle('Best Route Across the World', fontsize=18)

    # if we consider the start location, we will have the starting and ending city arleady determined. If we don't
    # we want to make sure our last city is the same as our starting city to have a complete trip around the world
    if not start_boolean:
        solution.append(solution[0])

    # this isolates the x, y and z values into their own arrays
    x_vals = [city_data[city][0] for city in solution]
    y_vals = [city_data[city][1] for city in solution]
    z_vals = [city_data[city][2] for city in solution]

    ax.scatter(x_vals, y_vals, z_vals, color='r')
    ax.plot(x_vals, y_vals, z_vals, color='b')

    # Here we go through each point in the graph and label the cities associated with each
    # point
    for i in range(len(solution)):
        ax.text(x_vals[i], y_vals[i], z_vals[i], '%s' % str(solution[i]))

    plt.show()

def plotSolutionEfficacy(plot_efficacy):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1,1,1)

    fig.suptitle('Efficacy of Results', fontsize=18)

    # x values represent the generation number and the y values represent the sum distance of the best path
    x_vals = [data[0] for data in plot_efficacy]
    y_vals = [data[1] for data in plot_efficacy]

    ax.plot(x_vals, y_vals, color='b')

    plt.fill_between(x_vals, y_vals, y_vals[-1], color='c')

    plt.xlabel("Generation")
    plt.ylabel("Total Distance of Path")

    plt.show()


if __name__ == '__main__':
    # initialize all the parameter values that are taken into consideration. One can play with different values to find
    # what works best. When tinkering, it is important to consider the amount of cities that are being used
    # population_size = 15
    population_size = 45
    mating_pool_size = 20
    elitism_size = 7
    tournament_size = 10
    starting_bool = False
    
    # I have two data sets, one for 10 cities and one for 30
    nameFile = 'Data/cityNames.txt'
    coordinatesFile = 'Data/cityCoordinates.txt'

    # get the city names and coordinates from the files
    city_names = nameParser(nameFile)
    city_coordiantes = coordinatesParser(coordinatesFile)

    # map the city names and coordinates into a dictionary
    city_data = cityMapping(city_coordiantes, city_names)

    # get initial starting city and generate population
    initial_city = city_names[random.randrange(len(city_names))]
    population = initialization(initial_city, population_size, city_names, starting_bool)

    # generate the fitness scores of the randomized population just generated
    pop_fitness = []
    for pop in population:
        fitness = fitnessEvaluation(pop, city_data)
        pop_fitness.append((pop, fitness))

    # sort the population from highest to lowest fitness and print the results of the initial starting point
    sorted_pop = sorted(pop_fitness, key=lambda path: path[1], reverse=True)
    print('Initial Population Best Score')
    print('Fitness score: ' + str(1 / sorted_pop[0][1]) + '         Best Solution: ' + str(sorted_pop[0][0]) + '\n')

    plot_efficacy = []
    for i in range(1, 201):
        # grab the parents from the parent selection, I have multiple options to choose from. Then I shuffle
        # the parents to have random two parents match for the offspring
        parents = parentSelection(sorted_pop, mating_pool_size, elitism_size)
        # parents = parentSelectionTournament(sorted_pop, mating_pool_size, elitism_size, tournament_size)
        random.shuffle(parents)

        # iterate over all the parents, depending on the probability, we apply the crossover and mutation
        children = []
        for index in range(0, len(parents), 2):
            # if we do crossover, we apply one of the options for recombination that I created. If not,
            # then keep the parents as offspring. We also consider start location
            if crossoverProbability():
                if starting_bool:
                    offspring1, offspring2 = recombination(parents[index][0], parents[index + 1][0])
                    # offspring1, offspring2 = uniformCrossover(parents[index][0], parents[index + 1][0], starting_bool)
                else:
                    offspring1, offspring2 = recombinationGeneral(parents[index][0], parents[index + 1][0])
                    # offspring1, offspring2 = uniformCrossover(parents[index][0], parents[index + 1][0], starting_bool)
            else:
                offspring1, offspring2 = parents[index][0], parents[index + 1][0]

            # Depending on the mutation probability, we apply one of the multiple mutation techniques
            if mutationProbability():
                offspring1 = mutation(offspring1, starting_bool)
                # offspring1 = mutationScramble(offspring1, starting_bool)
            if mutationProbability():
                offspring2 = mutation(offspring2, starting_bool)
                # offspring2 = mutationScramble(offspring2, starting_bool)

            # append children
            children.append((offspring1, fitnessEvaluation(offspring1, city_data)))
            children.append((offspring2, fitnessEvaluation(offspring2, city_data)))

        # depending on the survivor selection used, we get the next generation and then sort based on fitness
        next_generation = muAndLambdaSelection(sorted_pop, children)
        # next_generation = replacementSelection(sorted_pop, children)
        # next_generation = children
        sorted_pop = sorted(next_generation, key=lambda path: path[1], reverse=True)

        # append the generation number and the distance of the best solution to plot later
        plot_efficacy.append([i, 1 / sorted_pop[0][1]])

        # we then print the best solutions, I make the starting city one separate since we do fewer iterations for it
        if not starting_bool:
            if i % 5 == 0:
                print('Generation: ' + str(i))
                print('Fitness score: ' + str(1 / sorted_pop[0][1]) + '         Best Solution: ' + str(sorted_pop[0][0])
                      + '\n')
        else:
            print('Generation: ' + str(i))
            print('Fitness score: ' + str(1 / sorted_pop[0][1]) + '         Best Solution: ' + str(
                sorted_pop[0][0]) + '\n')

    # plot the final results
    plotMap(sorted_pop[0][0], city_data, starting_bool)
    plotSolutionEfficacy(plot_efficacy)
