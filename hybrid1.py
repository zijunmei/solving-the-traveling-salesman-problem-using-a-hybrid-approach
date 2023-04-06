import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import torch.nn as nn
import torch.nn.functional as F
#import tsplib95

def load_data(filename):
    """
    Load and preprocess the TSP instances from the TSPLIB dataset.

    :param filename: The filename of the TSPLIB dataset to load.
    :return: A tensor representing the distances between the cities in the dataset.
    """
    # Load the TSP instance from the TSPLIB file
    data = np.genfromtxt(filename, skip_header=7, dtype=None, encoding=None, comments="EOF")
    coords = np.array([[row[1], row[2]] for row in data], dtype=int)
    n_cities = len(coords)

    # Compute the distances between each pair of cities
    distances_matrix = cdist(coords, coords)

    # Convert the distances to a PyTorch tensor
    distances_tensor = torch.from_numpy(distances_matrix).float()

    return distances_tensor

class Generator(nn.Module):
    def __init__(self, latent_dim, n_cities):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, n_cities)
        self.activation = nn.LeakyReLU(0.2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        eps = 1e-8
        x = x + eps
        x = x - x.max(dim=1, keepdim=True)[0]
        x = F.softmax(x, dim=1)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_cities):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(n_cities, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.fc1_bn = torch.nn.BatchNorm1d(128)
    def forward(self, x):
        x = self.activation(self.fc1_bn(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_gan(generator, discriminator, optimizer_g, optimizer_d, dataloader, n_epochs, device):
    """
    Train the GAN on the TSP instances and update the generator and discriminator.

    :param generator: The generator network.
    :param discriminator: The discriminator network.
    :param optimizer_g: The optimizer for the generator.
    :param optimizer_d: The optimizer for the discriminator.
    :param dataloader: The PyTorch DataLoader for the TSP instances.
    :param n_epochs: The number of epochs to train for.
    :param device: The device to use for training (e.g., 'cuda' or 'cpu').
    """
    # Set the models to train mode
    generator.train()
    discriminator.train()

    # Set the maximum norm for gradient clipping
    max_norm = 0.1

    # Train the GAN for the specified number of epochs
    for epoch in range(n_epochs):
        for batch in dataloader:
            # Train the discriminator on real data
            optimizer_d.zero_grad()
            real_data = batch.to(device)
            d_real = discriminator(real_data)
            loss_real = torch.mean(torch.log(d_real))
            loss_real.backward()

            # Clip discriminator gradients
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)

            # Train the discriminator on fake data
            noise = torch.randn(len(batch), latent_dim, device=device)
            fake_data = generator(noise)
            d_fake = discriminator(fake_data)
            loss_fake = torch.mean(torch.log(1 - d_fake))
            loss_fake.backward()

            # Clip discriminator gradients
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)

            optimizer_d.step()

            # Train the generator
            optimizer_g.zero_grad()
            noise = torch.randn(len(batch), latent_dim, device=device)
            fake_data = generator(noise)
            d_fake = discriminator(fake_data)
            loss_g = torch.mean(torch.log(d_fake))
            loss_g.backward()

            # Clip generator gradients
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm)

            optimizer_g.step()
        print(f"Epoch [{epoch + 1}/{n_epochs}] Generator Loss: {loss_g.item()} Discriminator Loss: {loss_real.item() + loss_fake.item()}")    
    
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

def initialize_population(candidates, population_size):
    population = []
    for _ in range(population_size):
        candidate = candidates[np.random.randint(0, len(candidates))]
        population.append(np.random.permutation(candidate))
    return population

def selection(population, fitness, n_parents):
    """
    Select the parents for the genetic algorithm using roulette wheel selection.

    :param population: The population of candidate solutions.
    :param fitness: The fitness values of the candidate solutions.
    :param n_parents: The number of parents to select.
    :return: The selected parents.
    """# Calculate the selection probabilities for each candidate solution
    epsilon = 1e-8
    invalid_indices = np.isnan(fitness) | np.isinf(fitness)
    fitness[invalid_indices] = epsilon
    selection_probs = fitness / np.sum(fitness)

    # Select the parents using roulette wheel selection
    parents_indices = np.random.choice(len(population), size=n_parents, replace=False, p=selection_probs)
    parents = [population[i] for i in parents_indices]

    return parents

def crossover(parents):
    """
    Apply crossover to the parents to generate new offspring for the genetic algorithm.

    :param parents: The selected parents.
    :return: The generated offspring.
    """
    # Select two parents at random
    p1, p2 = np.random.choice(parents, size=2, replace=False)

    # Perform a single-point crossover operation
    crossover_point = np.random.randint(1, len(p1) - 1)
    offspring = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
   
    return offspring

def mutation(offsprings, p_mut):
    """
    Apply mutation to the offspring population with a given mutation probability.

    :param offsprings: The offspring population.
    :param p_mut: The probability of mutation.
    :return: The mutated offspring population.
    """
    for i in range(len(offsprings)):
        if np.random.rand() < p_mut:
            mutation_point1, mutation_point2 = np.random.choice(len(offsprings[i]), 2, replace=False)
            offsprings[i][mutation_point1], offsprings[i][mutation_point2] = offsprings[i][mutation_point2], offsprings[i][mutation_point1]
    print("Mutation NaN values:", np.isnan(offsprings[i]).any())

    return offsprings

def evaluate_fitness(population, distances_tensor):
    """
    Evaluate the fitness of the candidate solutions in the population.

    :param population: The population of candidate solutions.
    :param distances_tensor: The tensor representing the distances between the cities.
    :return: The fitness values of the candidate solutions.
    """
    fitness = []
    for idx, solution in enumerate(population):
        distance = 0
        for i in range(1, len(solution)):
            if np.isnan(solution[i - 1]) or np.isnan(solution[i]):
                print(f"NaN value found in solution {idx}: {solution}")
                print(f"Previous city index: {i-1}, Current city index: {i}")
                print(f"Previous city value: {solution[i - 1]}, Current city value: {solution[i]}")
            distance += distances_tensor[int(solution[i - 1]), int(solution[i])]
        distance += distances_tensor[int(solution[-1]), int(solution[0])]
        fitness.append(distance)
    return np.array(fitness)

def run_ga(population, distances_tensor, n_generations, n_parents, crossover_prob, mutation_prob):
    """
    Run the genetic algorithm to improve the candidate solutions.

    :param population: The population of candidate solutions.
    :param distances_tensor: The tensor representing the distances between the cities.
    :param n_generations: The number of generations to run the algorithm for.
    :param n_parents: The number of parents to select for each generation.
    :param crossover_prob: The probability of crossover.
    :param mutation_prob: The probability of mutation.
    :return: The best solution found by the genetic algorithm and its distance.
    """
    # Evaluate the initial population
    fitness = evaluate_fitness(population, distances_tensor)

    # Initialize the best solution and its distance
    best_solution = population[np.argmax(fitness)]
    epsilon = 1e-8
    best_distance = 1 / (np.max(fitness)+ epsilon)

    # Keep track of the best solution found in each generation
    best_distances = [best_distance]

    # Run the genetic algorithm for the specified number of generations
    for i in range(n_generations):
        # Select the parents using roulette wheel selection
        parents = selection(population, fitness, n_parents)

        # Generate new offspring using crossover
        offsprings = []
        for j in range(len(population) - n_parents):
            # Apply crossover to the parents with a certain probability
            if np.random.rand() < crossover_prob:
                offspring = crossover(parents)
            else: # Select a random parent if crossover is not applied
                offspring = np.random.permutation(population[0])
            offsprings.append(offspring)

        # Apply mutation to the offspring population with a certain probability
        offsprings = mutation(offsprings, mutation_prob)

        # Evaluate the fitness of the offspring population
        offsprings_fitness = evaluate_fitness(offsprings, distances_tensor)

        # Select the best solutions from the current population and the offspring population
        combined_population = np.concatenate([population, offsprings])
        combined_fitness = np.concatenate([fitness, offsprings_fitness])
        best_indices = np.argsort(combined_fitness)[::-1][:len(population)]
        population = [combined_population[i] for i in best_indices]
        fitness = [combined_fitness[i] for i in best_indices]

        # Update the best solution found and its distance
        if fitness[0] > 1 / best_distance:
            best_solution = population[0]
            best_distance = 1 / fitness[0]

        # Record the best distance in this generation
        best_distances.append(best_distance)

    return best_solution, best_distance, best_distances
def evaluate_population(population, distances_matrix):
    fitness_scores = []
    for individual in population:
        fitness = 0
        for i in range(len(individual) - 1):
            fitness += distances_matrix[individual[i], individual[i+1]]
        fitness_scores.append(fitness)
    best_fitness = min(fitness_scores)
    best_solution = population[fitness_scores.index(best_fitness)]
    return best_solution, best_fitness, None

if __name__ == '__main__':

     # Set the parameters for the hybrid approach
    filename = 'pr1002.tsp'
    n_epochs_gan = 500
    n_generations_ga = 500
    population_size = 100
    n_parents = 20
    crossover_prob = 0.8
    mutation_prob = 0.2
    latent_dim = 32
    print("CUDA available:", torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the TSP instances from the TSPLIB dataset
    distances_tensor = load_data(filename)
    n_cities = distances_tensor.shape[0]
    # Create the generator, discriminator, and optimizers
    generator = Generator(latent_dim, n_cities).to(device)
    discriminator = Discriminator(n_cities).to(device)

    # Create the optimizers for the generator and discriminator
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

    # Create the dataloader
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(distances_tensor, batch_size=batch_size, shuffle=True, num_workers=4)
    # Check input data: Make sure that the input data does not contain any NaN or infinity values
    assert not torch.isnan(distances_tensor).any(), "Input data contains NaN values"
    assert not torch.isinf(distances_tensor).any(), "Input data contains Inf values"
    # Train the GAN on the TSP instances and generate a pool of candidate solutions
    train_gan(generator, discriminator, optimizer_g, optimizer_d, dataloader, n_epochs_gan, device)
    candidates = generator(torch.randn(10000, latent_dim, device=device)).detach().cpu().numpy()
    print("Candidates with NaN values:", np.isnan(candidates).any(axis=1).sum())

    
    best_distances = [1e9]

    # Initialize the population for the genetic algorithm with the candidate solutions from the GAN
    population = initialize_population(candidates, population_size)

    # Improve the candidate solutions using the genetic algorithm
    best_solution, best_distance, best_distances_ga = run_ga(population, distances_tensor, n_generations_ga, n_parents, crossover_prob, mutation_prob)
    best_distances += best_distances_ga

    # Plot the improvement of the solution over time
    plt.plot(best_distances)
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.title('Hybrid Approach')
    plt.show()

    # Plot the final solution
    fig, ax = plt.subplots()
    coords = np.loadtxt(filename, skiprows=6, usecols=[1, 2], dtype=int)
    for i in range(len(best_solution)):
        x1, y1 = coords[best_solution[i - 1]]
        x2, y2 = coords[best_solution[i]]
        ax.plot([x1, x2], [y1, y2], marker='o', color='b')
    ax.scatter(coords[:, 0], coords[:, 1], marker='o', color='r')
    plt.title(f'Final solution (distance = {best_distance:.2f})')
    plt.show()

    # Check if the solution was improved by the GA
    improved = best_distances[-1] < best_distances[-2]
    print(f'The solution was improved by the GA: {improved}')
    # Plot the candidate solutions generated by the GAN
    fig, ax = plt.subplots()
    coords = np.loadtxt(filename, skiprows=6, usecols=[1, 2], dtype=int)
    for i in range(len(candidates)):
        solution = np.argsort(-candidates[i])
        x = [coords[j, 0] for j in solution]
        y = [coords[j, 1] for j in solution]
        ax.plot(x + [x[0]], y + [y[0]], alpha=0.1, color='b')
    ax.scatter(coords[:, 0], coords[:, 1], marker='o', color='r')
    plt.title('Candidate Solutions Generated by GAN')
    plt.show()

    # Plot the performance of the candidate solutions generated by the GAN
    plt.plot(best_distances)
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.title('Candidate Solution Performance')
    plt.show()

    # Plot the fitness of the candidate solutions evaluated by the GA
    _, fitness, _ = evaluate_population(population, distances_tensor)
    plt.hist(fitness, bins=30)
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('Candidate Solution Evaluation')
    plt.show()

    # Plot the improvement of the solution over time
    plt.plot(best_distances_ga)
    plt.xlabel('Generations')
    plt.ylabel('Distance')
    plt.title('Genetic Algorithm Performance')
    plt.show()

    # Check if the solution was improved by the GA
    improved = best_distances_ga[-1] < best_distances_ga[-2]
    print(f'The solution was improved by the GA: {improved}')