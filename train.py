
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

def load_data(filename):
    # Load the TSP instance from the TSPLIB file
    data = np.genfromtxt(filename, skip_header=7, dtype=None, encoding=None, comments="EOF")
    coords = np.array([[row[1], row[2]] for row in data], dtype=int)
    n_cities = len(coords)

    # Compute the distances between each pair of cities
    distances_matrix = cdist(coords, coords)

    # Convert the distances to a PyTorch tensor
    distances_tensor = torch.from_numpy(distances_matrix).float()

    # Normalize the distances to have zero mean and unit variance
    distances_mean = distances_tensor.mean()
    distances_std = distances_tensor.std()
    distances_tensor = (distances_tensor - distances_mean) / distances_std

    return distances_tensor, coords

class Generator(nn.Module):
    def __init__(self, latent_dim, n_cities):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(latent_dim, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, n_cities)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(x[:, -1])
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

class Discriminator(nn.Module):
    def __init__(self, n_cities):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(n_cities, 128)
        self.fc2 = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(128)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.activation(self.fc1(x))
        x = self.layer_norm(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def train_gan(generator, discriminator, optimizer_g, optimizer_d, dataloader, n_epochs, device):
    generator.train()
    discriminator.train()

    max_norm = 0.1
    lambda_gp = 10

    for epoch in range(n_epochs):
        for batch in dataloader:
            # Train the discriminator on real data
            optimizer_d.zero_grad()
            real_data = batch.unsqueeze(1).to(device)
            with torch.backends.cudnn.flags(enabled=False):
                d_real = discriminator(real_data)
            loss_real = -torch.mean(d_real)

            # Train the discriminator on fake data
            noise = torch.randn(len(batch), 1, latent_dim, device=device)
            fake_data = generator(noise)
            with torch.backends.cudnn.flags(enabled=False):
                d_fake = discriminator(fake_data.unsqueeze(1))
            loss_fake = torch.mean(d_fake)

            # Compute gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data.unsqueeze(1))
            loss_d = loss_real + loss_fake + lambda_gp * gradient_penalty

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm)
            optimizer_d.step()

            # Train the generator
            optimizer_g.zero_grad()
            noise = torch.randn(len(batch), 1, latent_dim, device=device)
            fake_data = generator(noise)
            with torch.backends.cudnn.flags(enabled=False):
                d_fake = discriminator(fake_data.unsqueeze(1))
            loss_g = -torch.mean(d_fake)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm)
            optimizer_g.step()

        print(f"Epoch [{epoch + 1}/{n_epochs}] Generator Loss: {loss_g.item()} Discriminator Loss: {loss_d.item()}")

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')


def compute_gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1).to(real_data.device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), device=real_data.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def probabilities_to_tours(candidates):
    tours = [np.argsort(candidate) for candidate in candidates]
    return np.array(tours)

def compute_tour_lengths(tours, distances_matrix):
    tour_lengths = []
    for tour in tours:
        tour_length = 0
        for i in range(len(tour) - 1):
            tour_length += distances_matrix[tour[i], tour[i+1]]
        tour_length += distances_matrix[tour[-1], tour[0]]  # Return to the starting city
        tour_lengths.append(tour_length)
    return np.array(tour_lengths)

import matplotlib.pyplot as plt

def plot_tour(coords, tour):
    plt.figure(figsize=(10, 10))
    tour_coords = np.array([coords[city] for city in tour])
    tour_coords = np.append(tour_coords, [tour_coords[0]], axis=0)  # Close the tour
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], marker='o', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Best Tour')
    plt.show()

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
    distances_tensor, coords = load_data(filename)
    distances_matrix = distances_tensor.numpy()
    n_cities = distances_tensor.shape[0]
    # Create the generator, discriminator, and optimizers
    generator = Generator(latent_dim, n_cities).to(device)
    discriminator = Discriminator(n_cities).to(device)

    # Create the optimizers for the generator and discriminator
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-6)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-6)

    # Create the dataloader
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(distances_tensor, batch_size=batch_size, shuffle=True, num_workers=4)
    # Check input data: Make sure that the input data does not contain any NaN or infinity values
    assert not torch.isnan(distances_tensor).any(), "Input data contains NaN values"
    assert not torch.isinf(distances_tensor).any(), "Input data contains Inf values"
    # Train the GAN on the TSP instances and generate a pool of candidate solutions
    train_gan(generator, discriminator, optimizer_g, optimizer_d, dataloader, n_epochs_gan, device)
    candidates = generator(torch.randn(10000, 1, latent_dim, device=device)).detach().cpu().numpy()
    print("Candidates with NaN values:", np.isnan(candidates).any(axis=1).sum())
    tours = probabilities_to_tours(candidates)
    tour_lengths = compute_tour_lengths(tours, distances_matrix)

    min_length = np.min(tour_lengths)
    mean_length = np.mean(tour_lengths)
    std_length = np.std(tour_lengths)

    print("Minimum tour length:", min_length)
    print("Mean tour length:", mean_length)
    print("Standard deviation of tour lengths:", std_length)
    best_tour = tours[np.argmin(tour_lengths)]
    plot_tour(coords, best_tour)
    best_distances = [1e9]

  