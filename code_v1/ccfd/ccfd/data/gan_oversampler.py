# ccfd/data/gan_oversampler.py
import torch
import torch.nn as nn
import torch.optim as optim
import cupy as cp
import numpy as np

class Generator(nn.Module):
    """ Takes random noise (latent space) and generates samples"""
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Normalize between -1 and 1
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module): # Used for 'vanilla' GAN
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)        


def train_gan(X_real, num_epochs=1000, latent_dim=10, batch_size=64, use_gpu=False):
    """
    Trains a GAN to generate synthetic samples on GPU (CUDA) or CPU.

    Args:
        X_real (numpy.ndarray or cupy.ndarray): The real dataset.
        num_epochs (int): Number of training iterations.
        latent_dim (int): Size of the random noise input to the generator.
        batch_size (int): Training batch size.
        use_gpu (bool): If True, trains on GPU, otherwise on CPU.

    Returns:
        Generator model (trained).
    """
    # Set device based on user choice
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert X_real to PyTorch Tensor
    if isinstance(X_real, cp.ndarray):  # CuPy GPU array
        X_real = torch.tensor(cp.asnumpy(X_real), dtype=torch.float32, device=device)
    else:  # NumPy CPU array
        X_real = torch.tensor(X_real, dtype=torch.float32, device=device)

    input_dim = X_real.shape[1]

    # Initialize Generator and Discriminator
    generator = Generator(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # Optimizers and Loss
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    loss_function = nn.BCELoss()

    for epoch in range(num_epochs):
        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # Train on real data
        idx = torch.randint(0, X_real.size(0), (batch_size,), device=device)
        real_data = X_real[idx]
        real_output = discriminator(real_data)
        real_loss = loss_function(real_output, real_labels)

        # Train on fake data
        z = torch.randn((batch_size, latent_dim), device=device)
        fake_data = generator(z)
        fake_output = discriminator(fake_data.detach())  # Detach to avoid updating Generator
        fake_loss = loss_function(fake_output, fake_labels)

        # Total Discriminator loss
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_output = discriminator(fake_data)  # Re-evaluate fake data
        loss_G = loss_function(fake_output, real_labels)  # Flip labels to fool discriminator
        loss_G.backward()
        optimizer_G.step()

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    return generator



# ============== WGAN ===================

class Critic(nn.Module):  # Discriminator in WGAN is called "Critic"
    """WGAN Critic: Evaluates real vs fake samples using Wasserstein loss"""
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a scalar score (no Sigmoid)
        )

    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cupy as cp

def train_wgan(X_real, num_epochs=1000, latent_dim=10, batch_size=64, 
               critic_iterations=5, weight_clip=0.01, use_gpu=False):
    """
    Trains a Wasserstein GAN (WGAN) using weight clipping. Supports both CPU and GPU.

    Args:
        X_real (numpy.ndarray or cupy.ndarray): Training dataset.
        num_epochs (int): Number of training iterations.
        latent_dim (int): Size of the random noise input.
        batch_size (int): Training batch size.
        critic_iterations (int): How many times to train Critic per Generator update.
        weight_clip (float): Clipping range for Critic weights.
        use_gpu (bool): If True, trains on GPU, otherwise on CPU.

    Returns:
        Generator model (trained).
    """
    # ✅ Set device based on user choice
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # ✅ Convert X_real to PyTorch Tensor
    if isinstance(X_real, cp.ndarray):  # CuPy GPU array
        X_real = torch.tensor(cp.asnumpy(X_real), dtype=torch.float32, device=device)
    else:  # NumPy CPU array
        X_real = torch.tensor(X_real, dtype=torch.float32, device=device)

    input_dim = X_real.shape[1]

    # ✅ Initialize Generator and Critic
    generator = Generator(latent_dim, input_dim).to(device)
    critic = Critic(input_dim).to(device)

    # ✅ Optimizers (no momentum in WGAN)
    optimizer_G = optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_C = optim.RMSprop(critic.parameters(), lr=0.00005)

    for epoch in range(num_epochs):
        for _ in range(critic_iterations):  # Train Critic multiple times per Generator update
            optimizer_C.zero_grad()
            real_idx = torch.randint(0, X_real.size(0), (batch_size,), device=device)
            real_data = X_real[real_idx]

            # ✅ Train on real data
            real_output = critic(real_data)
            loss_real = -torch.mean(real_output)  # Critic should maximize score for real data

            # ✅ Train on fake data
            z = torch.randn((batch_size, latent_dim), device=device)
            fake_data = generator(z)
            fake_output = critic(fake_data.detach())
            loss_fake = torch.mean(fake_output)  # Critic should minimize score for fake data

            # ✅ Total loss
            loss_C = loss_real + loss_fake
            loss_C.backward()
            optimizer_C.step()

            # ✅ Apply weight clipping
            for p in critic.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        # ✅ Train Generator
        optimizer_G.zero_grad()
        fake_data = generator(z)
        fake_output = critic(fake_data)
        loss_G = -torch.mean(fake_output)  # Generator should maximize Critic's score
        loss_G.backward()
        optimizer_G.step()

        # ✅ Print training progress
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | Critic Loss: {loss_C.item():.4f} | Generator Loss: {loss_G.item():.4f}")

    return generator

import torch
import numpy as np

def generate_synthetic_samples(generator, num_samples=1000, latent_dim=10, use_gpu=False):
    """
    Uses a trained GAN generator to create synthetic samples on GPU or CPU.

    Args:
        generator (torch.nn.Module): The trained generator model.
        num_samples (int): Number of synthetic samples to generate.
        latent_dim (int): Size of noise input.
        use_gpu (bool): If True, generates samples on GPU, otherwise on CPU.

    Returns:
        numpy.ndarray: Generated synthetic samples.
    """
    # Set the device based on user choice
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Ensure the generator is on the correct device
    generator = generator.to(device)

    # Generate random noise (z) on the same device
    z = torch.randn((num_samples, latent_dim), device=device)

    # Generate synthetic data and move to CPU before converting to NumPy
    synthetic_data = generator(z).detach().cpu().numpy()

    return synthetic_data


def generate_synthetic_samples2(generator, num_samples=1000, latent_dim=10):
    """
    Uses a trained GAN generator to create synthetic samples.
    :param generator: The trained generator model.
    :param num_samples: Number of synthetic samples to generate.
    :param latent_dim: Size of noise input.
    :return: Generated synthetic samples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn((num_samples, latent_dim)).to(device)
    synthetic_data = generator(z).cpu().detach().numpy()
    return synthetic_data

