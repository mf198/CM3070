# ccfd/data/gan_oversampler.py
import torch
import torch.nn as nn
import torch.optim as optim
import cudf
import numpy as np

class Generator(nn.Module):
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


class Discriminator(nn.Module):
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

def train_gan(X_real, num_epochs=1000, latent_dim=10, batch_size=64):
    """
    Trains a GAN to generate synthetic samples.
    :param X_real: The real dataset (numpy or tensor).
    :param num_epochs: Number of training iterations.
    :param latent_dim: Size of the random noise input to the generator.
    :param batch_size: Training batch size.
    :return: Trained generator model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert X_real to Tensor
    X_real = torch.tensor(X_real, dtype=torch.float32).to(device)
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
        real_labels = torch.ones((batch_size, 1)).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        # Train on real data
        idx = torch.randint(0, X_real.size(0), (batch_size,))
        real_data = X_real[idx]
        real_output = discriminator(real_data)
        real_loss = loss_function(real_output, real_labels)

        # Train on fake data
        z = torch.randn((batch_size, latent_dim)).to(device)
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


def generate_synthetic_samples(generator, num_samples=1000, latent_dim=10):
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


