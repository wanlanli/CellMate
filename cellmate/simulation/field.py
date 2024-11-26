# // Copyright 2024 wlli
# // 
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# // 
# //     https://www.apache.org/licenses/LICENSE-2.0
# // 
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import numpy as np
import matplotlib.pyplot as plt

class DiffusionSimulator:
    def __init__(self, size, D, dx=1.0, dt=0.1):
        """
        Initialize the diffusion simulator.
        
        Args:
            size (int): Size of the field (size x size).
            D (float): Diffusion coefficient.
            dx (float): Grid spacing.
            dt (float): Time step for the simulation.
        """
        self.size = size
        self.D = D
        self.dx = dx
        self.dt = dt
        self.field = np.zeros((size, size))  # Initialize field as a zero matrix

    def initialize_field(self, initial_positions, initial_concentration=100):
        """
        Set the initial concentration in the field.

        Args:
            initial_positions (list of tuples): List of (x, y) positions for initial concentrations.
            initial_concentration (float): Initial concentration at the specified positions.
        """
        for pos in initial_positions:
            x, y = pos
            self.field[x, y] = initial_concentration

    def add_elements(self, positions, concentration):
        """
        Add new elements into the field at specified positions with given concentration.
        
        Args:
            positions (list of tuples): List of (x, y) positions to add new elements.
            concentration (float): Concentration value to add at each position.
        """
        for pos in positions:
            x, y = pos
            x = int(x)
            y = int(y)
            if 0 <= x < self.size and 0 <= y < self.size:
                self.field[x, y] += concentration
            else:
                print(f"Warning: Position {pos} is out of bounds and will be ignored.")

    def diffuse(self, steps):
        """
        Simulate the diffusion process over a number of steps.

        Args:
            steps (int): Number of simulation steps.
        """
        for step in range(steps):
            # Compute the Laplacian (∇²C)
            laplacian = (
                np.roll(self.field, 1, axis=0) +  # Up
                np.roll(self.field, -1, axis=0) +  # Down
                np.roll(self.field, 1, axis=1) +  # Left
                np.roll(self.field, -1, axis=1) -  # Right
                4 * self.field  # Center
            ) / (self.dx ** 2)

            # Update the field using the diffusion equation
            self.field += self.D * laplacian * self.dt

            # Ensure no negative concentrations
            self.field = np.maximum(self.field, 0)

            # Plot the field at each step
            # self.plot_field(step)

    def plot_field(self, step):
        """
        Visualize the current state of the field as a heatmap.
        
        Args:
            step (int): Current simulation step.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(self.field, origin="lower", cmap="viridis", extent=(0, self.size, 0, self.size))
        plt.colorbar(label="Concentration")
        plt.title(f"Peptide Diffusion - Step {step}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)
        plt.show()


def main():
    # Parameters
    size = 100  # Size of the field (100x100)
    D = 1.0  # Diffusion coefficient
    dx = 1.0  # Grid spacing
    dt = 0.1  # Time step
    steps = 10  # Number of time steps
    initial_positions = [(50, 50)]  # Initial positions of the peptide

    # Create an instance of the simulator
    simulator = DiffusionSimulator(size=size, D=D, dx=dx, dt=dt)

    # Initialize the field
    simulator.initialize_field(initial_positions)

    # Simulate diffusion for 5 steps
    simulator.diffuse(5)

    # Add new elements to the field
    new_positions = [(40, 40), (60, 60)]
    simulator.add_elements(new_positions, concentration=500)

    # Continue diffusion for another 5 steps
    simulator.diffuse(5)
