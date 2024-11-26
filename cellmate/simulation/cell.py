# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import random


class Cell:
    def __init__(self, x, y, width, height, angle, id, cell_type):
        self.x = x  # Cell center x-coordinate
        self.y = y  # Cell center y-coordinate
        self.width = width  # Ellipse width
        self.height = height  # Ellipse height
        self.angle = angle
        self.id = id  # Cell ID
        self.cell_type = cell_type  # Cell type (1 or 2)
        self.dot_angle = random.choice([0, 180]) #self.angle  # Start dot at the rightmost tip of the ellipse
        self.sensed_inputs = []  # Inputs sensed by the dot

    def update_dot_position(self, field, resolution=100):
        """
        Orient the dot on the cell's periphery toward the location with the highest concentration in the field.
        If the concentration is homogeneous, the dot randomly switches between the two tips of the ellipse.

        Args:
            field (np.ndarray): The 2D concentration field.
            resolution (int): Number of points sampled along the cell's periphery.
        """
        angles = np.linspace(0, 360, resolution)
        best_concentration = -np.inf
        best_angle = None

        detected_concentration = []
        for angle in angles:
            rad = np.radians(angle)
            orientation_rad = np.radians(self.angle)
            test_x = self.x + (self.width / 2) * np.cos(rad) * np.cos(orientation_rad) - (self.height / 2) * np.sin(rad) * np.sin(orientation_rad)
            test_y = self.y + (self.width / 2) * np.cos(rad) * np.sin(orientation_rad) + (self.height / 2) * np.sin(rad) * np.cos(orientation_rad)

            # Convert coordinates to integers for field indexing
            ix, iy = int(test_x), int(test_y)

            # Check if the sampled point is within the field bounds
            if 0 <= ix < field.shape[0] and 0 <= iy < field.shape[1]:
                concentration = field[ix, iy]
                detected_concentration.append(concentration)
                if concentration > best_concentration:
                    best_concentration = concentration
                    best_angle = angle

        # Handle homogeneous concentration
        detected_concentration = np.array(detected_concentration)
        if best_concentration == -np.inf or np.all(detected_concentration == detected_concentration[0]):
            # Randomly switch between the two tips of the major axis
            self.dot_angle = random.choice([0, 180])  # 0 and 180 degrees represent the two tips
            print(f"Cell {self.id} Random Inputs: {self.dot_angle}")
        else:
            # Orient the dot to the angle with the highest concentration
            self.dot_angle = best_angle
            print(f"Cell {self.id} Sensed Inputs: {self.dot_angle}, concentration = {best_concentration}")

    def get_dot_coordinates(self):
        """Get the (x, y) position of the dot on the ellipse's periphery."""
        rad = np.radians(self.dot_angle)
        orientation_rad = np.radians(self.angle)  # Orientation of the ellipse's major axis
        # dot_x = self.x + (self.width / 2) * np.cos(rad)
        # dot_y = self.y + (self.height / 2) * np.sin(rad)
        dot_x = self.x + (self.width / 2) * np.cos(rad) * np.cos(orientation_rad) - (self.height / 2) * np.sin(rad) * np.sin(orientation_rad)
        dot_y = self.y + (self.width / 2) * np.cos(rad) * np.sin(orientation_rad) + (self.height / 2) * np.sin(rad) * np.cos(orientation_rad)
        return dot_x, dot_y

    def sense_other_cells(self, other_cells):
        """Simulate sensing other cells near the dot's current position."""
        dot_x, dot_y = self.get_dot_coordinates()
        self.sensed_inputs = []
        for other in other_cells:
            if other.cell_type != self.cell_type:  # Only sense cells of the opposite type
                other_dot_x, other_dot_y = other.get_dot_coordinates()
                distance = np.sqrt((dot_x - other_dot_x) ** 2 + (dot_y - other_dot_y) ** 2)
                if distance < 20:  # Sensing radius
                    self.sensed_inputs.append(f"Sensed Cell {other.id} (Type {other.cell_type}) at distance {distance:.2f}")

    def output_info(self):
        """Output the sensed information."""
        dot_x, dot_y = self.get_dot_coordinates()
        print(f"Cell {self.id} (Type {self.cell_type}) Dot Position: ({dot_x:.2f}, {dot_y:.2f})")
        print(f"Cell {self.id} Sensed Inputs: {self.sensed_inputs}")


# Plotting the field and cells
def plot_field(cells, step, field_size=100):
    plt.figure(figsize=(6, 6))
    plt.xlim(0, field_size)
    plt.ylim(0, field_size)
    for cell in cells:
        # Plot cell ellipse
        color = 'red' if cell.cell_type == 1 else 'green'
        ellipse = patches.Ellipse((cell.x, cell.y), cell.width, cell.height, cell.angle, color=color, alpha=0.3)
        plt.gca().add_patch(ellipse)
        plt.text(cell.x, cell.y, f"Cell {cell.id}", color="black", ha="center")

        # Plot dot
        dot_x, dot_y = cell.get_dot_coordinates()
        # plt.plot(dot_x, dot_y, 'ro', label=f"Dot (Cell {cell.id})" if cell.id == 1 else None)
        plt.plot(dot_x, dot_y, 'ro', c=color)

    plt.title(f"Cell Field at Step {step}")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # Simulation setup
    cells = [
        Cell(50, 50, 20, 10, 90, id=1, cell_type=1),
        Cell(63, 40, 15, 10, 90, id=2, cell_type=2),
        Cell(50, 70, 20, 12, 30, id=3, cell_type=2),
        Cell(33, 45, 18, 9, 20, id=4, cell_type=2),
    ]
    # Simulation loop
    steps = 10
    for step in range(steps):
        print(f"\nStep {step + 1}")
        for cell in cells:
            cell.update_dot_position(cells)  # Update the dot's position based on field information
            cell.sense_other_cells(cells)  # Sense nearby cells using the dot
            cell.output_info()  # Output sensing results
        plot_field(cells, step + 1)
