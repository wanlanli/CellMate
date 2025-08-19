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
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import numpy as np
import math


red_cmap = LinearSegmentedColormap.from_list("red_cmap", [(1, 1, 1, 0), (1, 0, 0, 1)])  # White to red
green_cmap = LinearSegmentedColormap.from_list("green_cmap", [(1, 1, 1, 0), (0, 1, 0, 1)])  # White to green


def plot_field(field, cells, step, field_size=100):
    plt.figure(figsize=(6, 6))
    plt.xlim(0, field_size)
    plt.ylim(0, field_size)
    plt.imshow(field[0].T, origin="lower", cmap=red_cmap, extent=(0, field[0].shape[0], 0, field[0].shape[1]), alpha=0.5)
    plt.colorbar(label="Concentration")
    plt.imshow(field[1].T, origin="lower", cmap=green_cmap, extent=(0, field[1].shape[0], 0, field[1].shape[1]), alpha=0.5)
    plt.colorbar(label="Concentration")
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


class simulation_animation():
    def __init__(self, data):
        self.data = data

    def init_figure(self):
        # fig, ax = plt.subplots()
        # self.fig = fig
        # self.ax = ax
        self.ax.clear()  # Clear the axis
        self.ax.set_title("Dynamic Animation")
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)

    def update(self, frame):
        # ax.xlim(0, field_size)
        # ax.ylim(0, field_size)
        field, cells, step = self.data[frame]
        self.ax.clear()
        self.ax.imshow(field[0].T, origin="lower",  cmap=red_cmap, extent=(0, field[0].shape[0], 0, field[0].shape[1]), alpha=0.5)
        # ax.colorbar(label="Concentration")
        self.ax.imshow(field[1].T, origin="lower",  cmap=green_cmap, extent=(0, field[1].shape[0], 0, field[1].shape[1]), alpha=0.5)
        # ax.colorbar(label="Concentration")
        for cell in cells:
            # Plot cell ellipse
            color = 'red' if cell.cell_type == 1 else 'green'
            ellipse = patches.Ellipse((cell.x, cell.y), cell.width, cell.height, cell.angle, color=color, alpha=0.3)
            self.ax.add_patch(ellipse)
            self.ax.text(cell.x, cell.y, f"Cell {cell.id}", color="black", ha="center")

            # Plot dot
            dot_x, dot_y = cell.get_dot_coordinates()
            # plt.plot(dot_x, dot_y, 'ro', label=f"Dot (Cell {cell.id})" if cell.id == 1 else None)
            self.ax.plot(dot_x, dot_y, 'ro', c=color)
        self.ax.set_title(f"Cell Field at Step {step}")
        self.ax.grid(True)

    def animate(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=len(self.data),
            init_func=self.init_figure, blit=False, interval=500
        )
        plt.close(fig)
        return ani


def angle_transfer(angle):
    angle = angle % 360
    if (angle > 90) & (angle <= 180):
        angle = 180 - angle
    if (angle > 180) & (angle <= 270):
        angle = angle - 180
    if angle > 270:
        angle = 360 - angle
    return angle


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generate_random_ellipse_params(num_ellipses=4, max_attempts=100):
    ellipses = []
    attempts = 0

    while len(ellipses) < num_ellipses and attempts < max_attempts:
        # Generate random parameters
        center_x = np.random.uniform(20, 80)
        center_y = np.random.uniform(20, 80)
        major_axis = np.random.uniform(10, 25)
        minor_axis = np.random.uniform(5, major_axis-1)
        angle = np.random.uniform(0, 360)

        # Create a new ellipse
        new_ellipse = {
            'center_x': center_x,
            'center_y': center_y,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle
        }

        # Check if it overlaps with existing ellipses
        overlap = False
        for e in ellipses:
            if check_overlap(new_ellipse, e):
                overlap = True
                break

        if not overlap:
            ellipses.append(new_ellipse)
        attempts += 1

    if len(ellipses) < num_ellipses:
        print("Could not generate enough non-overlapping ellipses.")
    return ellipses


def check_overlap(e1, e2):
    """Check if two ellipses overlap using approximate bounding box logic."""
    dx = e1['center_x'] - e2['center_x']
    dy = e1['center_y'] - e2['center_y']
    distance = np.sqrt(dx**2 + dy**2)

    # Calculate the maximum possible radii sum (rough approximation)
    max_radii_sum = max(e1['major_axis'], e1['minor_axis']) + max(e2['major_axis'], e2['minor_axis'])
    return distance < max_radii_sum
