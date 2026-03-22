# Copyright 2026 wlli
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


def diffusion_1d(x, t, D=1.0, M=1.0):
    """
    1D diffusion from point source.

    Parameters
    ----------
    x : float or np.ndarray
        position(s)
    t : float or np.ndarray
        time(s), must be > 0
    D : float
        diffusion coefficient
    M : float
        total released amount

    Returns
    -------
    c : float or np.ndarray
        concentration
    """
    x = np.asarray(x)
    t = np.asarray(t)

    # avoid division by zero
    t = np.maximum(t, 1e-12)

    return M / np.sqrt(4 * np.pi * D * t) * np.exp(-x**2 / (4 * D * t))


def distance_concentration_kernel(x, spread=0.1, amplitude=1.0, eps=1e-12):
    """
    Distance–Concentration Kernel (Gaussian-like decay)

    This function models how signal intensity decays with spatial distance
    in a single sensing step. It is inspired by the spatial profile of diffusion,
    but does NOT model time evolution or mass conservation.

    Mathematical form:
        c(x) = A * exp( - x^2 / spread )

    where:
        x       : distance
        spread  : controls how far the signal reaches (effective diffusion range)
        A       : maximum signal intensity at x = 0

    Parameters
    ----------
    x : array-like or float
        Distance(s) from the signal source (>= 0).
    spread : float, default=0.1
        Effective spread parameter (∝ diffusion_rate × sensing_time).
        Larger values -> slower decay (longer-range influence).
        Smaller values -> faster decay (more local signal).
    amplitude : float, default=1.0
        Peak signal value at x = 0.
    eps : float
        Small constant to avoid numerical issues.

    Returns
    -------
    c : ndarray or float
        Signal intensity at distance x.

    Notes
    -----
    - This is NOT a full diffusion solution.
    - Time is assumed constant (single-step sensing).
    - Total signal is NOT conserved.
    - This is effectively a spatial weighting / interaction kernel.

    Interpretation in your model
    ----------------------------
    - "spread" encodes how far a cell's signal propagates.
    - Larger spread ≈ faster diffusion / longer sensing range.
    - Smaller spread ≈ slower diffusion / local sensing.

    Typical usage
    -------------
    >>> x = np.linspace(0, 1, 100)
    >>> c = distance_concentration_kernel(x, spread=0.2)
    """

    x = np.asarray(x, dtype=float)
    spread = max(spread, eps)

    return amplitude * np.exp(-(x**2) / spread)
