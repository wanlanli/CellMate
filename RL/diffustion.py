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
