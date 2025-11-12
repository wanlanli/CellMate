import numpy as np


class MultiArmedBandit:
    def __init__(self,
                 probabilities=None,
                 n_arms=10,
                 mode="bernoulli",
                 noise_std=0.1):
        """
        Initialize a multi-armed bandit environment.

        Parameters
        ----------
        probabilities : list or np.ndarray, optional
            The true mean reward (probability of success) for each arm.
            If None, random values between 0 and 1 are generated.
        n_arms : int, default=10
            Number of arms (machines) available.
        mode : {"bernoulli", "continuous"}, default="bernoulli"
            Reward distribution type:
              - "bernoulli": binary rewards (0 or 1) based on success probability.
              - "continuous": continuous rewards sampled from a Gaussian distribution
                              centered on the true mean.
        noise_std : float, default=0.1
            Standard deviation of the Gaussian noise for continuous mode.
        """
        # Validate mode input
        if mode not in {"bernoulli", "continuous"}:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of \'bernoulli\', \'continuous\'.")
        # If no custom probabilities are provided, randomly assign success probabilities to each arm
        if probabilities is None:
            probabilities = np.random.normal(0, 1, n_arms)
        self.probabilities = probabilities
        self.n_arms = len(probabilities)
        self.mode = mode
        self.noise_std = noise_std

    def pull(self, arm):
        """
        Simulate pulling one of the bandit's arms.

        Parameters
        ----------
        arm : int
            Index of the arm to pull (0-based).

        Returns
        -------
        float
            Reward obtained from pulling the chosen arm.
            - In "bernoulli" mode: returns 0 or 1 (failure/success).
            - In "continuous" mode: returns a noisy reward value in [0, 1].
        """
        # Check if arm index is valid
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm index {arm}. Must be between 0 and {self.n_arms - 1}.")

        # Bernoulli mode: reward = 1 with probability p, else 0
        if self.mode == "bernoulli":
            noisy_p = np.clip(self.probabilities[arm] + np.random.normal(0, 1), 0, 1)
            reward = 1.0 if np.random.rand() < noisy_p else 0.0

        # Continuous mode: reward sampled from N(mean=p, std=noise_std), clipped to [0, 1]
        elif self.mode == "continuous":
            reward = self.probabilities[arm]+np.random.normal(0, 1)
        return reward
