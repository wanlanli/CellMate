
import numpy as np


class MultiArmedBanditStrategy:
    def __init__(self, bandit, method="epsilon_greedy"):
        self.bandit = bandit
        self.method = method
        self.__init_params()

    def run(self, method, iterations, epsilon=0):
        self.__init_params()
        if method == "epsilon_greedy":
            reward, values = self.epsilon_greedy_strategy(iterations, epsilon)
        elif method == "greedy":
            reward, values = self.epsilon_greedy_strategy(iterations, epsilon=0)
        elif method == "random":
            reward, values = self.epsilon_greedy_strategy(iterations, epsilon=1)
        return reward, values

    def epsilon_greedy_strategy(self, iterations, epsilon):
        for i in range(iterations):
            # Exploration vs exploitation decision
            if np.random.random() < epsilon:
                arm = np.random.choice(self.bandit.n_arms)  # Explore: random arm
            else:
                arm = np.argmax(self.arm_values)  # Exploit: best known arm

            reward = self.bandit.pull(arm)
            self.__update_counts(arm, reward)
        return self.step_reward, self.step_values

    def __update_counts(self, arm, reward):
        self.arm_counts[arm] += 1
        self.arm_values[arm] += (reward - self.arm_values[arm])/self.arm_counts[arm]

        self.step_reward.append(reward)
        self.step_values.append(self.arm_values[arm])

    def __init_params(self):
        self.arm_counts = np.zeros(self.bandit.n_arms)
        self.arm_values = np.zeros(self.bandit.n_arms)
        self.step_reward = []
        self.step_values = []
