import numpy as np

class CurriculumManager:
    def __init__(
        self,
        total_timesteps,
        num_workers=1,
        mode=2,              # 0=none, 1=fixed-linear, 2=adaptive-plateau
        current_timestep=0,
        window_size=50,           # adaptive mode: steps in each comparison window
        min_improvement=1e-3,     # adaptive mode: min avg improvement to avoid plateau
        patience=3                # adaptive mode: how many plateau checks before difficulty increase
    ):
        self.mode = mode
        self.difficulty = 10
        self.total_timesteps = total_timesteps / num_workers
        self.current_timestep = current_timestep

        # reward tracking
        self.episode_rewards = []   # rewards for current episode
        self.reward_history = []    # aggregated per-episode rewards

        # adaptive mode params
        self.window_size = window_size
        self.min_improvement = min_improvement
        self.patience = patience
        self.no_improvement_streak = 0

    def report_reward(self, reward):
        self.current_timestep += 1
        self.episode_rewards.append(reward)

    def end_episode(self):
        if self.episode_rewards:
            total_reward = sum(self.episode_rewards)
            self.reward_history.append(total_reward)
            self.episode_rewards.clear()

    def _difficulty_linear(self):
        """
        Mode 1: Linear progression from 10 → 1 over total_timesteps.
        """
        progress = self.current_timestep / self.total_timesteps
        difficulty = 10 - int(progress * 10)
        return max(difficulty, 1)

    def _difficulty_adaptive(self):
        """
        Mode 2: Adaptive progression via plateau detection.
        """
        if len(self.reward_history) < 2 * self.window_size:
            return self.difficulty  # not enough data yet

        recent_mean = np.mean(self.reward_history[-self.window_size:])
        prev_mean = np.mean(self.reward_history[-2*self.window_size:-self.window_size])
        improvement = recent_mean - prev_mean

        if improvement < self.min_improvement:
            self.no_improvement_streak += 1
            if self.no_improvement_streak >= self.patience:
                # increase difficulty (reduce size of button, etc.)
                self.difficulty = max(self.difficulty - 1, 1)
                self.no_improvement_streak = 0
        else:
            self.no_improvement_streak = 0

        return self.difficulty

    def get_difficulty(self):
        self.end_episode()
        if self.mode == 1:
            return self._difficulty_linear()
        elif self.mode == 2:
            return self._difficulty_adaptive()
        else:
            return self.difficulty
