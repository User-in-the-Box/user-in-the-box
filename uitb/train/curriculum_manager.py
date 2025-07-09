class CurriculumManager:
    def __init__(self, total_timesteps, num_workers=1, mode="none", current_timestep=0):
        self.mode = mode
        self.difficulty = 10
        self.total_timesteps = total_timesteps / num_workers
        self.current_timestep = current_timestep

    def report_reward(self, reward):
        self.current_timestep += 1
        print(f"reward reported at {self.current_timestep=}, and the reward is {reward}")
        # Here should be some reward gathering logix ;)

    def _caculate_difficulty(self):
        print(f"difficulty will be calculated from {self.current_timestep=} and {self.total_timesteps}")
        progress = self.current_timestep / self.total_timesteps
        difficulty = 10 - int(progress * 10)
        if difficulty < 1:
            difficulty = 1
        return difficulty

    def get_difficulty(self):
        return self._caculate_difficulty()

