import numpy as np
import torch

from typing import Any, Dict, Optional
import os
import warnings

from stable_baselines3.common.callbacks import BaseCallback, EventCallback
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from .dummy_vec_env import DummyVecEnv
from .evaluation import evaluate_policy

class LinearStdDecayCallback(BaseCallback):
    """
    Linearly decaying standard deviation

    :param initial_log_value: Log initial standard deviation value
    :param threshold: Threshold for progress remaining until decay begins
    :param min_value: Minimum value for standard deviation
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, initial_log_value, threshold, min_value, verbose=0):
      super(LinearStdDecayCallback, self).__init__(verbose)
      self.initial_value = np.exp(initial_log_value)
      self.threshold = threshold
      self.min_value = min_value

    def _on_rollout_start(self) -> None:
      progress_remaining = self.model._current_progress_remaining
      if progress_remaining > self.threshold:
        pass
      else:
        new_std = self.min_value + (progress_remaining/self.threshold) * (self.initial_value-self.min_value)
        self.model.policy.log_std.data = torch.tensor(np.log(new_std)).float()

    def _on_training_start(self) -> None:
      pass

    def _on_step(self) -> bool:
      return True

    def _on_rollout_end(self) -> None:
      pass

    def _on_training_end(self) -> None:
      pass


class LinearCurriculum(BaseCallback):
  """
  A callback to implement linear curriculum for one parameter

  :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
  """

  def __init__(self, name, start_value, end_value, end_timestep, start_timestep=0, verbose=0):
    super().__init__(verbose)
    self.name = name
    self.variable = start_value
    self.start_value = start_value
    self.end_value = end_value
    self.start_timestep = start_timestep
    self.end_timestep = end_timestep
    self.coeff = (end_value - start_value) / (end_timestep - start_timestep)

  def value(self):
    return self.variable

  def update(self, num_timesteps):
    if num_timesteps <= self.start_timestep:
      self.variable = self.start_value
    elif self.end_timestep >= num_timesteps > self.start_timestep:
      self.variable = self.start_value + self.coeff * (num_timesteps - self.start_timestep)
    else:
      self.variable = self.end_value

  def _on_training_start(self) -> None:
    pass

  def _on_rollout_start(self) -> None:
    self.training_env.env_method("callback", self.name, self.num_timesteps)

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    pass

  def _on_training_end(self) -> None:
    pass


# class CustomTrainLogCallback(BaseCallback):
#     """
#     Log custom values from training envs at each time step.
#     """

#     def __init__(self, name, verbose: int = 0):
#         super().__init__(verbose=verbose)
#         self.name = name

#     def _on_rollout_start(self) -> None:
#         # Get list of logging variables
#         # self._log_variables = self.training_env.get_attr("task")[0]._info["log_dict"].keys()
#         self._log_variables = self.training_env.env_method("get_logdict_keys")[0]
#         # print(f"LOG VARIABLES: {self._log_variables}")
    
#     def _on_step(self) -> bool:
#         # Check that the `_info` local variable is defined
#         for log_var_key in self._log_variables:
#           log_var_val = self.training_env.env_method("get_logdict_value", key=log_var_key)
#           # print(f"rollout/{log_var_key}: {log_var_val}")
#           log_var_val = np.mean(log_var_val)  #TODO: does not work correctly (sometimes None values are returned...)
          
#           self.logger.record(f"rollout/{log_var_key}", log_var_val)

#         return True

#     def update(self, num_timesteps):
#         pass


class EvalCallback(EventCallback):
  """
    A custom callback for evaluating an agent that derives from ``EventCallback``.
  .. warning::
    When using multiple environments, each call to  ``env.step()``
    will effectively correspond to ``n_envs`` steps.
    To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
  :param eval_env: The environment used for initialization
  :param callback_on_new_best: Callback to trigger
      when there is a new best model according to the ``mean_reward``
  :param callback_after_eval: Callback to trigger after every evaluation
  :param n_eval_episodes: The number of episodes to test the agent
  :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
  :param best_model_save_path: Path to a folder where the best model
      according to performance on the eval env will be saved.
  :param deterministic: Whether the evaluation should
      use a stochastic or deterministic actions.
  :param info_keywords: extra information to log, from the information return of env.step()
  :param render: Whether to render or not the environment during evaluation
  :param verbose: (int) Verbosity level 0: no output 1: info 2: debug
  :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
  """

  def __init__(self, eval_env, 
                callback_on_new_best: Optional[BaseCallback] = None,
                callback_after_eval: Optional[BaseCallback] = None,
                n_eval_episodes: int = 5,
                eval_freq: int = 10000,
                best_model_save_path: Optional[str] = None,
                deterministic: bool = True,
                info_keywords: tuple = (),
                render: bool = False,
                verbose: int = 1,
                warn: bool = True):
    super().__init__(callback_after_eval, verbose=verbose)
    
    self.callback_on_new_best = callback_on_new_best
    if self.callback_on_new_best is not None:
        # Give access to the parent
        self.callback_on_new_best.parent = self
    
    self.n_eval_episodes = n_eval_episodes
    self.eval_freq = eval_freq
    self.best_mean_reward = -np.inf
    self.last_mean_reward = -np.inf
    self.deterministic = deterministic
    self.info_keywords = info_keywords
    self.render = render
    self.warn = warn
    
    # Convert to VecEnv for consistency
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    self.eval_env = eval_env
    self.best_model_save_path = best_model_save_path
    self._is_success_buffer = []
    
  def _init_callback(self) -> None:
    # Does not work in some corner cases, where the wrapper is not the same
    if not isinstance(self.training_env, type(self.eval_env)):
        warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

    # Create folders if needed
    if self.best_model_save_path is not None:
        os.makedirs(self.best_model_save_path, exist_ok=True)

    # Init callback called on new best model
    if self.callback_on_new_best is not None:
        self.callback_on_new_best.init_callback(self.model)
        
  def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
    """
    Callback passed to the  ``evaluate_policy`` function
    in order to log the success rate (when applicable),
    for instance when using HER.
    :param locals_:
    :param globals_:
    """
    info = locals_["info"]

    if locals_["terminated"] or locals_["truncated"]:
        maybe_is_success = info.get("is_success")
        if maybe_is_success is not None:
            self._is_success_buffer.append(maybe_is_success)

  def _on_training_start(self) -> None:
    pass

  def _on_rollout_start(self) -> None:
    pass

  def _on_step(self) -> bool:
        
    continue_training = True

    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        # Reset success rate buffer
        self._is_success_buffer = []
        
        episode_rewards, episode_lengths, episode_customlogs = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            info_keywords=self.info_keywords,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )
        
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        mean_episode_customlogs, std_episode_customlogs = {k: np.mean(v) for k, v in episode_customlogs.items()}, {k: np.std(v) for k, v in episode_customlogs.items()}
        self.last_mean_reward = mean_reward

        if self.verbose >= 1:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            for key in episode_customlogs:
                print(f"{key}: {mean_episode_customlogs[key]:.2f} +/- {std_episode_customlogs[key]:.2f}")
        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        for key in episode_customlogs:
            self.logger.record(f"eval/mean_{key}", mean_episode_customlogs[key])

#         # Run a few episodes to evaluate progress with deterministic actions
#         det_info = self.evaluate(deterministic=True)

#         # Log evaluations
#         self.logger.record("evaluate/deterministic/ep_rew_mean", det_info[0])
#         self.logger.record("evaluate/deterministic/ep_len_mean", det_info[1])
#         self.logger.record("evaluate/deterministic/ep_targets_hit_mean", det_info[2])

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)


        # # Run a few more episodes to evaluate progress without deterministic actions
        # if self.stochastic_evals:
        #     sto_info = self.evaluate(deterministic=False)
        #     self.logger.record("evaluate/stochastic/ep_rew_mean", sto_info[0])
        #     self.logger.record("evaluate/stochastic/ep_len_mean", sto_info[1])
        #     self.logger.record("evaluate/stochastic/ep_targets_hit_mean", sto_info[2])  

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

        # mean_reward = det_info[0]
        # self.last_mean_reward = mean_reward
        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_reward
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

    return continue_training

  def _on_rollout_end(self) -> None:
    pass

  def _on_training_end(self) -> None:
    pass

#   def evaluate(self, deterministic):
#     rewards = np.zeros((self.n_eval_episodes,))
#     episode_lengths = np.zeros((self.n_eval_episodes,))
#     # episode_returns = np.zeros((self.n_eval_episodes,))
#     # episode_times = np.zeros((self.n_eval_episodes,))
#     targets_hit = np.zeros((self.n_eval_episodes,))

#     for i in range(self.n_eval_episodes):
#       obs = self.eval_env.reset()
#       terminated = False
#       truncated = False
#       while not terminated and not truncated:
#         action, _ = self.model.predict(obs, deterministic=deterministic)
#         obs, r, terminated, truncated, info = self.eval_env.step(action)
#         rewards[i] += r
#       input(info)
#       episode_lengths[i] = info["episode"]["l"]  #self.eval_env.steps
#       # episode_returns[i] = info["episode"]["r"]  #self.eval_env.steps
#       # episode_times[i] = info["episode"]["t"]  #self.eval_env.steps
#       targets_hit[i] = self.eval_env.trial_idx
#       # assert np.allclose(episode_returns[i], rewards[i])

#     return np.mean(rewards), np.mean(episode_lengths), np.mean(targets_hit)

  def update_child_locals(self, locals_: Dict[str, Any]) -> None:
    """
    Update the references to the local variables.
    :param locals_: the local variables during rollout collection
    """
    if self.callback:
        self.callback.update_locals(locals_)


class CurriculumCallback(BaseCallback):
    def __init__(self, manager, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # self.update_count += 1
        """# Get some performance metric (you must track/store this yourself)
        mean_reward = self.locals['rollout_buffer'].rewards.mean()

        if self.verbose:
            print(f"Update #{self.update_count}, mean reward: {mean_reward:.2f}")

        # Here you can modify environment difficulty
        for env_idx in range(self.training_env.num_envs):
            env = self.training_env.envs[env_idx]
            if hasattr(env, 'set_difficulty'):
                env.set_difficulty(self.update_count)"""

