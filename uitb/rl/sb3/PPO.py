import os
import importlib
import numpy as np
import pathlib

from stable_baselines3 import PPO as PPO_sb3
from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback  #, EvalCallback

from typing import TypeVar
import sys, time
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import safe_mean
SelfPPO = TypeVar("SelfPPO", bound="PPO")

from typing import Any, Dict, Optional, SupportsFloat, Tuple
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from collections import defaultdict

from typing import  Callable, Optional, Type, Union
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from ..base import BaseRLModel
from .callbacks import EvalCallback


class PPO(BaseRLModel):

  def __init__(self, simulator, checkpoint_path=None, wandb_id=None, info_keywords=()):
    super().__init__()

    rl_config = self.load_config(simulator)
    run_parameters = simulator.run_parameters
    simulator_folder = simulator.simulator_folder

    # Get total timesteps
    self.total_timesteps = rl_config["total_timesteps"]

    # Combine info keywords to be logged from config and from passed kwarg.
    # Note: Each entry of info_keywords needs to be of type Tuple[str, str], 
    # with the variable name as first and the episode operation (e.g., "sum", "mean", "final" (default)) as second string
    self.info_keywords = tuple({tuple(k) for k in (run_parameters.get("info_keywords", []) + list(info_keywords))})
    
    # Initialise parallel envs
    self.n_envs = rl_config["num_workers"]
    Monitor = Monitor_customops  #use Monitor_customops instead of Monitor class in make_vec_env
    parallel_envs = make_vec_env(simulator.__class__, n_envs=self.n_envs,
                                 seed=run_parameters.get("random_seed", None), vec_env_cls=SubprocVecEnv,
                                 monitor_kwargs={"info_keywords": self.info_keywords},
                                 env_kwargs={"simulator_folder": simulator_folder})
    
    if checkpoint_path is not None:
        # Resume training
        self.model = PPO_sb3_customlogs.load(checkpoint_path, parallel_envs, verbose=1, #policy_kwargs=rl_config["policy_kwargs"], 
                                  tensorboard_log=simulator_folder, n_steps=rl_config["nsteps"],
                                  batch_size=rl_config["batch_size"], target_kl=rl_config["target_kl"],
                                  learning_rate=rl_config["lr"], device=rl_config["device"])
        self.training_resumed = True
    else:
        # Add feature and stateful information encoders to policy_kwargs
        encoders = simulator.perception.encoders
        if simulator.task.get_stateful_information_space_params()["shape"] != (0,):
          #TODO: define stateful_information (and encoder) that can be used as default, if no stateful information is provided (zero-size array do not work with sb3 currently...)
          encoders["stateful_information"] = simulator.task.stateful_information_encoder
        rl_config["policy_kwargs"]["features_extractor_kwargs"] = {"encoders": encoders}
        rl_config["policy_kwargs"]["wandb_id"] = wandb_id

        # Initialise model
        self.model = PPO_sb3_customlogs(rl_config["policy_type"], parallel_envs, verbose=1, policy_kwargs=rl_config["policy_kwargs"],
                             tensorboard_log=simulator_folder, n_steps=rl_config["nsteps"],
                             batch_size=rl_config["batch_size"], target_kl=rl_config["target_kl"],
                             learning_rate=rl_config["lr"], device=rl_config["device"])
        self.training_resumed = False

        if "policy_init" in rl_config:
            params = os.path.join(pathlib.Path(__file__).parent, rl_config["policy_init"])
            self.model.policy.load_from_vector(np.load(params))
    
    # Create a checkpoint callback
    save_freq = rl_config["save_freq"] // self.n_envs
    checkpoint_folder = os.path.join(simulator_folder, 'checkpoints')
    self.checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                  save_path=checkpoint_folder,
                                                  name_prefix='model',
                                                  save_replay_buffer=True,
                                                  save_vecnormalize=True)

    # Get callbacks as a list
    self.callbacks = [*simulator.callbacks.values()]
    
    # Create an evaluation env (only used if eval_callback=True is passed to learn())
    self.eval_env = simulator.__class__(**{"simulator_folder": simulator_folder})
    
  def load_config(self, simulator):
    config = simulator.config["rl"]

    # Need to translate strings into classes
    config["policy_type"] = simulator.get_class("rl.sb3", config["policy_type"])

    if "activation_fn" in config["policy_kwargs"]:
      mods = config["policy_kwargs"]["activation_fn"].split(".")
      config["policy_kwargs"]["activation_fn"] = getattr(importlib.import_module(".".join(mods[:-1])), mods[-1])

    config["policy_kwargs"]["features_extractor_class"] = \
      simulator.get_class("rl.sb3", config["policy_kwargs"]["features_extractor_class"])

    if "lr" in config:
      if isinstance(config["lr"], dict):
        config["lr"] = simulator.get_class("rl.sb3", config["lr"]["function"])(**config["lr"]["kwargs"])

    return config

  def learn(self, callbacks, with_evaluation=False, eval_freq=400000, n_eval_episodes=5, eval_info_keywords=()):
    if with_evaluation:
        self.eval_env = Monitor(self.eval_env, info_keywords=eval_info_keywords)
        self.eval_freq = eval_freq // self.n_envs
        self.eval_callback = EvalCallback(self.eval_env, eval_freq=self.eval_freq, n_eval_episodes=n_eval_episodes, info_keywords=eval_info_keywords)

        self.model.learn(total_timesteps=self.total_timesteps,
                         callback=callbacks + [self.checkpoint_callback, self.eval_callback, *self.callbacks],
                         info_keywords=self.info_keywords,
                         reset_num_timesteps=not self.training_resumed)
    else:
        self.model.learn(total_timesteps=self.total_timesteps,
                         callback=callbacks + [self.checkpoint_callback, *self.callbacks],
                         info_keywords=self.info_keywords,
                         reset_num_timesteps=not self.training_resumed)

class PPO_sb3_customlogs(PPO_sb3):
   def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        info_keywords : tuple = (),
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    for keyword, operation in info_keywords:
                          self.logger.record(f"rollout/ep_{keyword}_{operation}", safe_mean([ep_info[keyword] for ep_info in self.ep_info_buffer if keyword in ep_info]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

class Monitor_customops(Monitor):
    """    
    Modified monitor wrapper for Gym environments, which allows to accumulate logged values per episode (e.g., store sum or mean of a logged variable per episode).
    To this end, info_keywords is a tuple containing (str, str) tuples, with variable name as first string and episode operation (e.g., "sum", "mean", or "final" (default)) as second string.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step() [see note above]
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """
    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[Tuple[str, str], ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env=env, filename=filename, allow_early_resets=allow_early_resets, reset_keywords=reset_keywords, info_keywords=info_keywords, override_existing=override_existing)


    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor_customops(env, path, allow_early_resets=True)"
            )
        self.info_keywords_acc_valuedict = defaultdict(list)
        return super().reset(**kwargs)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.rewards.append(float(reward))
        for key, op in self.info_keywords:
            if op in ["sum", "mean"]:
                self.info_keywords_acc_valuedict[key].append(float(info[key]))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key, op in self.info_keywords:
                if op == "sum":
                   ep_info[key] = sum(self.info_keywords_acc_valuedict[key])
                elif op == "mean":
                   ep_info[key] = safe_mean(self.info_keywords_acc_valuedict[key])
                else:
                  ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor_customops wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
        Note: the wrapper specified by this parameter will be applied after the ``Monitor_customops`` wrapper.
        if some cases (e.g. with TimeLimit wrapper) this can lead to undesired behavior.
        See here for more details: https://github.com/DLR-RM/stable-baselines3/issues/894
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor_customops`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor_customops(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

