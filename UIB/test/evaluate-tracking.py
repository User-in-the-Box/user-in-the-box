import gym
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage
import pickle
from pathlib import Path

from UIB.utils.logger import StateLogger, ActionLogger
from UIB.utils.functions import output_path

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'
  env.width = 120
  env.height = 80
  # Grab images
  width, height = env.metadata["imagesize"]
  img = np.flipud(env.sim.render(height=height, width=width, camera_name='for_testing'))
  ocular_img = np.flipud(env.sim.render(height=env.height, width=env.width, camera_name='oculomotor'))

  # Resample
  resample_factor = 3
  resample_height = env.height*resample_factor
  resample_width = env.width*resample_factor
  resampled_img = np.zeros((resample_height, resample_width, 3), dtype=np.uint8)
  for channel in range(3):
    resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)

  # Embed ocular image into free image
  i = height - resample_height
  j = width - resample_width
  img[i:, j:] = resampled_img

  return img


def generate_trajectory(env):
  sin_y = generate_sine_wave(env, env.target_limits_y, num_components=5)
  sin_z = generate_sine_wave(env, env.target_limits_z, num_components=5)
  return sin_y, sin_z


def generate_sine_wave(env, limits, num_components=5, min_amplitude=1, max_amplitude=5):
  min_frequency = 0.01
  max_frequency = 0.5

  # Generate a sine wave with multiple components
  t = np.arange(1000 + 1) * env.dt
  sine = np.zeros((t.size,))
  for _ in range(num_components):
    sine += env.rng.uniform(min_amplitude, max_amplitude) * \
            np.sin(env.rng.uniform(min_frequency, max_frequency) * 2 * np.pi * t + env.rng.uniform(0, 2 * np.pi))

  # Normalise to fit limits
  sine = sine - np.min(sine)
  sine = sine / np.max(sine)
  sine = limits[0] + (limits[1] - limits[0]) * sine

  return sine


def update_target_location(env, sin_y, sin_z):
  env.target_position[0] = 0
  env.target_position[1] = sin_y[env.steps]
  env.target_position[2] = sin_z[env.steps]
  env.model.body_pos[env.model._body_name2id["target"]] = env.target_origin + env.target_position
  env.sim.forward()


if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('config_file', type=str,
                      help='config file used for training the model')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='filename of a specific checkpoint (default: None, latest checkpoint is used)')
  parser.add_argument('--num_episodes', type=int, default=1,
                      help='how many episodes are evaluated (default: 1)')
  parser.add_argument('--record', action='store_true', help='enable recording')
  parser.add_argument('--out_file', type=str, default='evaluate_tracking.mp4',
                      help='output file for recording if recording is enabled (default: ./evaluate_tracking.mp4)')
  parser.add_argument('--logging', action='store_true', help='enable logging')
  parser.add_argument('--state_log_file', default='state_log',
                      help='output file for state log if logging is enabled (default: ./state_log)')
  parser.add_argument('--action_log_file', default='action_log',
                      help='output file for action log if logging is enabled (default: ./action_log)')
  args = parser.parse_args()


  # If config file is given load that
  with open(args.config_file, 'rb') as file:
    config = pickle.load(file)

  # Define output directories
  env_name = config["env_name"]
  run_folder = Path(args.config_file).parent.absolute()
  checkpoint_dir = os.path.join(run_folder, 'checkpoints')
  evaluate_dir = os.path.join(output_path(), config["env_name"], config["name"])

  # Make sure output dir exists
  os.makedirs(evaluate_dir, exist_ok=True)

  # Load latest model if filename not given
  if args.checkpoint is not None:
    model_file = args.checkpoint
  else:
    files = natural_sort(os.listdir(checkpoint_dir))
    model_file = files[-1]

  # Load policy
  print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')
  model = PPO.load(os.path.join(checkpoint_dir, model_file))

  # Load env_kwargs
  env_kwargs = config["env_kwargs"]

  # Override kwargs
  env_kwargs["action_sample_freq"] = 100
  env_kwargs["target_radius_limit"] = [0.05, 0.05]
  #env_kwargs["freq_curriculum"] = lambda : 0.0

  print(f"env_kwargs are: {env_kwargs}")

  # Initialise environment
  env = gym.make(env_name, **env_kwargs)

  # Visualise target plane
  env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.1

  if args.logging:

    # Initialise log
    state_logger = StateLogger(args.num_episodes)

    # Actions are logged separately to make things easier
    action_logger = ActionLogger(args.num_episodes)

  # Visualise evaluations
  episode_lengths = []
  num_trials = []
  rewards = []
  imgs = []
  for episode_idx in range(args.num_episodes):

    # Generate a sine wave with 5 components, different phases and amplitudes
    sin_y, sin_z = generate_trajectory(env)

    # Reset environment
    env.reset()
    done = False
    reward = 0

    update_target_location(env, sin_y, sin_z)
    obs = env.get_observation()

    if args.logging:
      state = env.get_state()
      state_logger.log(episode_idx, {**state, "termination": False, "target_hit": False})

    if args.record:
      imgs.append(grab_pip_image(env))

    # Loop until episode ends
    for i in range(sin_y.size-1):

      # Get actions from policy
      action, _states = model.predict(obs, deterministic=False)

      # Take a step
      _, r, done, info = env.step(action)
      reward += r

      if args.logging:
        action_logger.log(episode_idx, {"step": state["step"], "timestep": state["timestep"], "action": action.copy(),
                                        "ctrl": env.sim.data.ctrl.copy()})
        state = env.get_state()
        state_logger.log(episode_idx, {**state, **info})

      if args.record:
        # Visualise muscle activation
        env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl * 0.7
        imgs.append(grab_pip_image(env))

      # Update target location
      update_target_location(env, sin_y, sin_z)
      obs = env.get_observation()

  if args.logging:
    # Output log
    state_logger.save(os.path.join(evaluate_dir, args.state_log_file))
    action_logger.save(os.path.join(evaluate_dir, args.action_log_file))
    print(f'Log files have been saved files {os.path.join(evaluate_dir, args.state_log_file)}.pickle and '
          f'{os.path.join(evaluate_dir, args.action_log_file)}.pickle')

  if args.record:
    # Write the video
    env.write_video(imgs, os.path.join(evaluate_dir, args.out_file))
    print(f'A recording has been saved to file {os.path.join(evaluate_dir, args.out_file)}')