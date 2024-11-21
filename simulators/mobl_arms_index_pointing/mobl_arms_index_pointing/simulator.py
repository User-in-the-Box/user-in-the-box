import gymnasium as gym
from gymnasium import spaces
import pygame
import mujoco
import os
import numpy as np
import scipy
import matplotlib
import sys
import importlib
import shutil
import inspect
import pathlib
from datetime import datetime
import copy
from collections import defaultdict
import xml.etree.ElementTree as ET

from stable_baselines3 import PPO  #required to load a trained LLC policy in HRL approach

from .perception.base import Perception
from .utils.rendering import Camera, Context
from .utils.functions import output_path, parent_path, is_suitable_package_name, parse_yaml, write_yaml


class Simulator(gym.Env):
  """
  The Simulator class contains functionality to build a standalone Python package from a config file. The built package
   integrates a biomechanical model, a task model, and a perception model into one simulator that implements a gym.Env
   interface.
  """

  # May be useful for later, the three digit number suffix is of format X.Y.Z where X is a major version.
  version = "1.1.0"

  @classmethod
  def get_class(cls, *args):
    """ Returns a class from given strings. The last element in args should contain the class name. """
    # TODO check for incorrect module names etc
    modules = ".".join(args[:-1])
    if "." in args[-1]:
      splitted = args[-1].split(".")
      if modules == "":
        modules = ".".join(splitted[:-1])
      else:
        modules += "." + ".".join(splitted[:-1])
      cls_name = splitted[-1]
    else:
      cls_name = args[-1]
    module = cls.get_module(modules)
    return getattr(module, cls_name)

  @classmethod
  def get_module(cls, *args):
    """ Returns a module from given strings. """
    src = __name__.split(".")[0]
    modules = ".".join(args)
    return importlib.import_module(src + "." + modules)

  @classmethod
  def build(cls, config):
    """ Builds a simulator based on a config. The input 'config' may be a dict (parsed from YAML) or path to a YAML file

    Args:
      config:
        - A dict containing configuration information. See example configs in folder uitb/configs/
        - A path to a config file
    """

    # If config is a path to the config file, parse it first
    if isinstance(config, str):
      if not os.path.isfile(config):
        raise FileNotFoundError(f"Given config file {config} does not exist")
      config = parse_yaml(config)

    # Make sure required things are defined in config
    assert "simulation" in config, "Simulation specs (simulation) must be defined in config"
    assert "bm_model" in config["simulation"], "Biomechanical model (bm_model) must be defined in config"
    assert "task" in config["simulation"], "task (task) must be defined in config"

    assert "run_parameters" in config["simulation"], "Run parameters (run_parameters) must be defined in config"
    run_parameters = config["simulation"]["run_parameters"].copy()
    assert "action_sample_freq" in run_parameters, "Action sampling frequency (action_sample_freq) must be defined " \
                                                   "in run parameters"

    # Set simulator version
    config["version"] = cls.version

    # Save generated simulators to uitb/simulators
    if "simulator_folder" in config:
      simulator_folder = os.path.normpath(config["simulator_folder"])
    else:
      simulator_folder = os.path.join(output_path(), config["simulator_name"])

    # If 'package_name' is not defined use 'simulator_name'
    if "package_name" not in config:
      config["package_name"] = config["simulator_name"]
    if not is_suitable_package_name(config["package_name"]):
      raise NameError("Package name defined in the config file (either through 'package_name' or 'simulator_name') is "
                      "not a suitable Python package name. Use only lower-case letters and underscores instead of "
                      "spaces, and the name cannot start with a number.")

    # The name used in gym has a suffix -v0
    config["gym_name"] = "uitb:" + config["package_name"] + "-v0"

    # Create a simulator in the simulator folder
    cls._clone(simulator_folder, config["package_name"])

    # Load task class
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_cls.clone(simulator_folder, config["package_name"], app_executable=config["simulation"]["task"].get("kwargs", {}).get("unity_executable", None))
    simulation = task_cls.initialise(config["simulation"]["task"].get("kwargs", {}))

    # Set some compiler options
    # TODO: would make more sense to have a separate "environment" class / xml file that defines all these defaults,
    #  including e.g. cameras, lighting, etc., so that they could be easily changed. Task and biomechanical model would
    #  be integrated into that object
    compiler_defaults = {"inertiafromgeom": "auto", "balanceinertia": "true", "boundmass": "0.001",
                         "boundinertia": "0.001", "inertiagrouprange": "0 1"}
    compiler = simulation.find("compiler")
    if compiler is None:
      ET.SubElement(simulation, "compiler", compiler_defaults)
    else:
      compiler.attrib.update(compiler_defaults)

    # Load biomechanical model class
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_cls.clone(simulator_folder, config["package_name"])
    bm_cls.insert(simulation)

    # Add perception modules
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      module_cls.clone(simulator_folder, config["package_name"])
      module_cls.insert(simulation, **module_kwargs)

    # Clone also RL library files so the package will be completely standalone
    rl_cls = cls.get_class("rl", config["rl"]["algorithm"])
    rl_cls.clone(simulator_folder, config["package_name"])

    # TODO read the xml file directly from task.getroot() instead of writing it to a file first; need to input a dict
    #  of assets to mujoco.MjModel.from_xml_path
    simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")
    with open(simulation_file+".xml", 'w') as file:
      simulation.write(file, encoding='unicode')

    # Initialise the simulator
    model, _, _, _, _, _ = \
      cls._initialise(config, simulator_folder, {**run_parameters, "build": True})

    # Now that simulator has been initialised, everything should be set. Now we want to save the xml file again, but
    # mujoco only is able to save the latest loaded xml file (which is either the task or bm model xml files which are
    # are read in their __init__ functions), hence we need to read the file we've generated again before saving the
    # modified model
    mujoco.MjModel.from_xml_path(simulation_file+".xml")
    mujoco.mj_saveLastXML(simulation_file+".xml", model)

    # Save the modified model also as binary for faster loading
    mujoco.mj_saveModel(model, simulation_file+".mjcf", None)

    # Input built time into config
    config["built"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save config
    write_yaml(config, os.path.join(simulator_folder, "config.yaml"))

    return simulator_folder

  @classmethod
  def _clone(cls, simulator_folder, package_name):
    """ Create a folder for the simulator being built, and copy or create relevant files.

    Args:
       simulator_folder: Location of the simulator.
       package_name: Name of the simulator (which is a python package).
    """

    # Create the folder
    dst = os.path.join(simulator_folder, package_name)
    os.makedirs(dst, exist_ok=True)

    # Copy simulator
    src = pathlib.Path(inspect.getfile(cls))
    shutil.copyfile(src, os.path.join(dst, src.name))

    # Create __init__.py with env registration
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from .simulator import Simulator\n\n")
      file.write("from gymnasium.envs.registration import register\n")
      file.write("import pathlib\n\n")
      file.write("module_folder = pathlib.Path(__file__).parent\n")
      file.write("simulator_folder = module_folder.parent\n")
      file.write("kwargs = {'simulator_folder': simulator_folder}\n")
      file.write("register(id=f'{module_folder.stem}-v0', entry_point=f'{module_folder.stem}.simulator:Simulator', kwargs=kwargs)\n")

    # Copy utils
    shutil.copytree(os.path.join(parent_path(src), "utils"), os.path.join(simulator_folder, package_name, "utils"),
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))
    # Copy train
    shutil.copytree(os.path.join(parent_path(src), "train"), os.path.join(simulator_folder, package_name, "train"),
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))
    # Copy test
    shutil.copytree(os.path.join(parent_path(src), "test"), os.path.join(simulator_folder, package_name, "test"),
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

  @classmethod
  def _initialise(cls, config, simulator_folder, run_parameters):
    """ Initialise a simulator -- i.e., create a MjModel, MjData, and initialise all necessary variables.

    Args:
        config: A config dict.
        simulator_folder: Location of the simulator.
        run_parameters: Important run time variables that may also be used to override parameters.
    """

    # Get task class and kwargs
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_kwargs = config["simulation"]["task"].get("kwargs", {})

    # Get bm class and kwargs
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_kwargs = config["simulation"]["bm_model"].get("kwargs", {})

    # Initialise perception modules
    perception_modules = {}
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      perception_modules[module_cls] = module_kwargs

    # Get simulation file
    simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")

    # Load the mujoco model; try first with the binary model (faster, contains some parameters that may be lost when
    # re-saving xml files like body mass). For some reason the binary model fails to load in some situations (like
    # when the simulator has been built on a different computer)
    # TODO loading from binary disabled, weird problems (like a body not found from model when loaded from binary, but
    #  found correctly when model loaded from xml)
    # try:
    #  model = mujoco.MjModel.from_binary_path(simulation_file + ".mjcf")
    # except: # TODO what was the exception type
    model = mujoco.MjModel.from_xml_path(simulation_file + ".xml")

    # Initialise MjData
    data = mujoco.MjData(model)

    # Add frame skip and dt to run parameters
    run_parameters["frame_skip"] = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
    run_parameters["dt"] = model.opt.timestep*run_parameters["frame_skip"]

    # Initialise a rendering context, required for e.g. some vision modules
    run_parameters["rendering_context"] = Context(model,
                                                  max_resolution=run_parameters.get("max_resolution", [1280, 960]))

    # Initialise callbacks
    callbacks = {}
    for cb in run_parameters.get("callbacks", []):
      callbacks[cb["name"]] = cls.get_class(cb["cls"])(cb["name"], **cb["kwargs"])

    # Now initialise the actual classes; model and data are input to the inits so that stuff can be modified if needed
    # (e.g. move target to a specific position wrt to a body part)
    task = task_cls(model, data, **{**task_kwargs, **callbacks, **run_parameters})
    bm_model = bm_cls(model, data, **{**bm_kwargs, **callbacks, **run_parameters})
    perception = Perception(model, data, bm_model, perception_modules, {**callbacks, **run_parameters})

    return model, data, task, bm_model, perception, callbacks

  @classmethod
  def get(cls, simulator_folder, render_mode="rgb_array", render_mode_perception="embed", render_show_depths=False, run_parameters=None, use_cloned=True):
    """ Returns a Simulator that is located in given folder.

    Args:
      simulator_folder: Location of the simulator.
      render_mode: Whether render() will return a single rgb array (render_mode="rgb_array"),
        a list of rgb arrays (render_mode="rgb_array_list";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/render_collection.py),
        or None while the frames in a separate PyGame window are updated directly when calling
        step() or reset() (render_mode="human";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/human_rendering.py)).
      render_mode_perception: Whether images of visual perception modules should be directly embedded into main camera view ("embed"), stored as separate videos ("separate"), or not used at all [which allows to watch vision in Unity Editor if debug mode is enabled/standalone app is disabled] (None)
      render_show_depths: Whether depth images of visual perception modules should be included in rendering.
      run_parameters: Can be used to override parameters during run time.
      use_cloned: Can be useful for debugging. Set to False to use original files instead of the ones that have been
        cloned/copied during building phase.
    """

    # Read config file
    config_file = os.path.join(simulator_folder, "config.yaml")
    try:
      config = parse_yaml(config_file)
    except:
      raise FileNotFoundError(f"Could not open file {config_file}")

    # Make sure the simulator has been built
    if "built" not in config:
      raise RuntimeError("Simulator has not been built")

    # Make sure simulator_folder is in path (used to import gen_cls_cloned)
    if simulator_folder not in sys.path:
      sys.path.insert(0, simulator_folder)

    # Get Simulator class
    gen_cls_cloned = getattr(importlib.import_module(config["package_name"]), "Simulator")
    if hasattr(gen_cls_cloned, "version"):
      _legacy_mode = False
      gen_cls_cloned_version = gen_cls_cloned.version.split("-v")[-1]
    else:
      _legacy_mode = True
      gen_cls_cloned_version = gen_cls_cloned.id.split("-v")[-1]  #deprecated
    if use_cloned:
      gen_cls = gen_cls_cloned
    else:
      gen_cls = cls
      gen_cls_version = gen_cls.version.split("-v")[-1]

      if gen_cls_version.split(".")[0] > gen_cls_cloned_version.split(".")[0]:
        raise RuntimeError(
          f"""Severe version mismatch. The simulator '{config["simulator_name"]}' has version {gen_cls_cloned_version}, while your uitb package has version {gen_cls_version}.\nTo run with version {gen_cls_cloned_version}, set 'use_cloned=True'.""")
      elif gen_cls_version.split(".")[1] > gen_cls_cloned_version.split(".")[1]:
        print(
          f"""WARNING: Version mismatch. The simulator '{config["simulator_name"]}' has version {gen_cls_cloned_version}, while your uitb package has version {gen_cls_version}.\nTo run with version {gen_cls_version}, set 'use_cloned=True'.""")

    if _legacy_mode:
      _simulator = gen_cls(simulator_folder, run_parameters=run_parameters)
    else:
      try:
        _simulator = gen_cls(simulator_folder, render_mode=render_mode, render_mode_perception=render_mode_perception, render_show_depths=render_show_depths,
                          run_parameters=run_parameters)
      except TypeError:
        _simulator = gen_cls(simulator_folder, render_mode=render_mode, render_show_depths=render_show_depths,
                          run_parameters=run_parameters)

    # Return Simulator object
    return _simulator

  def __init__(self, simulator_folder, render_mode="rgb_array", render_mode_perception="embed", render_show_depths=False, run_parameters=None):
    """ Initialise a new `Simulator`.

    Args:
      simulator_folder: Location of a simulator.
      render_mode: Whether render() will return a single rgb array (render_mode="rgb_array"),
        a list of rgb arrays (render_mode="rgb_array_list";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/render_collection.py),
        or None while the frames in a separate PyGame window are updated directly when calling
        step() or reset() (render_mode="human";
        adapted from https://github.com/openai/gym/blob/master/gym/wrappers/human_rendering.py)).
      render_mode_perception: Whether images of visual perception modules should be directly embedded into main camera view ("embed"), stored as separate videos ("separate"), or not used at all [which allows to watch vision in Unity Editor if debug mode is enabled/standalone app is disabled] (None)
      render_show_depths: Whether depth images of visual perception modules should be included in rendering.
      run_parameters: Can be used to override parameters during run time.
    """

    # Make sure simulator exists
    if not os.path.exists(simulator_folder):
      raise FileNotFoundError(f"Simulator folder {simulator_folder} does not exists")
    self._simulator_folder = simulator_folder

    # Read config
    self._config = parse_yaml(os.path.join(self._simulator_folder, "config.yaml"))

    # Get run parameters: these parameters can be used to override parameters used during training
    self._run_parameters = self._config["simulation"]["run_parameters"].copy()
    self._run_parameters.update(run_parameters or {})

    # Initialise simulation
    self._model, self._data, self.task, self.bm_model, self.perception, self.callbacks = \
      self._initialise(self._config, self._simulator_folder, self._run_parameters)

    # Set action space TODO for now we assume all actuators have control signals between [-1, 1]
    self.action_space = self._initialise_action_space()

    # Set observation space
    self.observation_space = self._initialise_observation_space()

    # Collect some episode statistics
    self._episode_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    # Initialise viewer
    self._GUI_camera = Camera(self._run_parameters["rendering_context"], self._model, self._data, camera_id='for_testing',
                         dt=self._run_parameters["dt"])

    self._render_mode = render_mode
    self._render_mode_perception = render_mode_perception  #whether perception camera views should be directly embedded into camera view of camera_id ("embed"), stored in self._render_stack_perception ("separate"), or not used at all "separate"), or not used at all [which allows to watch vision in Unity Editor if debug mode is enabled/standalone app is disabled] (None)
    self._render_stack = []  #only used if render_mode == "rgb_array_list"
    self._render_stack_perception = defaultdict(list)  #only used if render_mode == "rgb_array_list" and self._render_mode_perception == "separate"
    self._render_stack_pop = True  #If True, clear the render stack after .render() is called.
    self._render_stack_clean_at_reset = True  #If True, clear the render stack when .reset() is called.
    self._render_show_depths = render_show_depths  #If True, depth images of visual perception modules are included in GUI rendering.
    self._render_screen_size = None  #only used if render_mode == "human"
    self._render_window = None  #only used if render_mode == "human"
    self._render_clock = None  #only used if render_mode == "human"

    if 'llc' in self.config:  #if HRL approach is used
        llc_simulator_folder = os.path.join(output_path(), self.config["llc"]["simulator_name"])
        if llc_simulator_folder not in sys.path:
            sys.path.insert(0, llc_simulator_folder)
        if not os.path.exists(llc_simulator_folder):
            raise FileNotFoundError(f"Simulator folder {llc_simulator_folder} does not exists")
        llccheckpoint_dir = os.path.join(llc_simulator_folder, 'checkpoints')
        # Load policy TODO should create a load method for uitb.rl.BaseRLModel
        print(f'Loading model: {os.path.join(llccheckpoint_dir, self.config["llc"]["checkpoint"])}\n')
        self.llc_model = PPO.load(os.path.join(llccheckpoint_dir, self.config["llc"]["checkpoint"]))
        self.action_space = self._initialise_HRL_action_space()
        self._max_steps = self.config["llc"]["llc_ratio"]
        self._dwell_threshold = int(0.5*self._max_steps)
        self._target_radius = 0.05
        self._independent_dofs = []
        self._independent_joints = []
        joints = self.config["llc"]["joints"]
        for joint in joints:
          joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint)
          if self._model.jnt_type[joint_id] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            raise NotImplementedError(f"Only 'hinge' and 'slide' joints are supported, joint "
                                  f"{joint} is of type {mujoco.mjtJoint(self._model.jnt_type[joint_id]).name}")
          self._independent_dofs.append(self._model.jnt_qposadr[joint_id])
          self._independent_joints.append(joint_id)
        self._jnt_range = self._model.jnt_range[self._independent_joints]


    #To normalize joint ranges for llc
  def _normalise_qpos(self, qpos):
    # Normalise to [0, 1]
    qpos = (qpos - self._jnt_range[:, 0]) / (self._jnt_range[:, 1] - self._jnt_range[:, 0])
    # Normalise to [-1, 1]
    qpos = (qpos - 0.5) * 2
    return qpos

  def _initialise_action_space(self):
    """ Initialise action space. """
    num_actuators = self.bm_model.nu + self.perception.nu
    actuator_limits = np.ones((num_actuators,2)) * np.array([-1.0, 1.0])
    return spaces.Box(low=np.float32(actuator_limits[:, 0]), high=np.float32(actuator_limits[:, 1]))

  def _initialise_HRL_action_space(self):
    bm_nu = self.bm_model.nu
    bm_jnt_range = np.ones((bm_nu,2)) * np.array([-1.0, 1.0])
    perception_nu = self.perception.nu
    perception_jnt_range = np.ones((perception_nu,2)) * np.array([-1.0, 1.0])
    jnt_range = np.concatenate((bm_jnt_range, perception_jnt_range), axis=0)
    action_space = gym.spaces.Box(low=jnt_range[:,0], high=jnt_range[:,1])
    return action_space

  def _initialise_observation_space(self):
    """ Initialise observation space. """
    observation = self.get_observation()
    obs_dict = dict()
    for module in self.perception.perception_modules:
      obs_dict[module.modality] = spaces.Box(dtype=np.float32, **module.get_observation_space_params())
    if "stateful_information" in observation:
      obs_dict["stateful_information"] = spaces.Box(dtype=np.float32,
                                                    **self.task.get_stateful_information_space_params())
    return spaces.Dict(obs_dict)

  def _get_qpos(self, model, data):
    qpos = data.qpos[self._independent_dofs].copy()
    qpos = self._normalise_qpos(qpos)
    return qpos

  def step(self, action):
    """ Step simulation forward with given actions.

    Args:
      action: Actions sampled from a policy. Limited to range [-1, 1].
    """
    if 'llc' in self.config:  #if HRL approach is used
        self.task._target_qpos = action # action to pass to LLC
        self._steps = 0 # Initialise loop control to 0
        #acc_reward = 0 #To be used when rewards are being accumulated in llc steps

        while self._steps < self._max_steps: # loop for llc controls based on llc_ratio

            llc_action, _states = self.llc_model.predict(self.get_llcobservation(action), deterministic=True) # Get BM action from LLC
            # Set control for the bm model
            self.bm_model.set_ctrl(self._model, self._data, llc_action)

            # Set control for perception modules (e.g. eye movements)
            self.perception.set_ctrl(self._model, self._data, action[self.bm_model.nu:])

            # Advance the simulation
            mujoco.mj_step(self._model, self._data, nstep=int(self._run_parameters["frame_skip"])) # Number of timesteps to skip for LLC

            # Update bm model (e.g. update constraints); updates also effort model
            self.bm_model.update(self._model, self._data)

            # Update perception modules
            self.perception.update(self._model, self._data)

            dist = np.abs(action - self._get_qpos(self._model, self._data))

            # Update environment        
            reward, terminated, truncated, info = self.task.update(self._model, self._data)

            # Add an effort cost to reward
            reward -= self.bm_model.get_effort_cost(self._model, self._data)

            #acc_reward += reward #To be used when rewards are being accumulated in llc steps

            # Get observation
            obs = self.get_observation()

            # Add frame to stack
            if self._render_mode == "rgb_array_list":
              self._render_stack.append(self._GUI_rendering())
            elif self._render_mode == "human":
              self._GUI_rendering_pygame()

            if truncated or terminated:
                break

            # Pointing
            if "target_spawned" in info:
                if info["target_spawned"] or info["target_hit"]:
                    break

            # Choice Reaction
            elif "new_button_generated" in info:
                if info["new_button_generated"] or info["target_hit"]:
                    break

            self._steps += 1
            if np.all(dist < self._target_radius):
                break

        return obs, reward, terminated, truncated, info

    else:
            # Set control for the bm model
        self.bm_model.set_ctrl(self._model, self._data, action[:self.bm_model.nu])

        # Set control for perception modules (e.g. eye movements)
        self.perception.set_ctrl(self._model, self._data, action[self.bm_model.nu:])

        # Advance the simulation
        mujoco.mj_step(self._model, self._data, nstep=self._run_parameters["frame_skip"])

        # Update bm model (e.g. update constraints); updates also effort model
        self.bm_model.update(self._model, self._data)

        # Update perception modules
        self.perception.update(self._model, self._data)

        # Update environment
        reward, terminated, truncated, info = self.task.update(self._model, self._data)

        # Add an effort cost to reward
        effort_cost = self.bm_model.get_effort_cost(self._model, self._data)
        info["EffortCost"] = effort_cost
        reward -= effort_cost

        # Get observation
        obs = self.get_observation(info)

        # Add frame to stack
        if self._render_mode == "rgb_array_list":
          self._render_stack.append(self._GUI_rendering())
        elif self._render_mode == "human":
          self._GUI_rendering_pygame()

        return obs, reward, terminated, truncated, info


  def get_observation(self, info=None):
    """ Returns an observation from the perception model.

    Returns:
      A dict with observations from individual perception modules. May also contain stateful information from a task.
    """

    # Get observation from perception
    observation = self.perception.get_observation(self._model, self._data, info)

    # Add any stateful information that is required
    stateful_information = self.task.get_stateful_information(self._model, self._data)
    if stateful_information.size > 0:  #TODO: define stateful_information (and encoder) that can be used as default, if no stateful information is provided (zero-size arrays do not work with sb3 currently...)
      observation["stateful_information"] = stateful_information

    return observation

  def get_llcobservation(self,action):
    """ Returns an observation from the perception model.

    Returns:
      A dict with observations from individual perception modules. May also contain stateful information from a task.
    """

    # Get observation from perception
    observation = self.perception.get_observation(self._model, self._data)
    
    # Remove Vision for LLC
    observation.pop("vision")
    qpos = self._get_qpos(self._model, self._data)
    qpos_diff = action - qpos
    
    # Stateful Information for LLC policy
    stateful_information = qpos_diff
    if stateful_information is not None:
      observation["stateful_information"] = stateful_information
    
    return observation


  def reset(self, seed=None):
    """ Reset the simulator and return an observation. """

    super().reset(seed=seed)

    # Reset sim
    mujoco.mj_resetData(self._model, self._data)

    # Reset all models
    self.bm_model.reset(self._model, self._data)
    self.perception.reset(self._model, self._data)
    info = self.task.reset(self._model, self._data)

    # Do a forward so everything will be set
    mujoco.mj_forward(self._model, self._data)

    if self._render_mode == "rgb_array_list":
      if self._render_stack_clean_at_reset:
        self._render_stack = []
        self._render_stack_perception = defaultdict(list)
      self._render_stack.append(self._GUI_rendering())
    elif self._render_mode == "human":
      self._GUI_rendering_pygame()

    return self.get_observation(), info

  def render(self):
    if self._render_mode == "rgb_array_list":
      render_stack = self._render_stack
      if self._render_stack_pop:
        self._render_stack = []
      return render_stack
    elif self._render_mode == "rgb_array":
      return self._GUI_rendering()
    else:
      return None
    
  def get_render_stack_perception(self):
      render_stack_perception = self._render_stack_perception
      # if self._render_stack_pop:
      #   self._render_stack_perception = defaultdict(list)
      return render_stack_perception

  def _GUI_rendering(self):
    # Grab an image from the 'for_testing' camera and grab all GUI-prepared images from included visual perception modules, and display them 'picture-in-picture'

    # Grab images
    img, _ = self._GUI_camera.render()

    if self._render_mode_perception == "embed":
      # Embed perception camera images into main camera image
        
      # perception_camera_images = [rgb_or_depth_array for camera in self.perception.cameras
      #                             for rgb_or_depth_array in camera.render() if rgb_or_depth_array is not None]
      perception_camera_images = self.perception.get_renders()

      # TODO: add text annotations to perception camera images
      if len(perception_camera_images) > 0:
        _img_size = img.shape[:2]  #(height, width)


        # Vertical alignment of perception camera images, from bottom right to top right
        ## TODO: allow for different inset locations
        _desired_subwindow_height = np.round(_img_size[0] / len(perception_camera_images)).astype(int)
        _maximum_subwindow_width = np.round(0.2 * _img_size[1]).astype(int)

        perception_camera_images_resampled = []
        for ocular_img in perception_camera_images:
          # Convert 2D depth arrays to 3D heatmap arrays
          if ocular_img.ndim == 2:
            if self._render_show_depths:
              ocular_img = matplotlib.pyplot.imshow(ocular_img, cmap=matplotlib.pyplot.cm.jet, interpolation='bicubic').make_image('TkAgg', unsampled=True)[0][
              ..., :3]
              matplotlib.pyplot.close()  #delete image
            else:
              continue

          resample_factor = min(_desired_subwindow_height / ocular_img.shape[0], _maximum_subwindow_width / ocular_img.shape[1])

          resample_height = np.round(ocular_img.shape[0] * resample_factor).astype(int)
          resample_width = np.round(ocular_img.shape[1] * resample_factor).astype(int)
          resampled_img = np.zeros((resample_height, resample_width, ocular_img.shape[2]), dtype=np.uint8)
          for channel in range(ocular_img.shape[2]):
            resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)

          perception_camera_images_resampled.append(resampled_img)

        ocular_img_bottom = _img_size[0]
        for ocular_img_idx, ocular_img in enumerate(perception_camera_images_resampled):
          #print(f"Modify ({ocular_img_bottom - ocular_img.shape[0]}, { _img_size[1] - ocular_img.shape[1]})-({ocular_img_bottom}, {img.shape[1]}).")
          img[ocular_img_bottom - ocular_img.shape[0]:ocular_img_bottom, _img_size[1] - ocular_img.shape[1]:] = ocular_img
          ocular_img_bottom -= ocular_img.shape[0]
        # input((len(perception_camera_images_resampled), perception_camera_images_resampled[0].shape, img.shape))
    elif self._render_mode_perception == "separate":
      for module, camera_list in self.perception.cameras_dict.items():
        for camera in camera_list:
          for rgb_or_depth_array in camera.render():
            if rgb_or_depth_array is not None:
              self._render_stack_perception[f"{module.modality}/{type(camera).__name__}"].append(rgb_or_depth_array)

    return img

  def _GUI_rendering_pygame(self):
    rgb_array = np.transpose(self._GUI_rendering(), axes=(1, 0, 2))

    if self._render_screen_size is None:
      self._render_screen_size = rgb_array.shape[:2]

    assert self._render_screen_size == rgb_array.shape[
                                       :2], f"Expected an rgb array of shape {self._render_screen_size} from self._GUI_camera, but received an rgb array of shape {rgb_array.shape[:2]}. "

    if self._render_window is None:
      pygame.init()
      pygame.display.init()
      self._render_window = pygame.display.set_mode(self._render_screen_size)

    if self._render_clock is None:
      self._render_clock = pygame.time.Clock()

    surf = pygame.surfarray.make_surface(rgb_array)
    self._render_window.blit(surf, (0, 0))
    pygame.event.pump()
    self._render_clock.tick(self.fps)
    pygame.display.flip()

  def close(self):
    """ Close the rendering window (if self._render_mode == 'human')."""
    super().close()
    if self._render_window is not None:
      import pygame

      pygame.display.quit()
      pygame.quit()

  @property
  def fps(self):
    return self._GUI_camera._fps

  def callback(self, callback_name, num_timesteps):
    """ Update a callback -- may be useful during training, e.g. for curriculum learning. """
    self.callbacks[callback_name].update(num_timesteps)

  def update_callbacks(self, num_timesteps):
    """ Update all callbacks. """
    for callback_name in self.callbacks:
      self.callback(callback_name, num_timesteps)

  # def get_logdict_keys(self):
  #   return list(self.task._info["log_dict"].keys())

  # def get_logdict_value(self, key):
  #   return self.task._info["log_dict"].get(key)

  @property
  def config(self):
    """ Return config. """
    return copy.deepcopy(self._config)

  @property
  def run_parameters(self):
    """ Return run parameters. """
    # Context cannot be deep copied
    exclude = {"rendering_context"}
    run_params = {k: copy.deepcopy(self._run_parameters[k]) for k in self._run_parameters.keys() - exclude}
    run_params["rendering_context"] = self._run_parameters["rendering_context"]
    return run_params

  @property
  def simulator_folder(self):
    """ Return simulator folder. """
    return self._simulator_folder

  @property
  def render_mode(self):
    """ Return render mode. """
    return self._render_mode

  def get_state(self):
    """ Return a state of the simulator / individual components (biomechanical model, perception model, task).

    This function is used for logging/evaluation purposes, not for RL training.

    Returns:
      A dict with one float or numpy vector per keyword.
    """

    # Get time, qpos, qvel, qacc, act_force, act, ctrl of the current simulation
    state = {"timestep": self._data.time,
             "qpos": self._data.qpos.copy(),
             "qvel": self._data.qvel.copy(),
             "qacc": self._data.qacc.copy(),
             "act_force": self._data.actuator_force.copy(),
             "act": self._data.act.copy(),
             "ctrl": self._data.ctrl.copy()}

    # Add state from the task
    state.update(self.task.get_state(self._model, self._data))

    # Add state from the biomechanical model
    state.update(self.bm_model.get_state(self._model, self._data))

    # Add state from the perception model
    state.update(self.perception.get_state(self._model, self._data))

    return state

  def close(self, **kwargs):
    """ Perform any necessary clean up.

    This function is inherited from gym.Env. It should be automatically called when this object is garbage collected
     or the program exists, but that doesn't seem to be the case. This function will be called if this object has been
     initialised in the context manager fashion (i.e. using the 'with' statement). """
    self.task.close(**kwargs)
    self.perception.close(**kwargs)
    self.bm_model.close(**kwargs)
