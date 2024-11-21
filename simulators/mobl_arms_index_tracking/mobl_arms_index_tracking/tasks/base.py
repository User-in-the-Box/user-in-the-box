from abc import ABC, abstractmethod
import os
import shutil
import inspect
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import pathlib
import importlib
from typing import final

from ..utils.functions import parent_path
from ..utils.functions import importer
from ..rl.encoders import BaseEncoder


class BaseTask(ABC):

  def __init__(self, model, data, **kwargs):
    """ Initialises a new `BaseTask`.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      **kwargs: Many kwargs that should be documented somewhere.
    """

    # Initialise mujoco model of the task, easier to manipulate things
    task_model = mujoco.MjModel.from_xml_path(self.get_xml_file())

    # Get action sample freq
    self._action_sample_freq = kwargs["action_sample_freq"]

    # Get dt
    self._dt = kwargs["dt"]

    # Get an rng
    self._rng = np.random.default_rng(kwargs.get("random_seed", None))

    # Get actuator names and joint names (if any)
    self._actuator_names = [mujoco.mj_id2name(task_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(task_model.nu)]
    self._joint_names = [mujoco.mj_id2name(task_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(task_model.njnt)]

    # Find actuator indices in the simulation
    self._actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                      for actuator_name in self._actuator_names]

    # Find joint indices in the simulation
    self._joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                   for joint_name in self._joint_names]

    # Shape of stateful information will be set later
    self._stateful_information_shape = None

    # Keep track of simulated steps
    self._steps = 0

    # # Used for logging states
    # self._info = {"terminated": False,
    #               "truncated": False,
    #               "log_dict": {}}


  def __init_subclass__(cls, *args, **kwargs):
    """ Define a new __init__ method with a hook that automatically sets stateful information shape after a child
    instance has been initialised. This is only for convenience, otherwise we would need to set the stateful information
    shape separately in each child class constructor, or after a a child of BaseTask has been initialised."""
    super().__init_subclass__(*args, **kwargs)
    def init_with_hook(self, model, data, init=cls.__init__, **init_kwargs):
      init(self, model, data, **init_kwargs)

      # Set stateful information shape
      self._stateful_information_shape = self.get_stateful_information(model, data).shape

      # Initialise encoder: use Identity encoder by default (which is no-op) if encoder has not been defined in config
      encoder_info = init_kwargs.get("stateful_information_encoder", None) or self._default_stateful_information_encoder
      self._initialise_stateful_information_encoder(encoder_info)

    cls.__init__ = init_with_hook


  ############ The methods below you should definitely overwrite as they are important ############

  @abstractmethod
  def _update(self, model, data):
    """ Updates the task/environment after a step.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.

    Returns:
      A float indicating the reward received from the task/environment, a boolean indicating whether the episode
        has terminated (e.g., episode success or failure), a boolean indicating whether the episode has been
        truncated (e.g., timeout), and a dict containing information about the states of the task/environment
    """
    pass

  @abstractmethod
  def _reset(self, model, data):
    """ Resets the task/environment.

    Returns:
      A dict containing information about the states of the task/environment
    """
    return dict()


  ############ The methods below are overwritable but often don't need to be overwritten ############

  def get_stateful_information(self, model, data):
    """ Returns stateful information pertinent to a task (like time left to achieve the task).

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.

    Returns:
      * An empty array if no stateful information is used for RL training
      * A numpy array, typically a vector but can be higher dimensional as well. If higher dimensional, then the
        property 'stateful_information_encoder' must be overwritten to map the higher dimensional array into a vector.

    """
    return np.array([])

  def _get_stateful_information_range(self):
    """ Return limits for stateful information. These limits aren't currently used for anything (AFAIK, not in gym or
    stable-baselines3; only to initialise the observation space required by gym.Env), so let's just use a default of
    -inf to inf. Overwrite this method to use different ranges.

    Returns:
        A dict with format {"low": float-or-array, "high": float-or-array} where the values indicate lowest and highest
          values the observation can have. The values can be floats or numpy arrays -- if arrays, they must have the
          same shape as the returned observation from method 'get_observation'
    """
    return {"low": float('-inf'), "high": float('inf')}

  @property
  def _default_stateful_information_encoder(self):
    """ Returns default encoder info as a dict. Overwrite this property in derived task class to use different default
     encoders for stateful information. """
    return {"module": "rl.encoders", "cls": "Identity"}

  def _get_state(self, model, data):
    """ Return the state of the task/environment. These states are used only for logging/evaluation, not for RL
    training

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.

    Returns:
      A dict where each key should have a float or a numpy vector as their value
    """
    return dict()

  @classmethod
  def get_xml_file(cls):
    """ Overwrite if you want to call the task xml file something else than 'task.xml'. """
    return os.path.join(parent_path(inspect.getfile(cls)), "task.xml")

  @classmethod
  def _get_reward_function(cls, specs, module_name="reward_functions"):
    """ Returns a reward function. One does not need to use this method when creating new tasks, it's just for
    convenience.

    Args:
      specs: Specifications of a reward function class, in format
        {"cls": "name-of-class", "kwargs": {"kw1": value1, "kw2": value2}}
      module_name: Name of the module, defaults to 'reward_functions'
    """
    module = importlib.import_module(".".join(cls.__module__.split(".")[:-1]) + f".{module_name}")
    return getattr(module, specs["cls"])(**specs.get("kwargs", {}))

  @classmethod
  def clone(cls, simulator_folder, package_name, app_executable=None):
    """ Clones (i.e. copies) the relevant python files into a new location.

    Args:
       simulator_folder: Location of the simulator.
       package_name: Name of the simulator (which is a python package)
       app_executable: (relative) path of app executable, if an external application
        is used by the task instance (e.g., a Unity app)
    """

    # Create 'tasks' folder
    dst = os.path.join(simulator_folder, package_name, "tasks")
    os.makedirs(dst, exist_ok=True)

    # Copy this file and __init__.py
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__)

    # Copy env folder (without apps subdirectory)
    src = parent_path(inspect.getfile(cls))
    # shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True, ignore=shutil.ignore_patterns('apps', '*.pyc'))

    # Copy application subdir (optional)
    if app_executable is not None:
      src_app = parent_path(os.path.join(src, app_executable))
      shutil.copytree(src_app, os.path.join(dst, src.stem, os.path.dirname(app_executable)), dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

    # Copy assets if they exist
    if os.path.isdir(os.path.join(src, "assets")):
      shutil.copytree(os.path.join(src, "assets"), os.path.join(simulator_folder, package_name, "assets"),
                      dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

  @classmethod
  def initialise(cls, task_kwargs):
    """ Initialise the simulator xml file into which bm_model will be integrated.

    Args:
      task_kwargs: kwargs for the task class that inherits from this base class.

    Returns:
      An `xml.etree.ElementTree` object which will be the base for the newly created simulator.

    """
    # Parse xml file and return the tree
    return ET.parse(cls.get_xml_file())

  def close(self, **kwargs):
    pass

  ############ The methods below you should not overwrite ############

  @final
  def update(self, model, data):
    """ Keeps track of how many steps have been taken (usually required to e.g. determine when a task has ended) and
    updates the task/environment. """
    self._steps += 1
    return self._update(model, data)

  @final
  def reset(self, model, data):
    """ Resets the number of steps taken and the task/environment. """
    self._steps = 0
    info = self._reset(model, data)
    mujoco.mj_forward(model, data)
    return info

  @final
  def get_state(self, model, data):
    """ Returns the number of steps taken as well as any relevant states of the task/environment. """
    state = {"steps": self._steps}
    state.update(self._get_state(model, data))
    return state

  @final
  def get_stateful_information_space_params(self):
    """ Returns stateful information space parameters. """
    return {**self._get_stateful_information_range(), "shape": self._stateful_information_shape}

  @final
  def _initialise_stateful_information_encoder(self, encoder_info):
    """ Import an encoder. We assume the encoder inherits from uitb.rl.encoders.BaseEncoder. """
    encoder_cls = importer(encoder_info)
    if not issubclass(encoder_cls, BaseEncoder):
      raise RuntimeError(f"Encoder {encoder_cls} does not inherit from uitb.rl.encoders.BaseEncoder")
    self._stateful_information_encoder = \
      encoder_cls(observation_shape=self._stateful_information_shape, **encoder_info.get("kwargs", {}))

  @final
  @property
  def stateful_information_encoder(self):
    """ If 'get_stateful_information' returns a higher dimensional numpy array, then this method must return an encoder
      (e.g. a PyTorch neural network) to map it into a vector. """
    return self._stateful_information_encoder
