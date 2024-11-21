from abc import ABC, abstractmethod
import os
import shutil
import inspect
import numpy as np
import mujoco
import pathlib
from typing import final

from ..utils.functions import parent_path
from ..utils.functions import importer
from ..rl.encoders import BaseEncoder


class BaseModule(ABC):

  def __init__(self, model, data, bm_model, **kwargs):
    """ Initialises aF new `BaseModule`. One module represents one perception capability.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      bm_model: An instance which inherits from the uitb.bm_models.base.BaseBMModel class
      **kwargs: Many kwargs that should be documented somewhere.
    """

    self._bm_model = bm_model
    self._actuator_names = []
    self._joint_names = []

    # Get an rng
    self._rng = np.random.default_rng(kwargs.get("random_seed", None))

    # Get modality  TODO this modality layer is somewhat unnecessary and may be removed in the future
    self._modality = parent_path(inspect.getfile(self.__class__)).parent.stem

    # Observation shape and encoder will be set in the subclass hook
    self._observation_shape = None
    self._encoder = None

    # List of cameras provided by the module (cameras should be appended by subclasses)
    self._cameras = []

  def __init_subclass__(cls, *args, **kwargs):
    """ Define a new __init__ method with a hook that automatically sets observation shape after a child instance
    has been initialised. This is only for convenience, otherwise we would need to set the observation shape separately
    in each child class constructor, or after a child of BaseModule has been initialised."""
    super().__init_subclass__(*args, **kwargs)
    def init_with_hook(self, model, data, bm_model, init=cls.__init__, **init_kwargs):
      init(self, model, data, bm_model, **init_kwargs)

      # Set observation shape
      self._observation_shape = self.get_observation(model, data).shape

      # Initialise encoder: use Identity encoder by default (which is no-op) if encoder has not been defined in config
      encoder_info = init_kwargs.get("encoder", None) or self._default_encoder
      self._initialise_encoder(encoder_info)

    cls.__init__ = init_with_hook


  ############ The methods below you should definitely overwrite as they are important ############

  @staticmethod
  @abstractmethod
  def insert(simulator_tree, **kwargs):
    """ Insert required elements into the simulation xml file.

    Args:
      simulator_tree: An `xml.etree.ElementTree` containing the parsed simulator xml file
    """
    pass

  @abstractmethod
  def get_observation(self, model, data, info=None):
    """ Return an observation from this perception module. These observations are used for RL training.

    Returns:
      A numpy array, which can be a vector or a higher dimensional array. If a higher dimensional array, then the
        property 'encoder' must be implemented.
    """
    pass

  @property
  def _default_encoder(self):
    """ Returns default encoder info as a dict. Overwrite this property in perception modules to use different encoders
    by default. """
    return {"module": "rl.encoders", "cls": "Identity"}


  ############ The methods below are overwritable but often don't need to be overwritten ############

  def _reset(self, model, data):
    """ Reset the perception module. """
    pass

  def _update(self, model, data):
    """ Update the perception module after a step has been taken. """
    pass

  def _get_state(self, model, data):
    """ Return the state of the perception module. These states are used only for logging/evaluation, not for RL
    training

    Returns:
      A dict where each key should have a float or a numpy vector as their value
    """
    return dict()

  def set_ctrl(self, model, data, action):
    """ Set control signal (e.g. for eye movements).

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      action: An action value from a policy, limited to range [-1, 1]
    """
    pass

  def _get_observation_range(self):
    """ Return limits for the observations. These limits aren't currently used for anything (AFAIK, not in gym(nasium) or
    stable-baselines3; only to initialise the observation space required by gym.Env), so let's just use a default of
    -inf to inf. Overwrite this method to use different ranges.

    Returns:
        A dict with format {"low": float-or-array, "high": float-or-array} where the values indicate lowest and highest
          values the observation can have. The values can be floats or numpy arrays -- if arrays, they must have the
          same shape as the returned observation from method 'get_observation'
    """
    return {"low": float('-inf'), "high": float('inf')}

  @classmethod
  def clone(cls, simulator_folder, package_name):
    """ Clones (i.e. copies) the relevant python files into a new location.

    Args:
       simulator_folder: Location of the simulator.
       package_name: Name of the simulator (which is a python package)
    """

    # Create "perception" folder if needed
    dst = os.path.join(simulator_folder, package_name, "perception")
    os.makedirs(dst, exist_ok=True)

    # Copy this file and __init__.py
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))
    shutil.copyfile(os.path.join(base_file.parent, "__init__.py"), os.path.join(dst, "__init__.py"))

    # Get src for this module
    src = parent_path(inspect.getfile(cls))

    # Create upper level folder (if needed)
    modality = os.path.join(dst, src.parent.stem)
    os.makedirs(modality, exist_ok=True)

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(modality, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[3:]) + " import " + cls.__name__)

    # Copy module folder
    shutil.copytree(src, os.path.join(modality, src.stem), dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

    # Copy assets if they exist
    if os.path.isdir(os.path.join(src, "assets")):
      shutil.copytree(os.path.join(src, "assets"), os.path.join(simulator_folder, package_name, "assets"),
                      dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

  def close(self, **kwargs):
    """ Perform any necessary clean up. """
    pass

  ############ The methods below you should not overwrite ############

  @final
  def reset(self, model, data):
    """ Reset (and update) the perception module. """
    self._reset(model, data)
    self._update(model, data)
    mujoco.mj_forward(model, data)

  @final
  def update(self, model, data):
    """ Update the perception module after a step has been taken. """
    self._update(model, data)

  @final
  def get_state(self, model, data):
    """ Returns the state of the perception module. """
    state = dict()
    state.update(self._get_state(model, data))
    return state

  @final
  def get_observation_space_params(self):
    """ Returns the observation space parameters. """
    return {**self._get_observation_range(), "shape": self._observation_shape}

  @final
  def _initialise_encoder(self, encoder_info):
    """ Import an encoder. We assume the encoder inherits from uitb.rl.encoders.BaseEncoder. """
    encoder_cls = importer(encoder_info)
    if not issubclass(encoder_cls, BaseEncoder):
      raise RuntimeError(f"Encoder {encoder_cls} does not inherit from uitb.rl.encoders.BaseEncoder")
    self._encoder = encoder_cls(observation_shape=self._observation_shape, **encoder_info.get("kwargs", {}))

  @final
  @property
  def encoder(self):
    """ An encoder (typically a PyTorch neural network) that encodes observations into a latent space. If observations
    are higher dimensional (dim > 1), then the encoder must encode those observations into one dimensional vectors. """
    return self._encoder

  @final
  @property
  def nu(self):
    """ Return number of actuators. """
    return len(self._actuator_names)

  @final
  @property
  def actuator_names(self):
    """ Return actuator names. """
    return self._actuator_names.copy()

  @final
  @property
  def joint_names(self):
    """ Return joint names. """
    return self._joint_names.copy()

  @final
  @property
  def modality(self):
    """ Return modality. """
    return self._modality


@final
class Perception:
  """
  This class implements a Perception model, which consists of multiple perception modules.
  """

  def __init__(self, model, data, bm_model, perception_modules, run_parameters):
    """ Initialises a new `Perception` instance.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      bm_model: An instance that inherits from the uitb.bm_models.base.BaseBMModel class.
      perception_modules: A list of dicts in format [{"cls": perception-module-class1, "kwargs": given-kwargs1},
        {"cls": perception-module-class2, "kwargs": given-kwargs2}, ...].
      run_parameters: A dict of run parameters that contain important run time variables (and can be used to override
        parameters after a simulator has been built)
    """

    # Get names of any (controllable) actuators that might be used in perception modules, like for e.g. eye movements
    self._actuator_names = []
    self._joint_names = []

    # Get encoders
    self.encoders = dict()

    self.perception_modules = []
    self._cameras = {}
    for module_cls, kwargs in perception_modules.items():
      module = module_cls(model, data, bm_model, **{**kwargs, **run_parameters})
      self.perception_modules.append(module)
      self._actuator_names.extend(module.actuator_names)
      self._joint_names.extend(module.joint_names)
      self.encoders[module.modality] = module.encoder

      if module.modality == "vision":
        self._cameras[module] = module._cameras

    # Find actuators in the simulation
    self._actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                      for actuator_name in self._actuator_names]
    self._nu = len(self._actuators)

    # Find joints in the simulation
    self._joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                   for joint_name in self._joint_names]

  def set_ctrl(self, model, data, action):
    """ Set control signals for perception modules.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      action: Action values sampled from a policy, limited to range [-1, 1]
    """
    num = 0
    for module in self.perception_modules:
      module.set_ctrl(model, data, action[num:num+module.nu])
      num += module.nu

  def reset(self, model, data):
    """ Reset perception modules. """
    for module in self.perception_modules:
      module.reset(model, data)
    self.update(model, data)

  def update(self, model, data):
    """ Update perception modules after a step has been taken (or after a reset). """
    for module in self.perception_modules:
      module.update(model, data)

  def get_state(self, model, data):
    """ Returns the state of all perception modules. These states are used only for logging/evaluation, not RL
    training. """
    state = dict()
    for module in self.perception_modules:
      state.update(module.get_state(model, data))
    return state

  def get_observation(self, model, data, info=None):
    """ Return the observation from all perception modules. These observations are used for RL training. """
    observations = {}
    for module in self.perception_modules:
      observations[module.modality] = module.get_observation(model, data, info)
    return observations

  def close(self, **kwargs):
    """ Perform any necessary clean up. """
    for module in self.perception_modules:
      module.close(**kwargs)

  def get_renders(self):
    """ Return rgb and/or depth arrays from all registered cameras."""
    perception_camera_images = []
    # Use self.cameras_dict to have access to the module where a camera lives
    for module in self.cameras_dict:
      for camera in self.cameras_dict[module]:
        for rgb_or_depth_array in camera.render():
          # camera.render() returns a tuple (rgb_array, depth_array), with entries set to None if not available
          if rgb_or_depth_array is not None:
            if hasattr(module, "camera_active") and module.camera_active == False:
              # return zero (black) array if module.camera_active == False (can be used to temporarily disable visual input)
              rgb_or_depth_array *= 0
            perception_camera_images.append(rgb_or_depth_array)
    return perception_camera_images

  @property
  def cameras(self):
    return [_cam for cameras in self._cameras.values() for _cam in cameras]

  @property
  def cameras_dict(self):
    return self._cameras

  @property
  def actuators(self):
    """ Return actuator IDs of all perception modules. """
    return self._actuators.copy()

  @property
  def joints(self):
    """ Return joint IDs of all perception modules. """
    return self._joints.copy()

  @property
  def nu(self):
    """ Return total number of actuators. """
    return self._nu
