from abc import ABC, abstractmethod
import os
import shutil
import inspect
import pathlib

from ..utils.functions import parent_path

class BaseRLModel(ABC):

  def __init__(self, **kwargs):
    pass

  @abstractmethod
  def learn(self, wandb_callback):
    pass

  @classmethod
  def clone(cls, simulator_folder, package_name):

    # Create 'rl' folder
    dst = os.path.join(simulator_folder, package_name, "rl")
    os.makedirs(dst, exist_ok=True)

    # Copy the rl library folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pyc'))

    # Copy this file
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))

    # Copy the file with encoders
    encoder_file = os.path.join(base_file.parent, "encoders.py")
    shutil.copyfile(encoder_file, os.path.join(dst, "encoders.py"))

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__)