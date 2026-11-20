import xml.etree.ElementTree as ET
import numpy as np
import itertools
import warnings
from scipy.spatial.transform.rotation import Rotation

from ...base import BaseModule

class RectangularCuboidGrid(BaseModule):

  def __init__(self, model, data, bm_model, geom, resolution, margin=0.001, **kwargs):
    super().__init__(model, data, bm_model, **kwargs)

    self._geom = geom
    self._resolution = resolution

    # Get geom position and quat (transform quat into scalar-last format)
    geom_pos = model.geom(geom).pos.copy()
    geom_quat = model.geom(geom).quat.copy()
    geom_quat = np.concatenate([geom_quat[1:], geom_quat[:1]])

    # Get transformation matrix to geom (from parent body)
    T_geom = np.eye(4)
    T_geom[:3, 3] = geom_pos
    T_geom[:3, :3] = Rotation.from_quat(geom_quat).as_matrix()

    # Get geom (half-)size, also add some margin to make sure all contacts will be inside the volume
    geom_size = model.geom(geom).size.copy() + margin

    # Raise an error if there's no zero point defined in resolution
    resolution = np.array(resolution)  # Take a copy so we don't modify the given original resolution
    zero_idx = np.where(np.array(resolution)==0)[0]
    if zero_idx.size != 1:
      warnings.warn("One of the dimensions of 'resolution' should be zero unless you _really_ know what you're doing",
                    RuntimeWarning)

    # Then transform that zero into a two (could be >2 but the generated sites would be inside the geom)
    resolution[zero_idx] = 2

    # Calculate positions for the sites
    def midpoints_for_axis(axis):
      midpoints = np.arange(geom_size[axis]/resolution[axis], 2*geom_size[axis], 2*geom_size[axis]/resolution[axis])
      return -geom_size[axis] + midpoints
    midpoints = [midpoints_for_axis(0), midpoints_for_axis(1), midpoints_for_axis(2)]

    # Set positions and sizes
    site_size = geom_size / resolution
    self._sites = []
    self._sensors = []
    for i, midpoint in enumerate(itertools.product(midpoints[0], midpoints[1], midpoints[2])):
      site_name = f"{geom}-site-{i}"
      site_in_body = np.matmul(T_geom, np.concatenate([midpoint, np.array([1])]))
      model.site(site_name).pos = site_in_body[:3]
      model.site(site_name).size = site_size
      model.site(site_name).quat = model.geom(geom).quat.copy()  # Copy the original geom quat in scalar-first format

      self._sites.append(site_name)
      self._sensors.append(f"{geom}-touch-{i}")

  @staticmethod
  def insert(simulation, **kwargs):

    # Get root
    root = simulation.getroot()

    # Get the parent body of the geom
    body = root.find(f".//geom[@name='{kwargs['geom']}']...")

    # Make sure sensor element exists
    sensors = root.find('sensor')
    if sensors is None:
      sensors = ET.Element('sensor')
      root.append(sensors)

    # If there are zero dimensions make them twos
    resolution = np.array(kwargs["resolution"])
    zero_idx = np.where(resolution==0)[0]
    for idx in zero_idx:
      resolution[idx] = 2

    # Add sites and sensors
    for i in range(int(np.prod(resolution))):

      site_name = f"{kwargs['geom']}-site-{i}"

      # Add box type sites to the geom; correct pos, quat, and size will be set later in __init__
      body.append(ET.Element("site", name=site_name, type="box", size="0.01 0.01 0.01"))

      # Add sensors
      sensors.append(ET.Element("touch", name=f"{kwargs['geom']}-touch-{i}", site=site_name))

  def get_observation(self, model, data, info=None):
    obs = np.zeros(len(self._sensors),)
    for idx, sensor in enumerate(self._sensors):
      obs[idx] = np.asarray(data.sensor(sensor).data).item() / 100
    return obs
