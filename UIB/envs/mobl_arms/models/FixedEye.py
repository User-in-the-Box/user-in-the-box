import gym
from gym import spaces
import mujoco_py
import numpy as np
import os
from abc import ABC, abstractmethod

from UIB.utils.functions import project_path
from UIB.utils import effort_terms


class FixedEye(ABC, gym.Env):

  # Model file
  #xml_file = os.path.join(project_path(), "envs/mobl_arms/models/variants/mobl_arms_muscles_v2_modified.xml")
  xml_file = os.path.join(project_path(), "envs/mobl_arms/models/variants/mobl_arms_muscles_v2_modified_bright.xml")

  # Fingertip
  fingertip = "hand_2distph"

  def __init__(self, shoulder_variant="original", **kwargs):

    # Set shoulder model variant
    self.shoulder_variant = shoulder_variant

    # Set action sampling
    self.action_sample_freq = kwargs.get('action_sample_freq', 10)
    self.timestep = 0.002
    self.frame_skip = int(1/(self.timestep*self.action_sample_freq))

    # RNG in case we need it
    self.rng = np.random.default_rng()

    # Initialise model and sim
    self.model = mujoco_py.load_model_from_path(self.xml_file)
    self.sim = mujoco_py.MjSim(self.model, nsubsteps=self.frame_skip)

    # Get indices of dependent and independent joints
    self.dependent_joints = np.unique(self.model.eq_obj1id[self.model.eq_active.astype(bool)])
    self.independent_joints = list(set(np.arange(self.model.njnt)) - set(self.dependent_joints))

    # Get equality constraint ID of shoulder1_r2
    if self.shoulder_variant.startswith("patch"):
      eq_ID_shoulder1_r2 = ([idx for idx, i in enumerate(self.sim.model.eq_data) if
                            self.sim.model.eq_type[idx] == 2 and (id_1 := self.sim.model.eq_obj1id[idx]) > 0 and (
                            id_2 := self.sim.model.eq_obj2id[idx]) and {self.sim.model.joint_id2name(id_1),
                                                                       self.sim.model.joint_id2name(id_2)} == {
                            "shoulder1_r2",
                            "elv_angle"}])
      assert len(eq_ID_shoulder1_r2) == 1
      self.eq_ID_shoulder1_r2 = eq_ID_shoulder1_r2[0]

    # Set action space
    muscles_limits = np.ones((self.model.na,2)) * np.array([-1.0, 1.0])
    self.action_space = spaces.Box(low=np.float32(muscles_limits[:, 0]), high=np.float32(muscles_limits[:, 1]))

    # Get reward function and effort term
    self.reward_function = kwargs.get('reward_function', None)
    self.effort_term = kwargs.get('effort_term', effort_terms.Zero())

    # Observations from eye shouldn't be rendered when they are not needed
    self.render_observations = kwargs.get('render_observations', True)

    # Size of ocular image
    self.ocular_image_height = kwargs.get('ocular_image_height', 80)
    self.ocular_image_width = kwargs.get('ocular_image_width', 120)

    # Set camera stuff, self._viewers needs to be initialised before self.get_observation() is called
    self.viewer = None
    self._viewers = {}
    self.metadata = {
      'render.modes': ['human', 'rgb_array', 'depth_array'],
      'video.frames_per_second': int(np.round(1.0 / (self.model.opt.timestep * self.frame_skip))),
      "imagesize": (1280, 800)
    }
    self.sim.model.cam_pos[self.sim.model._camera_name2id['for_testing']] = np.array([1.5, -1.5, 0.9])
    self.sim.model.cam_quat[self.sim.model._camera_name2id['for_testing']] = np.array([0.6582, 0.6577, 0.2590, 0.2588])

    # Get callbacks
    self.callbacks = {callback.name: callback for callback in kwargs.get('callbacks', [])}

  def set_ctrl(self, action):
    self.sim.data.ctrl[:] = np.clip(self.sim.data.act[:] + action, 0, 1)

    if self.shoulder_variant.startswith("patch"):
      #self.sim.data.qpos[self.sim.model.joint_name2id('shoulder1_r2')] = -((np.pi - 2*self.sim.data.qpos[self.sim.model.joint_name2id('shoulder_elv')])/np.pi) * self.sim.data.qpos[self.sim.model.joint_name2id('elv_angle')]
      self.sim.model.eq_data[self.eq_ID_shoulder1_r2, 1] = -((np.pi - 2 * self.sim.data.qpos[self.sim.model.joint_name2id('shoulder_elv')]) / np.pi)

      if self.shoulder_variant == "patch-v2":
        self.sim.model.jnt_range[self.sim.model.joint_name2id('shoulder_rot'), :] = np.array(
          [-np.pi / 2, np.pi / 9]) - 2 * np.min((self.sim.data.qpos[self.sim.model.joint_name2id('shoulder_elv')], np.pi - self.sim.data.qpos[self.sim.model.joint_name2id('shoulder_elv')])) / np.pi * self.sim.data.qpos[self.sim.model.joint_name2id('elv_angle')]

  @abstractmethod
  def step(self, action):
    pass

  def get_observation(self):

    # Normalise qpos
    jnt_range = self.sim.model.jnt_range[self.independent_joints]
    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5) * 2

    # Get qvel, qacc
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    # Get fingertip position; not normalised
    fingertip_position = self.sim.data.get_geom_xpos(self.fingertip)

    # Normalise act
    act = (self.sim.data.act.copy() - 0.5) * 2

    # Proprioception features
    proprioception = np.concatenate([qpos, qvel, qacc, fingertip_position, act])

    if self.render_observations:
      # Get visual observation and normalize
      render = self.sim.render(width=self.ocular_image_width, height=self.ocular_image_height, camera_name='oculomotor', depth=True)
      depth = render[1]
      depth = np.flipud((depth - 0.5) * 2)
      rgb = render[0]
      rgb = np.flipud((rgb/255.0 - 0.5)*2)
      visual = np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2)
    else:
      visual = None

    return {'proprioception': proprioception, 'visual': visual}

  def reset(self):

    self.sim.reset()

    # Randomly sample qpos, qvel, act
    nq = len(self.independent_joints)
    qpos = self.rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    qvel = self.rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    act = self.rng.uniform(low=np.zeros((self.model.na,)), high=np.ones((self.model.na,)))

    # Set qpos and qvel
    self.sim.data.qpos.fill(0)
    self.sim.data.qpos[self.independent_joints] = qpos
    self.sim.data.qvel.fill(0)
    self.sim.data.qvel[self.independent_joints] = qvel
    self.sim.data.act[:] = act

    # # Start with T-Pose
    # self.sim.data.qpos[self.sim.model.joint_name2id("shoulder_elv")] = 1.57

    # Some effort terms may be stateful and need to be reset
    self.effort_term.reset()

    # Do a forward so everything will be set
    self.sim.forward()

    return self.get_observation()

  def callback(self, callback_name, num_timesteps):
    self.callbacks[callback_name].update(num_timesteps)

  def grab_image(self, height, width):

    # Make sure estimate is not in the image
    self.model.geom_rgba[self.model._geom_name2id["target-sphere-estimate"]][-1] = 0

    rendered = self.sim.render(height=height, width=width, camera_name='oculomotor', depth=True)
    rgb = ((np.flipud(rendered[0]) / 255.0) - 0.5) * 2
    depth = (np.flipud(rendered[1]) - 0.5) * 2
    #return np.expand_dims(np.flipud(depth), 0)
    #return np.concatenate([rgb.transpose([2, 0, 1]), np.expand_dims(depth, 0)])
    return np.concatenate([np.expand_dims(rgb[:, :, 1], 0), np.expand_dims(depth, 0)])

  def grab_proprioception(self):

    # Ignore eye qpos and qvel for now
    jnt_range = self.sim.model.jnt_range[self.independent_joints]

    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5) * 2
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    finger_position = self.sim.data.get_geom_xpos(self.fingertip).copy()
    return np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position])
    #return np.concatenate([qpos[2:], qvel[2:], qacc[2:]])

  def grab_target(self):
    # Use self.target_position for normalised position around self.target_origin
    # Make target radius zero mean using known limits
    normalised_radius = self.target_radius - (self.target_radius_limit[1]-self.target_radius_limit[0])
    return np.concatenate([self.target_position[1:].copy(), np.array([normalised_radius])])

  def get_state(self):
    state = {"step": self.steps, "timestep": self.sim.data.time,
             "qpos": self.sim.data.qpos[self.independent_joints].copy(),
             "qvel": self.sim.data.qvel[self.independent_joints].copy(),
             "qacc": self.sim.data.qacc[self.independent_joints].copy(),
             "act": self.sim.data.act.copy(),
             "fingertip_xpos": self.sim.data.get_geom_xpos(self.fingertip).copy(),
             "fingertip_xmat": self.sim.data.get_geom_xmat(self.fingertip).copy(),
             "fingertip_xvelp": self.sim.data.get_geom_xvelp(self.fingertip).copy(),
             "fingertip_xvelr": self.sim.data.get_geom_xvelr(self.fingertip).copy(),
             "termination": False}
    return state

  def render(self, mode='human', width=1280, height=800, camera_id=None, camera_name=None):

    if mode == 'rgb_array' or mode == 'depth_array':
        if camera_id is not None and camera_name is not None:
            raise ValueError("Both `camera_id` and `camera_name` cannot be"
                             " specified at the same time.")

        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = 'track'

        if camera_id is None and camera_name in self.model._camera_name2id:
            camera_id = self.model.camera_name2id(camera_name)

        self._get_viewer(mode).render(width, height, camera_id=camera_id)

    if mode == 'rgb_array':
        # window size used for old mujoco-py:
        data = self._get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]
    elif mode == 'depth_array':
        self._get_viewer(mode).render(width, height)
        # window size used for old mujoco-py:
        # Extract depth part of the read_pixels() tuple
        data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
        # original image is upside-down, so flip it
        return data[::-1, :]
    elif mode == 'human':
        self._get_viewer(mode).render()

  def _get_viewer(self, mode):
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
      if mode == 'human':
        self.viewer = mujoco_py.MjViewer(self.sim)
      elif mode == 'rgb_array' or mode == 'depth_array':
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

      self._viewers[mode] = self.viewer
    return self.viewer

  def close(self):
    pass

  @property
  def dt(self):
    return self.model.opt.timestep * self.frame_skip

  def write_video(self, imgs, filepath):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, self.metadata["video.frames_per_second"], self.metadata["imagesize"])
    for img in imgs:
      out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()
