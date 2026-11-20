import numpy as np
import mujoco

from ..base import BaseTask

class LowLevelController(BaseTask):

  def __init__(self, model, data, joints, **kwargs):
    super().__init__(model, data, **kwargs)

    # Define number of trials in episode
    self._max_trials = 10
    self._trial_idx = 0
    self._targets_hit = 0
    self._trials_ep = 0

    self._target_radius = 0.05  #maximum permitted distance between target and current joint posture (using the Euclidean norm)
    self._steps_inside_target = 0
    self._dwell_threshold = int(kwargs.get("dwell_time", 0.5)*self._action_sample_freq)

    # Use early termination if target is not hit in time
    self._steps_since_last_hit = 0
    self._max_steps_without_hit = self._action_sample_freq*kwargs.get("max_trial_time",  2)

    # Whether to penalize distance of all joints (including dependent/virtual joints), or only of joints that are actively actuated
    self._track_all_joints = kwargs.get("track_all_joints", False)
    
    # Get independent dofs -- would be easier to just grab these from the bm model
    self._independent_dofs = []
    self._independent_joints = []
    for joint in joints:
      joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
      if model.jnt_type[joint_id] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        raise NotImplementedError(f"Only 'hinge' and 'slide' joints are supported, joint "
                                  f"{joint} is of type {mujoco.mjtJoint(model.jnt_type[joint_id]).name}")
      self._independent_dofs.append(model.jnt_qposadr[joint_id])
      self._independent_joints.append(joint_id)


    # Get joint range for normalisation
    if self._track_all_joints:
      self._jnt_range = model.jnt_range.copy()
    else:
      self._jnt_range = model.jnt_range[self._independent_joints]

    # Initialise qpos
    self._qpos = None
    self._get_qpos(model, data)

    # Initialise target qpos
    self._target_qpos = None
    self._sample_target_qpos(model, data)

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([0.6582, 0.6577, 0.2590, 0.2588])

  def _update(self, model, data):

    # Get qpos
    self._get_qpos(model, data)

    # Set some defaults
    terminated = False
    truncated = False
    self._info["target_sampled"] = False

    # Distance to target
    dist = np.linalg.norm(self._target_qpos - self._qpos)

    # Check if target is close enough
    if dist < self._target_radius:
      self._steps_inside_target += 1
      self._info["inside_target"] = True
    else:
      self._steps_inside_target = 0
      self._info["inside_target"] = False

    if self._info["inside_target"] and self._steps_inside_target >= self._dwell_threshold:

      # Update counters
      self._info["target_hit"] = True
      self._trial_idx += 1
      self._targets_hit += 1
      self._steps_since_last_hit = 0
      self._steps_inside_target = 0
      self._info["acc_dist"] += np.mean(dist)
      self._sample_target_qpos(model, data)
      self._info["target_sampled"] = True

    else:
      self._info["target_hit"] = False

    # Check if time limit has been reached
    self._steps_since_last_hit += 1
    if self._steps_since_last_hit >= self._max_steps_without_hit:
      # Spawn a new target
      self._steps_since_last_hit = 0
      self._trial_idx += 1
      self._info["acc_dist"] += np.mean(dist)
      self._sample_target_qpos(model, data)
      self._info["target_sampled"] = True

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      self._info["mean_dist"] = self._info["acc_dist"]/self._trial_idx
      truncated = True
      self._info["termination"] = "max_trials_reached"
   
    reward = self.get_reward_with_hit_bonus(dist-self._target_radius, self._info.copy())

    return reward, terminated, truncated, self._info

  def get_reward_with_hit_bonus(self, dist, info):
    if info["target_hit"]:
      #_reward = 8
      _reward = (self._max_steps_without_hit / self._max_trials) / 10
    else:
      _reward = np.exp(-dist * 3) / 10
    if info["inside_target"]:
      _reward += 1 / 10
    # else:
    #   return np.prod(np.exp(-dist*3))
    return _reward

  def _normalise_qpos(self, qpos):
    # Normalise to [0, 1]
    qpos = (qpos - self._jnt_range[:, 0]) / (self._jnt_range[:, 1] - self._jnt_range[:, 0])
    # Normalise to [-1, 1]
    qpos = (qpos - 0.5) * 2
    return qpos
  
  def _unnormalise_qpos(self, qpos):
    return (qpos/2 + 0.5) * (self._jnt_range[:, 1] - self._jnt_range[:, 0]) + self._jnt_range[:, 0]

  def _get_qpos(self, model, data):
    if self._track_all_joints:
      qpos = data.qpos.copy()
    else:
      qpos = data.qpos[self._independent_dofs].copy()
    self._qpos = self._normalise_qpos(qpos)
  
  @staticmethod
  def _ensure_joint_eq_constraints(model, qpos):
    # adjust virtual joints according to active constraints (MuJoCo 2.x eq_active / 3.x+ eq_active0):
    eq_active = getattr(model, "eq_active0", getattr(model, "eq_active", None))
    _eq_constraints = zip(model.eq_obj1id[
                        (model.eq_type == 2) & (eq_active == 1)],
                    model.eq_obj2id[
                        (model.eq_type == 2) & (eq_active == 1)],
                    model.eq_data[(model.eq_type == 2) &
                                  (eq_active == 1), 4::-1])
    for (virtual_joint_id, physical_joint_id, poly_coefs) in _eq_constraints:
      qpos[virtual_joint_id] = np.polyval(poly_coefs, qpos[physical_joint_id])

  def _reset(self, model, data):
    self._steps_since_last_hit = 0
    self._steps_inside_target = 0
    self._trial_idx = 0
    self._targets_hit = 0

    self._info = {"target_hit": False, "inside_target": False, "target_sampled": False,
                  "terminated": False, "truncated": False,
                  "termination": False, "mean_dist": 0, "acc_dist": 0}

    self._sample_target_qpos(model, data)

  def _sample_target_qpos(self, model, data):
    jnt_range = np.array([[-0.2, 2.27],
                          [0, 2.5],
                          [-0.8, 0.34],
                          [0, 2.27],
                          [-0.6, 0.6]])
    target_qpos = self._rng.uniform(low=jnt_range[:,0], high=jnt_range[:,1], size=(len(self._independent_dofs,)))
    if self._track_all_joints:
      dummy_qpos = np.zeros(model.nq)
      dummy_qpos[self._independent_dofs] = target_qpos
      # ensure that joint equality constraints hold for target qpos
      self._ensure_joint_eq_constraints(model, dummy_qpos)
      target_qpos = dummy_qpos
    self._target_qpos = self._normalise_qpos(target_qpos)

  def get_stateful_information(self, model, data):
    # Need to output target qpos here
    qpos_diff = self._target_qpos - self._qpos
    return qpos_diff
