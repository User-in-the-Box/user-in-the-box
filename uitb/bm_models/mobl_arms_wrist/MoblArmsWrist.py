from ..base import BaseBMModel
from ...utils.functions import initialise_pos_and_quat

import numpy as np
import mujoco


class MoblArmsWrist(BaseBMModel):

  def __init__(self, model, data, **kwargs):
    super().__init__(model, data, **kwargs)

    # Set shoulder variant
    self.shoulder_variant = kwargs.get("shoulder_variant", "patch-v1")

    # Update skull rotation if necessary
    if "skull_rotation" in kwargs:

      # Rotate skull
      model.body("skull").quat = kwargs["skull_rotation"]

      # Do a forward
      mujoco.mj_forward(model, data)

      # Check if there are any bodies whose positions need to be updated
      # must be 1) weld equality constraint, and 2) obj2_id refer to the skull body.
      # It would be nicer if each equality were updated in their respective classes (like unity headset in the unity
      # task), but cannot be done with current ordering of task -> bm model -> perception
      cond1 = model.eq_type == mujoco.mjtEq.mjEQ_WELD
      cond2 = model.eq_obj2id == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "skull")
      idxs = np.where(np.logical_and(cond1, cond2))[0]

      # Initialise pos and quat according to the relpose for each found body.
      for idx in idxs:
        initialise_pos_and_quat(model, data, "skull", model.eq_data[idx][3:10],
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[idx]))

  def _update(self, model, data):

    # Update shoulder equality constraints
    if self.shoulder_variant.startswith("patch"):
      sh_elv = np.asarray(data.joint('shoulder_elv').qpos).item()
      model.equality("shoulder1_r2_con").data[1] = \
        -((np.pi - 2 * sh_elv) / np.pi)

      if self.shoulder_variant == "patch-v2":
        elv_ang = np.asarray(data.joint('elv_angle').qpos).item()
        model.joint('shoulder_rot').range[:] = \
          np.array([-np.pi / 2, np.pi / 9]) - \
          2 * min(sh_elv, np.pi - sh_elv) / np.pi * elv_ang

      # Do a forward calculation
      mujoco.mj_forward(model, data)

  @classmethod
  def _get_floor(cls):
    return None
