from abc import ABC, abstractmethod
import mujoco
import numpy as np


class BaseEffortModel(ABC):

  def __init__(self, bm_model, **kwargs):
    self._bm_model = bm_model

  @abstractmethod
  def cost(self, model, data):
    pass

  @abstractmethod
  def reset(self, model, data):
    pass

  @abstractmethod
  def update(self, model, data):
    # If needed to e.g. reduce max force output
    pass

  def _get_state(self, model, data):
    """ Return the state of the effort model. These states are used only for logging/evaluation, not for RL
    training

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.

    Returns:
      A dict where each key should have a float or a numpy vector as their value
    """
    return dict()


class Composite(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-7, **kwargs):
    super().__init__(bm_model)
    self._weight = weight

  def cost(self, model, data):
    mujoco.mj_inverse(model, data)
    angle_acceleration = np.sum(data.qacc[self._bm_model.independent_dofs] ** 2)
    energy = np.sum(data.qvel[self._bm_model.independent_dofs] ** 2
                    * data.qfrc_inverse[self._bm_model.independent_dofs] ** 2)
    return self._weight * (energy + 0.05 * angle_acceleration)

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass


class CumulativeFatigue(BaseEffortModel):

  # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles -- using control signals
  # instead of torque or force
  def __init__(self, bm_model, dt, **kwargs):
    super().__init__(bm_model)
    self._r = 7.5
    self._F = 0.0146
    self._R = 0.0022
    self._LD = 10
    self._LR = 10
    self._MA = None
    self._MR = None
    self._weight = 0.01
    self._dt = dt

  def cost(self, model, data):

    # Initialise MA if not yet initialised
    if self._MA is None:
      self._MA = np.zeros((model.na,))
      self._MR = np.ones((model.na,))

    # Get target load (actual activation, which might be reached only with some "effort", depending on how many muscles can be activated (fast enough) and how many are in fatigue state)
    TL = data.act.copy()

    # Calculate C(t)
    C = np.zeros_like(self._MA)
    idxs = (self._MA < TL) & (self._MR > (TL - self._MA))
    C[idxs] = self._LD * (TL[idxs] - self._MA[idxs])
    idxs = (self._MA < TL) & (self._MR <= (TL - self._MA))
    C[idxs] = self._LD * self._MR[idxs]
    idxs = self._MA >= TL
    C[idxs] = self._LR * (TL[idxs] - self._MA[idxs])

    # Calculate rR
    rR = np.zeros_like(self._MA)
    idxs = self._MA >= TL
    rR[idxs] = self._r*self._R
    idxs = self._MA < TL
    rR[idxs] = self._R

    # Calculate MA, MR
    self._MA += (C - self._F*self._MA)*self._dt
    self._MR += (-C + rR*self._MR)*self._dt

    # Not sure if these are needed
    self._MA = np.clip(self._MA, 0, 1)
    self._MR = np.clip(self._MR, 0, 1)

    # Calculate effort
    effort = np.linalg.norm(self._MA - TL)

    return self._weight*effort

  def reset(self, model, data):
    self._MA = None
    self._MR = None

  def update(self, model, data):
    pass


class CumulativeFatigue3CCr(BaseEffortModel):

  # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles -- using control signals
  # instead of torque or force
  # v2: now with additional muscle fatigue state
  def __init__(self, bm_model, dt, weight=0.01, **kwargs):
    super().__init__(bm_model)
    self._r = 7.5
    self._F = 0.0146
    self._R = 0.0022
    self._LD = 10
    self._LR = 10
    self._MA = None
    self._MR = None
    self._MF = None
    self._TL = None
    self._dt = dt
    self._weight = weight
    self._effort_cost = None

  def cost(self, model, data):
    # Calculate effort
    effort = np.linalg.norm(self._MA - self._TL)
    self._effort_cost = self._weight*effort
    return self._effort_cost

  def reset(self, model, data):
    self._MA = np.zeros((model.na,))
    self._MR = np.ones((model.na,))
    self._MF = np.zeros((model.na,))

  def update(self, model, data):
    # Get target load
    TL = data.act.copy()
    self._TL = TL

    # Calculate C(t)
    C = np.zeros_like(self._MA)
    idxs = (self._MA < TL) & (self._MR > (TL - self._MA))
    C[idxs] = self._LD * (TL[idxs] - self._MA[idxs])
    idxs = (self._MA < TL) & (self._MR <= (TL - self._MA))
    C[idxs] = self._LD * self._MR[idxs]
    idxs = self._MA >= TL
    C[idxs] = self._LR * (TL[idxs] - self._MA[idxs])

    # Calculate rR
    rR = np.zeros_like(self._MA)
    idxs = self._MA >= TL
    rR[idxs] = self._r*self._R
    idxs = self._MA < TL
    rR[idxs] = self._R

    # Clip C(t) if needed, to ensure that MA, MR, and MF remain between 0 and 1
    C = np.clip(C, np.maximum(-self._MA/self._dt + self._F*self._MA, (self._MR - 1)/self._dt + rR*self._MF),
                np.minimum((1 - self._MA)/self._dt + self._F*self._MA, self._MR/self._dt + rR*self._MF))

    # Update MA, MR, MF
    dMA = (C - self._F*self._MA)*self._dt
    dMR = (-C + rR*self._MF)*self._dt
    dMF = (self._F*self._MA - rR*self._MF)*self._dt
    self._MA += dMA
    self._MR += dMR
    self._MF += dMF
    
  def _get_state(self, model, data):
    state = {"3CCr_MA": self._MA,
         "3CCr_MR": self._MR,
         "3CCr_MF": self._MF,
         "effort_cost": self._effort_cost}
    return state


class ConsumedEndurance(BaseEffortModel):

  lifting_muscles = ["DELT1", "DELT2", "DELT3", "SUPSP", "INFSP", "SUBSC", "TMIN", "BIClong", "BICshort", "TRIlong", "TRIlat", "TRImed"]  

  # consumed endurance model, taken from https://dl.acm.org/doi/pdf/10.1145/2556288.2557130
  def __init__(self, bm_model, dt, weight=0.01, **kwargs):
    super().__init__(bm_model)
    self._dt = dt
    self._weight = weight
    self._endurance = None
    self._consumed_endurance = None
    self._effort_cost = None
    
  def get_endurance(self, model, data):
    #applied_shoulder_torque = np.linalg.norm(data.qfrc_inverse[:])
    #applied_shoulder_torque = np.linalg.norm(data.qfrc_actuator[:])
    lifting_indices = [model.actuator(_i).id for _i in self.lifting_muscles]
    applied_shoulder_torques = data.actuator_force[lifting_indices]
    maximum_shoulder_torques = model.actuator_gainprm[lifting_indices, 2]
    #assert np.all(applied_shoulder_torque <= 0), "Expected only negative values in data.actuator_force."
    #strength = np.mean((applied_shoulder_torques/maximum_shoulder_torques)**2)
    strength = np.abs(applied_shoulder_torques/maximum_shoulder_torques)  #compute strength per muscle
    # assert np.all(strength <= 1), f"Applied torque is larger than maximum torque! strength:{strength}, applied:{applied_shoulder_torques}, max:{maximum_shoulder_torques}"
    strength = strength.clip(0, 1)
    
    # if strength > 0.15:
    #     endurance = (1236.5/((strength*100 - 15)**0.618)) - 72.5
    # else:
    #     endurance = np.inf
    
    endurance = np.inf * np.ones_like(strength)
    endurance[strength > 0.15] = (1236.5/((strength[strength > 0.15]*100 - 15)**0.618)) - 72.5
    
    minimum_endurance = np.min(endurance)
    # TODO: take minimum of each muscle synergy, and then apply sum/mean
    
    return minimum_endurance
    
  def cost(self, model, data):
    # Calculate consumed endurance
    self._endurance = self.get_endurance(model, data)
    #total_time = data.time
    consumed_time = self._dt
    
    if self._endurance < np.inf:
        self._consumed_endurance = (consumed_time/self._endurance)*100
    else:
        self._consumed_endurance = 0.0
    
    self._effort_cost = self._weight*self._consumed_endurance
    return self._effort_cost

  def reset(self, model, data):
    #WARNING: bm_model.reset() should reset simulation time (i.e., data.time==0 before the next costs are calculated)
    pass

  def update(self, model, data):
    pass

  def _get_state(self, model, data):
    state = {"consumed_endurance": self._consumed_endurance,
             "effort_cost": self._effort_cost}
    return state

    
class MuscleState(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-4, **kwargs):
    super().__init__(bm_model)
    self._weight = weight

  def cost(self, model, data):
    return self._weight * np.sum(data.act ** 2)

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass


class Neural(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-4, **kwargs):
    super().__init__(bm_model)
    self._weight = weight
    self._effort_cost = None

  def cost(self, model, data):
    self._effort_cost = self._weight * np.sum(data.ctrl ** 2)
    return self._effort_cost

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass
  
  def _get_state(self, model, data):
    state = {"effort_cost": self._effort_cost}
    return state


class Zero(BaseEffortModel):

  def cost(self, model, data):
    return 0

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass