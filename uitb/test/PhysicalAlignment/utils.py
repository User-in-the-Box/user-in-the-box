import pandas as pd
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

## used for coordinate transformations
def transformation_matrix(pos, quat):
  quat = np.roll(quat, -1)
  matrix = Rotation.from_quat(quat).as_matrix()

  # Create the matrix
  T = np.eye(4)
  T[:3, :3] = matrix
  T[:3, 3] = pos
  return T


def ReachEnvelopeCheck(env,
                       use_vr_controller,
                       welded_body,
                       relpose,
                       endeffector_name,
                       trajectories_table_columns,
                       num_episodes,
                       video_output: bool,
                       figure_filename,
                       table_filename,
                       video_filename):
    #return 123

    trajectories_table = pd.DataFrame(columns=trajectories_table_columns)
    #with imageio.get_writer(video_filename, fps=int(1 / (env._model.opt.timestep))) as video:
    video = []

    for video_episode_index in range(num_episodes):
        print('Video - Episode {}/{}'.format(video_episode_index + 1, num_episodes))
        obs, info = env.reset()
        print("Successfully reset.")
        if video_output:
            video.append(env.render())

        nsteps_to_test = 100
        ##TODO: generalize to other biom. models (allow to define initial posture and list of joints over which to iterate)
        range_elevation_angle = np.linspace(
            env._model.joint('elv_angle').range[0],
            env._model.joint('elv_angle').range[1], nsteps_to_test)
        range_elevation = np.linspace(
            env._model.joint('shoulder_elv').range[0],
            env._model.joint('shoulder_elv').range[1], nsteps_to_test)
        _steps = 0
        for i in range(nsteps_to_test):
            for j in range(nsteps_to_test):
                if (100*(_steps+1)/(nsteps_to_test**2) % 10) == 0:
                    print(f"{100*(_steps+1)/(nsteps_to_test**2)}% reached.")
                
                # if use_vr_controller:
                #     obs, info = env.reset()
                env._data.joint('elv_angle').qpos = range_elevation_angle[i]
                env._data.joint('shoulder_elv').qpos = range_elevation[j]
                
                # maximally extend the arm:
                env._data.joint("elbow_flexion").qpos = env._model.joint("elbow_flexion").range[0]

                # adjust virtual joints according to active constraints (MuJoCo 2.x eq_active / 3.x+ eq_active0):
                eq_active = getattr(env._model, "eq_active0", getattr(env._model, "eq_active", None))
                for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
                        env._model.eq_obj1id[
                            (env._model.eq_type == 2) & (eq_active == 1)],
                        env._model.eq_obj2id[
                            (env._model.eq_type == 2) & (eq_active == 1)],
                        env._model.eq_data[(env._model.eq_type == 2) &
                                                    (eq_active == 1), 4::-1]):
                    if physical_joint_id >= 0:
                        env._data.joint(virtual_joint_id).qpos = np.polyval(poly_coefs, env._data.joint(physical_joint_id).qpos)
                # qpos[env._model.joint_name2id('shoulder1_r2')] = -qpos[env._model.joint_name2id(
                #     'elv_angle')]  # joint equality constraint between "elv_angle" and "shoulder1_r2"

                #env._data.qpos[:] = qpos
                # if use_vr_controller:
                #     mujoco.mj_step(env._model, env._data)
                # else:
                #     mujoco.mj_forward(env._model, env._data)
                mujoco.mj_forward(env._model, env._data)
                if video_output:
                    video.append(env.render())
                trajectories_table.loc[len(trajectories_table), :"shoulder_elv_pos"] = np.concatenate((env._data.joint('elv_angle').qpos, env._data.joint('shoulder_elv').qpos))
                if use_vr_controller:
                    #manually compute VR controller position using "right_controller_relpose" from config file and current position of the aux body/welded body
                    T1 = transformation_matrix(pos=env._data.body(welded_body).xpos, quat=env._data.body(welded_body).xquat)
                    T2 = transformation_matrix(pos=relpose[:3], quat=relpose[3:])
                    T = np.matmul(T1, np.linalg.inv(T2))
                    T[:3, 3]
                    trajectories_table.loc[len(trajectories_table) - 1,
                                "end-effector_xpos_x":"end-effector_xpos_z"] = T[:3, 3]
                    # _controller_quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)
                else:
                    trajectories_table.loc[len(trajectories_table) - 1,
                        "end-effector_xpos_x":"end-effector_xpos_z"] = env._data.body(endeffector_name).xpos
                # print(env._data.body(endeffector_name).xpos)

                _steps += 1
    
    ## -> Store csv file
    trajectories_table.to_csv(table_filename)
    print(f"ReachEnvelope data stored at '{table_filename}'.")


    ## -> Store mp4 file
    print("\nCreating ReachEnvelope video...")
    # temporarily override fps
    _orig_fps = env._GUI_camera._fps
    env._GUI_camera._fps = 100
    env._GUI_camera.write_video_set_path(video_filename)
    # Write the video
    # simulator._camera.write_video(imgs, os.path.join(evaluate_dir, args.out_file))
    for _img in video:
        env._GUI_camera.write_video_add_frame(_img)
    env._GUI_camera.write_video_close()
    # reset to original fps from env
    env._GUI_camera._fps = _orig_fps
    print(f"ReachEnvelope video stored at '{video_filename}'.")
