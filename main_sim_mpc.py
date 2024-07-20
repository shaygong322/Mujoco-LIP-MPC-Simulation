import numpy as np
import time
import pdb
import os
import shutil
import math
from matplotlib import pyplot as plt

import MPC_LIP_modi
import MPC_LIP_cbf
import gymnasium as gym
from gymnasium.envs.registration import register
from data_procs.logger_mpc import Logger


if __name__ == "__main__":

	# Initilize MPC 
	hd_init  = 0
	leg_ind  = -1															# negative left, positive right
	step_t	 = 0.4
	safe_dis = 0.4
	goal 	 = [[10, 10]]
	map_init = [0.0, 0.0]
	margin 	 = [-0.5, 10.5]
	path = 'data_log/LIP_mexy_'
	obs_cir_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [6, 4, 0.8], [6.4, 7.2, 1.0], [4.8, 0.8, 0.4],
					     [2, 6, 0.3]])
	obs_elp_list = []
	obs_elp_safe = []

	# obs_elp_list = np.array([[4.8, 0.8, 0.4, 0.32, math.pi/12],
    #         [1.5, 1.5, 1.1, 0.7, math.pi/4],
    #         [3, 5, 0.6, 1.2, math.pi/3]])

	# obs_cir_list = np.array([[6, 4, 0.8], 
    #         [6.4, 7.2, 1.0],
    #         [8, 2, 0.3]])

	# obs_cir_list =  [[1.83, 4.14, 0.83], [2.9, 1.23, 0.94], [4.43, 8.02, 0.87], [6.12, 6.53, 0.51]]
	# obs_elp_list =  [[4.88, 4.48, 0.85, 0.65, 0.19], [0.32, 2.58, 0.25, 0.22, 2.74], [8.7, 3.47, 0.96, 0.52, 0.47], [8.37, 9.47, 0.52, 0.29, 3.09]]

	obs_cir_safe = np.array(obs_cir_list) + [0, 0, safe_dis]
	# obs_elp_safe = np.array(obs_elp_list) + [0, 0, safe_dis, safe_dis, 0]
	mpc_cbf1 = MPC_LIP_modi.MPCCBF(goal, obs_cir_list, obs_cir_safe, obs_elp_list, obs_elp_safe, margin)

	# Initilize logger
	logger = Logger(step_t, map_init, hd_init, goal, margin)

	# Digit initilize
	start_timer = time.time()
	render_mode = True
	dynamics_randomization = False

	# Fixed torso and rope mode added for debugging purposes
	fb_idx = [0, 1, 2, 3, 4, 5, 6]
	fb_vel_idx = [0, 1, 2]
	fb_pos = np.array([0, 0.0, 0.950181, 1.0, 0.0, 0.0, 0.0])  #[0, 1, 0]
	fb_vel = np.array([0, 0, 0 ])

	# Register Custom Digit environment -------------------------------------------------------------------------------------------------------------
	register(id='Digit-v1',
			entry_point='digit.digit_tsc_nosprings:DigitEnv',
			kwargs =   {'dynamics_randomization': dynamics_randomization}) 	
	

	#Create environment
	if render_mode:	
		env = gym.make('Digit-v1', render_mode="human")
	else:
		env = gym.make('Digit-v1')	

	state, info = env.reset()
	done = False 

	# Define simulation parameters/conditions
	max_traj_len = 150000 #max_episode_length in walking steps
	i = 0
	f_cyc = 40
	traj_len = 0
	num_step = 1
	torso_fixed = False
	real_close = False
	action = np.zeros(20)
	real_str_traj = []
	feasi_traj_list = []
	fail_traj_list = []
	full_traj_list = []
	pre_traj_list = []
	pred_str_traj_list = []

	# init vel desire
	vel_des = mpc_cbf1.alip_des_vel(0.6, leg_ind)
	vel_log = []

	while traj_len < max_traj_len:

		if torso_fixed == True:
			pos = np.copy(env.sim.data.qpos)
			vel = np.copy(env.sim.data.qvel)
			pos[fb_idx] = fb_pos
			vel[fb_vel_idx] = fb_vel
			env.set_state(pos,vel)
		
		logger.update_n_record(env, leg_ind, step_t/f_cyc)
		rest_t = step_t - (i)*(step_t/f_cyc)

		if i == 0:
			logger.set_stf_head(num_step)

		if np.mod(i, 1) == 0:
			print('++++++++++++++++++++++++++++++')
			print('mpc')
			traj_rest, nex_pos, plan_traj, feasi = logger.gen_nex_foot_input(mpc_cbf1, leg_ind, rest_t, vel_des, num_step)
			pre_traj_list.append(traj_rest)
			vel_des = logger.mpc_state_tar[0][2:4]
			logger.print_states(rest_t, num_step)

		high_level_action = logger.gen_tsc_control(i, f_cyc)
		
		last_stance_sign = env.stance_sign
		next_state, reward, done, info = env.step(high_level_action)
		i += 1
		
		if last_stance_sign != env.stance_sign:
			# Plot trajectories
			print('@@@@@@@@@@@@@@@@@@@@@@@@@ change foot @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
			# if num_step >= 27:
			# 	logger.plot_each_pre_trajects(pre_traj_list, real_traj, af_pred_traj_list, obs_list)
			# logger.plot_each_pre_trajects(pre_traj_list, real_traj, af_pred_traj_list, obs_list)
			
			# Update index
			i = 0
			num_step += 1
			leg_ind = -1*leg_ind
			vel_des = logger.mpc_state_tar[1][2:4]
			real_str_traj.append(logger.pos_com_map_glo_fram)
			nex_traj = np.array([logger.pos_com_map_glo_fram, nex_pos])
			pred_str_traj_list.append(nex_traj)
			full_traj_list.append(plan_traj)
			if feasi != 2:
				feasi_traj_list.append(plan_traj)
			else:
				fail_traj_list.append(plan_traj)
			pre_traj_list = []

			if real_close:
				print('close')
				break

		if logger.height < 0.4:
			print('fall')
			break

		if logger.close2goal:
			real_close = True
		# dis2goal = math.sqrt((logger.pos_com_map_glo_fram[0]-goal[0][0])**2 +\
		# 			   (logger.pos_com_map_glo_fram[1]-goal[0][1])**2) 

		# if dis2goal <= 0.25:
		# 	print('close')
		# 	break

		if render_mode and traj_len % 2 == 0:
			env.render()
		traj_len += 1

	elapsed_time = time.time() - start_timer
	print("elapsed time = ", elapsed_time)
	logger.plot_each_pre_trajects(pre_traj_list, real_str_traj, pred_str_traj_list, obs_cir_list, \
							   obs_elp_list, feasi_traj_list, fail_traj_list, full_traj_list, path)

