import numpy as np
import time
import pdb
import os
import shutil
import math
from matplotlib import pyplot as plt

import MPC_LIP_sig_step
import gymnasium as gym
from gymnasium.envs.registration import register
from data_procs.logger import Logger


if __name__ == "__main__":

	# Initilize MPC 
	hd_init  = 0
	leg_ind  = -1															# negative left, positive right
	step_t	 = 0.4
	safe_dis = 0.4
	goal 	 = [[10, 10]]
	map_init = [0.0, 0.0]
	margin 	 = [-0.5, 10.5]

	obs_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [6, 4, 0.8], [6.4, 7.2, 1.0], [4.8, 0.8, 0.4],
					     [2, 6, 0.3]])
	# obs_list = np.array([[3, 0, 0.5]])
	obs_safe = obs_list
	obs_safe = obs_safe + [0, 0, safe_dis]
	mpc_cbf1 = MPC_LIP_sig_step.MPCCBF(goal, obs_list, obs_safe, margin)

	# Initilize logger
	logger = Logger(step_t, map_init, hd_init, goal)

	# Digit initilize
	start_timer = time.time()
	render_mode = True
	dynamics_randomization = False

	# Fixed torso and rope mode added for debuggind purposes
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
	max_traj_len = 1500 #max_episode_length in walking steps
	i = 0
	traj_len = 0
	num_step = 1
	torso_fixed = False
	action = np.zeros(20)
	real_traj = []
	pre_traj_list = []
	af_pred_traj_list = []

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
		
		logger.update_n_record(env, leg_ind)
		rest_t = step_t - (i)*0.02

		# Bug testing
		if rest_t < 0:
			print(num_step)
			logger.plot_each_pre_trajects(pre_traj_list, real_traj, af_pred_traj_list, obs_list)

		if i == 0:
			logger.set_stf_head(num_step)

		s_mpc = 15
		if i == s_mpc:
			print('++++++++++++++++++++++++++++++')
			print('mpc')
			traj_rest = logger.gen_nex_foot_input(mpc_cbf1, leg_ind, rest_t, num_step)
			vel_des = logger.mpc_state_tar[0][2:4]
		else:
			print('++++++++++++++++++++++++++++++')
			print('alip')
			traj_rest, af_pred_traj = logger.cal_foot_input(mpc_cbf1, rest_t, vel_des, num_step)
			af_pred_traj_list.append(af_pred_traj)
		
		real_traj.append(logger.pos_com_map_glo_fram)
		pre_traj_list.append(traj_rest)
		high_level_action = logger.gen_tsc_control(i)
		
		logger.print_states(rest_t, num_step)

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
			real_traj = []
			pre_traj_list = []
			af_pred_traj_list = []
			
		if logger.close2goal:
			break

		if render_mode and traj_len % 2 == 0:
			env.render()
		traj_len += 1

	elapsed_time = time.time() - start_timer
	print("elapsed time = ", elapsed_time)
	logger.plot_each_pre_trajects(pre_traj_list, real_traj, af_pred_traj_list, obs_list)

