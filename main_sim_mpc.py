import numpy as np
import time
import MPC_LIP_modi
import gymnasium as gym
from gymnasium.envs.registration import register
from data_procs.logger_mpc import Logger


if __name__ == "__main__":
	# Obstacle list and arrays
	safe_dis = 0.4
	obs_cir_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [6, 4, 0.8], [6.4, 7.2, 1.0], [4.8, 0.8, 0.4],
					     [2, 6, 0.3]])
	obs_cir_safe = np.array(obs_cir_list) + [0, 0, safe_dis]
	obs_elp_list = []
	obs_elp_safe = []

	# Initilize MPCCBF
	goal = [[10, 10]]
	margin = [-0.5, 10.5]
	mpc_cbf1 = MPC_LIP_modi.MPCCBF(goal, obs_cir_list, obs_cir_safe, obs_elp_list, obs_elp_safe, margin)

	# Initilize Logger
	step_t = 0.4
	map_init = [0.0, 0.0]
	hd_init = 0.0
	logger = Logger(step_t, map_init, hd_init, goal, margin)

	# Register Custom Digit environment with gym
	dynamics_randomization = False # Set to False to enable dynamics randomization
	register(id='Digit-v1', 
		  entry_point='digit.digit_tsc_nosprings:DigitEnv', 
		  kwargs={'dynamics_randomization': dynamics_randomization})
	render = True
	env = gym.make('Digit-v1', render_mode="human") # Set render_mode to "human" to render the simulation
	state, info = env.reset()

	# Define simulation parameters/conditions
	max_traj_len = 150000 # max_episode_length in walking steps
	i = 0
	f_cyc = 40
	leg_ind = -1 # negative left, positive right
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
	vel_des = mpc_cbf1.alip_des_vel(0.6, leg_ind)
	vel_log = []
	fb_idx = [0, 1, 2, 3, 4, 5, 6]
	fb_vel_idx = [0, 1, 2]
	fb_pos = np.array([0, 0.0, 0.950181, 1.0, 0.0, 0.0, 0.0])
	fb_vel = np.array([0, 0, 0 ])
	path = 'data_log/LIP_mexy_' # Path to save data
	done = False  # Task done flag
	start_timer = time.time() # Start timer
	
	# Main loop
	while traj_len < max_traj_len:

		if torso_fixed == True:
			pos = np.copy(env.sim.data.qpos)
			vel = np.copy(env.sim.data.qvel)
			pos[fb_idx] = fb_pos
			vel[fb_vel_idx] = fb_vel
			env.set_state(pos,vel)
		
		# Update logger
		logger.update_n_record(env, leg_ind, step_t/f_cyc)

		# Calculate rest time
		rest_t = step_t - i*(step_t/f_cyc)

		# If first step, set the head of the STF
		if i == 0:
			logger.set_stf_head(num_step)

		# If the step is a multiple of 1(which is always true), generate the next foot input
		if np.mod(i, 1) == 0:
			print('++++++++++++++++++++++++++++++')
			print('mpc')
			traj_rest, nex_pos, plan_traj, feasi = logger.gen_nex_foot_input(mpc_cbf1, leg_ind, rest_t, vel_des, num_step)
			pre_traj_list.append(traj_rest)
			vel_des = logger.mpc_state_tar[0][2:4]
			logger.print_states(rest_t, num_step)

		# Generate high level action
		high_level_action = logger.gen_tsc_control(i, f_cyc)
		
		# Log last stance sign
		last_stance_sign = env.stance_sign

		# Take a step in the environment
		next_state, reward, done, info = env.step(high_level_action)

		# Update i
		i += 1
		
		# Check if foot has changed
		if last_stance_sign != env.stance_sign:
			# Plot trajectories
			print('@@@@@@@@@@@@@@@@@@@@@@@@@ change foot @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
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

		# Check if the robot has fallen
		if logger.height < 0.4:
			print('fall')
			break

		# Check if the robot is really close to the goal
		if logger.close2goal:
			real_close = True

		if render and traj_len % 2 == 0:
			env.render()
		traj_len += 1

	elapsed_time = time.time() - start_timer
	print("elapsed time = ", elapsed_time)
	logger.plot_each_pre_trajects(pre_traj_list, real_str_traj, pred_str_traj_list, obs_cir_list, \
							   obs_elp_list, feasi_traj_list, fail_traj_list, full_traj_list, path)
