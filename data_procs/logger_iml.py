import time
import math
import pdb
import shutil
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Logger:
	def __init__(self, time_step, obs, map_init_pos, hd_init_ang, goal, margin):
		# time param
		self.dt = time_step
		self.t_tot = 0.0
		self.t_list = []

		# map initial
		self.map_init = map_init_pos
		self.hd_init  = hd_init_ang
		self.goal = np.array(goal)
		self.margin = margin

		# stance foot angle
		self.ang_stf_rob_glo_fram = None
		self.ang_stf_map_glo_fram = None

		self.nex_cur_hd = [0.0, 0.0, 0.0]
		self.mpc_hds_list = [0.0, 0.0, 0.0]

		self.list_ang_stf_rob_glo_fram = []
		self.list_ang_stf_map_glo_fram = []

		# base (from TSC wrapper states)
		self.pos_base_rob_glo_fram = None				# Based on robot global frame
		self.vel_base_rob_glo_fram = None				# Based on robot global frame
		self.hd_base_rob_glo_fram  = None				# Based on robot global frame

		self.pos_base_map_glo_fram = None				# Based on map global frame
		self.vel_base_map_glo_fram = None				# Based on map global frame
		self.hd_base_map_glo_fram  = None				# Based on map global frame
		self.body_vx	= None							# Based on robot body frame

		self.list_pos_base_rob_glo_fram = []
		self.list_vel_base_rob_glo_fram = []
		self.list_hd_base_rob_glo_fram	= []

		self.list_pos_base_map_glo_fram = []
		self.list_vel_base_map_glo_fram = []
		self.list_hd_base_map_glo_fram	= []

		# CoM (from TSC wrapper)
		self.pos_com_fot_fram = None					# Based on fot frame
		self.vel_com_fot_fram = None					# Based on fot frame

		self.pos_com_map_glo_fram = None				# Based on map global frame
		self.vel_com_map_glo_fram = None				# Based on map global frame
		self.state_com_map_glo 	  = None

		self.list_pos_com_fot_fram = []
		self.list_vel_com_fot_fram = []

		self.list_pos_com_map_glo_fram = []
		self.list_vel_com_map_glo_fram = []
		self.list_state_com_map_glo	   = []

		# Training dataset
		self.obs_info = np.ravel(obs)
		self.state_X = []
		self.mpc_res_y = []

		# stance foot
		self.leg_ind = -1
		self.pos_stf_rob_glo_frame = None				# Base on robot global frame
		self.pos_stf_map_glo_frame = None				# Base on map global frame

		self.list_pos_stf_rob_glo_frame = []
		self.list_pos_stf_map_glo_frame = []

		# swing foot
		self.pos_swf_rob_glo_frame = None
		self.list_pos_swf_rob_glo_frame = []

		# MPC-LIP
		self.lip_pred_pos = None						# Based on map global frame
		self.lip_pred_vel = None						# Based on map global frame
		self.mpc_state_tar  = None						# Based on map global frame
		self.fot_loc_map  = None						# Based on map global frame
		self.fot_loc_robo = None						# Based on robot global frame

		# Control relevent
		self.fot_loc_tar_map = None						# Based on map global frame
		self.fot_loc_tar_rob = None						# Based on robot global frame
		self.foot_input   = None						# Based on fot frame
		self.hd_input_pr  = None						# Based on robot global frame
		self.hd_input_cos = None						# Based on robot global frame
		self.nex_turn 	  = 0.0
		self.close2goal   = False

		self.list_lip_pred_pos = []
		self.list_lip_pred_vel = []
		self.list_fot_loc_map  = []
		self.list_fot_loc_robo = []
		self.list_foot_input   = []
		self.list_hd_input_pr  = []
		self.list_hd_input_cos = []
		self.list_fot_loc_tar_map = []
		self.list_fot_loc_tar_rob = []

	# Pos Trans: robot global to map global
	def pos_robo_glo_2_map_glo(self, pos_robo_glo):
		pos = np.array([pos_robo_glo]).T
		M_T = np.array([[math.cos(self.hd_init), -math.sin(self.hd_init)], 
				  		[math.sin(self.hd_init), math.cos(self.hd_init)]])
		# pos
		rot_pos = M_T @ pos
		trans   = np.array(self.map_init)
		map_pos = np.ravel(rot_pos) + trans
		return map_pos

	# Vel Trans: robot global to map global
	def vel_robo_glo_2_map_glo(self, vel_robo_glo):
		vel = np.array([vel_robo_glo]).T
		M_T = np.array([[math.cos(self.hd_init), -math.sin(self.hd_init)], 
				  		[math.sin(self.hd_init), math.cos(self.hd_init)]])
		# vel
		map_vel = np.ravel(M_T @ vel)
		return map_vel

	# Head angle Trans: robot global to map global
	def hd_robo_glo_2_map_glo(self, heading_glo):
		map_head = heading_glo + self.hd_init
		return map_head

	# Head angle Trans: map global to robot global
	def hd_map_glo_2_robo_glo(self, heading_map):
		robo_head = self.angle_A_minus_B(heading_map, self.hd_init)
		return robo_head
	
	# Pos Trans: map global to robot global
	def pos_map_glo_2_robo_glo(self, pos_map_glo):
		pos_map = np.array([pos_map_glo])
		trans 	 = np.array([self.map_init])
		robo_pos = pos_map - trans
		M = np.array([[math.cos(self.hd_init), math.sin(self.hd_init)], 
				  	  [-math.sin(self.hd_init), math.cos(self.hd_init)]])
		rot_pos = M @ robo_pos.T
		return np.ravel(rot_pos)
	
	# Vel Trans: map global to robot global
	def vel_map_glo_2_robo_glo(self, vel_map_glo):
		vel = np.array([vel_map_glo]).T
		M_T = np.array([[math.cos(self.hd_init), math.sin(self.hd_init)], 
				  		[-math.sin(self.hd_init), math.cos(self.hd_init)]])
		# vel
		robo_vel = np.ravel(M_T @ vel)
		return robo_vel

	# Pos Trans: fot loc to map global
	def pos_fot_loc_2_map_glo(self, fot_pos, map_hd):
		M_T = np.array([[math.cos(map_hd), -math.sin(map_hd)], 
						[math.sin(map_hd), math.cos(map_hd)]])
		
		map_pos = np.ravel(M_T @ np.array([self.pos_com_fot_fram]).T) + fot_pos
		return map_pos
	
	# Vel Trans: fot loc to map global
	def vel_fot_loc_2_map_glo(self, map_hd):
		M_T = np.array([[math.cos(map_hd), -math.sin(map_hd)], 
						[math.sin(map_hd), math.cos(map_hd)]])
		
		map_vel = np.ravel(M_T @ np.array([self.vel_com_fot_fram]).T)
		return map_vel
		
	# Angle difference
	def angle_A_minus_B(self, A, B):
		res_ang = A - B
		if res_ang < 0 and abs(res_ang) > math.pi:
			res_ang += 2*math.pi
		elif res_ang > 0 and abs(res_ang) > math.pi:
			res_ang -= 2*math.pi
		return res_ang

	# Trans quat to angle
	def quat_2_head(self, quat):
		x = quat[0]
		y = quat[1]
		z = quat[2]
		w = quat[3]

		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)
		if roll_x >= 0:
			map_hd = math.pi - roll_x
		else:
			map_hd = -roll_x -math.pi

		return map_hd


	# Get stance foot location in robot global from digit
	def get_cur_fot_loc(self, env):
		left_p, rigt_p = env.get_ft_glo_from_sensor()

		if self.leg_ind < 0:
			robo_glo_p_cur = left_p
			robo_glo_swf = rigt_p
		else:
			robo_glo_p_cur = rigt_p
			robo_glo_swf = left_p
		return robo_glo_p_cur, robo_glo_swf
	
	# Average three mpc heading angle difference and one predict differece.
	def avg_hd(self, cur_hd):
		sum_diff = self.nex_turn
		self.nex_cur_hd = [cur_hd] + self.mpc_hds_list[0:2]
		for i in range(len(self.nex_cur_hd)):
			diff = self.angle_A_minus_B(self.mpc_hds_list[i], self.nex_cur_hd[i])
			sum_diff += diff
		hd_avg = sum_diff/(i+1+1)
		return hd_avg
	

	# Update and record the robot states/ foot location current time step
	def update_n_record(self, env, leg_ind, dt):
		self.t_list.append(self.t_tot)
		self.t_tot += dt
		self.leg_ind = leg_ind
		# get current stance foot position and record
		self.pos_stf_rob_glo_frame, self.pos_swf_rob_glo_frame = self.get_cur_fot_loc(env)
		self.list_pos_stf_rob_glo_frame.append(self.pos_stf_rob_glo_frame)
		self.list_pos_swf_rob_glo_frame.append(self.pos_swf_rob_glo_frame)

		# get current stance foot position in map global and record
		self.pos_stf_map_glo_frame = self.pos_robo_glo_2_map_glo(self.pos_stf_rob_glo_frame)
		self.list_pos_stf_map_glo_frame.append(self.pos_stf_map_glo_frame)
		
		# read com data from digit wraper
		fot_frm_pos, fot_frm_vel, robo_quat = env.get_fot_loc_com_from_sensor()

		fot_pos 	= np.array(self.pos_stf_map_glo_frame)
		robo_hd 	= self.quat_2_head(robo_quat)
		map_hd      = self.hd_robo_glo_2_map_glo(robo_hd)
		# map_hd		= 0.0
		self.height = fot_frm_pos[2]
		self.pos_com_fot_fram = fot_frm_pos[0:2]
		self.vel_com_fot_fram = fot_frm_vel[0:2]
		self.hd_base_rob_glo_fram = robo_hd
		self.hd_base_map_glo_fram = map_hd
		self.pos_com_map_glo_fram = self.pos_fot_loc_2_map_glo(fot_pos, map_hd)
		self.vel_com_map_glo_fram = self.vel_fot_loc_2_map_glo(map_hd)
		# record com data
		self.list_pos_com_fot_fram.append(self.pos_com_fot_fram)
		self.list_vel_com_fot_fram.append(self.vel_com_fot_fram)
		self.list_hd_base_rob_glo_fram.append(self.hd_base_rob_glo_fram)
		self.list_hd_base_map_glo_fram.append(self.hd_base_map_glo_fram)
		self.list_pos_com_map_glo_fram.append(self.pos_com_map_glo_fram)
		self.list_vel_com_map_glo_fram.append(self.vel_com_map_glo_fram)

		# read base data from simu
		self.pos_base_rob_glo_fram, self.vel_base_rob_glo_fram = env.get_robo_glo_base_from_sensor()
		self.pos_base_map_glo_fram = self.pos_robo_glo_2_map_glo(self.pos_base_rob_glo_fram)
		self.vel_base_map_glo_fram = self.vel_robo_glo_2_map_glo(self.vel_base_rob_glo_fram)
		# record base data 
		self.list_pos_base_rob_glo_fram.append(self.pos_base_rob_glo_fram)
		self.list_vel_base_rob_glo_fram.append(self.vel_base_rob_glo_fram)
		self.list_pos_base_map_glo_fram.append(self.pos_base_map_glo_fram)
		self.list_vel_base_map_glo_fram.append(self.vel_base_map_glo_fram)

		self.body_vx = self.vel_com_map_glo_fram[0]*math.cos(map_hd)+\
					   self.vel_com_map_glo_fram[1]*math.sin(map_hd)
	

	# For MPC Method
	# Set heading angle input
	def set_stf_head(self, num_step):
		# Set stance foot
		self.ang_stf_rob_glo_fram = self.hd_base_rob_glo_fram
		self.ang_stf_map_glo_fram = self.hd_base_map_glo_fram
		self.list_ang_stf_rob_glo_fram.append(self.ang_stf_rob_glo_fram)
		self.list_ang_stf_map_glo_fram.append(self.ang_stf_map_glo_fram)

		cur_hd = self.hd_base_rob_glo_fram
		self.hd_input_cos = cur_hd
		self.nex_turn = self.tube_func(self.nex_turn, cur_hd)
		self.hd_input_pr = self.avg_hd(cur_hd)


	def tube_func(self, turning, init_tube_value):
		tube_upper = 0.15
		tube_lower = -0.15
		tube_value = init_tube_value
		d_head_value = turning
		print('turning ang', turning)
		print('tube_value', tube_value)
		if d_head_value > 0:
			if tube_upper > d_head_value:
				tube_value += 0.4*d_head_value
			else:
				tube_value += 0.7*d_head_value
		elif d_head_value < 0:
			if tube_lower < d_head_value:
				tube_value += 0.4*d_head_value
			else:
				tube_value += 0.7*d_head_value
		return self.angle_A_minus_B(tube_value, init_tube_value)
	

	# Make LIP prediction with current states 
	def predict_dt_state_traj(self, mpc_cbf1, rest_t):
		# Predict next states
		map_pos = self.list_pos_com_map_glo_fram[-1]
		map_vel = self.list_vel_com_map_glo_fram[-1]
		map_p 	= self.list_pos_stf_map_glo_frame[-1]
		map_p = np.append(map_p, self.hd_input_pr)
		map_hd = self.hd_base_map_glo_fram

		x_nex, temp_pos = mpc_cbf1.get_next_states(map_pos, map_vel, map_hd, map_p, rest_t)
		hd_ang_rob = self.hd_map_glo_2_robo_glo(x_nex[4])
		return x_nex, hd_ang_rob, temp_pos #### have an error in calculate hd_ang
	

	# Generate next foot step with predict states
	def gen_nex_foot_input(self, mpc_cbf1, leg_ind, rest_t, v_des_map, num_step):
		x_nex, hd_ang_rob, traj_rest = self.predict_dt_state_traj(mpc_cbf1,  rest_t)

		self.hd_tar_rob = hd_ang_rob
		self.lip_pred_pos = x_nex[0:2]
		self.lip_pred_vel = x_nex[2:4]
		nex_pos_rob = self.pos_map_glo_2_robo_glo(x_nex[0:2])
		nex_vel_rob = self.vel_map_glo_2_robo_glo(v_des_map)

		if self.mpc_state_tar == None:
			self.mpc_state_tar = np.ravel([x_nex, x_nex, x_nex])

		if num_step == 0:
			guess = np.ravel([self.mpc_state_tar[1], self.mpc_state_tar[2], self.mpc_state_tar[2]])
		else:
			guess = np.ravel(self.mpc_state_tar)

		# Generate new step
		x_mpc_tar, nex_contr, self.mpc_hds_list, close_2_goal, feasi, plan_traj = mpc_cbf1.gen_control_test(x_nex, (-1)*leg_ind, guess)
		# x_mpc_tar, nex_contr, self.mpc_hds_list, close_2_goal, feasi = mpc_cbf1.gen_control_test(x_nex, (-1)*leg_ind, guess, True)
		nex_stf_rob = self.pos_map_glo_2_robo_glo(nex_contr[0:2])

		self.nex_turn = nex_contr[2]
		self.mpc_state_tar = x_mpc_tar
		self.close2goal = close_2_goal
		self.fot_loc_map = nex_contr[0:2]
		self.fot_loc_robo = nex_stf_rob
		self.fot_loc_tar_map = self.fot_loc_map
		self.fot_loc_tar_rob = self.fot_loc_robo
		
		# Generate foot location input
		cur_stf_rob = self.pos_stf_rob_glo_frame
		fot_vec = np.array([nex_stf_rob - cur_stf_rob]).T
		cur_base_ang = self.hd_base_rob_glo_fram

		M_T = np.array([[math.cos(cur_base_ang), math.sin(cur_base_ang)], 
					  [-math.sin(cur_base_ang), math.cos(cur_base_ang)]])
		M = np.array([[math.cos(self.ang_stf_rob_glo_fram), math.sin(self.ang_stf_rob_glo_fram)], 
					  [-math.sin(self.ang_stf_rob_glo_fram), math.cos(self.ang_stf_rob_glo_fram)]])
		self.foot_input = np.ravel(M_T @ fot_vec)
		nex_pos_vec = np.array([nex_pos_rob - cur_stf_rob]).T
		self.nex_pos_fot_loc = np.ravel(M_T @ nex_pos_vec)
		self.nex_vel_fot_loc = np.ravel(M_T @ nex_vel_rob)

		# Decord data
		self.list_foot_input.append(self.foot_input)
		self.list_fot_loc_map.append(self.fot_loc_map)		
		self.list_lip_pred_pos.append(self.lip_pred_pos)		# continuous record data will plot a jump graph
		self.list_lip_pred_vel.append(self.lip_pred_vel)		# not a seperate trajectory
		self.list_fot_loc_robo.append(self.fot_loc_robo)
		self.list_fot_loc_tar_map.append(self.fot_loc_tar_map)
		self.list_fot_loc_tar_rob.append(self.fot_loc_tar_rob)

		# Get state_X for learning
		fea = self.obs_info
		fea = np.concatenate([fea, np.ravel(self.pos_com_map_glo_fram)])
		fea = np.concatenate([fea, np.ravel(self.vel_com_map_glo_fram)])
		fea = np.concatenate([fea, np.ravel(self.hd_base_map_glo_fram)])
		fea = np.concatenate([fea, np.ravel(self.pos_stf_map_glo_frame)])
		fea = np.concatenate([fea, np.ravel(self.goal)])
		fea = np.concatenate([fea, [leg_ind, rest_t]])
		if self.state_X == []:
			self.state_X = np.array([fea])
		else:
			self.state_X = np.concatenate([self.state_X, [fea]])
		print('state_X shape: ', self.state_X.shape)
		# pdb.set_trace()

		# Get MPC global for learning
		out_fea = np.ravel(self.fot_loc_map)
		out_fea = np.concatenate([out_fea, [0, self.hd_input_pr+self.hd_input_cos]])
		out_fea = np.concatenate([out_fea, np.ravel(x_nex[0:2])])
		out_fea = np.concatenate([out_fea, np.ravel(v_des_map)])
		if self.mpc_res_y == []:
			self.mpc_res_y = np.array([out_fea])
		else:
			self.mpc_res_y = np.concatenate([self.mpc_res_y, [out_fea]])
		print('mpc_res shape: ', self.mpc_res_y.shape)


		return traj_rest, x_mpc_tar[0][0:2], plan_traj, feasi


	def gen_tsc_control(self, i, n_cyc):
		head_ang_new = self.hd_input_pr/n_cyc*(i+4.5) + self.hd_input_cos
		c1 = [self.foot_input[0], self.foot_input[1], 0, head_ang_new]
		c2 = [self.nex_pos_fot_loc[0], self.nex_pos_fot_loc[1], self.nex_vel_fot_loc[0], 0]
		self.list_hd_input_pr.append(self.hd_input_pr)
		self.list_hd_input_cos.append(self.hd_input_cos)
		high_level_action = np.r_[c1, c2]
		return high_level_action
	
	def gen_act_res(self, cyc, gap):
		out_act_fea = []
		for i in range(int(cyc/gap)):
			act_fea = np.ravel(self.pos_stf_map_glo_frame)
			act_fea = np.concatenate([act_fea, [0, self.hd_base_map_glo_fram]])
			act_fea = np.concatenate([act_fea, self.pos_com_map_glo_fram])
			act_fea = np.concatenate([act_fea, self.vel_com_map_glo_fram])
			if out_act_fea == []:
				out_act_fea = np.array([act_fea])
			else:
				out_act_fea = np.concatenate([out_act_fea, [act_fea]])

		return out_act_fea

	# Tools Functions
	# Print 
	def print_states(self, i, num_step):
		# print('----relative time----')
		# print('t: ', 0.4-i)
		# print('remain t: ', i)
		print('num of step', num_step)




        
