import time
import math
import pickle
import shutil
import numpy as np
from matplotlib import pyplot as plt

class Logger:
	def __init__(self, time_step, map_init_pos, hd_init_ang, goal):
		# time param
		self.dt = 0.02
		self.t_tot = 0.0
		self.t_list = []

		# map initial
		self.map_init = map_init_pos
		self.hd_init  = hd_init_ang
		self.goal = np.array(goal)

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
		self.body_vx			   = None				# Based on robot body frame

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
		self.close2goal   = False

		self.list_lip_pred_pos = []
		self.list_lip_pred_vel = []
		# self.list_mpc_pos_tar  = []
		# self.list_mpc_vel_tar  = []
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
	def avg_hd(self, tar_hd1, cur_hd):
		sum_diff = self.angle_A_minus_B(tar_hd1, cur_hd)
		print('init_tar', sum_diff)
		self.nex_cur_hd = [cur_hd] + self.mpc_hds_list[0:2]
		print('cur_hd list', self.nex_cur_hd)
		print('mpc_hd list', self.mpc_hds_list)
		for i in range(len(self.nex_cur_hd)):
			diff = self.angle_A_minus_B(self.mpc_hds_list[i], self.nex_cur_hd[i])
			print('mpc_diff', diff)
			sum_diff += diff
		hd_avg = sum_diff/(i+1+1)
		print('avg_hd', hd_avg)
		return hd_avg
	

	# Update and record the robot states/ foot location current time step
	def update_n_record(self, env, leg_ind):
		self.t_list.append(self.t_tot)
		self.t_tot += self.dt
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
	def set_stf_head(self, mpc_cbf1, num_step):
		self.step_init_pos = self.pos_com_map_glo_fram
		x_nex, hd_ang_rob, hd_ang_map, temp_pos = self.predict_dt_state_traj(mpc_cbf1, self.dt)

		# Set stance foot angle
		self.ang_stf_rob_glo_fram = self.hd_base_rob_glo_fram
		self.ang_stf_map_glo_fram = self.hd_base_map_glo_fram
		self.list_ang_stf_rob_glo_fram.append(self.ang_stf_rob_glo_fram)
		self.list_ang_stf_map_glo_fram.append(self.ang_stf_map_glo_fram)

		cur_hd = self.hd_base_rob_glo_fram
		# Generate hd input
		if num_step <= 3:
			self.hd_input_cos = 0.0
			self.hd_input_pr = 0.0
		else:
			self.hd_input_cos = cur_hd
			self.hd_input_pr = self.avg_hd(hd_ang_rob, cur_hd)
			# if self.hd_input_pr > 0.3:
			# 	self.hd_input_pr = 0.3
			# elif self.hd_input_pr < -0.3:
			# 	self.hd_input_pr = -0.3
		
		# Generate heading target

	# update hd input each time
	def set_hd_input(self, hd_rob_tar, num_step):
		cur_hd = self.hd_base_rob_glo_fram
		# Generate hd input
		if num_step <= 3:
			self.hd_input_cos = 0.0
			self.hd_input_pr = 0.0
		else:
			self.hd_input_cos = cur_hd
			self.hd_input_pr = self.angle_A_minus_B(hd_rob_tar, cur_hd)
			# if self.hd_input_pr > 0.3:
			# 	self.hd_input_pr = 0.3
			# elif self.hd_input_pr < -0.3:
			# 	self.hd_input_pr = -0.3
		
		

	# Make LIP prediction with current states 
	def predict_dt_state_traj(self, mpc_cbf1, rest_t):
		# Predict next states
		map_pos = self.list_pos_com_map_glo_fram[-1]
		map_vel = self.list_vel_com_map_glo_fram[-1]
		map_p 	= self.list_pos_stf_map_glo_frame[-1]

		x_nex, hd_ang_map, temp_pos = mpc_cbf1.get_next_states(map_pos, map_vel, map_p, rest_t, self.step_init_pos)
		hd_ang_rob = self.hd_map_glo_2_robo_glo(hd_ang_map)
		return x_nex, hd_ang_rob, hd_ang_map, temp_pos #### have an error in calculate hd_ang
	

	# Generate next foot step with predict states
	def gen_nex_foot_input(self, mpc_cbf1, leg_ind, rest_t, num_step):
		x_nex, hd_ang_rob, hd_ang_map, traj_rest = self.predict_dt_state_traj(mpc_cbf1,  rest_t)
		# set heading input
		# self.set_hd_input(hd_ang_rob, num_step)

		self.hd_tar_rob = hd_ang_rob
		self.lip_pred_pos = x_nex[0:2]
		self.lip_pred_vel = x_nex[2:4]
		nex_pos_rob = self.pos_map_glo_2_robo_glo(x_nex[0:2])
		nex_vel_rob = self.vel_map_glo_2_robo_glo(x_nex[2:4])

		# Generate new step
		x_mpc_tar, nex_stf_map, self.mpc_hds_list, close_2_goal = mpc_cbf1.gen_control_test(x_nex, hd_ang_map, (-1)*leg_ind, self.mpc_state_tar)
		# x_mpc_tar, nex_stf_map, self.mpc_hds_list, close_2_goal = mpc_cbf1.gen_control_test(x_nex, hd_ang_map, (-1)*leg_ind, self.mpc_state_tar, True)
		nex_stf_rob = self.pos_map_glo_2_robo_glo(nex_stf_map)

		self.close2goal = close_2_goal
		self.fot_loc_map = nex_stf_map
		self.fot_loc_robo = nex_stf_rob
		self.mpc_state_tar = x_mpc_tar
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
		# self.nex_vel_fot_loc = [mpc_cbf1.bvx_max, 0]

		# Decord data
		self.list_foot_input.append(self.foot_input)
		# self.list_mpc_pos_tar.append(self.mpc_pos_tar)
		# self.list_mpc_vel_tar.append(self.mpc_vel_tar)
		self.list_fot_loc_map.append(self.fot_loc_map)
		self.list_lip_pred_pos.append(self.lip_pred_pos)		# continuous record data will plot a jump graph
		self.list_lip_pred_vel.append(self.lip_pred_vel)		# not a seperate trajectory
		self.list_fot_loc_robo.append(self.fot_loc_robo)
		self.list_hd_input_cos.append(self.hd_input_cos)
		self.list_fot_loc_tar_map.append(self.fot_loc_tar_map)
		self.list_fot_loc_tar_rob.append(self.fot_loc_tar_rob)

		return traj_rest
	
	# Calculate foot location with desire pos
	def cal_foot_input(self, mpc_cbf1, rest_t, v_des_map, num_step):
		x_nex, hd_ang_rob, hd_ang_map, traj_rest = self.predict_dt_state_traj(mpc_cbf1, rest_t)
		# set heading input
		# self.set_hd_input(hd_ang_rob, num_step)
		self.hd_tar_rob = hd_ang_rob

		nex_pos_rob = self.pos_map_glo_2_robo_glo(x_nex[0:2])
		# nex_vel_rob = self.vel_map_glo_2_robo_glo(x_nex[2:4])
		nex_vel_rob = self.vel_map_glo_2_robo_glo(v_des_map)

		# Generate foot input
		cur_stf_rob = self.pos_stf_rob_glo_frame
		cur_base_ang = self.hd_base_rob_glo_fram
		
		nex_stf_map = mpc_cbf1.cal_foot_with_veldes(x_nex, v_des_map)
		nex_stf_rob = self.pos_map_glo_2_robo_glo(np.ravel(nex_stf_map))

		vec_fot = np.array([nex_stf_rob - cur_stf_rob]).T
		M_T = np.array([[math.cos(cur_base_ang), math.sin(cur_base_ang)], 
					  [-math.sin(cur_base_ang), math.cos(cur_base_ang)]])
		
		M = np.array([[math.cos(self.ang_stf_rob_glo_fram), math.sin(self.ang_stf_rob_glo_fram)], 
					  [-math.sin(self.ang_stf_rob_glo_fram), math.cos(self.ang_stf_rob_glo_fram)]])
		self.foot_input = np.ravel(M_T @ vec_fot)
		nex_pos_vec = np.array([nex_pos_rob - cur_stf_rob]).T
		self.nex_pos_fot_loc = np.ravel(M_T @ nex_pos_vec)
		self.nex_vel_fot_loc = np.ravel(M_T @ nex_vel_rob)
		# self.nex_vel_fot_loc = [mpc_cbf1.bvx_max, 0]
		self.fot_loc_tar_map = nex_stf_map
		self.fot_loc_tar_rob = nex_stf_rob
		self.list_foot_input.append(self.foot_input)
		self.list_fot_loc_tar_map.append(self.fot_loc_tar_map)
		self.list_fot_loc_tar_rob.append(self.fot_loc_tar_rob)

		# Test: draw the traject with next stace foot to x_des
		map_pos = np.ravel(x_nex[0:2])
		map_vel = np.ravel(x_nex[2:4])
		x_nex_nex, hd_ang_map, traj_nex = mpc_cbf1.get_next_states(map_pos, map_vel, nex_stf_map, self.dt, map_pos)

		return traj_rest, traj_nex


	# For ALIP Method
	def alip_planning(self, mpc_cbf1, leg_ind, rest_t):
		x_nex, hd_ang_rob, hd_ang_map, temp_pos = self.predict_dt_state_traj(mpc_cbf1, rest_t)
		# print('nex state', x_nex)
		cur_stf_rob = self.pos_stf_rob_glo_frame
		self.state_com_map_glo = x_nex

		vel_des = mpc_cbf1.alip_des_vel(leg_ind)
		# print('vel_des', vel_des)
		nex_stf_map = mpc_cbf1.cal_foot_with_veldes(x_nex, vel_des)
		nex_stf_rob = self.pos_map_glo_2_robo_glo(np.ravel(nex_stf_map))
		self.foot_input = np.array([nex_stf_rob - cur_stf_rob]).T
		self.fot_loc_tar_map = nex_stf_map
		self.fot_loc_tar_rob = nex_stf_rob

		self.list_foot_input.append(self.foot_input)
		self.list_fot_loc_tar_map.append(self.fot_loc_tar_map)
		self.list_fot_loc_tar_rob.append(self.fot_loc_tar_rob)
		self.list_state_com_map_glo.append(self.state_com_map_glo)

		map_pos = np.ravel(x_nex[0:2])
		map_vel = np.ravel(x_nex[2:4])
		x_nex_nex, hd_ang_map, trajec_nex = mpc_cbf1.get_next_states(map_pos, map_vel, nex_stf_map, self.dt, map_pos)
		return temp_pos, trajec_nex, vel_des
	

	# For mpc method only
	def mpc_nex_foot_input(self, mpc_cbf1, leg_ind, rest_t, num_step):
		x_nex, hd_ang_rob, hd_ang_map, traj_rest = self.predict_dt_state_traj(mpc_cbf1,  rest_t)
		# set heading input
		# self.set_hd_input(hd_ang_rob, num_step)

		self.hd_tar_rob = hd_ang_rob
		self.lip_pred_pos = x_nex[0:2]
		self.lip_pred_vel = x_nex[2:4]
		nex_pos_rob = self.pos_map_glo_2_robo_glo(x_nex[0:2])
		nex_vel_rob = self.vel_map_glo_2_robo_glo(x_nex[2:4])

		# Generate new step
		# x_mpc_tar, nex_stf_map, close_2_goal = mpc_cbf1.gen_control_test(x_nex, hd_ang_map, (-1)*leg_ind)
		x_mpc_tar, nex_stf_map, close_2_goal = mpc_cbf1.gen_control_test(x_nex, hd_ang_map, (-1)*leg_ind, True)
		nex_stf_rob = self.pos_map_glo_2_robo_glo(nex_stf_map)

		self.close2goal = close_2_goal
		self.fot_loc_map = nex_stf_map
		self.fot_loc_robo = nex_stf_rob
		self.mpc_state_tar = x_mpc_tar
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
		# self.nex_vel_fot_loc = np.ravel(M_T @ nex_vel_rob)
		self.nex_vel_fot_loc = [mpc_cbf1.bvx_max, 0]

		# Decord data
		self.list_foot_input.append(self.foot_input)
		# self.list_mpc_pos_tar.append(self.mpc_pos_tar)
		# self.list_mpc_vel_tar.append(self.mpc_vel_tar)
		self.list_fot_loc_map.append(self.fot_loc_map)
		self.list_lip_pred_pos.append(self.lip_pred_pos)		# continuous record data will plot a jump graph
		self.list_lip_pred_vel.append(self.lip_pred_vel)		# not a seperate trajectory
		self.list_fot_loc_robo.append(self.fot_loc_robo)
		self.list_fot_loc_tar_map.append(self.fot_loc_tar_map)
		self.list_fot_loc_tar_rob.append(self.fot_loc_tar_rob)

		return traj_rest


	def gen_tsc_control(self, i):
		# exp turning signal
		# tau = 0.2
		# head_ang_new = self.hd_input_pr*(1-math.exp(-tau*i)) + self.hd_input_cos
		head_ang_new = self.hd_input_pr/20*(i+4.5) + self.hd_input_cos
		c1 = [self.foot_input[0], self.foot_input[1], 0, head_ang_new]
		# c2 = [self.nex_pos_fot_loc[0], self.nex_pos_fot_loc[1], self.nex_vel_fot_loc[0], -self.nex_vel_fot_loc[1]]
		c2 = [self.nex_pos_fot_loc[0], self.nex_pos_fot_loc[1], self.nex_vel_fot_loc[0], 0]
		self.list_hd_input_pr.append(self.hd_input_pr)
		self.list_hd_input_cos.append(self.hd_input_cos)
		high_level_action = np.r_[c1, c2]
		return high_level_action

	# Tools Functions

	# Print 
	def print_states(self, i, num_step):
		print('----relative time----')
		print('t: ', 0.4-i)
		print('remain t: ', i)
		print('num of step', num_step)
		print('-------robot states-------')
		print('stf pos map', self.pos_stf_map_glo_frame)
		print('swf pos rob', self.pos_swf_rob_glo_frame)
		print('stf hed map', self.ang_stf_map_glo_fram)
		print('stf hed robo', self.ang_stf_rob_glo_fram)
		print('rob pos map', self.pos_com_map_glo_fram)
		print('rob vel map', self.vel_com_map_glo_fram)
		print('rob hd map', self.hd_base_map_glo_fram)
		print('rob hd robo', self.hd_base_rob_glo_fram)
		# print('next state', self.state_com_map_glo)
		print('body velo x', self.body_vx)
		print('-------control input--------')
		print('next pos/(vel) des', np.matrix(self.mpc_state_tar))
		print('next vel des', self.nex_vel_fot_loc)
		print('tar angle', self.hd_tar_rob)
		print('next foot map', self.fot_loc_tar_map)
		print('next foot rob', self. fot_loc_tar_rob)
		print('foot input', self.foot_input)
		print('hd input cos(base)', self.hd_input_cos)
		print('hd input pr(tar)', self.hd_input_pr)

	def plot_cir(self, cir):
		theta = np.linspace(0, 2 * np.pi, 100)
		x = cir[2]*np.cos(theta)+cir[0]
		y = cir[2]*np.sin(theta)+cir[1]
		plt.plot(x, y, color='#696969', linewidth=2.5)

	# Plot predict and actual trajectories
	def plot_trajects(self, temp_pos):
		real_com_traj = np.array(self.list_pos_com_map_glo_fram)
		real_base_traj = np.array(self.list_pos_base_map_glo_fram)
		map_p = np.array(self.list_pos_stf_map_glo_frame)
		plt.figure()
		plt.plot(real_com_traj[:, 0], real_com_traj[:, 1], 'ro-')
		plt.plot(real_base_traj[:, 0], real_base_traj[:, 1], 'go-')
		plt.plot(temp_pos[:, 0], temp_pos[:, 1], 'b--')
		plt.plot(map_p[:, 0], map_p[:, 1], 'rx')
		plt.plot(self.fot_loc_map[0], self.fot_loc_map[1], 'bx')
		plt.grid(True)
		plt.xlim([-0.5, 3])
		plt.ylim([-5, 5])
		plt.show()

	# Test: Plot predict trajectories every state
	def plot_each_pre_trajects(self, list, part_real, af_pred, circles):
		# real_com_traj = np.array(part_real)
		real_com_traj = np.array(self.list_pos_com_map_glo_fram)
		# real_base_traj = np.array(self.list_pos_base_map_glo_fram)
		heading = np.array(self.list_hd_base_map_glo_fram)
		map_p_real = np.array(self.list_pos_stf_map_glo_frame)
		target_p = np.array(self.list_fot_loc_tar_map)
		swf_real = np.array(self.list_pos_swf_rob_glo_frame)
		t_list = np.array(self.t_list)
		body_vel = np.array(self.list_vel_com_fot_fram)
		turning = np.array(self.list_hd_input_pr)
		path = 'data_log/OLIP_mexx_'

		with open(path + 'time.pkl', 'wb') as file:
			pickle.dump(t_list, file)
		with open(path + 'heading.pkl', 'wb') as file:
			pickle.dump(heading, file)
		with open(path + 'turning.pkl', 'wb') as file:
			pickle.dump(turning, file)
		with open(path + 'body_vel.pkl', 'wb') as file:
			pickle.dump(body_vel, file)

		plt.rcParams.update({'font.size': 15})
		figure, axes = plt.subplots(figsize=(5,5))
		plt.plot(real_com_traj[:, 0], real_com_traj[:, 1], 'r', linewidth=2.5)
		# plt.plot(real_base_traj[:, 0], real_base_traj[:, 1], 'go-')
		# for i in range(len(list)):
		# 	temp_pos =list[i]
		# 	plt.plot(temp_pos[:, 0], temp_pos[:, 1])
		plt.plot(0, 0, '^r', markersize=10)
		plt.plot(self.goal[0][0], self.goal[0][1], '*g', markersize=15)

		# for i in range(len(af_pred)):
		# 	af_pred_pos = af_pred[i]
		# 	plt.plot(af_pred_pos[:, 0], af_pred_pos[:, 1], 'b--')
		plt.plot(map_p_real[:, 0], map_p_real[:, 1], '.b')
		Drawing_colored_circle = plt.Circle((10, 10), 0.35, color = "#F5DEB3")
		axes.add_artist( Drawing_colored_circle )
		plt.legend(['CoM', 'Start', 'Goal', 'Foot', 'Goal region'],
           fontsize="15", loc ="upper left")
		# plt.plot(self.nex_pos[0], self.nex_pos[1], 'b*')
		# plt.plot(self.fot_loc_tar_map[0], self.fot_loc_tar_map[1], 'bx')
		# plt.plot(target_p[:, 0], target_p[:, 1], 'rx')
		# plt.plot(self.fot_loc_map[0], self.fot_loc_map[1], 'go')
		# plt.plot(swf_real[:, 0], swf_real[:, 1], 'co')
		for each in circles:
			self.plot_cir(each)
		# plt.grid(True)
		plt.xlim([-0.5, 10.5])
		plt.ylim([-0.5, 10.5])
		plt.axis('equal')

		plt.figure(figsize=(5,5))
		plt.plot(t_list[0:-1:20], body_vel[0:-1:20, 0], linewidth=2.0)
		# plt.plot(t_list, body_vel[:, 1], linewidth=2.0)
		# plt.grid(True)
		plt.legend(['vx', 'vy'])
		# plt.title('Body velocity with time')

		plt.figure(figsize=(5,5))
		plt.plot(t_list[0:-1:20], heading[0:-1:20], linewidth=2.0)
		plt.ylim([-1.57, 1.57])
		# plt.grid(True)
		# plt.title('Heading angle with time')

		plt.figure(figsize=(5,5))
		plt.plot(t_list, turning, linewidth=2.0)
		plt.ylim([-1.57, 1.57])
		plt.grid(True)
		plt.title('Turning angle with time')
		plt.show()


        
