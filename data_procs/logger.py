import time
import math
import shutil
import numpy as np
from matplotlib import pyplot as plt

class Logger:
	def __init__(self, time_step, map_init_pos, hd_init_ang, goal):
		# time param
		self.dt = time_step
		self.t_abs = None
		self.t_ref = None 
		self.t_abs_list = [0]
		self.t_ref_list = [0]

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
	def update_n_record(self, env, leg_ind):
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
		# Generate hd input
		# if num_step <= 3:
		# 	self.hd_input_cos = 0.0
		# 	self.hd_input_pr = 0.0
		# else:
		# 	self.hd_input_cos = cur_hd
		# 	self.hd_input_pr = self.tube_func(self.nex_turn, cur_hd)
		self.hd_input_cos = cur_hd
		self.nex_turn = self.tube_func(self.nex_turn, cur_hd)
		self.hd_input_pr = self.avg_hd(cur_hd)
		# self.hd_input_pr = self.nex_turn

		# Generate heading target
		self.list_hd_input_pr.append(self.hd_input_pr)
		self.list_hd_input_cos.append(self.hd_input_cos)


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
		# set heading input
		# self.set_hd_input(hd_ang_rob, num_step)

		self.hd_tar_rob = hd_ang_rob
		self.lip_pred_pos = x_nex[0:2]
		self.lip_pred_vel = x_nex[2:4]
		nex_pos_rob = self.pos_map_glo_2_robo_glo(x_nex[0:2])
		nex_vel_rob = self.vel_map_glo_2_robo_glo(v_des_map)
		print('predic_states', x_nex)
		print('init_guess', np.matrix(self.mpc_state_tar))

		# Generate new step
		# x_mpc_tar, nex_contr, self.mpc_hds_list, close_2_goal = mpc_cbf1.gen_control_test(x_nex, (-1)*leg_ind, self.mpc_state_tar)
		x_mpc_tar, nex_contr, self.mpc_hds_list, close_2_goal = mpc_cbf1.gen_control_test(x_nex, (-1)*leg_ind, self.mpc_state_tar, True)
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
		self.list_hd_input_pr.append(self.hd_input_pr)
		self.list_lip_pred_pos.append(self.lip_pred_pos)		# continuous record data will plot a jump graph
		self.list_lip_pred_vel.append(self.lip_pred_vel)		# not a seperate trajectory
		self.list_fot_loc_robo.append(self.fot_loc_robo)
		self.list_hd_input_cos.append(self.hd_input_cos)
		self.list_fot_loc_tar_map.append(self.fot_loc_tar_map)
		self.list_fot_loc_tar_rob.append(self.fot_loc_tar_rob)

		return traj_rest
	
	# Calculate foot location with desire pos
	def cal_foot_input(self, mpc_cbf1, rest_t, v_des_map, num_step):
		x_nex, hd_ang_rob, traj_rest = self.predict_dt_state_traj(mpc_cbf1, rest_t)

		self.hd_tar_rob = hd_ang_rob
		nex_pos_rob = self.pos_map_glo_2_robo_glo(x_nex[0:2])
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
		self.fot_loc_tar_map = nex_stf_map
		self.fot_loc_tar_rob = nex_stf_rob

		self.list_foot_input.append(self.foot_input)
		self.list_fot_loc_tar_map.append(self.fot_loc_tar_map)
		self.list_fot_loc_tar_rob.append(self.fot_loc_tar_rob)

		# Test: draw the traject with next stace foot to x_des
		map_pos = np.ravel(x_nex[0:2])
		map_vel = np.ravel(x_nex[2:4])
		nex_stf_map = np.append(nex_stf_map, 0.0)
		map_hd = float(x_nex[4])
		x_nex_nex, traj_nex = mpc_cbf1.get_next_states(map_pos, map_vel, map_hd, nex_stf_map, self.dt)

		return traj_rest, traj_nex


	def gen_tsc_control(self, i):
		# exp turning signal
		# tau = 0.2
		# head_ang_new = self.hd_input_pr*(1-math.exp(-tau*i)) + self.hd_input_cos
		head_ang_new = self.hd_input_pr/20*(i+4.5) + self.hd_input_cos
		c1 = [self.foot_input[0], self.foot_input[1], 0, head_ang_new]
		c2 = [self.nex_pos_fot_loc[0], self.nex_pos_fot_loc[1], self.nex_vel_fot_loc[0], 0]

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
		# print('stf pos map', self.pos_stf_map_glo_frame)
		print('swf pos rob', self.pos_swf_rob_glo_frame)
		# print('stf hed map', self.ang_stf_map_glo_fram)
		# print('stf hed robo', self.ang_stf_rob_glo_fram)
		print('rob pos map', self.pos_com_map_glo_fram)
		print('rob vel map', self.vel_com_map_glo_fram)
		# print('rob hd map', self.hd_base_map_glo_fram)
		print('rob hd robo', self.hd_base_rob_glo_fram)
		# print('next state', self.state_com_map_glo)
		print('body vel x', self.body_vx)
		print('-------control input--------')
		print('mpc states', np.matrix(self.mpc_state_tar))
		print('next vel des', self.nex_vel_fot_loc)
		print('nex ang cont', self.nex_turn)
		print('tar angle(by perdition)', self.hd_tar_rob)
		# print('next foot map', self.fot_loc_tar_map)
		print('next foot rob', self. fot_loc_tar_rob)
		# print('foot input', self.foot_input)
		print('hd input cos(base)', self.hd_input_cos)
		print('hd input pr(tar)', self.hd_input_pr)

	def plot_cir(self, cir):
		theta = np.linspace(0, 2 * np.pi, 100)
		x = cir[2]*np.cos(theta)+cir[0]
		y = cir[2]*np.sin(theta)+cir[1]
		plt.plot(x, y)

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
		map_p_real = np.array(self.list_pos_stf_map_glo_frame)
		target_p = np.array(self.list_fot_loc_tar_map)
		swf_real = np.array(self.list_pos_swf_rob_glo_frame)

		plt.figure()
		plt.plot(real_com_traj[:, 0], real_com_traj[:, 1], 'ro-')
		# plt.plot(real_base_traj[:, 0], real_base_traj[:, 1], 'go-')
		for i in range(len(list)):
			temp_pos =list[i]
			plt.plot(temp_pos[:, 0], temp_pos[:, 1])

		for i in range(len(af_pred)):
			af_pred_pos = af_pred[i]
			plt.plot(af_pred_pos[:, 0], af_pred_pos[:, 1], 'b--')
		for each in circles:
			self.plot_cir(each)

		plt.plot(map_p_real[:, 0], map_p_real[:, 1], 'bx')
		# plt.plot(self.nex_pos[0], self.nex_pos[1], 'b*')
		# plt.plot(self.fot_loc_tar_map[0], self.fot_loc_tar_map[1], 'bx')
		plt.plot(target_p[:, 0], target_p[:, 1], 'rx')
		# plt.plot(self.fot_loc_map[0], self.fot_loc_map[1], 'go')
		plt.plot(swf_real[:, 0], swf_real[:, 1], 'co')
		plt.grid(True)
		plt.xlim([-0.5, 10])
		plt.ylim([-0.5, 10])
		plt.show()


        
