import time
import math
import shutil
import pickle
import numpy as np
from ALIP import ALIP
from matplotlib import pyplot as plt

class Logger:
	def __init__(self, time_step, map_init_pos, hd_init_ang, goal, margin):
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

		# angluar momentum
		self.x_alip = None
		self.y_alip = None
		self.x_alip_list = []
		self.y_alip_list = []

		# swing foot
		self.pos_swf_rob_glo_frame = None
		self.list_pos_swf_rob_glo_frame = []

		# MPC-LIP
		self.lip_pred_pos = None						# Based on map global frame
		self.lip_pred_vel = None						# Based on map global frame
		self.mpc_state_tar  = None						# Based on map global frame
		self.fot_loc_map  = None						# Based on map global frame
		self.fot_loc_robo = None						# Based on robot global frame
		self.mpc_contr = [[0.8, 0], [0.8, 0], [0.8, 0]]

		# Control relevent
		self.fot_loc_tar_map = None						# Based on map global frame
		self.fot_loc_tar_rob = None						# Based on robot global frame
		self.foot_input   = None						# Based on fot frame
		self.hd_input_pr  = None						# Based on robot global frame
		self.hd_input_cos = None						# Based on robot global frame
		self.nex_turn 	  = 0.0
		self.close2goal   = False

		self.v_contr = []
		self.w_contr = []
		self.list_lip_pred_pos = []
		self.list_lip_pred_vel = []
		self.list_fot_loc_map  = []
		self.list_fot_loc_robo = []
		self.list_foot_input   = []
		self.list_hd_input_pr  = []
		self.list_hd_input_cos = []
		self.list_fot_loc_tar_map = []
		self.list_fot_loc_tar_rob = []

		self.alip = ALIP(1.0, 0.4, 40, 0.4)

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
		sum_diff = self.angle_A_minus_B(self.vel_hd, cur_hd)
		for i in range(len(self.mpc_contr)-1):
			sum_diff += self.mpc_contr[i+1][1]
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
		self.x_alip, self.y_alip = env.get_fot_loc_angm()

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
		self.vel_hd = math.atan2(self.vel_com_map_glo_fram[1],self.vel_com_map_glo_fram[0])
		# record com data
		self.x_alip_list.append(self.x_alip)
		self.y_alip_list.append(self.y_alip)
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
		self.hd_input_pr = self.nex_turn



	def tube_func(self, turning, init_tube_value):
		tube_upper = 0.15
		tube_lower = -0.15
		tube_value = init_tube_value
		d_head_value = turning
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
	def gen_nex_vel_input(self, mpc_cbf1, rest_t, num_step):
		x_nex, hd_ang_rob, temp_pos = self.predict_dt_state_traj(mpc_cbf1,  rest_t)

		self.hd_tar_rob = hd_ang_rob
		self.lip_pred_pos = x_nex[0:2]
		self.lip_pred_vel = x_nex[2:4]
		dd_state = np.concatenate([x_nex[0:2], [x_nex[4]]])

		if num_step == 0:
			guess = np.ravel([self.mpc_contr[1], self.mpc_contr[2], self.mpc_contr[2]])
		else:
			guess = np.ravel(self.mpc_contr)

		# Generate new step
		# x_mpc_tar, self.mpc_hds_list, self.mpc_contr, close_2_goal, feasi = mpc_cbf1.gen_dd_control(dd_state, guess, np.ravel(self.mpc_contr[0]), True, temp_pos)
		x_mpc_tar, self.mpc_hds_list, self.mpc_contr, close_2_goal, feasi = mpc_cbf1.gen_dd_control(dd_state, guess, np.ravel(self.mpc_contr[0]))
		self.nex_turn = self.mpc_contr[0][1]
		self.v_contr.append(self.mpc_contr[0][0])
		self.w_contr.append(self.mpc_contr[0][1])
		self.mpc_state_tar = x_mpc_tar
		self.close2goal = close_2_goal
		return x_mpc_tar, feasi
		

	def ALIP_gen_foot_input(self, time):
		# ALIP CONTROLLER
		speed = float(self.mpc_contr[0][0])
		px, py, Ly_est, Lx_est = self.alip.getFootPlacement(speed, self.leg_ind, time, self.x_alip, self.y_alip)
		x, y = self.alip.getTimedState(self.x_alip, self.y_alip, self.dt-time)
		self.foot_input = [px, py]
		self.nex_pos_fot_loc = [float(x[0]), float(y[0])]
		self.nex_vel_fot_loc = [speed, 0]


	def gen_tsc_control(self, i, n_cyc):
		# exp turning signal
		# tau = 0.2
		# head_ang_new = self.hd_input_pr*(1-math.exp(-tau*i)) + self.hd_input_cos
		head_ang_new = self.hd_input_pr/n_cyc*(i+4.5) + self.hd_input_cos
		c1 = [self.foot_input[0], self.foot_input[1], 0, head_ang_new]
		c2 = [self.nex_pos_fot_loc[0], self.nex_pos_fot_loc[1], self.nex_vel_fot_loc[0], 0]
		self.list_hd_input_pr.append(self.hd_input_pr)
		self.list_hd_input_cos.append(self.hd_input_cos)
		high_level_action = np.r_[c1, c2]
		return high_level_action
	

	# Tools Functions
	# Print 
	def print_states(self, i, num_step):
		# print('----relative time----')
		# print('t: ', 0.4-i)
		# print('remain t: ', i)
		print('num of step', num_step)
		# print('-------robot states-------')
		# # print('stf pos map', self.pos_stf_map_glo_frame)
		# print('swf pos rob', self.pos_swf_rob_glo_frame)
		# # print('stf hed map', self.ang_stf_map_glo_fram)
		# # print('stf hed robo', self.ang_stf_rob_glo_fram)
		# print('rob pos map', self.pos_com_map_glo_fram)
		# print('rob vel map', self.vel_com_map_glo_fram)
		# # print('rob hd map', self.hd_base_map_glo_fram)
		# print('rob hd robo', self.hd_base_rob_glo_fram)
		# # print('next state', self.state_com_map_glo)
		# print('body vel x', self.body_vx)
		# print('-------control input--------')
		# print('mpc states', np.matrix(self.mpc_state_tar))
		# print('next vel des', self.nex_vel_fot_loc)
		# print('nex ang cont', self.nex_turn)
		# print('tar angle(by perdition)', self.hd_tar_rob)
		# # print('next foot map', self.fot_loc_tar_map)
		# print('next foot rob', self.fot_loc_tar_rob)
		# print('foot input', self.foot_input)
		# print('hd input cos(base)', self.hd_input_cos)
		# print('hd input pr(tar)', self.hd_input_pr)

	def plot_cir(self, cir):
		theta = np.linspace(0, 2 * np.pi, 100)
		x = cir[2]*np.cos(theta)+cir[0]
		y = cir[2]*np.sin(theta)+cir[1]
		plt.plot(x, y, color='#696969')

	def elp_func(self, elp, x, y):
		a = (elp[3]*math.cos(elp[4]))**2+(elp[2]*math.sin(elp[4]))**2
		b = 2*math.cos(elp[4])*math.sin(elp[4])*(elp[3]**2-elp[2]**2)
		c = (elp[3]*math.sin(elp[4]))**2+(elp[2]*math.cos(elp[4]))**2
		v1 = a*(x-elp[0])**2
		v2 = b*(x-elp[0])*(y-elp[1])
		v3 = c*(y-elp[1])**2
		return v1+v2+v3-(elp[3]*elp[2])**2

	def plot_elp(self, elp):            # [x_c, y_c, A, B, C, r]
		x1 = np.arange(self.margin[0], self.margin[1], 0.1)
		x2 = np.arange(self.margin[0], self.margin[1], 0.1)
		X1, X2 = np.meshgrid(x1, x2)
		z = self.elp_func(elp, X1, X2)
		plt.contour(X1, X2, z, [0], colors='#696969')

	# Test: Plot predict trajectories every state
	def plot_each_pre_trajects(self, real_str, feasi_str, fail_str, circles, elips, ful_traj, path):
		real_com_traj = np.array(self.list_pos_com_map_glo_fram)
		map_p_real = np.array(self.list_pos_stf_map_glo_frame)
		heading = np.array(self.list_hd_base_map_glo_fram)
		body_vel = np.array(self.list_vel_com_fot_fram)
		target_p = np.array(self.list_fot_loc_tar_map)
		turning = np.array(self.list_hd_input_pr)
		v_contr = np.array(self.v_contr)
		w_contr = np.array(self.w_contr)
		t_contr = np.linspace(0, self.t_list[-1], len(v_contr))
		real_str = np.array(real_str)
		t_list = np.array(self.t_list)


		with open(path + 'pos.pkl', 'wb') as file:
			pickle.dump(real_com_traj, file)
		with open(path + 'time.pkl', 'wb') as file:
			pickle.dump(t_list, file)
		with open(path + 'foot.pkl', 'wb') as file:
			pickle.dump(map_p_real, file)
		with open(path + 'heading.pkl', 'wb') as file:
			pickle.dump(heading, file)
		with open(path + 'body_vel.pkl', 'wb') as file:
			pickle.dump(body_vel, file)
		with open(path + 'ellp.pkl', 'wb') as file:
			pickle.dump(elips, file)
		with open(path + 'cir.pkl', 'wb') as file:
			pickle.dump(circles, file)
		with open(path + 'real_end.pkl', 'wb') as file:
			pickle.dump(real_str, file)
		with open(path + 'pred_feasi_end.pkl', 'wb') as file:
			pickle.dump(feasi_str, file)
		with open(path + 'pred_fail_end.pkl', 'wb') as file:
			pickle.dump(fail_str, file)
		with open(path + 'pred_full_end.pkl', 'wb') as file:
			pickle.dump(ful_traj, file)

		plt.figure(figsize=(5,5))
		plt.plot(real_com_traj[:, 0], real_com_traj[:, 1], 'r', linewidth=2.0)
		plt.plot(self.goal[0][0], self.goal[0][1], '*g')

		# plt.plot(target_p[:, 0], target_p[:, 1], 'rx')
		plt.plot(map_p_real[:, 0], map_p_real[:, 1], '.b')
		plt.legend(['com trajectory', 'goal', 'foot placement'])
		
		for each in circles:
			self.plot_cir(each)
		for each in elips:
			self.plot_elp(each)
		plt.grid(True)
		plt.axis('equal')
		plt.xlim(self.margin)
		plt.ylim(self.margin)

		plt.figure(figsize=(5,5))
		plt.plot(t_list, body_vel[:, 0], linewidth=2.0)
		plt.plot(t_list, body_vel[:, 1], linewidth=2.0)
		plt.plot(t_contr, v_contr, linewidth=2.0)
		plt.grid(True)
		plt.legend(['vx', 'vy'])
		plt.title('Body velocity with time')

		plt.figure(figsize=(5,5))
		plt.plot(t_list, heading, linewidth=2.0)
		plt.ylim([-1.57, 1.57])
		plt.grid(True)
		plt.title('Heading angle with time')

		plt.figure(figsize=(5,5))
		plt.plot(t_list, turning, linewidth=2.0)
		plt.plot(t_contr, w_contr, linewidth=2.0)
		plt.ylim([-1.57, 1.57])
		plt.grid(True)
		plt.title('Turning angle with time')

		plt.figure(figsize=(5,5))
		plt.plot(real_str[:, 0], real_str[:, 1], linewidth=3.0)
		for each in feasi_str:
			temp = np.array(each)
			plt.plot(temp[:, 0], temp[:, 1], '-or', linewidth=2.0)
		for each in fail_str:
			temp = np.array(each)
			plt.plot(temp[:, 0], temp[:, 1], '-o', linewidth=2.0, color='#C0C0C0')
		for each in circles:
			self.plot_cir(each)
		for each in elips:
			self.plot_elp(each)
		plt.legend(['real trajectory', 'planned trajectory', 'infeasible trajectoty'])
		plt.xlim(self.margin)
		plt.ylim(self.margin)
		plt.axis('equal')
		plt.grid(True)
		plt.title('Trajectory compare')
	
		plt.show()


        
