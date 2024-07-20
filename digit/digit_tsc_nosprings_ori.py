from math import floor

import numpy as np 
import os
import random
import shutil
import pickle
import time

import mujoco
import matplotlib.pyplot as plt


from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# import os.path
# this_directory = os.getcwd()
# parent_directory = os.path.dirname(this_directory)
# filename = "tsc_wrapper.cpython-38-x86_64-linux-gnu.so"

# if not os.path.exists(filename):
# 	# src_folder = parent_directory + "/build/py_tsc/"
# 	src_folder = "/home/lab/catkin_ws/devel/lib"
# 	dst_folder = this_directory
# 	src_file = os.path.join(src_folder, filename)
# 	dst_file = os.path.join(dst_folder, filename)
# 	shutil.copy(src_file, dst_folder)
# 	print(f"The file {filename} has been copied from {src_folder} to {dst_folder}.")

import cppimport.import_hook
import tsc_wrapper




DEFAULT_CAMERA_CONFIG = {
	"trackbodyid": 1,
	"distance": 4.0,
	"lookat": np.array((0.0, 0.0, 1.15)),
	"elevation": -20.0,
}


class DigitEnv(MujocoEnv, utils.EzPickle):    
	metadata = {
		"render_modes": [
			"human",
			"rgb_array",
			"depth_array",
		],
		"render_fps": 1000,
	}

	def __init__(self, dynamics_randomization=False, **kwargs):
		# self.sim = CassieSim("./cassie/cassiemujoco/cassie.xml")

		self.vis = None

		self.dynamics_randomization = dynamics_randomization
		print(self.dynamics_randomization)

		state_est_size = 71
		clock_size     = 2
		speed_size     = 1

		self.observation_space = np.zeros(state_est_size + clock_size + speed_size)
		self.action_space      = np.zeros(20)


		# self.trajectory = DigitTrajectorySet(traj_path)

		self.P = np.array([100,  100,  88,  96,  50]) 
		self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])


		self.action_leg_max = np.array([1.5835, 1.5835, 13.55799, 14.457309375, 0.839518, 0.839518, 1.583530725, 1.583530725, 13.55799, 14.457309375, 0.839518, 0.839518])
		self.action_leg_min = np.array([-1.5835, -1.5835, -13.55799, -14.457309375, -0.839518, -0.839518, -1.583530725, -1.583530725, -13.55799, -14.457309375, -0.839518, -0.839518])
		self.action_arm_max = np.array([1.5835, 1.5835, 1.5835, 1.5835, 1.5835, 1.5835, 1.5835, 1.5835])
		self.action_arm_min = np.array([-1.5835, -1.5835, -1.5835, -1.5835, -1.5835, -1.5835,-1.5835, -1.5835])

		self.torque_leg_max = np.array([126.682, 79.1765, 216.928, 231.317, 41.9759, 41.9759, 126.682, 79.1765, 216.928, 231.317, 41.9759, 41.9759])
		self.torque_leg_min = np.array([-126.682, -79.1765, -216.928, -231.317, -41.9759, -41.9759, -126.682, -79.1765, -216.928, -231.317, -41.9759, -41.9759])
		self.torque_arm_max = np.array([126.68, 126.68, 79.17, 126.68, 126.68, 126.68, 79.17, 126.68])
		self.torque_arm_min = np.array([-126.68, -126.68, -79.17, -126.68, -126.68, -126.68, -79.17, -126.68])

		self.gearbox_leg = np.array([80,50,16,16,50,50,80,50,16,16,50,50])
		self.gearbox_arm = np.array([80,80,50,80,80,80,50,80])
  
		self.P_leg = np.array([1500,  1500,  1500,  2000, 1000, 1000, 1500,  1500,  1500,  2000, 1000, 1000])
		self.D_leg = np.array([66.849, 26.1129, 38.05, 38.05, 20, 20, 66.849, 26.1129, 38.05, 38.05, 20, 20])
		self.P_arm = np.array([500, 500, 500, 500, 500, 500, 500, 500]) 
		self.D_arm = np.array([66.849, 66.849, 26.1129, 66.849, 66.849, 66.849, 26.1129, 66.849])

		self.u = np.zeros(20)

		self.simrate = 15 # simulate X mujoco steps with same pd target
						  # 30 brings simulation from 2000Hz to roughly 30Hz (considering frameskip=2 -> each sim step is 0.001s)
						  # 15 brings simulation from 2000Hz to roughly 60Hz (considering frameskip=2 -> each sim step is 0.001s)
		self.time    = 0  # number of time steps in current episode
		self.phase   = 0  # portion of the phase the robot is in
		self.counter = 0  # number of phase cycles completed in episode

		# NOTE: a reference trajectory represents ONE phase cycle (2 steps), which last 1 second for Digit
		self.time_cycle = 1 
		self.freq_updateNN = 60



		# see include/cassiemujoco.h for meaning of these indices	
		self.pos_leg_idx = [7,8,9,14,18,23,34,35,36,41,45,50]
		self.pos_arm_idx = [30,31,32,33,57,58,59,60]
		self.pos_idx = np.concatenate((self.pos_leg_idx,self.pos_arm_idx),axis=0)
		self.pos_passJoint_idx = [15,16,17,28,29, 42,43,44,55,56]   #[lShin,lTarsus,lHeelSpring,lToePitch,lToeRoll, rShin,rTarsus,rHeelSpring,rToePitch,rToeRoll]

		self.vel_leg_idx = [6,7,8,12,16,20, 30,31,32,36,40,44]
		self.vel_arm_idx = [26,27,28,29,50,51,52,53]
		self.vel_passJoint_idx = [13,14,15,24,25, 37,38,39,48,49]		
		self.vel_idx = np.concatenate((self.vel_leg_idx,self.vel_arm_idx),axis=0)

		self.offset_leg  = np.array([0.29169603, -0.07416819, 0.35135351, 0.37290114, -0.2184496, 0.13822394, 
 									 -0.3366968, 0.00432535, -0.30720362, -0.29399701, 0.14286116, -0.0999864])
		self.offset_arm  = np.array([-0.14962254, 1.06077823, -0.00359455, -0.14791954, 
									 0.1477781, -1.13863746, 0.00304071, 0.12687265])
		self.offset = np.concatenate((self.offset_leg, self.offset_arm),axis=0)
		self.speed     = 0.0
		self.phase_add = 1

		self.qpos_idx_matlab   = [0, 1, 2, 3, 4, 5, 6,
								7, 8, 9, 14, 15, 16, 17, 18, 23, 28, 29, 30, 31, 32, 33, 
								34, 35, 36, 41, 42, 43, 44, 45, 50, 55, 56, 57, 58, 59, 60]
		self.qvel_idx_matlab   = [0, 1, 2, 3, 4, 5, 
								6, 7, 8, 12, 13, 14, 15, 16, 20, 24, 25, 26, 27, 28, 29,
								30, 31, 32, 36, 37, 38, 39, 40, 44, 47, 49, 50, 51, 52, 53]

		# # Record default dynamics parameters
		# self.default_damping = self.sim.get_dof_damping()
		# self.default_mass = self.sim.get_body_mass()
		# self.default_ipos = self.sim.get_body_ipos()

		self.qpos = None
		self.qvel = None

		# self.joint_position_seq = [] 
		# self.joint_command_seq = []
		# self.joint_torque_seq = []
		self.t_aux = 0
		# self.t_abs = 0

		# qpos, _ = self.get_ref_state(self.phase)
		# self.joints_old = qpos[self.pos_idx]

		self.qpos_wbc_idx = []
		self.vel_wbc_idx  = []
		self.acc_wbc_idx  = []
		self.torque_wbc2mujoco_idx = [0,1,2,3,5,6,	9,10,11,12, 	13,14,15,16,18,19,	 22,23,24,25]
		# self.gearbox_wbc = np.array([80,50,16,16,1,50,50,1,1,80,80,50,80,80,50,16,16,1,50,50,1,1,80,80,50,80])		
		self.gearbox_wbc = np.array([80,50,16,16,50,50,80,80,50,80,80,50,16,16,50,50,80,80,50,80])		

		#Init TSC class and task space targets
		self.stand_time = 0.0 #seconds
		self.save_logs = False
		self.tsc = tsc_wrapper.TSC_wrapper(self.stand_time, self.save_logs)
		self.iter = 0
		self.number_walking_step = 0
		self.iter_step_increment = 10	#Defatul value 10 -> means every iteration of the TSC (each 1 ms) the iter counter increments in 10, which corresponds to a step cycle of 0.4 seconds.
										#Higher values than 10 make the phase of the step increment faster and the step cycle shorter than 0.4 seconds.
										#Lower values than 10 make the phase of the step increment slower and the step cycle longer than 0.4 seconds.

		self.pfx_rel = 0.0
		self.pfy_rel = 0.0
		self.pfz_rel = 0.0
		self.swfoot_clearance = 0.0


		self.desired_velocity = np.array([0.0, 0.0, 0.0])

		#Define Floating Base Tasks
		pos_floatingBaseTask        = np.array([0.1, 0.00, 0.95])
		vel_floatingBaseTask        = np.zeros(3)
		acc_floatingBaseTask        = np.zeros(3)
		omega_dot_floatingBaseTask  = np.zeros(3)
		R_wb_floatingBaseTask       = np.eye(3).flatten()
		# R_wb_floatingBaseTask       = np.array([0.9950042, 0.0, 0.0998334, 0.0, 1.0, 0.0, -0.0998334, 0.0, 0.9950042])
		omega_floatingBaseTask      = np.zeros(3)

		#Define End-effector (feet) Tasks
		pos_leftFootTask	= np.array([0.0638677, 0.134745, 0.0619495])
		R_wb_leftFootTask	= np.array([0.0308726,  -0.499956, 0.865501,
										0.999523,    0.0153944,   -0.0267606, 
										5.52244e-05, 0.865914,    0.500193 ])
		vel_leftFootTask	= np.array([0, 0, 0])
		acc_leftFootTask	= np.array([0, 0, 0])
		pos_rightFootTask	= np.array([0.0634209, -0.134967, 0.0619513])
		R_wb_rightFootTask	= np.array([0.0305188,    0.49996,     0.865511,   
										-0.999534,    0.0152711,    0.0264233,   
										-6.67367e-06, -0.865914,    0.500193])
		vel_rightFootTask	= np.array([0, 0, 0])
		acc_rightFootTask	= np.array([0, 0, 0])

		#Define additional tasks
		desired_velTask = np.array([0.0, 0.0, 0.0])

		self.tsc.setAllTasks(pos_floatingBaseTask, vel_floatingBaseTask, acc_floatingBaseTask, omega_dot_floatingBaseTask, R_wb_floatingBaseTask, omega_floatingBaseTask, 
					pos_leftFootTask, R_wb_leftFootTask, vel_leftFootTask, acc_leftFootTask, pos_rightFootTask, R_wb_rightFootTask, vel_rightFootTask, acc_rightFootTask,
					desired_velTask)

		# kwargs = {'render_mode': 'human'}

		observation_space = Box(
			low=-np.inf, high=np.inf, shape=(107,), dtype=np.float64
		)

		this_directory = os.getcwd()
		model_path = this_directory + "/digit/model/assets/digit_new_model_no_springs.xml"

		MujocoEnv.__init__(
			self,
			model_path,
			2,
			observation_space=observation_space,
			default_camera_config=DEFAULT_CAMERA_CONFIG,
			**kwargs
		)
		utils.EzPickle.__init__(self)

	
		np.set_printoptions(suppress=True)





	def step_simulation(self):
			
			q_raw, dq_raw, acc_raw = self.get_wbc_raw_state_from_sensor()
			self.tsc.updateStateWrapper(q_raw, dq_raw, acc_raw)

			mask = np.zeros(8)	#updated inside the task space controller
			self.tsc.run(mask, self.iter_step_increment)
			torque = self.tsc.output()
			action = self.action_map_wbc2mujoco(torque)
			action = self.transform_torque_wbc(action)
			self.do_simulation(action, self.frame_skip)

			self.time += 0.001
			self.iter += 1


	def step(self, action_rl):
			heading_angle = action_rl[5]
		# print("action_rl: ",action_rl)
			desired_vel_x = action_rl[3]
			desired_vel_y = action_rl[4]
			if self.iter > self.stand_time*1000:
				# print("\nQPOS: \n",self.data.qpos)
				# print("\nQVEL: \n",self.data.qvel)
				# exit()
				phase_ramp = np.clip((self.iter-self.stand_time*1000)/4000, 0,1)

				#Update task space trajectories
				pos_floatingBaseTask        = np.array([0.1, 0.00, 0.95])
				vel_floatingBaseTask        = np.zeros(3)
				acc_floatingBaseTask        = np.zeros(3)
				desired_velTask		        = np.zeros(3)
				

				pos_floatingBaseTask[0] = self.data.qpos[0]
				pos_floatingBaseTask[1] = self.data.qpos[1]
				pos_floatingBaseTask[2] = 1.0 #+ 0.1*np.sin((self.iter-start_walking)/1000)

				
				# Set desired heading angle by rotating the floating base by delta_yaw
				delta_yaw = 3.0*(heading_angle - self.yaw)
				# print("omega:", delta_yaw)
				# print("yaw",  self.yaw)
				# print("delta yaw",  delta_yaw)
				
				R_z = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw), 0],
								[np.sin(delta_yaw), np.cos(delta_yaw), 0],
								[0, 0, 1]])
				R_wb_floatingBaseTask = R_z


				desired_velTask[0] 		= desired_vel_x*phase_ramp
				vel_floatingBaseTask[0] = desired_vel_x*phase_ramp
				desired_velTask[1] 		= desired_vel_y*phase_ramp
				vel_floatingBaseTask[1] = desired_vel_y*phase_ramp
				# vel_floatingBaseTask [2] = 0.1*np.cos((self.iter-start_walking)/1000)*1000/1000


				#Update tasks in TSC wrapper
				self.tsc.setTask("torso", "pos", pos_floatingBaseTask)
				self.tsc.setTask("torso", "vel", vel_floatingBaseTask)
				self.tsc.setTask("torso", "R_wb", R_wb_floatingBaseTask)
				self.tsc.setTask("desired_vel", "vel", desired_velTask)

				self.pfx_rel = self.qpos[0] + action_rl[0]
				self.pfy_rel = self.qpos[1] + action_rl[1]

				self.pfz_rel = 0 #0.3*action_rl[2]
				self.swfoot_clearance = 0.12

				terrain_angle = 0.0
				self.tsc.updateSwingFootTarget(self.pfx_rel, self.pfy_rel, self.pfz_rel, terrain_angle, self.swfoot_clearance)



			for i in range(self.simrate):
				self.step_simulation()
		
			height = self.data.qpos[2]
			# print("height: ", height)
			# Early termination
			done = not(height > 0.8 and height < 2.0)
			reward = 0
			reward = self.compute_reward(action_rl)
			if reward < 0.3:
					done = True
			# return self.get_full_state(), reward, done, {}
			# return self.get_wbc_state_from_sensor(), reward, done, {}
			return self.get_learning_state(), reward, done, {}


	def compute_reward(self, action):
		qpos = np.copy(self.data.qpos)
		qvel = np.copy(self.data.qvel)
		reward = 0.0

		vx = qvel[0]

		time_touchdown_error = 0
		
		weight_tracking = [0.2, 0.2, 0.3, 0.3]	#[ya_pCoM, ya_qTor, ya_xNsf, ya_zNsf]

		tracking_error = 0
		time_touchdown_error = 0


		forward_diff = np.abs(vx - self.speed)
		if forward_diff < 0.05:
				forward_diff = 0

		y_vel = np.abs(qvel[1])
		if y_vel < 0.03:
			y_vel = 0

		straight_diff = np.abs(qpos[1])
		if straight_diff < 0.05:
			straight_diff = 0

		actual_q = qpos[3:7]
		target_q = [1, 0, 0, 0]
		orientation_error = 5 * (1 - np.inner(actual_q, target_q) ** 2)

		reward = 0.000 + \
							0.30 * np.exp(-orientation_error) + \
							0.30 * np.exp(-forward_diff) +      \
							0.30 * np.exp(-y_vel) +             \
							0.10 * np.exp(-straight_diff)

		return reward


	def reset_model(self):
			# self.phase = random.randint(0, self.phaselen)
			# self.phase = 0
			# print(self.phase)
			
			self.time = 0
			self.counter = 0
			self.tsc = tsc_wrapper.TSC_wrapper(self.stand_time, self.save_logs)
			self.tsc.init()
			self.iter = 0
			self.number_walking_step = 0


			# qpos = np.array([0.1001457,  0.00017802,  0.95621484,  0.99998875, -0.00024166, -0.00473079,
			# 				0.00026102,  0.32515658,  0.00305146,  0.19527317,  0.99863832, -0.00299799,
			# 				0.00047012,  0.05207974,  0.10461138, -0.10193585, -0.02604453,  0.99904385,
			# 				0.04148622,  0.00095081,  0.01376137,  0.00025904,  0.99805544,  0.06232509,
			# 				-0.00095378,  0.00005199,  0.01383774, -0.05710495, -0.28577198,  0.94747364,
			# 				-0.0321669 ,  0.02194378, -0.32551434, -0.00214094, -0.1942266,   0.9986845,
			# 				0.00298946,  0.00047922, -0.05118704, -0.10267876,  0.10247357,  0.03226578,
			# 				0.99899293, -0.04154225,  0.0009827,  -0.01692336,  0.00292482,  0.99805039,
			# 				-0.06238458, -0.00095507, -0.00163732, -0.01555447,  0.05800225,  0.28461554,
			# 				-1.0051888 , -0.01143921,  0.01979053])

			# qpos = np.array([-0.000064, 0.000070, 0.950181, 0.999999, 0.000030, -0.000027, 0.001287,
			# 	   			 0.325685, 0.003982, 0.060953, 0.998499, -0.002613, 0.000547, 0.054705, 0.109731, -0.109664,
			# 	   			 0.027702, 0.999890, -0.004864, 0.001440, -0.013921, -0.061125, 0.999420, 0.011867, -0.002226,
			# 	   			 0.031841, -0.046928, -0.057329, -0.308679, 0.937404, -0.037069, 0.020408, -0.325474, -0.000611,
			# 	   			 -0.062242, 0.998510, 0.002614, 0.000544, -0.054497, -0.109314, 0.109222, -0.026694, 0.999897,
			# 	   			 0.004863, 0.001417, 0.013403, 0.060098, 0.999437, -0.011867, -0.002203, -0.031310, 0.045831,
			# 	   			 0.057292, 0.309157, -1.012092, 0.001563, 0.019892])

			# qpos = np.array([-0.000064, 0.000070, 0.950181, 0.999999, 0.000030, -0.000027, 0.001287,
			# 	   0.325685, 0.003982, 0.060953, 0.998499, -0.002613, 0.000547, 0.054705, 0.109731, 0, -0.109664, 0,
			# 	   0.027702, 0.999890, -0.004864, 0.001440, -0.013921, -0.061125, 0.999420, 0.011867, -0.002226,
			# 	   0.031841, -0.046928, -0.057329, -0.308679, 0.937404, -0.037069, 0.020408, -0.325474, -0.000611,
			# 	   -0.062242, 0.998510, 0.002614, 0.000544, -0.054497, -0.109314, 0, 0.109222, 0, -0.026694, 0.999897,
			# 	   0.004863, 0.001417, 0.013403, 0.060098, 0.999437, -0.011867, -0.002203, -0.031310, 0.045831,
			# 	   0.057292, 0.309157, -1.012092, 0.001563, 0.019892])

			#Robot standing position from hardware using AR default controller:
			qpos = np.array([ 0.04923635,  0.0265547,   1.03003979,  0.99997924,  0.00344193,  0.00544639,
							 -0.00003055,  0.36097882, -0.04035668,  0.31081243,  0.98415015,  0.0327087,
							  0.00939374,  0.17404135,  0.35667201, -0.32580705, -0.10452634,  0.8888311,
							 -0.45508734, -0.02668737,  0.04650351,  0.09044927,  0.84316593, -0.53569291,
							  0.02746639, -0.0367414,   0.10156386, -0.02265031, -0.15497474,  0.98888628,
							  0.01899998, -0.18698789, -0.3406291,  -0.01964617, -0.30025788,  0.98409474,
							 -0.03969807,  0.01052491, -0.17283181, -0.35526966,  0.3191397,   0.08962223,
							  0.88238801,  0.46824786, -0.02326101, -0.03992837, -0.05540693,  0.83569464,
							  0.54849576,  0.01686036,  0.02196827, -0.07609346,  0.05501961,  0.14297165,
							 -1.16685966,  0.0030991,   0.11499239])
	
			qvel = np.zeros(len(self.data.qvel))
			self.set_joint_pos(qpos,qvel,1)

			# self.desired_velocity = np.array([np.random.uniform(-0.6, 0.6), np.random.uniform(-0.3, 0.3), 0])
			# self.desired_velocity = np.array([np.random.uniform(0.0, 0.5), np.random.uniform(-0.0, 0.0), 0])
			# self.desired_velocity = np.array([0.4, 0.0, 0])	

			q_raw, dq_raw, acc_raw = self.get_wbc_raw_state_from_sensor()
			self.tsc.updateStateWrapper(q_raw, dq_raw, acc_raw)
			self.target_yaw = self.tsc.getYawGlobal()		#Captures initial yaw angle to use it as target. 			

			# return self.get_full_state()
			# return self.get_wbc_state()
			# return self.get_wbc_state_from_sensor()
			return self.get_learning_state()



	def get_learning_state(self):	# Extract the data from the XML sensors and order it according to the WBC state vector (qpos, qvel, qacc)
		self.qpos, self.qvel, self.acc, self.velocity_avg, self.yaw = self.get_wbc_state_from_wrapper()
		# print("====================================================================================================")

		# self.qpos = np.concatenate((self.data.sensordata[0:3], 										# pelvis position 
		#                             self.data.sensordata[4:7], [self.data.sensordata[3]],		    # pelvis orientation(quaternion order x,y,z,w) 
		#                             self.data.sensordata[10:10+26])) 								# joint positions
		
		# quat_wxyz = self.data.sensordata[3:7]
		# R_wb = quaternion_to_rotation_matrix(quat_wxyz)
		
		# self.qvel = np.concatenate((np.matmul(R_wb.T,self.data.sensordata[36:39]), 					# pelvis linear velocity 
		#                             np.matmul(R_wb.T,self.data.sensordata[39:42]), 	    			# pelvis angular velocity (omega)
		#                             self.data.sensordata[42:42+26])) 	
		

		# # self.qvel = np.concatenate((self.data.sensordata[36:39], self.data.sensordata[39:42], 	    # pelvis velocity and angular velocity (omega)
		# #                             self.data.sensordata[42:42+26])) 								# joint velocities

		# self.base_acc = self.data.sensordata[7:10]													# pelvis acceleration
		
		
		leftFoot_stanceTimeRemain = self.tsc.getGaitSchedulerLeftFoot()[0] 		# get_gaitScheduler_leftFoot() -> [stanceTime, swingTime]
		# leftFoot_swingTimeRemain  = self.tsc.getGaitSchedulerLeftFoot()[1]
		# rightFoot_stanceTimeRemain = self.tsc.getGaitSchedulerRightFoot()[0] 		# get_gaitScheduler_leftFoot() -> [stanceTime, swingTime]
		# rightFoot_swingTimeRemain  = self.tsc.getGaitSchedulerRightFoot()[1]

		# contact_pos_toe_pitch_mujoco = np.zeros(3)
		# contact_pos_foot_wrapper = np.zeros(3)
		stance_swing_foot_state = self.tsc.getStanceSwingFootState()
		if (leftFoot_stanceTimeRemain > 0.0):	# left foot is in stance phase - > contact point is LeftFoot				                
			#getting the position of the stance foot directly from the mujoco python bindings is ~3 times faster than the wrapper, and values are the same
			# contact_pos_toe_pitch_mujoco = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'left-toe-roll')]
			# contact_pos_toe_pitch_mujoco = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'left-foot')]
			# contact_pos_foot_wrapper = self.tsc.getStanceSwingFootState()[0:3]
			pos_sw_foot_wrapper = stance_swing_foot_state[3:6]
			vel_sw_foot_wrapper = stance_swing_foot_state[9:12]
			contact_pos_foot_mujoco = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'left-foot')]
			self.stance_sign = -1  
			self.next_stance_sign = 1  
		else:								# right foot is in stance phase - > contact point is RightFoot
			#getting the position of the stance foot directly from the mujoco python bindings is ~3 times faster than the wrapper, and values are the same
			# contact_pos_toe_pitch_mujoco = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'right-toe-roll')]
			# contact_pos_toe_pitch_mujoco = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'right-foot')]
			# contact_pos_foot_wrapper = self.tsc.getStanceSwingFootState()[0:3]
			pos_sw_foot_wrapper = stance_swing_foot_state[3:6]
			vel_sw_foot_wrapper = stance_swing_foot_state[6:9]
			contact_pos_foot_mujoco = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'right-foot')]
			self.stance_sign = 1  
			self.next_stance_sign = -1


		self.number_walking_step = int(self.tsc.getWalkingSteps()[0])

		self.test_mode = True
		if self.test_mode:
			# print("current foothold: 		 ", 	contact_pos_foot_mujoco)
			# print("next foothold: 		 ", 	[self.next_foothold_x, self.next_foothold_y])

			Rz = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
						   [np.sin(self.yaw),  np.cos(self.yaw), 0],
						   [0, 0, 1]])
			# next_foothold = np.array([self.next_foothold_x, self.next_foothold_y, 0.001])
			next_foothold = np.array([self.pfx_rel, self.pfy_rel, self.pfz_rel])
			next_foothold_wrt_yaw = np.matmul(Rz, next_foothold)
			world_pos_next_foot_hold = contact_pos_foot_mujoco + next_foothold_wrt_yaw

			# next_next_foothold = np.array([self.next_next_foothold_x, self.next_next_foothold_y, 0.001])
			# next_next_foothold_wrt_yaw = np.matmul(Rz, next_next_foothold)
			# world_pos_next_next_foot_hold = world_pos_next_foot_hold + next_next_foothold_wrt_yaw

			self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'current_foothold')] = contact_pos_foot_mujoco
			self.data.site_xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'current_foothold')] = Rz.flatten()
			self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'next_foothold')]    = world_pos_next_foot_hold 
			# self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'next_next_foothold')] = world_pos_next_next_foot_hold 




		self.error_vel_avg  = self.velocity_avg - self.desired_velocity


		robot_state = np.concatenate((
				self.qpos[0:3],
				[self.qvel[0], self.qvel[1]],
				pos_sw_foot_wrapper,
				vel_sw_foot_wrapper,
				self.error_vel_avg[0:2],
				self.desired_velocity[0:2],
		), axis = 0)
		





		return np.array(robot_state)




	def set_joint_pos(self, qpos, qvel, iters=10):
		"""
		This takes a floating base position and some joint positions
		and abuses the MuJoCo solver to get the constrained forward
		kinematics. 
		There might be a better way to do this, e.g. using mj_kinematics
		"""
	
		# fb_idx = [0, 1, 2, 3, 4, 5, 6]
		# fb_pos = np.array([0, 0, 1.2, 1.0, 0.0, 0.0, 0.0])
		for _ in range(iters):
			# qpos = np.copy(self.data.qpos) #Create copy of data returned by the simulator to override only the states I am interested in (defined the the vectors above)
			# qvel = np.copy(self.data.qvel)

			# qpos[self.qpos_idx_matlab] = pos[self.qpos_idx_matlab] 
			# pos[2] = 1.03


			self.set_state(qpos, qvel)
			self.do_simulation(np.zeros(20), self.frame_skip)
			# self.render()



	def transform_torque(self, torque_leg, torque_arm):
		u_leg = torque_leg*(1/self.gearbox_leg)
		u_arm = torque_arm*(1/self.gearbox_arm)
		return u_leg, u_arm


	def update_speed(self, speed):
		self.speed = speed


	def action_map_wbc2mujoco(self, torque):
		action = torque[self.torque_wbc2mujoco_idx]
		# torque_leg = action[0:12]
		# torque_arm = action[12:20]
		# torque_leg, torque_arm = self.transform_torque(self, torque_leg, torque_arm)
		# return np.concatenate(torque_leg, torque_arm)
		return action


	def get_wbc_state(self):
		qpos = self.data.qpos #[self.qpos_wbc_idx]
		qvel = self.data.qvel #[self.vel_wbc_idx]
		qacc = self.data.qacc #[self.acc_wbc_idx]
		return np.concatenate((qpos, qvel, qacc))

	def get_wbc_state_from_sensor(self):	# Extract the data from the XML sensors and order it according to the WBC state vector (qpos, qvel, qacc)
			qpos = np.concatenate((self.data.sensordata[0:3], 										# pelvis position 
								   self.data.sensordata[4:7], [self.data.sensordata[3]],		# pelvis orientation(quaternion order x,y,z,w) 
								   self.data.sensordata[10:10+26])) 								# joint positions
			# print(qpos)
			qvel = np.concatenate((self.data.sensordata[36:39], self.data.sensordata[39:42],	# pelvis velocity and angular velocity (omega)
								   self.data.sensordata[42:42+26])) 								# joint velocities

			base_acc = self.data.sensordata[7:10]													# pelvis acceleration
			
			return np.concatenate((qpos, qvel, base_acc))		


	def get_wbc_state_from_wrapper(self):	# Extract the data from the XML sensors and order it according to the WBC state vector (qpos, qvel, qacc)
		stateWrtStanceFoot = self.tsc.getStateWrtStanceFoot()
		qpos   = stateWrtStanceFoot[0:33]
		qvel   = stateWrtStanceFoot[33:33+32]
		acc    = stateWrtStanceFoot[65:65+3]
		av_vel = stateWrtStanceFoot[68:68+3]
		yaw_global = stateWrtStanceFoot[71]
		return qpos, qvel, acc, av_vel, yaw_global


	def get_wbc_raw_state_from_sensor(self):
		pos_raw = np.concatenate((self.data.sensordata[0:3], 		    # pelvis position in world coordinates                                   
									self.data.sensordata[3:7],		    # pelvis orientation in world coordinates(quaternion order x,y,z,w) 
									self.data.sensordata[10:10+26]))	# joint positions
		vel_raw = np.concatenate((self.data.sensordata[36:39], 		# pelvis linear velocity 
									self.data.sensordata[39:42], 	    # pelvis angular velocity (omega)
									self.data.sensordata[42:42+26])) 	# joint velocities		
		acc_raw = self.data.sensordata[7:10]
		return pos_raw, vel_raw, acc_raw
	

	def get_robo_glo_state_from_sensor(self):
		pos = np.array(self.data.sensordata[0:2])                               
		quat = np.array(self.data.sensordata[3:7])						
		vel = np.array(self.data.sensordata[36:38])
		return pos, vel, quat
	
	
	def get_ft_glo_from_sensor(self):
		pos_left_foot_world_coord  = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'left-foot')]
		pos_right_foot_world_coord = self.data.xpos[mujoco.mj_name2id(self.model, 1, 'right-foot')]
		left_p = np.array(pos_left_foot_world_coord[0:2])
		rigt_p = np.array(pos_right_foot_world_coord[0:2])
		return left_p, rigt_p


	def transform_torque_wbc(self, torque):
		u = torque*(1/self.gearbox_wbc)
		return u
