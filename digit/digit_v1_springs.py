from math import floor

import numpy as np 
import os
import random


from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


import os.path
this_directory = os.getcwd()
parent_directory = os.path.dirname(this_directory)
filename = "tsc_wrapper.cpython-38-x86_64-linux-gnu.so"

if not os.path.exists(filename):
	src_folder = parent_directory + "/build/py_tsc/"
	dst_folder = this_directory
	src_file = os.path.join(src_folder, filename)
	dst_file = os.path.join(dst_folder, filename)
	shutil.copy(src_file, dst_folder)
	print(f"The file {filename} has been copied from {src_folder} to {dst_folder}.")
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

		dirname = os.path.dirname(__file__)


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

		# print("len_traj: ", len(self.trajectory))
		self.phaselen = int(self.time_cycle/(1/self.freq_updateNN)) 
		# print("phaselen: ", self.phaselen)
		self.phaserate = floor(1002 / self.phaselen) 
		# print("phaserate: ", self.phaserate)

		# self.phaselen = floor(len(self.trajectory) / self.simrate) - 1
		# print( " traje length",len(self.trajectory))
		# print( " phase length",self.phaselen)
		# print( " phase rate",self.phaserate)

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



		self.qpos_wbc_idx = []
		self.vel_wbc_idx  = []
		self.acc_wbc_idx  = []
		self.torque_wbc2mujoco_idx = [0,1,2,3,5,6,	9,10,11,12, 	13,14,15,16,18,19,	 22,23,24,25]

		# self.gearbox_wbc = np.array([80,50,16,16,1,50,50,1,1,80,80,50,80,80,50,16,16,1,50,50,1,1,80,80,50,80])		
		self.gearbox_wbc = np.array([80,50,16,16,50,50,80,80,50,80,80,50,16,16,50,50,80,80,50,80])		


		# kwargs = {'render_mode': 'human'}

		observation_space = Box(
			low=-np.inf, high=np.inf, shape=(115,), dtype=np.float64
		)



		# MujocoEnv.__init__(self, '/home/lab/digit_tsc/MujocoPySim/digit/digitmujoco/assets/digit_new_model.xml', 2, observation_space, **kwargs)
		MujocoEnv.__init__(
			self,
			"/home/lab/digit_tsc/MujocoPySim/digit/digitmujoco/assets/digit_new_model.xml",
			2,
			observation_space=observation_space,
			default_camera_config=DEFAULT_CAMERA_CONFIG,
			**kwargs
		)


		# MujocoEnv.__init__(self, 'digit_ysp_o.xml', 2)
		utils.EzPickle.__init__(self)

		# self.animate_trajectory()
		# self.phase = 0
	
		# np.set_printoptions(suppress=True)



	def animate_trajectory(self):
		self.phase = 0

		for t in range(self.phaselen*1):
			if self.phase > self.phaselen:
				self.phase = 0

			pos_ref, vel_ref = self.get_ref_state(self.phase)
			self.set_joint_pos(pos_ref, vel_ref,100)
			self.phase += self.phase_add
			for j in range(100):
				self.render()			
			
		exit()
		



	def step(self, action_wbc):

			height = self.data.qpos[2]
			# print("height: ", height)

			# u_leg = action[0:12]
			# u_arm = action[12:20]
			# self.do_simulation(np.concatenate((u_leg,u_arm),axis=0), self.frame_skip)
			# action = np.zeros(26)
			action = self.action_map_wbc2mujoco(action_wbc)

			action = self.transform_torque_wbc(action)
			self.do_simulation(action, self.frame_skip)
		

			# Early termination
			done = not(height > 0.8 and height < 2.0)
			reward = 0
			reward = self.compute_reward()
			if reward < 0.3:
					done = True
			# return self.get_full_state(), reward, done, {}
			return self.get_wbc_state_from_sensor(), reward, done, {}


	def compute_reward(self):
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

			self.speed  = np.random.uniform(-0.3, 0.5)

			qpos = np.array([-0.000064, 0.000070, 0.950181, 0.999999, 0.000030, -0.000027, 0.001287,
				   0.325685, 0.003982, 0.060953, 0.998499, -0.002613, 0.000547, 0.054705, 0.109731, 0, -0.109664, 0,
				   0.027702, 0.999890, -0.004864, 0.001440, -0.013921, -0.061125, 0.999420, 0.011867, -0.002226,
				   0.031841, -0.046928, -0.057329, -0.308679, 0.937404, -0.037069, 0.020408, -0.325474, -0.000611,
				   -0.062242, 0.998510, 0.002614, 0.000544, -0.054497, -0.109314, 0, 0.109222, 0, -0.026694, 0.999897,
				   0.004863, 0.001417, 0.013403, 0.060098, 0.999437, -0.011867, -0.002203, -0.031310, 0.045831,
				   0.057292, 0.309157, -1.012092, 0.001563, 0.019892])

			# print(qpos.shape)		



			qvel = np.zeros(len(self.data.qvel))
			# print(qvel.shape)		
			# exit()			
			self.set_joint_pos(qpos,qvel,100)

			# return self.get_full_state()
			# return self.get_wbc_state()
			return self.get_wbc_state_from_sensor()




	def get_damping(self):
		return np.array(self.sim.get_dof_damping())

	def get_mass(self):
		return np.array(self.sim.get_body_mass())

	def get_ipos(self):
		return np.array(self.sim.get_body_ipos()[3:6])




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



	def pd_leg_step(self, qd, qdotd, q, qdot):
		#print("Gains Kp,Kd = ",[self.kp, self.kd])
		error_pos = qd - q
		error_vel = qdotd - qdot
  
		output = self.P_leg*error_pos + self.D_leg*error_vel

		return np.clip(output, self.torque_leg_min, self.torque_leg_max)


	def pd_arm_step(self, qd, qdotd, q, qdot):
		#print("Gains Kp,Kd = ",[self.kp, self.kd])
		error_pos = qd - q
		error_vel = qdotd - qdot
  
		output = self.P_arm *error_pos + self.D_arm*error_vel

		return np.clip(output, self.torque_arm_min, self.torque_arm_max)


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
			# print(self.data.sensordata.shape)
			qpos = np.concatenate((self.data.sensordata[0:3], 										# pelvis position 
								   self.data.sensordata[4:7], [self.data.sensordata[3]],		# pelvis orientation(quaternion order x,y,z,w) 
								   self.data.sensordata[10:10+26])) 								# joint positions
			# print(qpos)
			qvel = np.concatenate((self.data.sensordata[36:39], self.data.sensordata[39:42],	# pelvis velocity and angular velocity (omega)
								   self.data.sensordata[42:42+26])) 								# joint velocities

			base_acc = self.data.sensordata[7:10]													# pelvis acceleration
			
			return np.concatenate((qpos, qvel, base_acc))		



	def transform_torque_wbc(self, torque):
		u = torque*(1/self.gearbox_wbc)
		return u
