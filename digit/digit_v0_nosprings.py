from math import floor

import numpy as np 
import os
import random
import os.path


from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

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

		self.gearbox_wbc = np.array([80,50,16,16,50,50,80,80,50,80,80,50,16,16,50,50,80,80,50,80])		

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
		# np.set_printoptions(suppress=True)


	def step(self, action):
			action = self.transform_torque_wbc(action)
			self.do_simulation(action, self.frame_skip)
		
			# Early termination
			height = self.data.qpos[2]
			done = not(height > 0.8 and height < 2.0)
			reward = 0
			return self.get_state_from_sensor(), reward, done, {}


	def reset_model(self):

			qpos = np.array([-0.000064, 0.000070, 0.950181, 0.999999, 0.000030, -0.000027, 0.001287,
				   			 0.325685, 0.003982, 0.060953, 0.998499, -0.002613, 0.000547, 0.054705, 0.109731, -0.109664,
				   			 0.027702, 0.999890, -0.004864, 0.001440, -0.013921, -0.061125, 0.999420, 0.011867, -0.002226,
				   			 0.031841, -0.046928, -0.057329, -0.308679, 0.937404, -0.037069, 0.020408, -0.325474, -0.000611,
				   			 -0.062242, 0.998510, 0.002614, 0.000544, -0.054497, -0.109314, 0.109222, -0.026694, 0.999897,
				   			 0.004863, 0.001417, 0.013403, 0.060098, 0.999437, -0.011867, -0.002203, -0.031310, 0.045831,
				   			 0.057292, 0.309157, -1.012092, 0.001563, 0.019892])

			qvel = np.zeros(len(self.data.qvel))
			self.set_joint_pos(qpos,qvel,100)
			return self.get_state_from_sensor()



	def set_joint_pos(self, qpos, qvel, iters=10):
		"""
		This takes a floating base position and some joint positions
		and abuses the MuJoCo solver to get the constrained forward
		kinematics. 
		There might be a better way to do this, e.g. using mj_kinematics
		"""
		for _ in range(iters):
			self.set_state(qpos, qvel)
			self.do_simulation(np.zeros(20), self.frame_skip)



	def get_state_from_sensor(self):	# Extract the data from the XML sensors and order it according to the WBC state vector (qpos, qvel, qacc)
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
