from matplotlib import pyplot as plt
import numpy as np

AR_SIM = False
# AR_SIM = False


# if AR_SIM:
#     path = "build/User/"
# else:
#     path = "build/"

path = "tsc_logs/"

#TSC variables. 
data_raw = np.loadtxt(path+"datasets_tsc.txt", delimiter=',', dtype=str)
print("TSC output. Shape: ", [data_raw.shape[0],data_raw.shape[1]-1])
data_tsc_outputs = np.zeros((data_raw.shape[0],data_raw.shape[1]-1))

for i in range(data_raw.shape[0]):
    for j in range(data_raw.shape[1]-1):    #remove the last column because it is the new line character (empty)
        # print("ITERATION: ", i, j)
        # print(data_raw[i, j])
        data_tsc_outputs[i, j] = float(data_raw[i, j])

time_tsc_output     = data_tsc_outputs[:,0]
torque_tsc          = data_tsc_outputs[:,1:27]
contact_force_tsc   = data_tsc_outputs[:,27:27+24]

# # PLOT DATA TSC OUTPUT
# fig, ax = plt.subplots()   #left leg
# ax.plot(time_tsc_output, torque_tsc[:,0:9])
# ax.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9']) 

# # fig, ax = plt.subplots()   #left arm
# # ax.plot(time_tsc_output, torque_tsc[:,9:13])
# # ax.legend(['10','11','12','13'])

# fig, ax = plt.subplots()   #right leg
# ax.plot(time_tsc_output, torque_tsc[:,13:22])
# ax.legend(['14', '15', '16', '17', '18', '19', '20', '21', '22'])

# # fig, ax = plt.subplots()   #right arm
# # ax.plot(time_tsc_output, torque_tsc[:,22:26])
# # ax.legend(['23', '24', '25', '26'])
# plt.show()





#Robot's current state
data_raw = np.loadtxt(path+"datasets_state.txt", delimiter=',', dtype=str)
print("Data robot states. Shape: ", [data_raw.shape[0],data_raw.shape[1]-1])
data_states = np.zeros((data_raw.shape[0],data_raw.shape[1]-1))

for i in range(data_raw.shape[0]):
    for j in range(data_raw.shape[1]-1):    #remove the last column because it is the new line character (empty)
        # print("ITERATION: ", i, j)
        # print(data_raw[i, j])
        data_states[i, j] = float(data_raw[i, j])

time_states            = data_states[:,0]
base_position_raw      = data_states[:,1:4]
base_velocity_raw      = data_states[:,4:7]
base_quaternion_raw    = data_states[:,7:11]
joint_position_raw     = data_states[:,11:37]
joint_velocity_raw     = data_states[:,37:63]
base_omega_raw         = data_states[:,63:66]
base_acceleration_raw  = data_states[:,66:69]

base_position          = data_states[:,69:72]
base_orientation       = data_states[:,72:76]
joint_position         = data_states[:,76:102]
base_velocity          = data_states[:,102:105]
base_omega             = data_states[:,105:108]
joint_velocity         = data_states[:,108:134]
base_acceleration      = data_states[:,134:137]

lfoot_position         = data_states[:,137:140]
rfoot_position         = data_states[:,140:143]


# time_states     = data_states[:,0]
# base_position   = data_states[:,1:4]
# base_velocity      = data_states[:,4:7]
# base_quaternion = data_states[:,7:11]

# joint_position  = data_states[:,11:37]
# joint_velocity  = data_states[:,37:63]

# base_ang_velocity  = data_states[:,63:66]
# base_acceleration   = data_states[:,66:69]

# base_position_filtered = data_states[:,69:72]
# base_orientation_filtered = data_states[:,72:76]
# joint_position_filtered = data_states[:,76:102]

# base_velocity_filtered = data_states[:,102:105]
# base_ang_velocity_filtered = data_states[:,105:108]
# joint_velocity_filtered = data_states[:,108:134]

# base_acceleration_filtered = data_states[:,134:137]
# lfoot_position  = data_states[:,137:140]
# rfoot_position  = data_states[:,1340:143]




#Robot's task space planning
data_raw = np.loadtxt(path+"datasets_planning.txt", delimiter=',', dtype=str)
print("Data tsc planning. Shape: ", [data_raw.shape[0],data_raw.shape[1]-1])
data_tsc = np.zeros((data_raw.shape[0],data_raw.shape[1]-1))

for i in range(data_raw.shape[0]):
    for j in range(data_raw.shape[1]-1):    #remove the last column because it is the new line character (empty)
        # print("ITERATION: ", i, j)
        # print(data_raw[i, j])
        data_tsc[i, j] = float(data_raw[i, j])

time_task                   = data_tsc[:,0]
des_base_position           = data_tsc[:,1:4]
des_base_velocity           = data_tsc[:,4:7]
des_base_acceleration       = data_tsc[:,7:10]
des_base_omega              = data_tsc[:,10:13]
des_base_omega_dot          = data_tsc[:,13:16]
des_left_foot_position      = data_tsc[:,16:19]
des_left_foot_velocity      = data_tsc[:,19:22]
des_left_foot_acc           = data_tsc[:,22:25]
des_right_foot_position     = data_tsc[:,25:28]
des_right_foot_velocity     = data_tsc[:,28:31]
des_right_foot_acc          = data_tsc[:,31:34]
des_force                   = data_tsc[:,34:34+24]




# #Reinforcement Learning State and Action
# data_raw = np.loadtxt(path+"datasets_rl.txt", delimiter=',', dtype=str)
# print("Data RL state and action. Shape: ", [data_raw.shape[0],data_raw.shape[1]-1])
# data_rl = np.zeros((data_raw.shape[0],data_raw.shape[1]-1))

# for i in range(data_raw.shape[0]):
#     for j in range(data_raw.shape[1]-1):    #remove the last column because it is the new line character (empty)
#         # print("ITERATION: ", i, j)
#         # print(data_raw[i, j])
#         data_rl[i, j] = float(data_raw[i, j])
# start_idx = 3000
# end_idx = 8000
# time_rl     = data_rl[start_idx:end_idx,0]
# rl_state    = data_rl[start_idx:end_idx,1:26]
# rl_action   = data_rl[start_idx:end_idx,26:33]

# # PLOT DATA RL STATE AND ACTION
# fig, ax = plt.subplots()   #base position  x,y,z
# ax.plot(time_rl, rl_state[:,0:6])
# ax.legend(['x', 'y', 'z', 'vx', 'vy'])

# fig, ax = plt.subplots()   #base position  x,y,z
# ax.plot(time_rl, rl_state[:,6:12])
# ax.legend(['roll', 'pitch', 'next_x', 'next_next_x', 'next_y', 'next_next_y'])

# fig, ax = plt.subplots()   #base position  x,y,z
# ax.plot(time_rl, rl_action)
# ax.legend(['offset_foot_x', 'offset_foot_y','offset height', 'offset_vx', 'offset_vy'])


# # PLOT DATA RL STATE AND ACTION
# fig, ax = plt.subplots()   #base position  x,y,z, vx, vy
# ax.plot(time_rl, rl_state[:,0:5])
# ax.legend(['x', 'y', 'z', 'vx', 'vy'])

# fig, ax = plt.subplots()   #pos sw foot
# ax.plot(time_rl, rl_state[:,5:8])
# ax.legend(['psw_x', 'psw_y', 'psw_z'])

# fig, ax = plt.subplots()   #vel error and base vel
# ax.plot(time_rl, rl_state[:,8:12])
# ax.legend(['evx', 'evy', 'vx_d', 'vy_d'])


# fig, ax = plt.subplots()   #foothold positions
# ax.plot(time_rl, rl_state[:,12:18])
# ax.legend(['next fhx', 'next next fhx', 'next fhy', 'next next fhy', 'last fhx', 'last fhx','last fhy'])

# fig, ax = plt.subplots()   #last action
# ax.plot(time_rl, rl_state[:,18:25])
# ax.legend(['last 1', '2', '3', '4','5','6','7'])


# fig, ax = plt.subplots()   #base position  x,y,z
# ax.plot(time_rl, rl_action)
# ax.legend(['1', '2', '3', '4','5','6','7'])


# fig, ax = plt.subplots()   #base position  x,y,z
# ax.plot(time_rl, rl_state[:,18:20])
# ax.plot(time_rl, rl_action[:,0:2])
# ax.plot(time_rl, rl_state[:,14])
# ax.legend(["x_foot","y_foot","last_x_foot","last_y_foot","fhy"])



# plot data with legends
fig, ax = plt.subplots()   #base position  x,y,z
ax.plot(time_states, base_position)
# ax.plot(time_states, base_position[:,0:2])
ax.plot(time_task, des_base_position, '--')
# ax.plot(time_task, des_base_position[:,0:2], '--')
ax.legend(['x', 'y', 'z', 'des_x', 'des_y', 'des_z'])
# ax.legend(['x', 'y', 'des_x', 'des_y'])

# fig, ax = plt.subplots()   #base velocity  x,y,z
# ax.plot(time_states, base_velocity)
# ax.plot(time_task, des_base_velocity, '--')
# ax.legend(['vx', 'vy', 'vz', 'des_vx', 'des_vy', 'des_vz'])

# fig, ax = plt.subplots()   #base velocity  x,y,z
# ax.plot(time_states, base_omega)
# ax.plot(time_task, des_base_omega, '--')
# ax.legend(['wx', 'wy', 'wz', 'des_wx', 'des_wy', 'des_wz'])


# plot data with legends
fig, ax = plt.subplots()   #base position  x,y,z
ax.plot(time_states, base_position_raw)
ax.plot(time_task, des_base_position, '--')
ax.legend(['x_raw', 'y_raw', 'z_raw', 'des_x', 'des_y', 'des_z'])
# ax.plot(time_states, base_position_raw[:,0:2])
# ax.plot(time_task, des_base_position[:,0:2], '--')
# ax.legend(['x_raw', 'y_raw', 'des_x', 'des_y'])

# fig, ax = plt.subplots()   #base velocity  x,y,z
# ax.plot(time_states, base_velocity_raw)
# ax.plot(time_task, des_base_velocity, '--')
# ax.legend(['vx_raw', 'vy_raw', 'vz_raw', 'des_vx', 'des_vy', 'des_vz'])

# fig, ax = plt.subplots()   #base velocity  x,y,z
# ax.plot(time_states, base_omega_raw)
# ax.plot(time_task, des_base_omega, '--')
# ax.legend(['wx_raw', 'wy_raw', 'wz_raw', 'des_wx', 'des_wy', 'des_wz'])


# ax.legend(['vx', 'vy', 'vz'])

# fig, ax = plt.subplots()   #base acceleration  x,y,z
# ax.plot(time_states, base_velocity[:,0], label='vx')
# ax.plot(time_task, des_base_velocity[:,0], '--', label='des_vx')

# fig, ax = plt.subplots()   #base acceleration  x,y,z
# ax.plot(time_states, base_velocity[:,1], label='vy')
# ax.plot(time_task, des_base_velocity[:,1], '--', label='des_vy')

# fig, ax = plt.subplots()   #base acceleration  x,y,z
# ax.plot(time_states, base_velocity[:,2], label='vz')
# ax.plot(time_task, des_base_velocity[:,2], '--', label='des_vz')


# fig, ax = plt.subplots()   #base acceleration  x,y,z
# ax.plot(time_states, base_velocity[:,2], label='vz')
# ax.plot(time_task, des_base_velocity[:,2], '--', label='des_vz')

# plt.show()
# exit()

# create a figure with 3 subplots and plot a vector in each subplot
fig, ax = plt.subplots(3,1)
ax[0].plot(time_states, lfoot_position[:,0], label='lfoot_x')
ax[0].plot(time_task, des_left_foot_position[:,0], '--', label='des_lfoot__x')
ax[0].legend()
ax[1].plot(time_states, lfoot_position[:,1], label='lfoot_y')
ax[1].plot(time_task, des_left_foot_position[:,1], '--', label='des_lfoot__y')
ax[1].legend()
ax[2].plot(time_states, lfoot_position[:,2], label='lfoot_z')
ax[2].plot(time_task, des_left_foot_position[:,2], '--', label='des_lfoot__z_')
ax[2].legend()


# create a figure with 3 subplots and plot a vector in each subplot
fig, ax = plt.subplots(3,1)
ax[0].plot(time_states, rfoot_position[:,0], label='rfoot_x')
ax[0].plot(time_task, des_right_foot_position[:,0], '--', label='des_rfoot__x')
ax[0].legend()
ax[1].plot(time_states, rfoot_position[:,1], label='rfoot_y')
ax[1].plot(time_task, des_right_foot_position[:,1], '--', label='des_rfoot__y')
ax[1].legend()
ax[2].plot(time_states, rfoot_position[:,2], label='rfoot_z')
ax[2].plot(time_task, des_right_foot_position[:,2], '--', label='des_rfoot__z_')
ax[2].legend()




plt.show()



# void Manager::saveAllData() {

#     /* gait */
#     for (int i = 0; i < 2; i++) {
#         out_gait << gaitScheduler.data().stanceTimeRemain(i) << ", ";
#         out_gait << gaitScheduler.data().swingTimeRemain(i) << ", ";
#     }

#     /* planning */
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.floatingBaseTask.pos(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.floatingBaseTask.vel(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.floatingBaseTask.acc(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.floatingBaseTask.omega(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.floatingBaseTask.omega_dot(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.leftFootTask.pos(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.leftFootTask.vel(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.leftFootTask.acc(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.rightFootTask.pos(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.rightFootTask.vel(i) << ", ";
#     }
#     for (int i = 0; i < 3; i++) {
#         out_planning << tasks.rightFootTask.acc(i) << ", ";
#     }
#     for (int i = 0; i < 24; i++) {
#         out_planning << tasks.forceTask(i) << ", ";
#     }