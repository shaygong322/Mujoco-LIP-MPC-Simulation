from matplotlib import pyplot as plt
import numpy as np

AR_SIM = True
AR_SIM = False


# if AR_SIM:
#     path = "build/User/"
# else:
#     path = "MujocoPySim/"
path = "tsc_logs/"


#Robot's current state
data_raw = np.loadtxt(path+"datasets_state.txt", delimiter=',', dtype=str)
print("Data robot states. Shape: ", [data_raw.shape[0],data_raw.shape[1]-1])
data_states = np.zeros((data_raw.shape[0],data_raw.shape[1]-1))

for i in range(data_raw.shape[0]):
    for j in range(data_raw.shape[1]-1):    #remove the last column because it is the new line character (empty)
        # print("ITERATION: ", i, j)
        # print(data_raw[i, j])
        data_states[i, j] = float(data_raw[i, j])

time_states     = data_states[:,0]
base_position   = data_states[:,1:4]
base_quaternion = data_states[:,4:8]
joint_position  = data_states[:,8:34]

base_velocity      = data_states[:,34:37]
base_ang_velocity  = data_states[:,37:40]
joint_velocity  = data_states[:,40:66]

base_acceleration   = data_states[:,66:69]

lfoot_position  = data_states[:,69:72]
rfoot_position  = data_states[:,72:75]


#Robot's task space planning
data_raw = np.loadtxt(path+"datasets_planning.txt", delimiter=',', dtype=str)
print("Data robot states. Shape: ", [data_raw.shape[0],data_raw.shape[1]-1])
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


# exit()

# plot data with legends
fig, ax = plt.subplots()   #base position  x,y,z
ax.plot(time_states, base_position)
ax.plot(time_task, des_base_position, '--')
ax.legend(['x', 'y', 'z', 'des_x', 'des_y', 'des_z'])

fig, ax = plt.subplots()   #base velocity  x,y,z
ax.plot(time_states, base_velocity)
ax.plot(time_task, des_base_velocity, '--')
ax.legend(['vx', 'vy', 'vz', 'des_vx', 'des_vy', 'des_vz'])
# ax.legend(['vx', 'vy', 'vz'])

# fig, ax = plt.subplots()   #base velocity  x,y,z
# ax.plot(time_states, base_velocity[:,0], label='vx')
# ax.plot(time_task, des_base_velocity[:,0], '--', label='des_vx')

# fig, ax = plt.subplots()   #base velocity  x,y,z
# ax.plot(time_states, base_velocity[:,1], label='vy')
# ax.plot(time_task, des_base_velocity[:,1], '--', label='des_vy')

# fig, ax = plt.subplots()   #base velocity  x,y,z
# ax.plot(time_states, base_velocity[:,2], label='vz')
# ax.plot(time_task, des_base_velocity[:,2], '--', label='des_vz')


fig, ax = plt.subplots()   #base acceleration  x,y,z
ax.plot(time_states, base_acceleration)
ax.plot(time_task, des_base_acceleration, '--')
ax.legend(['accx', 'accy', 'accz', 'des_accx', 'des_accy', 'des_accz'])



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