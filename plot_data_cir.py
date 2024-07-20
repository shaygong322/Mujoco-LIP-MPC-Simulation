import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

""" Plot data """

def plot_cir(cir, stri = None):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = cir[2]*np.cos(theta)+cir[0]
        y = cir[2]*np.sin(theta)+cir[1]
        if stri == None:
            plt.plot(x, y, color='#696969', linewidth=2.5)
        else:
            plt.plot(x, y, '--', color='#696969', linewidth=2.5)

    
def elp_func(elp, x, y):
		a = (elp[3]*math.cos(elp[4]))**2+(elp[2]*math.sin(elp[4]))**2
		b = 2*math.cos(elp[4])*math.sin(elp[4])*(elp[3]**2-elp[2]**2)
		c = (elp[3]*math.sin(elp[4]))**2+(elp[2]*math.cos(elp[4]))**2
		v1 = a*(x-elp[0])**2
		v2 = b*(x-elp[0])*(y-elp[1])
		v3 = c*(y-elp[1])**2
		return v1+v2+v3-(elp[3]*elp[2])**2

def plot_elp(elp, margin, stri = None):            # [x_c, y_c, A, B, C, r]
    x1 = np.arange(margin[0], margin[1], 0.1)
    x2 = np.arange(margin[0], margin[1], 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    z = elp_func(elp, X1, X2)
    if stri == None:
        plt.contour(X1, X2, z, [0], colors='#696969', linewidths=2.5)
    else:
        plt.contour(X1, X2, z, [0], colors='#696969', linewidths=2.5, linestyles='dashed')


# LIP-MPC model
num = 'xy'
path_l = 'MujocoPySim/data_log/LIP_me'+ str(num) +'_'
# DD-MPC model
num = 'xx'
path_d = 'MujocoPySim/data_log/DD_me'+ str(num) +'_'
path_o = 'MujocoPySim/data_log/OLIP_me'+ str(num) +'_'



start = [0, 0]
goal = [10, 10]
margin = [-0.5, 10.5]

with open(path_l + 'pos.pkl', 'rb') as file:
    real_com_traj_lip = pickle.load(file)
with open(path_l + 'time.pkl', 'rb') as file:
    t_list_lip = pickle.load(file)
with open(path_l + 'foot.pkl', 'rb') as file:
    map_p_real_lip = pickle.load(file)
with open(path_l + 'heading.pkl', 'rb') as file:
    heading_lip = pickle.load(file)
with open(path_l + 'turning.pkl', 'rb') as file:
    turning_lip = pickle.load(file)
with open(path_l + 'body_vel.pkl', 'rb') as file:
    body_vel_lip = pickle.load(file)
with open(path_l + 'ellp.pkl', 'rb') as file:
    elp_list_lip = pickle.load(file)
with open(path_l + 'cir.pkl', 'rb') as file:
    cir_list_lip = pickle.load(file)
with open(path_l + 'real_end.pkl', 'rb') as file:
    real_str_lip = pickle.load(file)
with open(path_l + 'pred_end.pkl', 'rb') as file:
    pred_str_lip = pickle.load(file)
with open(path_l + 'pred_feasi_end.pkl', 'rb') as file:
    feasi_traj = pickle.load(file)
with open(path_l + 'pred_fail_end.pkl', 'rb') as file:
    fail_traj = pickle.load(file)

with open(path_o + 'time.pkl', 'rb') as file:
    t_list_olip = pickle.load(file)
with open(path_o + 'heading.pkl', 'rb') as file:
    heading_olip = pickle.load(file)
with open(path_o + 'turning.pkl', 'rb') as file:
    turning_olip = pickle.load(file)
with open(path_o + 'body_vel.pkl', 'rb') as file:
    body_vel_olip = pickle.load(file)

with open(path_d + 'pos.pkl', 'rb') as file:
    real_com_traj_dd = pickle.load(file)
with open(path_d + 'time.pkl', 'rb') as file:
    t_list_dd = pickle.load(file)
with open(path_d + 'foot.pkl', 'rb') as file:
    map_p_real_dd = pickle.load(file)
with open(path_d + 'heading.pkl', 'rb') as file:
    heading_dd = pickle.load(file)
with open(path_d + 'body_vel.pkl', 'rb') as file:
    body_vel_dd = pickle.load(file)
with open(path_d + 'ellp.pkl', 'rb') as file:
    elp_list_dd = pickle.load(file)
with open(path_d + 'cir.pkl', 'rb') as file:
    cir_list_dd = pickle.load(file)
with open(path_d + 'real_end.pkl', 'rb') as file:
    real_str_dd = pickle.load(file)
with open(path_d + 'pred_feasi_end.pkl', 'rb') as file:
    feasi_str = pickle.load(file)
with open(path_d + 'pred_fail_end.pkl', 'rb') as file:
    fail_str = pickle.load(file)


safe_dis = 0.4
cir_safe_lip = np.array(cir_list_lip) + [0, 0, safe_dis]
# elp_safe_lip = np.array(elp_list_lip) + [0, 0, safe_dis, safe_dis, 0]
elp_safe_lip = []
    

# Calculate error
## LIP
# tracking_err_lip = [0]
# step_lip = [0]
# for i in range(len(pred_str_lip)-1):
#     d_dis = math.sqrt((pred_str_lip[i][1, 0]-pred_str_lip[i+1][0, 0])**2
#                       +(pred_str_lip[i][1, 1]-pred_str_lip[i+1][0, 1])**2)
#     tracking_err_lip.append(d_dis)
#     step_lip.append(i)

# tracking_err_lip = np.array(tracking_err_lip)
# step_lip = (t_list_lip[-1]/i)*np.array(step_lip)

## DD
# tracking_err_dd = [0]
# step_dd = [0]
# for i in range(len(pred_str_dd)-1):
#     d_dis = math.sqrt((pred_str_dd[i][1, 0]-pred_str_dd[i+1][0, 0])**2
#                       +(pred_str_dd[i][1, 1]-pred_str_dd[i+1][0, 1])**2)
#     tracking_err_dd.append(d_dis)
#     step_dd.append(i)

# tracking_err_dd = np.array(tracking_err_dd)
# step_dd = (t_list_dd[-1]/i)*np.array(step_dd)


plt.rcParams.update({'font.size': 15})
figure, axes = plt.subplots(figsize=(5,5))
plt.plot(real_com_traj_lip[:, 0], real_com_traj_lip[:, 1], 'r', linewidth=2.5)
plt.plot(start[0], start[1], '^r', markersize=10)
plt.plot(goal[0], goal[1], '*g', markersize=15)

# plt.plot(target_p[:, 0], target_p[:, 1], 'rx')
plt.plot(map_p_real_lip[:, 0], map_p_real_lip[:, 1], '.b', markersize=7)
Drawing_colored_circle = plt.Circle((10, 10), 0.35, color = "#F5DEB3")
axes.add_artist( Drawing_colored_circle )
plt.legend(['CoM', 'Start', 'Goal', 'Foot', 'Goal region'],
           fontsize="15", loc ="upper left")
for each in cir_list_lip:
    plot_cir(each)
for each in elp_list_lip:
    plot_elp(each, margin)
# plt.grid(True)
plt.axis('equal')
plt.xlim(margin)
plt.ylim(margin)


figure, axes = plt.subplots(figsize=(5,5))
# plt.plot(real_com_traj_dd[:2930, 0], real_com_traj_dd[:2930, 1], 'r', linewidth=2.5)
plt.plot(real_com_traj_dd[:, 0], real_com_traj_dd[:, 1], 'r', linewidth=2.5)
plt.plot(start[0], start[1], '^r', markersize=10)
plt.plot(goal[0], goal[1], '*g', markersize=15)

# plt.plot(target_p[:, 0], target_p[:, 1], 'rx')
# plt.plot(map_p_real_dd[:2930, 0], map_p_real_dd[:2930, 1], '.b', markersize=7)
plt.plot(map_p_real_dd[:, 0], map_p_real_dd[:, 1], '.b', markersize=7)
Drawing_colored_circle = plt.Circle((10, 10), 0.35, color = "#F5DEB3")
axes.add_artist( Drawing_colored_circle )
plt.legend(['CoM', 'Start', 'Goal', 'Foot', 'Goal region'],
           fontsize="15", loc ="upper left")
# plt.plot(real_com_traj_dd[-1, 0], real_com_traj_dd[-1, 1], 'rX', markersize=13)

for each in cir_list_lip:
    plot_cir(each)
for each in elp_list_lip:
    plot_elp(each, margin)
# plot_cir([10, 10, 0.4])
# plt.grid(True)
plt.axis('equal')
# plt.xlim(margin)
# plt.ylim(margin)


plt.figure(figsize=(5,5))
plt.plot(t_list_olip[0:-1:20], body_vel_olip[0:-1:20, 0], linewidth=2.0)
plt.plot(t_list_lip[0:-1:40], body_vel_lip[0:-1:40, 0], linewidth=2.0)
# plt.plot(t_list_lip, body_vel_lip[:, 1], linewidth=2.0)
# plt.grid(True)
# plt.legend(['Original', 'Modified'])
# plt.legend(['vx', 'vy'])
# plt.plot(t_list_lip[0:-1:80], body_vel_lip[0:-1:80, 0], linewidth=2.0)
print(len(body_vel_lip))
print(len(body_vel_dd))
plt.ylim([-0.05, 0.9])


plt.figure(figsize=(5,5))
plt.plot(t_list_olip[0:-1:20], heading_olip[0:-1:20], linewidth=2.0)
plt.plot(t_list_lip[0:-1:40], heading_lip[0:-1:40], linewidth=2.0)
# plt.legend(['Original', 'Modified'])
plt.ylim([-0.57, 1.57])
# plt.grid(True)

plt.figure(figsize=(5,5))
plt.plot(t_list_olip[0:-1:20], turning_olip[0:-1:20], linewidth=2.0)
plt.plot(t_list_lip[0:-1:80], turning_lip[0:-1:80], linewidth=2.0)
# plt.legend(['Original', 'Modified'])
plt.ylim([-0.57, 0.57])

# plt.figure(figsize=(5,5))
# plt.plot(t_list_lip, turning_lip, linewidth=2.0)
# plt.ylim([-1.57, 1.57])
# plt.grid(True)
# plt.title('Turning angle with time')

plt.figure(figsize=(5,5))
plt.plot(real_com_traj_lip[:, 0], real_com_traj_lip[:, 1], linewidth=3.0)
for each in feasi_traj:
    plt.plot(each[:, 0], each[:, 1], linewidth=2.0, color='red')
for each in fail_traj:
    plt.plot(each[:, 0], each[:, 1], linewidth=2.0, color='#C0C0C0')
for each in cir_list_lip:
    plot_cir(each)
for each in elp_list_lip:
    plot_elp(each, margin)
for each in cir_safe_lip:
    plot_cir(each, 'dash')
for each in elp_safe_lip:
    plot_elp(each, margin, 'dash')
# plt.legend(['real trajectory', 'planned trajecory'],
#            fontsize="15", loc ="upper left")
plt.axis('equal')
plt.xlim(margin)
plt.ylim(margin)
# plt.grid(True)

plt.figure(figsize=(5,5))
plt.plot(t_list_dd[0:-1:40], body_vel_dd[0:-1:40, 0], linewidth=2.0)
plt.plot(t_list_dd, body_vel_dd[:, 1], linewidth=2.0)
# plt.grid(True)
plt.legend(['vx', 'vy'])

plt.figure(figsize=(5,5))
plt.plot(t_list_dd[0:-1:40], heading_dd[0:-1:40], linewidth=2.0)
plt.ylim([-1.57, 1.57])
# plt.grid(True)

# plt.figure(figsize=(5,5))
# plt.plot(real_str_lip[:, 0], real_str_lip[:, 1], linewidth=3.0)
# for each in pred_str_lip:
#     plt.plot(each[:, 0], each[:, 1], linewidth=2.0, color = 'red')
# for each in cir_list_lip:
#     plot_cir(each)
# for each in elp_list_lip:
#     plot_elp(each, margin)
# plt.legend(['real trajectory', 'planned trajecory'],
#            fontsize="15", loc ="upper left")
# plt.axis('equal')
# plt.xlim(margin)
# plt.ylim(margin)


plt.figure(figsize=(5,5))
plt.plot(real_com_traj_dd[:, 0], real_com_traj_dd[:, 1], linewidth=3.0)
for each in feasi_str:
    temp = np.array(each)
    plt.plot(temp[:, 0], temp[:, 1], '-or', linewidth=2.0)
for each in fail_str:
    temp = np.array(each)
    plt.plot(temp[:, 0], temp[:, 1], '-o', linewidth=2.0, color='#C0C0C0')
for each in cir_list_dd:
    plot_cir(each)
for each in elp_list_dd:
    plot_elp(each, margin)
for each in cir_safe_lip:
    plot_cir(each, 'dash')
for each in elp_safe_lip:
    plot_elp(each, margin, 'dash')
# plt.legend(['real trajectory', 'planned trajecory'],
#            fontsize="15", loc ="upper left")
plt.axis('equal')
# plt.xlim(margin)
# plt.ylim(margin)
# plt.grid(True)

# plt.figure(figsize=(11,5))
# plt.plot(step_lip, tracking_err_lip, linewidth=3.0)
# # plt.plot(step_dd[:90], tracking_err_dd[:90], linewidth=3.0)
# plt.plot(step_dd, tracking_err_dd, linewidth=3.0)
# plt.plot(step_lip[-1], tracking_err_lip[-1], '*g', markersize=15)
# plt.plot(step_dd[-1], tracking_err_dd[-1], '*g', markersize=15)
# plt.xlabel('Time')
# plt.ylabel('Error')
# # plt.grid(True)
# plt.legend(['LIP', 'Diff_Drv', 'end'], fontsize="15", loc ="lower right")
plt.show()