import math
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt

def elp_func(elp, x, y):
    a = (elp[3]*math.cos(elp[4]))**2+(elp[2]*math.sin(elp[4]))**2
    b = 2*math.cos(elp[4])*math.sin(elp[4])*(elp[3]**2-elp[2]**2)
    c = (elp[3]*math.sin(elp[4]))**2+(elp[2]*math.cos(elp[4]))**2
    v1 = a*(x-elp[0])**2
    v2 = b*(x-elp[0])*(y-elp[1])
    v3 = c*(y-elp[1])**2
    return v1+v2+v3-(elp[3]*elp[2])**2

# newly
def plot_elp(elp):            # [x_c, y_c, A, B, C, r]
    x1 = np.arange(-1, 11, 0.1)
    x2 = np.arange(-1, 11, 0.1)
    X1, X2 = np.meshgrid(x1, x2)
    z = elp_func(elp, X1, X2)
    plt.contour(X1, X2, z, [0], colors='#696969')


def plot_cir(cir):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = cir[2]*np.cos(theta)+cir[0]
    y = cir[2]*np.sin(theta)+cir[1]
    plt.plot(x, y, color='#696969')


def random_circle(num, margin, radius, safe_dis):
    cir_list = [[10, 10, 0.3], [0, 0, 1.0]]
    while True:
        collapse = False
        x_c = round(margin*random.random(), 2)
        y_c = round(margin*random.random(), 2)
        r_c = round((radius-0.35)*random.random() + 0.35, 2)

        for each in cir_list:
            dis = (x_c-each[0])**2+(y_c-each[1])**2-(r_c+each[2]+2*safe_dis)**2
            if dis >= 0:
                pass
            else:
                collapse = True
                break

        if collapse == False:
            cir_list.append([x_c, y_c, r_c])

        if len(cir_list) >= num+2:
            break
    cir_list.remove([10, 10, 0.3])
    cir_list.remove([0, 0, 1.0])
    return cir_list


def random_obs(cir_list, obs_typ):
    out_cir = []
    out_elp = []
    if obs_typ == 'mix':
        for i in range(len(cir_list)):
            if np.mod(i, 2) == 0:
                out_cir.append(cir_list[i])
            else:
                a = cir_list[i][2]
                b = round((a/2)*random.random()+(a/2), 2)
                phi = round(random.randint(0, 180)*math.pi/180, 2)
                out_elp.append([cir_list[i][0], cir_list[i][1], a, b, phi])
    elif obs_typ == 'cir':
        out_cir = cir_list

    return out_cir, out_elp


def gen_ran_obs_list(num, obs_typ):
    cir_seed = random_circle(num, 8.5, 1, 0.8)
    cir_list, elp_list = random_obs(cir_seed, obs_typ)
    # print('obs_cir_list = ', cir_list)
    # print('obs_elp_list = ', elp_list)

    return cir_list, elp_list


def plot_demo(cir_list, elp_list):
    plt.figure(figsize=(5,5))

    for each in elp_list:
        plot_elp(each)
    for each in cir_list:
        plot_cir(each)
    plt.xlim([-0.5, 10.5])
    plt.ylim([-0.5, 10.5])
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    c_l, e_l = gen_ran_obs_list(6, 'cir')
    plot_demo(c_l, e_l)


