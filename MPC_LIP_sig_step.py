import math
import cyipopt
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

"""12.14 update: Use three control - foot placement(x, y), turning heading angle
                 Having velocity-turning angle constraints
                 Add target heading angle turning value to cost function 
                 Modify as single step for robot simulation
                 """

class MPCCBF:
    def __init__(self, goals, obs_param, obs_cbf, margin, step = 3):
        # Robot params
        self.goal = np.matrix(goals).T
        Hei = 1.0
        g = 9.81
        self.beta = math.sqrt(g/Hei)
        self.dt = 0.4

        # Obs env
        self.margin = margin
        self.obs_list = obs_param

        # MPC
        self.N = step

        # CBF 
        self.obs_safe = obs_cbf
        self.power = 4

        # Constraints param
        self.leg = 0.09                 ## leg length square
        self.x_max = 5
        self.bvx_max = 0.8
        self.bvx_min = 0.4
        self.bvy_max = 0.3
        self.bvy_min = 0.15
        self.ang_max = math.pi/16

        # ALIP param
        self.step_gap = 0.3
        self.sigma = float(self.beta*mp.coth(self.dt*self.beta/2))

        # dynamics
        self.A = np.matrix([[math.cosh(self.beta*self.dt), 0, math.sinh(self.beta*self.dt)/self.beta, 0, 0],
                            [0, math.cosh(self.beta*self.dt), 0, math.sinh(self.beta*self.dt)/self.beta, 0],
                            [math.sinh(self.beta*self.dt)*self.beta, 0, math.cosh(self.beta*self.dt), 0, 0],
                            [0, math.sinh(self.beta*self.dt)*self.beta, 0, math.cosh(self.beta*self.dt), 0], 
                            [0, 0, 0, 0, 1]])
        self.B = np.matrix([[1-math.cosh(self.beta*self.dt), 0, 0],
                            [0, 1-math.cosh(self.beta*self.dt), 0],
                            [-math.sinh(self.beta*self.dt)*self.beta, 0, 0],
                            [0, -math.sinh(self.beta*self.dt)*self.beta, 0],
                            [0, 0, 1]])
        
        a = 5
        b = 1
        D = a*(math.cosh(self.beta*self.dt)-1)**2 + b*(math.sinh(self.beta*self.dt)*self.beta)**2
        Ch = -a*(math.cosh(self.beta*self.dt)-1)/D
        Sh = -b*math.sinh(self.beta*self.dt)*self.beta/D
        self.W = np.matrix([[Ch, 0, Sh, 0, 0], [0, Ch, 0, Sh, 0], [0, 0, 0, 0, 1]])
        self.M_A = (self.A - self.B @ self.W @ self.A)
        self.M_B = self.B @ self.W
        self.B_vel_shr = self.B[2:4, 0:2]
        self.inv_B_vel_shr = np.linalg.inv(self.B_vel_shr)

        Pre_1 = self.M_B
        Pre_2 = self.M_A @ self.M_B
        Pre_3 = self.M_A @ self.M_A @ self.M_B
        zero_m= np.zeros_like(Pre_1)
        d0 = np.concatenate([zero_m, zero_m, zero_m], axis=1)
        d1 = np.concatenate([Pre_1, zero_m, zero_m], axis=1)
        d2 = np.concatenate([Pre_2, Pre_1, zero_m], axis=1)
        d3 = np.concatenate([Pre_3, Pre_2, Pre_1], axis=1)
        self.dx_du = np.concatenate([d0, d1, d2, d3])

        pla_1 = self.W
        pla_2 = -self.W @ self.A @ self.M_B
        pla_3 = -self.W @ self.A @ self.M_A @ self.M_B
        zero_p= np.zeros_like(pla_1)
        dP1 = np.concatenate([pla_1, zero_p, zero_p], axis=1)
        dP2 = np.concatenate([pla_2, pla_1, zero_p], axis=1)
        dP3 = np.concatenate([pla_3, pla_2, pla_1], axis=1)
        self.dP_du = np.concatenate([dP1, dP2, dP3])
    

    def gen_control_test(self, state, leg_ind, init_guess, plot = False, trajec = []):
        # self.init_state = np.matrix(np.concatenate([start, glo_v, [hd_ang]])).T
        close_2_goal = False
        self.init_state = np.matrix(state).T
        xk = self.init_state
        u = self.solveMPCCBF(xk, leg_ind, init_guess)           # array format
        p_list = []
        hd_list = []
        xk_list = [np.ravel(xk)]

        for i in range(self.N):
            uk = np.array([u[5*i:5*(i+1)]]).T
            pk = self.solve_footdisp(xk, uk)                     # matrix format
            xk = self.M_A @ xk + self.M_B @ uk                  # matrix format
                
            temp_pos = xk[0:2]
            dis2goal = math.sqrt((temp_pos-self.goal).T @ (temp_pos-self.goal))
            p_list.append(np.ravel(pk))
            xk_list.append(np.ravel(xk))
            hd_list.append(float(xk[4]))

            if dis2goal <= 0.35:
                close_2_goal = True

        if plot:
            for j in range(self.N):
                p = np.ravel(p_list[j])
                xk = np.ravel(xk_list[j])
                temp_pos_det = self.xk_track_det(xk, p, self.dt)
                plt.plot(temp_pos_det[1:, 0], temp_pos_det[1:, 1])
                plt.plot(p[0], p[1], '.b')
            plt.plot(xk_list[1][0], xk_list[1][1], '.r')
            if trajec != []:
                plt.plot(trajec[1:, 0], trajec[1:, 1], 'b--')
            for each in self.obs_list:
                self.plot_cir(each)
            # plt.xlim(self.margin)
            # plt.ylim(self.margin)
            # plt.xlim([-0.5, 6])
            # plt.ylim([-1, 1])
            plt.xlim([-0.5, 10])
            plt.ylim([-0.5, 10])
            plt.grid(True)
            plt.title('Pd foot loc')
        return xk_list[1:], p_list[0], hd_list, close_2_goal
    
    
    def get_next_states(self, glo_pos, glo_vel, glo_hd, glo_p, t_rest, plot = False):
        # dynamics
        A = np.matrix([[math.cosh(self.beta*t_rest),0,math.sinh(self.beta*t_rest)/self.beta,0,0],
                       [0,math.cosh(self.beta*t_rest),0,math.sinh(self.beta*t_rest)/self.beta,0],
                       [math.sinh(self.beta*t_rest)*self.beta,0,math.cosh(self.beta*t_rest),0,0],
                       [0,math.sinh(self.beta*t_rest)*self.beta,0,math.cosh(self.beta*t_rest),0],
                       [0, 0, 0, 0, 1]])
        B = np.matrix([[1-math.cosh(self.beta*t_rest),0,0],
                       [0,1-math.cosh(self.beta*t_rest),0],
                       [-math.sinh(self.beta*t_rest)*self.beta,0,0],
                       [0,-math.sinh(self.beta*t_rest)*self.beta,0],
                       [0, 0, t_rest*(1/self.dt)]])

        xk = np.matrix(np.concatenate([glo_pos, glo_vel, [glo_hd]])).T
        p = np.matrix(glo_p).T
        x_next = A @ xk + B @ p
        p = np.ravel(p)
        temp_pos_det = self.xk_track_det(np.ravel(xk), p, t_rest)

        if plot:
            # p = np.ravel(p)
            # temp_pos_det = self.xk_track_det(np.ravel(xk), p)
            plt.plot(temp_pos_det[1:, 0], temp_pos_det[1:, 1], 'r')
            plt.xlim([-0.5, 6])
            plt.ylim([-1, 1])
            plt.grid(True)
            plt.title('Pd next states with current foot loc')
            plt.scatter(p[0], p[1])
            plt.show()
        return np.ravel(x_next), temp_pos_det
    

    def alip_des_vel(self, vx_max, leg_ind):
        v_com_des = vx_max
        vdes_x = self.sigma*v_com_des*self.dt/2
        vdes_y = 0.5*(-0.5*leg_ind*self.step_gap)*\
            (self.beta*math.sinh(self.beta*self.dt)) / (math.cosh(self.beta*self.dt) + 1)
        return np.array([vdes_x, vdes_y])
    

    def cal_foot_with_veldes(self, x_state, vel_des_glo):
        xk = np.array([x_state]).T
        xk_nex = np.array([vel_des_glo]).T
        A_x = self.A @ xk
        p_f = self.inv_B_vel_shr@(xk_nex - A_x[2:4])
        return np.ravel(p_f)


    def solveMPCCBF(self, xk, od_ev, init_guess):
        xk_array = np.ravel(xk)
        if init_guess == None:
            u0 = np.ravel([xk_array, xk_array, xk_array])
        else:
            u0 = np.ravel([init_guess[1], init_guess[2], init_guess[2]])
        
        lb = None
        ub = None
        cl = []
        cu = [] 
        if od_ev > 0:
            for i in range(self.N):
                if np.mod(i, 2) == 0:
                    cl = np.append(cl, [self.bvx_min, self.bvy_min])
                    cl = np.append(cl, np.zeros(len(self.obs_safe)))
                    cl = np.append(cl, [0, -self.ang_max])
                    cu = np.append(cu, [self.bvx_max, self.bvy_max])
                    cu = np.append(cu, np.inf*np.ones(len(self.obs_safe)))
                    cu = np.append(cu, [self.leg, self.ang_max])
                else:
                    cl = np.append(cl, [self.bvx_min, -self.bvy_max])
                    cl = np.append(cl, np.zeros(len(self.obs_safe)))
                    cl = np.append(cl, [0, -self.ang_max])
                    cu = np.append(cu, [self.bvx_max, -self.bvy_min])
                    cu = np.append(cu, np.inf*np.ones(len(self.obs_safe)))
                    cu = np.append(cu, [self.leg, self.ang_max])

        else:
            for i in range(self.N):
                if np.mod(i, 2) == 0:
                    cl = np.append(cl, [self.bvx_min, -self.bvy_max])
                    cl = np.append(cl, np.zeros(len(self.obs_safe)))
                    cl = np.append(cl, [0, -self.ang_max])
                    cu = np.append(cu, [self.bvx_max, -self.bvy_min])
                    cu = np.append(cu, np.inf*np.ones(len(self.obs_safe)))
                    cu = np.append(cu, [self.leg, self.ang_max])
                else:
                    cl = np.append(cl, [self.bvx_min, self.bvy_min])
                    cl = np.append(cl, np.zeros(len(self.obs_safe)))
                    cl = np.append(cl, [0, -self.ang_max])
                    cu = np.append(cu, [self.bvx_max, self.bvy_max])
                    cu = np.append(cu, np.inf*np.ones(len(self.obs_safe)))
                    cu = np.append(cu, [self.leg, self.ang_max])

        goal = self.goal
        goal_array = np.ravel(self.goal)
        for i in range(len(self.obs_safe)):
            cir = self.obs_safe[i]
            temp_cen_dis = (xk_array[0]-cir[0])**2+(xk_array[1]-cir[1])**2
            temp_goal_dis = (xk_array[0]-goal_array[0])**2+(xk_array[1]-goal_array[1])**2

            if temp_cen_dis < temp_goal_dis and temp_cen_dis < 9*cir[2]**2:
                theta = math.atan2(goal_array[1]-xk_array[1], goal_array[0]-xk_array[0])
                alpha = math.atan2(cir[1]-xk_array[1], cir[0]-xk_array[0])
                d_the1 = theta-alpha
                if d_the1 < 0 and abs(d_the1) > math.pi:
                    d_the1 = d_the1 + 2*math.pi
                elif d_the1 > 0 and abs(d_the1) > math.pi:
                    d_the1 = d_the1 -2*math.pi

                if abs(d_the1) < (math.pi/12):
                    if d_the1 < 0:
                        new_ang = theta-math.pi/12
                    else:
                        new_ang = theta+math.pi/12
                    x = math.sqrt(temp_goal_dis)*math.cos(new_ang)
                    y = math.sqrt(temp_goal_dis)*math.sin(new_ang)
                    goal = np.matrix([xk_array[0]+x, xk_array[1]+y]).T
                    break


        nlp = cyipopt.Problem(
           n=len(u0),
           m=len(cl),
           problem_obj=LIP_Prob(xk, self.M_A, self.M_B, self.A, self.W, self.dx_du,
                                self.dP_du, self.obs_safe, goal, self.N),
           lb=lb,
           ub=ub,
           cl=cl,
           cu=cu,
        )
        # nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('linear_solver', 'ma57')
        nlp.add_option('hsllib', 'libcoinhsl.so')
        nlp.add_option('max_iter', 20)
        nlp.add_option('sb', 'yes')
        # nlp.add_option('print_timing_statistics', 'yes')
        nlp.add_option('print_level', 0)
        # nlp.add_option('linear_solver', 'ma27')
        # nlp.add_option('hessian_approximation', 'limited-memory')
        # nlp.add_option('derivative_test', 'first-order')

        u, info = nlp.solve(u0)
        return u
    
    
    def xk_track_det(self, xk, contr, t_rest):
        t_det = np.arange(0, t_rest+0.01, 0.01)
        temp_pos_det = [xk[0:2]]
        xk = np.matrix(xk).T
        p = np.matrix(contr).T
        for i in range(len(t_det)):
            A = np.matrix([[math.cosh(self.beta*t_det[i]), 0, math.sinh(self.beta*t_det[i])/self.beta, 0, 0],
                            [0, math.cosh(self.beta*t_det[i]), 0, math.sinh(self.beta*t_det[i])/self.beta, 0],
                            [math.sinh(self.beta*t_det[i])*self.beta, 0, math.cosh(self.beta*t_det[i]), 0, 0],
                            [0, math.sinh(self.beta*t_det[i])*self.beta, 0, math.cosh(self.beta*t_det[i]), 0],
                            [0, 0, 0, 0, 1]])
            B = np.matrix([[1-math.cosh(self.beta*t_det[i]), 0, 0],
                            [0, 1-math.cosh(self.beta*t_det[i]), 0],
                            [-math.sinh(self.beta*t_det[i])*self.beta, 0, 0],
                            [0, -math.sinh(self.beta*t_det[i])*self.beta, 0],
                            [0, 0, t_det[i]*(1/self.dt)]])
            temp_xk = A @ xk + B @ p
            temp_pos_det.append(list(np.ravel(temp_xk[0:2])))
        return np.array(temp_pos_det)

    
    def solve_footdisp(self, xk, u):
        xf = self.A @ xk
        dx = u - xf
        p = self.W @ dx
        return p
    

    def tube_func(self, heading_list, init_tube_value):
        tube_upper = 0.15
        tube_lower = -0.15
        new_heading = np.zeros_like(heading_list)
        tube_value = init_tube_value
        for i in range(len(heading_list)):
            d_head_value = heading_list[i]-tube_value
            if d_head_value > 0:
                if tube_upper > d_head_value:
                    tube_value += 0.5*d_head_value
                else:
                    tube_value += 0.7*d_head_value
            elif d_head_value < 0:
                if tube_lower < d_head_value:
                    tube_value += 0.5*d_head_value
                else:
                    tube_value += 0.7*d_head_value
            new_heading[i] = tube_value
        return new_heading
    

    def plot_cir(self, cir):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = cir[2]*np.cos(theta)+cir[0]
        y = cir[2]*np.sin(theta)+cir[1]
        plt.plot(x, y)


class LIP_Prob:
    def __init__(self, xk, M_A, M_B, A, W, dx, dp, obs_safe, goal, step):
        # MPC param
        self.p = 2
        self.q = 1
        self.r = 15
        self.P = self.p*np.identity(2)
        self.Q = self.q*np.identity(2)
        self.N = step

        # CBF param
        self.gama = 0.4
        self.obs_modi_pram = obs_safe
        self.power = 4

        # Energy
        self.s = 0.014*180/math.pi

        # Partial
        self.dx_du = dx
        self.dP_du = dp
        
        # Dynamics
        self.A = A
        self.W = W
        self.M_A = M_A
        self.M_B = M_B

        # Initial states
        self.xk = xk

        # Final goal
        self.goal = goal
    

    def objective(self, u):
        cost = 0.0
        # Compute states
        u = np.matrix(u).T
        x = np.matrix(np.zeros([5, self.N + 1]))
        p = np.matrix(np.zeros([3, self.N]))
        x[:, 0] = self.xk
        for i in range(self.N):
            p[:, i] = self.solve_footdisp(x[:, i], u[5*i:5*(i+1)])
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[5*i:5*(i+1)]
            tar_ang, d_plc = self.cal_tar_ang(x[:, i+1])
            cost += (x[0:2, i+1]-self.goal).T @ self.Q @ (x[0:2, i+1]-self.goal) + (self.r*(x[4, i+1]-tar_ang)**2)
        cost += (x[0:2, 1]-self.goal).T @ self.P @ (x[0:2, 1]-self.goal)     
        cost = float(cost)
        return cost
    

    def gradient(self, u):
        # Compute states
        u = np.matrix(u).T
        x = np.matrix(np.zeros([5, self.N + 1]))
        p = np.matrix(np.zeros([3, self.N]))
        d_tan_du = []
        x[:, 0] = self.xk
        for i in range(self.N):
            p[:, i] = self.solve_footdisp(x[:, i], u[5*i:5*(i+1)])
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[5*i:5*(i+1)]
            d_tan_du.append(self.cal_dtar_ang_du(x[:, i+1], i))
        
        jac = np.ravel(2*(self.q+self.p)*((x[0,1]-self.goal[0])*self.dx_du[5,:]+\
                                          (x[1,1]-self.goal[1])*self.dx_du[6,:])\
                +2*self.q*((x[0,2]-self.goal[0])*self.dx_du[10,:]+(x[1,2]-self.goal[1])*self.dx_du[11,:])\
                +2*self.q*((x[0,3]-self.goal[0])*self.dx_du[15,:]+(x[1,3]-self.goal[1])*self.dx_du[16,:]))\
                +2*self.r*(d_tan_du[0]+d_tan_du[1]+d_tan_du[2])
        
        return jac
    

    def constraints(self, u):
        u = np.matrix(u).T
        x = np.matrix(np.zeros([5, self.N + 1]))
        p = np.matrix(np.zeros([3, self.N]))
        x[:, 0] = self.xk
        cons = []
        for i in range(self.N):
            p[:, i] = self.solve_footdisp(x[:, i], u[5*i:5*(i+1)])
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[5*i:5*(i+1)]
            pos = x[0:2, i]
            ang_h = x[4, i+1]
            M = np.matrix([[math.cos(ang_h), math.sin(ang_h)],[-math.sin(ang_h), math.cos(ang_h)]])
            v_body = M @ x[2:4, i+1]

            dlx = pos[0, 0]-p[0, i]
            dly = pos[1, 0]-p[1, i]
            leg_len = dlx**2+dly**2
            
            hk = self.h(x[0, i], x[1, i])
            h_next = self.h(x[0, i+1], x[1, i+1])
            dh = h_next - hk
            cbf = dh + self.gama*hk
            d_ang = np.array(p[2, i])

            cons = np.append(cons, v_body)
            cons = np.append(cons, cbf)
            cons = np.append(cons, [leg_len, d_ang])
        return cons


    def jacobian(self, u):
        u = np.matrix(u).T
        x = np.matrix(np.zeros([5, self.N + 1]))
        p = np.matrix(np.zeros([3, self.N]))
        ang = np.zeros(self.N)
        x[:, 0] = self.xk
        for i in range(self.N):
            p[:, i] = self.solve_footdisp(x[:, i], u[5*i:5*(i+1)])
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[5*i:5*(i+1)]
            ang_h = x[4, i+1]
            ang[i] = ang_h
            
        # Velo-Jac
        v_para1 = self.dv_du_para(x[2, 1], x[3, 1], x[4, 1])
        v_para2 = self.dv_du_para(x[2, 2], x[3, 2], x[4, 2])
        v_para3 = self.dv_du_para(x[2, 3], x[3, 3], x[4, 3])
        dv11_du = np.array(v_para1[0,0]*self.dx_du[7,:]+v_para1[0,1]*self.dx_du[8,:]+v_para1[0,2]*self.dx_du[9,:])
        dv21_du = np.array(v_para1[1,0]*self.dx_du[7,:]+v_para1[1,1]*self.dx_du[8,:]+v_para1[1,2]*self.dx_du[9,:])
        dv12_du = np.array(v_para2[0,0]*self.dx_du[12,:]+v_para2[0,1]*self.dx_du[13,:]+v_para2[0,2]*self.dx_du[14,:])
        dv22_du = np.array(v_para2[1,0]*self.dx_du[12,:]+v_para2[1,1]*self.dx_du[13,:]+v_para2[1,2]*self.dx_du[14,:])
        dv13_du = np.array(v_para3[0,0]*self.dx_du[17,:]+v_para3[0,1]*self.dx_du[18,:]+v_para3[0,2]*self.dx_du[19,:])
        dv23_du = np.array(v_para3[1,0]*self.dx_du[17,:]+v_para3[1,1]*self.dx_du[18,:]+v_para3[1,2]*self.dx_du[19,:])

        # CBF-Jac
        dh0_du = np.zeros(5*self.N)
        r = len(self.obs_modi_pram)
        dCBF1 = np.zeros([r, 15])
        dCBF2 = np.zeros([r, 15])
        dCBF3 = np.zeros([r, 15])
        for i in range(r):
            dh1d1, dh1d2 = self.dh(self.obs_modi_pram[i], x[0, 1], x[1, 1])
            dh1_du = np.ravel(dh1d1*self.dx_du[5, :] + dh1d2*self.dx_du[6, :])
            dh2d1, dh2d2 = self.dh(self.obs_modi_pram[i], x[0, 2], x[1, 2])
            dh2_du = np.ravel(dh2d1*self.dx_du[10,:] + dh2d2*self.dx_du[11,:])
            dh3d1, dh3d2 = self.dh(self.obs_modi_pram[i], x[0, 3], x[1, 3])
            dh3_du = np.ravel(dh3d1*self.dx_du[15,:] + dh3d2*self.dx_du[16,:])
            dCBF1[i, :] = dh1_du + (self.gama - 1)*dh0_du
            dCBF2[i, :] = dh2_du + (self.gama - 1)*dh1_du
            dCBF3[i, :] = dh3_du + (self.gama - 1)*dh2_du

        # Leg length-Jac
        df1 = np.array(2*(x[0, 0]-p[0, 0])*(self.dx_du[0,:]-self.dP_du[0,:])+\
                       2*(x[1, 0]-p[1, 0])*(self.dx_du[1,:]-self.dP_du[1,:]))
        df2 = np.array(2*(x[0, 1]-p[0, 1])*(self.dx_du[5,:]-self.dP_du[3,:])+\
                       2*(x[1, 1]-p[1, 1])*(self.dx_du[6,:]-self.dP_du[4,:]))
        df3 = np.array(2*(x[0, 2]-p[0, 2])*(self.dx_du[10,:]-self.dP_du[6,:])+\
                       2*(x[1, 2]-p[1, 2])*(self.dx_du[11,:]-self.dP_du[7,:]))

        # Turning angle-Jac
        theta_con1 = np.array(self.dP_du[2,:])
        theta_con2 = np.array(self.dP_du[5,:])
        theta_con3 = np.array(self.dP_du[8,:])
        
        jac = np.concatenate([dv11_du, dv21_du, dCBF1, df1, theta_con1,
                              dv12_du, dv22_du, dCBF2, df2, theta_con2,
                              dv13_du, dv23_du, dCBF3, df3, theta_con3])
        return jac


    def h(self, x1, x2):                        # Cal descition cost in matrix format
        h_v = np.array([])
        for each in self.obs_modi_pram:
            temp = (x1-each[0])**2 + (x2-each[1])**2 - each[2]**2
            h_v = np.append(h_v, float(temp))
        return h_v
    

    def dh(self, cir, x1, x2):
        dx1 = 2*(x1-cir[0])
        dx2 = 2*(x2-cir[1])
        return dx1, dx2
    

    def dv_du_para(self, vx, vy, the):
        a1 = math.cos(the)
        a2 = math.sin(the)
        a3 = -a2*vx + a1*vy
        b1 = -a2
        b2 = a1
        b3 = -a1*vx - a2*vy
        return np.array([[a1, a2, a3], [b1, b2, b3]])
    

    def solve_footdisp(self, xk, u):
        xf = self.A @ xk
        dx = u - xf
        p = self.W @ dx
        return p
    

    def den_du(self, hd, d_hd, dv):
        if hd == 0:
            df_dhd = 0
        else:
            df_dhd = self.s*hd/abs(hd)
        den = df_dhd*d_hd+dv
        return den
    
    def cal_tar_ang(self, state):
        d_plc = self.goal - state[0:2]
        tar_ang = math.atan2(d_plc[1], d_plc[0])
        return tar_ang, d_plc
    
    def cal_dtar_ang_du(self, state, i):
        tar_ang, d_plc = self.cal_tar_ang(state)
        d_tar_du = ((d_plc[0]*(-self.dx_du[5*(i+1)+1, :])-\
                     d_plc[1]*(-self.dx_du[5*(i+1), :]))/(d_plc[0]**2+d_plc[1]**2))
        d_ang = (state[4]-tar_ang)*(self.dx_du[5*(i+1)+4,:] - d_tar_du)
        return d_ang
    
    

if __name__ == "__main__":  
    start = [0, 0]
    goal = [[10, 10]]
    heading = 0
    glo_v = [0.6, -0.3]
    body_v = [-0.6, -0.3]
    margin = [-0.5, 10.5]
    safe_dis = 0.32

    obs_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [6, 4, 0.8], [7, 7, 1]])
    obs_safe = obs_list
    obs_safe = obs_safe + [0, 0, safe_dis]
    
    mpc_cbf1 = MPCCBF(goal, obs_list, obs_safe, margin)
    leg_ind = 1
    init_guess = None
    state = np.concatenate([start, glo_v, [heading]])

    ## Check gen_control_test func
    for i in range(5):
        x_list, p_list, close2goal = mpc_cbf1.gen_control_test(state, leg_ind, init_guess, True)
        leg_ind = (-1)*leg_ind
        init_guess = x_list
        state = x_list[0]

    ## Check get_nex_states func
    # glo_p = [0.13, -0.2, 0.1]
    # t_rest = 0.25
    # x_next, temp_pos_det = mpc_cbf1.get_next_states(start, glo_v, heading, glo_p, t_rest, True)
    # print(x_next)

    plt.show()
   