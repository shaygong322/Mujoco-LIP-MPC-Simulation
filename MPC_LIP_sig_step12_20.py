import math
import cyipopt
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt


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

        # Constraints param
        self.leg = 0.09                 ## leg length square
        self.x_max = 5
        self.bvx_max = 0.8
        self.bvx_min = 0.0
        self.bvy_max = 0.35
        self.bvy_min = 0.15
        self.ang_max = math.pi/4

        # dynamics
        self.A = np.matrix([[math.cosh(self.beta*self.dt), 0, math.sinh(self.beta*self.dt)/self.beta, 0],
                            [0, math.cosh(self.beta*self.dt), 0, math.sinh(self.beta*self.dt)/self.beta],
                            [math.sinh(self.beta*self.dt)*self.beta, 0, math.cosh(self.beta*self.dt), 0],
                            [0, math.sinh(self.beta*self.dt)*self.beta, 0, math.cosh(self.beta*self.dt)]])
        self.B = np.matrix([[1-math.cosh(self.beta*self.dt), 0],
                            [0, 1-math.cosh(self.beta*self.dt)],
                            [-math.sinh(self.beta*self.dt)*self.beta, 0],
                            [0, -math.sinh(self.beta*self.dt)*self.beta]])
        
        a = 5
        b = 1
        D = a*(math.cosh(self.beta*self.dt)-1)**2 + b*(math.sinh(self.beta*self.dt)*self.beta)**2
        Ch = -a*(math.cosh(self.beta*self.dt)-1)/D
        Sh = -b*math.sinh(self.beta*self.dt)*self.beta/D
        self.W = np.matrix([[Ch, 0, Sh, 0], [0, Ch, 0, Sh]])
        self.M_A = (self.A - self.B @ self.W @ self.A)
        self.M_B = self.B @ self.W
        self.B_pos_shr = self.B[0:2, :]
        self.inv_B_pos_shr = np.linalg.inv(self.B_pos_shr)
        self.B_vel_shr = self.B[2:4, :]
        self.inv_B_vel_shr = np.linalg.inv(self.B_vel_shr)
        # self.inv_B_pos_shr = self.W[:, 0:2]
        # self.inv_B_vel_shr = self.W[:, 2:4]
    

    def generate_control_g_v(self, state, hd_ang, leg_ind, plot = False, trajec = []):
        # hd_ang: number, body_v: list, 
        # leg_ind: determine which leg is the swing leg.(odd for right, even for left)

        close_2_goal = False

        # Update
        self.init_state = np.matrix(state).T
        xk = self.init_state

        uk = self.solveMPCCBF(xk, leg_ind, hd_ang)           # array format
        uk = np.array([uk]).T
        p = self.solve_footdisp(xk, uk)                     # matrix format
        # pre_pos = xk[0:2]
        xk = self.M_A @ xk + self.M_B @ uk                  # matrix format
            
        temp_pos = xk[0:2]
        # temp_dis = list(np.ravel(temp_pos - pre_pos))
        # heading = math.atan2(temp_dis[1], temp_dis[0])
        dis2goal = math.sqrt((temp_pos-self.goal).T @ (temp_pos-self.goal))

        if dis2goal <= 0.35:
            close_2_goal = True

        if plot:
            p = np.ravel(p)
            temp_pos_det = self.xk_track_det(np.ravel(self.init_state), p, self.dt)
            if trajec != []:
                plt.plot(trajec[1:, 0], trajec[1:, 1], 'b--')
            plt.plot(temp_pos_det[1:, 0], temp_pos_det[1:, 1], 'r')
            # plt.xlim(self.margin)
            # plt.ylim(self.margin)
            plt.xlim([-0.5, 6])
            plt.ylim([-1, 1])
            plt.grid(True)
            plt.title('Pd foot loc')
            plt.scatter(p[0], p[1])
            plt.show()
        
        return np.ravel(xk), np.ravel(p), close_2_goal
    

    def get_next_states(self, glo_pos, glo_vel, glo_p, t_rest, init_pos, plot = False):
        # dynamics
        A = np.matrix([[math.cosh(self.beta*t_rest), 0, math.sinh(self.beta*t_rest)/self.beta, 0],
                        [0, math.cosh(self.beta*t_rest), 0, math.sinh(self.beta*t_rest)/self.beta],
                        [math.sinh(self.beta*t_rest)*self.beta, 0, math.cosh(self.beta*t_rest), 0],
                        [0, math.sinh(self.beta*t_rest)*self.beta, 0, math.cosh(self.beta*t_rest)]])
        B = np.matrix([[1-math.cosh(self.beta*t_rest), 0],
                        [0, 1-math.cosh(self.beta*t_rest)],
                        [-math.sinh(self.beta*t_rest)*self.beta, 0],
                        [0, -math.sinh(self.beta*t_rest)*self.beta]])

        xk = np.matrix(np.ravel([glo_pos, glo_vel])).T
        init_pos = np.matrix(np.ravel([init_pos])).T
        p = np.matrix(glo_p).T
        x_next = A @ xk + B @ p
        temp_pos = x_next[0:2]
        temp_dis = list(np.ravel(temp_pos - init_pos))
        p = np.ravel(p)
        temp_pos_det = self.xk_track_det(np.ravel(xk), p, t_rest)
        new_heading = math.atan2(temp_dis[1], temp_dis[0])

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

        return np.ravel(x_next), new_heading, temp_pos_det
    

    def alip_des_vel(self, leg_ind):
        v_com_des = self.bvx_max
        step_gap = 0.3

        sigma = float(self.beta*mp.coth(self.dt*self.beta/2))
        vdes_x = sigma*v_com_des*self.dt/2
        vdes_y = 0.5*(-0.5*leg_ind*step_gap)*\
            (self.beta*math.sinh(self.beta*self.dt)) / (math.cosh(self.beta*self.dt) + 1)
        return np.array([vdes_x, vdes_y])
    
    def cal_foot_with_veldes(self, x_state, vel_des_glo):
        xk = np.array([x_state]).T
        xk_nex = np.array([vel_des_glo]).T
        A_x = self.A @ xk
        p = self.inv_B_vel_shr@(xk_nex - A_x[2:4])
        return np.ravel(p)


    def cal_foot_with_posdes(self, x_state, pos_des_glo):
        xk = np.array([x_state]).T
        xk_nex = np.array([pos_des_glo]).T
        A_x = self.A @ xk
        p = self.inv_B_pos_shr@(xk_nex - A_x[0:2])
        return np.ravel(p)
    

    def solveMPCCBF(self, xk, od_ev, heading, init_guess):
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
        
        # print(goal)

        nlp = cyipopt.Problem(
           n=len(u0),
           m=len(cl),
           problem_obj=LIP_Prob(xk, self.M_A, self.M_B, self.A, self.W, 
                                self.obs_safe, goal, heading, self.N),
           lb=lb,
           ub=ub,
           cl=cl,
           cu=cu,
        )
        # nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('linear_solver', 'ma57')
        nlp.add_option('hsllib', 'libcoinhsl.so')
        nlp.add_option('max_iter', 30)
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        # nlp.add_option('sb', 'yes')
        # nlp.add_option('print_level', 0)
        # nlp.add_option('linear_solver', 'ma27')
        # nlp.add_option('hessian_approximation', 'limited-memory')
        # nlp.add_option('derivative_test', 'first-order')

        u, info = nlp.solve(u0)
        # print(u)
        return u
    
    
    def xk_track_det(self, xk, foot, t_rest):
        t_det = np.arange(0, t_rest+0.01, 0.01)
        temp_pos_det = [xk[0:2]]
        xk = np.matrix(xk).T
        p = np.matrix(foot).T
        for i in range(len(t_det)):
            if t_det[i] > t_rest:
                t_det[i] = t_rest
            A = np.matrix([[math.cosh(self.beta*t_det[i]), 0, math.sinh(self.beta*t_det[i])/self.beta, 0],
                            [0, math.cosh(self.beta*t_det[i]), 0, math.sinh(self.beta*t_det[i])/self.beta],
                            [math.sinh(self.beta*t_det[i])*self.beta, 0, math.cosh(self.beta*t_det[i]), 0],
                            [0, math.sinh(self.beta*t_det[i])*self.beta, 0, math.cosh(self.beta*t_det[i])]])
            B = np.matrix([[1-math.cosh(self.beta*t_det[i]), 0],
                            [0, 1-math.cosh(self.beta*t_det[i])],
                            [-math.sinh(self.beta*t_det[i])*self.beta, 0],
                            [0, -math.sinh(self.beta*t_det[i])*self.beta]])
            temp_xk = A @ xk + B @ p
            temp_pos_det.append(list(np.ravel(temp_xk[0:2])))

        return np.array(temp_pos_det)

    
    def solve_footdisp(self, xk, u):
        xf = self.A @ xk
        dx = u - xf
        p = self.W @ dx
        return p


    def plot_cir(self, cir):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = cir[2]*np.cos(theta)+cir[0]
        y = cir[2]*np.sin(theta)+cir[1]
        plt.plot(x, y)

    # for testing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def gen_control_test(self, state, hd_ang, leg_ind, init_guess, plot = False, trajec = []):
        # hd_ang: number, body_v: list, 
        # leg_ind: determine which leg is the swing leg.(odd for right, even for left)

        close_2_goal = False

        # Update
        self.init_state = np.matrix(state).T
        xk = self.init_state
        u = self.solveMPCCBF(xk, leg_ind, hd_ang, init_guess)           # array format
        p_list = []
        xk_list = [np.ravel(xk)]
        hd_list = []

        for i in range(self.N):
            uk = np.array([u[4*i:4*(i+1)]]).T
            pk = self.solve_footdisp(xk, uk)                     # matrix format
            pre_pos = xk[0:2]
            xk = self.M_A @ xk + self.M_B @ uk                  # matrix format
                
            temp_pos = xk[0:2]
            dx = temp_pos-pre_pos
            hd_list.append(math.atan2(dx[1], dx[0]))
            dis2goal = math.sqrt((temp_pos-self.goal).T @ (temp_pos-self.goal))
            p_list.append(pk)
            xk_list.append(np.ravel(xk))

            if dis2goal <= 0.35:
                close_2_goal = True

        if plot:
            for j in range(self.N):
                p = np.ravel(p_list[j])
                xk = np.ravel(xk_list[j])
                temp_pos_det = self.xk_track_det(xk, p, self.dt)
                plt.plot(temp_pos_det[1:, 0], temp_pos_det[1:, 1], 'r')
                plt.scatter(p[0], p[1])
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
            # plt.show()
        
        return xk_list[1:], np.ravel(p_list[0]), hd_list, close_2_goal

    # Testing end ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class LIP_Prob:
    def __init__(self, xk, M_A, M_B, A, W, obs_safe, goal, heading, step):
        # MPC param
        self.p = 20
        self.q = 10
        self.P = self.p*np.identity(2)
        self.Q = self.q*np.identity(2)
        self.N = step

        # CBF param
        self.gama = 0.4
        self.obs_modi_pram = obs_safe
        
        # Dynamics
        self.A = A
        self.W = W
        self.M_A = M_A
        self.M_B = M_B

        # Initial states
        self.xk = xk
        self.heading = heading

        # Final goal
        self.goal = goal
    
    def objective(self, u):
        cost = 0.0
        # Compute states
        u = np.matrix(u).T
        x = np.matrix(np.zeros([4, self.N + 1]))
        x[:, 0] = self.xk
        for i in range(self.N):
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[4*i:4*(i+1)]
            cost += (x[0:2, i]-self.goal).T @ self.Q @ (x[0:2, i]-self.goal)
        cost += (x[0:2, i+1]-self.goal).T @ self.Q @ (x[0:2, i+1]-self.goal) + \
                (x[0:2, 1]-self.goal).T @ self.P @ (x[0:2, 1]-self.goal)     
        cost = float(cost)
        return cost
    
    def gradient(self, u):
        Pre_1 = self.M_B
        Pre_2 = self.M_A @ self.M_B
        Pre_3 = (self.M_A @ self.M_A) @ self.M_B
        zero_array = np.zeros(4)

        # Compute states
        u = np.matrix(u).T
        x = np.matrix(np.zeros([4, self.N + 1]))
        x[:, 0] = self.xk
        for i in range(self.N):
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[4*i:4*(i+1)]

        dx11_du = np.append(np.append(np.ravel(Pre_1[0, 0:4]), zero_array), zero_array)
        dx12_du = np.append(np.append(np.ravel(Pre_1[1, 0:4]), zero_array), zero_array)
        dx21_du = np.append(np.append(np.ravel(Pre_2[0, 0:4]), np.ravel(Pre_1[0, 0:4])), zero_array)
        dx22_du = np.append(np.append(np.ravel(Pre_2[1, 0:4]), np.ravel(Pre_1[1, 0:4])), zero_array)
        dx31_du = np.append(np.append(np.ravel(Pre_3[0, 0:4]), np.ravel(Pre_2[0, 0:4])), np.ravel(Pre_1[0, 0:4]))
        dx32_du = np.append(np.append(np.ravel(Pre_3[1, 0:4]), np.ravel(Pre_2[1, 0:4])), np.ravel(Pre_1[1, 0:4]))

        jac = np.ravel(2*(self.q+self.p)*((x[0, 1]-self.goal[0])*dx11_du+(x[1, 1]-self.goal[1])*dx12_du) \
                +2*self.q*((x[0, 2]-self.goal[0])*dx21_du+(x[1, 2]-self.goal[1])*dx22_du) \
                +2*self.q*((x[0, 3]-self.goal[0])*dx31_du+(x[1, 3]-self.goal[1])*dx32_du))
        return jac
    
    def constraints(self, u):
        h0 = self.heading
        u = np.matrix(u).T
        x = np.matrix(np.zeros([4, self.N + 1]))
        p = np.matrix(np.zeros([2, self.N]))
        ang = np.zeros(self.N)
        x[:, 0] = self.xk
        cons = []
        for i in range(self.N):
            p[:, i] = self.solve_footdisp(x[:, i], u[4*i:4*(i+1)])
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[4*i:4*(i+1)]
            pos = x[0:2, i]
            dx = -pos[0, 0]+x[0, i+1]
            dy = -pos[1, 0]+x[1, i+1]
            ang_h = math.atan2(dy, dx)
            ang[i] = ang_h
            d_ang = ang_h-h0
            if d_ang < 0 and abs(d_ang) > math.pi:
                d_ang = d_ang + 2*math.pi
            elif d_ang > 0 and abs(d_ang) > math.pi:
                d_ang = d_ang - 2*math.pi
            h0 = ang_h
            M = np.matrix([[math.cos(ang_h), math.sin(ang_h)],[-math.sin(ang_h), math.cos(ang_h)]])
            v_body = M @ x[2:4, i+1]

            dlx = pos[0, 0]-p[0, i]
            dly = pos[1, 0]-p[1, i]
            leg_len = dlx**2+dly**2
            
            hk = self.h(x[0, i], x[1, i])
            h_next = self.h(x[0, i+1], x[1, i+1])
            dh = h_next - hk
            cbf = dh + self.gama*hk

            cons = np.append(cons, np.ravel(v_body))
            cons = np.append(cons, cbf)
            cons = np.append(cons, leg_len)
            cons = np.append(cons, d_ang)
        
        return cons

    def jacobian(self, u):
        Pre_1 = self.M_B
        Pre_2 = self.M_A @ self.M_B
        Pre_3 = self.M_A @ self.M_A @ self.M_B
        zero_array = np.zeros(4)

        dx01_du = np.append(np.append(zero_array, zero_array), zero_array)
        dx02_du = np.append(np.append(zero_array, zero_array), zero_array)
        dx03_du = np.append(np.append(zero_array, zero_array), zero_array)
        dx04_du = np.append(np.append(zero_array, zero_array), zero_array)
        dx11_du = np.append(np.append(np.ravel(Pre_1[0, 0:4]), zero_array), zero_array)
        dx12_du = np.append(np.append(np.ravel(Pre_1[1, 0:4]), zero_array), zero_array)
        dx13_du = np.append(np.append(np.ravel(Pre_1[2, 0:4]), zero_array), zero_array)
        dx14_du = np.append(np.append(np.ravel(Pre_1[3, 0:4]), zero_array), zero_array)
        dx21_du = np.append(np.append(np.ravel(Pre_2[0, 0:4]), np.ravel(Pre_1[0, 0:4])), zero_array)
        dx22_du = np.append(np.append(np.ravel(Pre_2[1, 0:4]), np.ravel(Pre_1[1, 0:4])), zero_array)
        dx23_du = np.append(np.append(np.ravel(Pre_2[2, 0:4]), np.ravel(Pre_1[2, 0:4])), zero_array)
        dx24_du = np.append(np.append(np.ravel(Pre_2[3, 0:4]), np.ravel(Pre_1[3, 0:4])), zero_array)
        dx31_du = np.append(np.append(np.ravel(Pre_3[0, 0:4]), np.ravel(Pre_2[0, 0:4])), np.ravel(Pre_1[0, 0:4]))
        dx32_du = np.append(np.append(np.ravel(Pre_3[1, 0:4]), np.ravel(Pre_2[1, 0:4])), np.ravel(Pre_1[1, 0:4]))
        dx33_du = np.append(np.append(np.ravel(Pre_3[2, 0:4]), np.ravel(Pre_2[2, 0:4])), np.ravel(Pre_1[2, 0:4]))
        dx34_du = np.append(np.append(np.ravel(Pre_3[3, 0:4]), np.ravel(Pre_2[3, 0:4])), np.ravel(Pre_1[3, 0:4]))
        dx_du = [dx01_du, dx02_du, dx03_du, dx04_du, dx11_du, dx12_du, dx13_du, dx14_du, \
                 dx21_du, dx22_du, dx23_du, dx24_du, dx31_du, dx32_du, dx33_du, dx34_du]
        
        u = np.matrix(u).T
        x = np.matrix(np.zeros([4, self.N + 1]))
        p = np.matrix(np.zeros([2, self.N]))
        ang = np.zeros(self.N)
        x[:, 0] = self.xk
        d_the = []
        for i in range(self.N):
            p[:, i] = self.solve_footdisp(x[:, i], u[4*i:4*(i+1)])
            x[:, i+1] = self.M_A @ x[:, i] + self.M_B @ u[4*i:4*(i+1)]
            pos = x[0:2, i]
            dx = -pos[0, 0]+x[0, i+1]
            dy = -pos[1, 0]+x[1, i+1]
            ang_h = math.atan2(dy, dx)
            ang[i] = ang_h
            d_the.append(np.ravel((dx*(dx_du[4*(i+1)+1]-dx_du[4*i+1])- \
                                   dy*(dx_du[4*(i+1)]-dx_du[4*i]))/(dx**2+dy**2)))

        # Velo-Jac
        dv11_du = -math.sin(ang[0])*d_the[0]*x[2, 1] + math.cos(ang[0])*dx13_du + \
               math.cos(ang[0])*d_the[0]*x[3, 1] + math.sin(ang[0])*dx14_du

        dv12_du = -math.cos(ang[0])*d_the[0]*x[2, 1] - math.sin(ang[0])*dx13_du + \
                -math.sin(ang[0])*d_the[0]*x[3, 1] + math.cos(ang[0])*dx14_du
        
        dv21_du = -math.sin(ang[1])*d_the[1]*x[2, 2] + math.cos(ang[1])*dx23_du + \
               math.cos(ang[1])*d_the[1]*x[3, 2] + math.sin(ang[1])*dx24_du

        dv22_du = -math.cos(ang[1])*d_the[1]*x[2, 2] - math.sin(ang[1])*dx23_du + \
                -math.sin(ang[1])*d_the[1]*x[3, 2] + math.cos(ang[1])*dx24_du
        
        dv31_du = -math.sin(ang[2])*d_the[2]*x[2, 3] + math.cos(ang[2])*dx33_du + \
               math.cos(ang[2])*d_the[2]*x[3, 3] + math.sin(ang[2])*dx34_du

        dv32_du = -math.cos(ang[2])*d_the[2]*x[2, 3] - math.sin(ang[2])*dx33_du + \
                -math.sin(ang[2])*d_the[2]*x[3, 3] + math.cos(ang[2])*dx34_du
        
        # CBF-Jac
        dh0_du = np.zeros(4*self.N)
        r = len(self.obs_modi_pram)
        dCBF1 = np.zeros([r, 12])
        dCBF2 = np.zeros([r, 12])
        dCBF3 = np.zeros([r, 12])
        for i in range(r):
            dh1d1, dh1d2 = self.dh(self.obs_modi_pram[i], x[0, 1], x[1, 1])
            dh1_du = dh1d1*dx11_du + dh1d2*dx12_du
            dh2d1, dh2d2 = self.dh(self.obs_modi_pram[i], x[0, 2], x[1, 2])
            dh2_du = dh2d1*dx21_du + dh2d2*dx22_du
            dh3d1, dh3d2 = self.dh(self.obs_modi_pram[i], x[0, 3], x[1, 3])
            dh3_du = dh3d1*dx31_du + dh3d2*dx32_du
            dCBF1[i, :] = dh1_du + (self.gama - 1)*dh0_du
            dCBF2[i, :] = dh2_du + (self.gama - 1)*dh1_du
            dCBF3[i, :] = dh3_du + (self.gama - 1)*dh2_du

        # Leg length-Jac
        pla_1 = self.W
        pla_2 = -self.W @ self.A @ self.M_B
        pla_3 = -self.W @ self.A @ self.M_A @ self.M_B
        dP11du = np.append(np.append(np.ravel(pla_1[0, 0:4]), zero_array), zero_array)
        dP12du = np.append(np.append(np.ravel(pla_1[1, 0:4]), zero_array), zero_array)
        dP21du = np.append(np.append(np.ravel(pla_2[0, 0:4]), np.ravel(pla_1[0, 0:4])), zero_array)
        dP22du = np.append(np.append(np.ravel(pla_2[1, 0:4]), np.ravel(pla_1[1, 0:4])), zero_array)
        dP31du = np.append(np.append(np.ravel(pla_3[0, 0:4]), np.ravel(pla_2[0, 0:4])), np.ravel(pla_1[0, 0:4]))
        dP32du = np.append(np.append(np.ravel(pla_3[1, 0:4]), np.ravel(pla_2[1, 0:4])), np.ravel(pla_1[1, 0:4]))
        df1 = 2*(x[0, 0]-p[0, 0])*(dx01_du-dP11du)+2*(x[1, 0]-p[1, 0])*(dx02_du-dP12du)
        df2 = 2*(x[0, 1]-p[0, 1])*(dx11_du-dP21du)+2*(x[1, 1]-p[1, 1])*(dx12_du-dP22du)
        df3 = 2*(x[0, 2]-p[0, 2])*(dx21_du-dP31du)+2*(x[1, 2]-p[1, 2])*(dx22_du-dP32du)

        # Heading-Jac
        theta_con1 = d_the[0]-dx01_du
        theta_con2 = d_the[1]-d_the[0]
        theta_con3 = d_the[2]-d_the[1]
        
        jac = np.append(dv11_du, dv12_du)
        jac = np.append(jac, dCBF1)
        jac = np.append(jac, [df1, theta_con1, dv21_du, dv22_du])
        jac = np.append(jac, dCBF2)
        jac = np.append(jac, [df2, theta_con2, dv31_du, dv32_du])
        jac = np.append(jac, dCBF3)
        jac = np.append(jac, [df3, theta_con3])
        jac = jac.reshape(self.N*(4+len(self.obs_modi_pram)), self.N*4)
        
        return jac


    def h(self, x1, x2):                        # Cal descition cost in matrix format
        h_v = np.array([])
        for each in self.obs_modi_pram:
            temp = (x1-each[0])**2 + (x2-each[1])**2 - each[2]**2
            h_v = np.append(h_v, float(temp))
        return h_v
    
    def solve_footdisp(self, xk, u):
        xf = self.A @ xk
        dx = u - xf
        p = self.W @ dx
        return p
    
    def dh(self, cir, x1, x2):
        dx1 = 2*(x1-cir[0])
        dx2 = 2*(x2-cir[1])
        return dx1, dx2



if __name__ == "__main__":  
    start = [10, 10]
    goal = [[0, 0]]
    heading = math.pi
    body_v = [0.6, -0.4]
    margin = [-0.5, 10.5]
    safe_dis = 0.32

    obs_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [4, 6, 0.8], [7, 7, 1]])
    obs_safe = obs_list
    obs_safe = obs_safe + [0, 0, safe_dis]
    
    mpc_cbf1 = MPCCBF(goal, obs_list, obs_safe, margin)
    foot, body_foot, heading = mpc_cbf1.generate_control(start, heading, body_v)