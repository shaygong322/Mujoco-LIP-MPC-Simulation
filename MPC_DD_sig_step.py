import math
import copy
import cyipopt
import numpy as np
import matplotlib.pyplot as plt

"""12.24 update: Use three control - foot placement(x, y), turning heading angle
                 Add target heading angle turning value to cost function 
                 Given large r value for heading angle in cost func     (seems like works)
                 """

class MPCCBF:
    def __init__(self, goals, cir_param, cir_cbf, elp_param, elp_cbf, margin, step = 3):
        # Robot params
        self.goal = np.matrix(goals).T
        Hei = 1.0
        g = 9.81
        self.beta = math.sqrt(g/Hei)
        self.dt = 0.4

        # Obs env
        self.margin = margin
        self.cir_list = cir_param
        self.elp_list = elp_param

        # MPC
        self.N = step

        # CBF 
        self.cir_safe = cir_cbf
        self.elp_safe = elp_cbf

        # Constraints param
        self.leg = 0.09                 ## leg length square
        self.x_max = 5
        self.v_max = 0.8
        self.v_min = 0.4
        self.ang_max = math.pi/16

        # Time
        self.tot_time = 80

        # dynamics diff_drive
        self.A = np.matrix([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        
        # dynamics lip
        self.A_L = np.matrix([[math.cosh(self.beta*self.dt), 0, math.sinh(self.beta*self.dt)/self.beta, 0, 0],
                            [0, math.cosh(self.beta*self.dt), 0, math.sinh(self.beta*self.dt)/self.beta, 0],
                            [math.sinh(self.beta*self.dt)*self.beta, 0, math.cosh(self.beta*self.dt), 0, 0],
                            [0, math.sinh(self.beta*self.dt)*self.beta, 0, math.cosh(self.beta*self.dt), 0], 
                            [0, 0, 0, 0, 1]])
        self.B_L = np.matrix([[1-math.cosh(self.beta*self.dt), 0, 0],
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
        self.M_A = (self.A_L - self.B_L @ self.W @ self.A_L)
        self.M_B = self.B_L @ self.W
        self.B_pos_shr = self.B_L[0:2, 0:2]
        self.inv_B_pos_shr = np.linalg.inv(self.B_pos_shr)
    

    def gen_dd_control(self, state, init_guess, last_u, plot = False, trajec = []):
        # Update
        close2goal = False
        self.init_state = np.matrix(state).T
        xk = self.init_state
        # self.select_obs(xk)
        print(last_u)
        u, fesi = self.solveMPCCBF(xk, init_guess, last_u)            # array format
        x_0 = list(np.ravel(xk))
        heading = []
        control = []
        states = [x_0]

        for i in range(self.N):
            uk = np.array([u[2*i:2*(i+1)]]).T
            B = np.matrix([[self.dt*math.cos(xk[2, 0]), 0], 
                           [self.dt*math.sin(xk[2, 0]), 0],
                           [0, 1]])
            xk = self.A @ xk + B @ uk

            temp_pos = xk[0:2]
            if i == 0:
                dis2goal = math.sqrt((temp_pos-self.goal).T @ (temp_pos-self.goal))        
            states.append(list(np.ravel(xk)))
            heading.append(float(xk[2]))
            control.append(uk)

        if dis2goal <= 0.35:
            close2goal = True

        if plot:
            # Generate plots
            state = np.array(states)
            plt.plot(state[:, 0], state[:, 1], '-or')
            # plt.plot(self.init_state[0], self.init_state[1], '^r')
            plt.plot(trajec[1:, 0], trajec[1:, 1], 'b')
            # plt.xlim(self.margin)
            # plt.ylim(self.margin)
            plt.grid(True)
            for each in self.cir_list:
                self.plot_cir(each)
            for each in self.elp_list:
                self.plot_elp(each)
            for each in self.cir_safe:
                self.plot_cir(each)
            for each in self.elp_safe:
                self.plot_elp(each)
            plt.axis('equal')
            plt.show()
        
        return states, heading, control, close2goal, fesi


    def solveMPCCBF(self, xk, init_guess, last_u):
        xk_array = np.ravel(xk)
        u0 = init_guess
        
        lb = []
        ub = []
        cl = []
        cu = []
        for i in range(self.N):
            lb_i = [self.v_min, -self.ang_max]
            ub_i = [self.v_max, self.ang_max]
            cl_i = np.zeros(len(self.cir_safe)+len(self.elp_safe))
            cl_i = np.append(cl_i, [self.v_min])
            cu_i = np.inf*np.ones(len(self.cir_safe)+len(self.elp_safe))
            cu_i = np.append(cu_i, [self.v_max])
            lb = np.append(lb, lb_i)
            ub = np.append(ub, ub_i)
            cl = np.append(cl, cl_i)
            cu = np.append(cu, cu_i)

        # modify the goal when in special cases
        goal = self.goal
        # goal_array = np.ravel(self.goal)
        # for i in range(len(self.cir_safe)):
        #     cir = self.cir_safe[i]
        #     temp_cen_dis = (xk_array[0]-cir[0])**2+(xk_array[1]-cir[1])**2
        #     temp_goal_dis = (xk_array[0]-goal_array[0])**2+(xk_array[1]-goal_array[1])**2

        #     if temp_cen_dis < temp_goal_dis and temp_cen_dis < 9*cir[2]**2:
        #         theta = math.atan2(goal_array[1]-xk_array[1], goal_array[0]-xk_array[0])
        #         alpha = math.atan2(cir[1]-xk_array[1], cir[0]-xk_array[0])
        #         d_the1 = theta-alpha
        #         if d_the1 < 0 and abs(d_the1) > math.pi:
        #             d_the1 = d_the1 + 2*math.pi
        #         elif d_the1 > 0 and abs(d_the1) > math.pi:
        #             d_the1 = d_the1 -2*math.pi

        #         if abs(d_the1) < (math.pi/12):
        #             if d_the1 < 0:
        #                 new_ang = theta-math.pi/12
        #             else:
        #                 new_ang = theta+math.pi/12
        #             x = math.sqrt(temp_goal_dis)*math.cos(new_ang)
        #             y = math.sqrt(temp_goal_dis)*math.sin(new_ang)
        #             goal = np.matrix([xk_array[0]+x, xk_array[1]+y]).T
        #             break


        nlp = cyipopt.Problem(
           n=len(u0),
           m=len(cl),
           problem_obj=LIP_Prob(xk, self.A, self.dt, self.cir_safe, 
                                self.elp_safe, goal, self.N, last_u),
           lb=lb,
           ub=ub,
           cl=cl,
           cu=cu,
        )
        nlp.add_option('linear_solver', 'ma57')
        nlp.add_option('hsllib', 'libcoinhsl.so')
        nlp.add_option('max_iter', 40)
        nlp.add_option('sb', 'yes')
        # nlp.add_option('print_timing_statistics', 'yes')
        # nlp.add_option('print_level', 0)
        # nlp.add_option('linear_solver', 'ma27')
        # nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('derivative_test', 'first-order')

        u, info = nlp.solve(u0)
        fesi = info["status"]
        return u, fesi
    

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


    def select_obs(self, xk):
        self.sel_cir = []
        self.sel_elp = []
        detec_range = 4**2
        xk = list(np.ravel(xk))
        for each in self.cir_safe:
            dis = float((xk[0]-each[0])**2+(xk[1]-each[1])**2-each[2]**2)
            if dis <= detec_range:
                self.sel_cir.append(each)
        for each in self.elp_safe:
            r = max(each[2], each[3])
            dis = float((xk[0]-each[0])**2+(xk[1]-each[1])**2-r**2)
            if dis <= detec_range:
                self.sel_elp.append(each)


    def cal_foot_with_posdes(self, x_state, pos_des_glo):
        xk = np.array([x_state]).T
        xk_nex = np.array([pos_des_glo]).T
        A_x = self.A_L @ xk
        p = self.inv_B_pos_shr@(xk_nex - A_x[0:2])
        return np.ravel(p)


    def plot_cir(self, cir):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = cir[2]*np.cos(theta)+cir[0]
        y = cir[2]*np.sin(theta)+cir[1]
        plt.plot(x, y, color='#696969')

    # newly
    def elp_func(self, elp, x, y):
        a = (elp[3]*math.cos(elp[4]))**2+(elp[2]*math.sin(elp[4]))**2
        b = 2*math.cos(elp[4])*math.sin(elp[4])*(elp[3]**2-elp[2]**2)
        c = (elp[3]*math.sin(elp[4]))**2+(elp[2]*math.cos(elp[4]))**2
        v1 = a*(x-elp[0])**2
        v2 = b*(x-elp[0])*(y-elp[1])
        v3 = c*(y-elp[1])**2
        return v1+v2+v3-(elp[3]*elp[2])**2
    
    # newly
    def plot_elp(self, elp):            # [x_c, y_c, A, B, C, r]
        x1 = np.arange(self.margin[0], self.margin[1], 0.1)
        x2 = np.arange(self.margin[0], self.margin[1], 0.1)
        X1, X2 = np.meshgrid(x1, x2)
        z = self.elp_func(elp, X1, X2)
        plt.contour(X1, X2, z, [0], colors='#696969')

    
    def tube_func(self, heading_list, init_tube_value):
        tube_upper = 0.2
        tube_lower = -0.2
        new_heading = np.zeros_like(heading_list)
        tube_value = init_tube_value
        for i in range(len(heading_list)):
            d_head_value = heading_list[i]-tube_value
            if d_head_value > 0:
                if tube_upper > d_head_value:
                    tube_value += 0.3*d_head_value
                else:
                    tube_value += 0.7*d_head_value
            elif d_head_value < 0:
                if tube_lower < d_head_value:
                    tube_value += 0.3*d_head_value
                else:
                    tube_value += 0.7*d_head_value
            new_heading[i] = tube_value
        return new_heading



class LIP_Prob:
    def __init__(self, xk, A, dt, cir_safe, elp_safe, goal, step, last_u):
        # MPC param
        self.p = 0
        self.q = 1
        self.r = 50
        self.P = self.p*np.identity(2)
        self.Q = self.q*np.identity(2)
        self.N = step
        self.t = 2

        # CBF param
        self.gama = 0.2
        self.cir_modi_pram = cir_safe
        self.elp_modi_pram = elp_safe
        self.power = 4

        # Energy
        self.s = 0.024*180/math.pi
        
        # Dynamics
        self.A = A
        self.dt = dt

        # Initial states
        self.xk = xk
        self.last_u = last_u

        # Final goal
        self.goal = goal
    
    def objective(self, u):
        cost = 0.0
        # Compute states
        u_p = np.matrix(self.last_u).T
        u = np.matrix(u).T
        x = np.matrix(np.zeros([3, self.N + 1]))
        x[:, 0] = self.xk
        for i in range(self.N):
            u_c = u[2*i:2*(i+1)]
            B = np.matrix([[self.dt*math.cos(x[2, i]), 0], 
                           [self.dt*math.sin(x[2, i]), 0],
                           [0, 1]])
            x[:, i+1] = self.A @ x[:, i] + B @ u_c
            tar_ang, d_plc = self.cal_tar_ang(x[:, i+1])
            cost += (x[0:2, i+1]-self.goal).T @ self.Q @ (x[0:2, i+1]-self.goal) + \
                    (self.r*(x[2, i+1]-tar_ang)**2) + self.t*(u_c-u_p).T @ (u_c-u_p)
            u_p = u_c
        cost += (x[0:2, 1]-self.goal).T @ self.P @ (x[0:2, 1]-self.goal)     
        return float(cost)
    
    def gradient(self, u):
        # Compute states
        u_p = np.matrix(self.last_u).T
        u = np.matrix(u).T
        x = np.matrix(np.zeros([3, self.N + 1]))
        x[:, 0] = self.xk
        d_tan_du = []
        for i in range(self.N):
            B = np.matrix([[self.dt*math.cos(x[2, i]), 0], 
                           [self.dt*math.sin(x[2, i]), 0],
                           [0, 1]])
            x[:, i+1] = self.A @ x[:, i] + B @ u[2*i:2*(i+1)]
        
        dx_du = self.cal_dx_du(x, u)
        du_du = self.cal_du_du(u)
        for i in range(self.N):
            d_tan_du.append(self.cal_dtar_ang_du(x[:, i+1], i, dx_du))
        
        jac = np.ravel(2*(self.q+self.p)*((x[0,1]-self.goal[0])*dx_du[3,:]+\
                                          (x[1,1]-self.goal[1])*dx_du[4,:])\
                +2*self.q*((x[0,2]-self.goal[0])*dx_du[6,:]+(x[1,2]-self.goal[1])*dx_du[7,:])\
                +2*self.q*((x[0,3]-self.goal[0])*dx_du[9,:]+(x[1,3]-self.goal[1])*dx_du[10,:]))\
                +2*self.t*(((u[0]-u_p[0])*du_du[0, :]+(u[1]-u_p[1])*du_du[1, :])\
                          +((u[2]-u[0])*(du_du[2, :]-du_du[0, :])+(u[3]-u[1])*(du_du[3, :]-du_du[1, :]))\
                          +((u[4]-u[2])*(du_du[4, :]-du_du[2, :])+(u[5]-u[3])*(du_du[5, :]-du_du[3, :])))\
                +2*self.r*(d_tan_du[0]+d_tan_du[1]+d_tan_du[2])
        return jac
    
    def constraints(self, u):
        u = np.matrix(u).T
        x = np.matrix(np.zeros([3, self.N + 1]))
        x[:, 0] = self.xk
        cons = []
        for i in range(self.N):
            B = np.matrix([[self.dt*math.cos(x[2, i]), 0], 
                           [self.dt*math.sin(x[2, i]), 0],
                           [0, 1]])
            x[:, i+1] = self.A @ x[:, i] + B @ u[2*i:2*(i+1)]
            
            hk_cir = self.h_cir(x[0, i], x[1, i])
            h_next_cir = self.h_cir(x[0, i+1], x[1, i+1])
            cbf_cir = h_next_cir + (self.gama-1)*hk_cir
            hk_elp = self.h_elp(x[0, i], x[1, i])
            h_next_elp = self.h_elp(x[0, i+1], x[1, i+1])
            cbf_elp = h_next_elp + (self.gama-1)*hk_elp
            f_en = float(self.s*abs(u[2*(i+1)-1])+u[2*i])

            cons = np.append(cons, cbf_cir)
            cons = np.append(cons, cbf_elp)
            cons = np.append(cons, f_en)
        return cons

    def jacobian(self, u):
        u = np.matrix(u).T
        x = np.matrix(np.zeros([3, self.N + 1]))
        x[:, 0] = self.xk
        for i in range(self.N):
            B = np.matrix([[self.dt*math.cos(x[2, i]), 0], 
                           [self.dt*math.sin(x[2, i]), 0],
                           [0, 1]])
            x[:, i+1] = self.A @ x[:, i] + B @ u[2*i:2*(i+1)]
        
        dx_du = self.cal_dx_du(x, u)
        
        # CBF-Jac
        dh0_du = np.zeros(2*self.N)
        r = len(self.cir_modi_pram)
        dCBF1_cir = np.zeros([r, 6])
        dCBF2_cir = np.zeros([r, 6])
        dCBF3_cir = np.zeros([r, 6])
        for i in range(r):
            dh1d1, dh1d2 = self.dh_cir(self.cir_modi_pram[i], x[0, 1], x[1, 1])
            dh1_du = np.ravel(dh1d1*dx_du[3,:] + dh1d2*dx_du[4,:])
            dh2d1, dh2d2 = self.dh_cir(self.cir_modi_pram[i], x[0, 2], x[1, 2])
            dh2_du = np.ravel(dh2d1*dx_du[6,:] + dh2d2*dx_du[7,:])
            dh3d1, dh3d2 = self.dh_cir(self.cir_modi_pram[i], x[0, 3], x[1, 3])
            dh3_du = np.ravel(dh3d1*dx_du[9,:] + dh3d2*dx_du[10,:])
            dCBF1_cir[i, :] = dh1_du + (self.gama - 1)*dh0_du
            dCBF2_cir[i, :] = dh2_du + (self.gama - 1)*dh1_du
            dCBF3_cir[i, :] = dh3_du + (self.gama - 1)*dh2_du
        
        r = len(self.elp_modi_pram)
        dCBF1_elp = np.zeros([r, 6])
        dCBF2_elp = np.zeros([r, 6])
        dCBF3_elp = np.zeros([r, 6])
        for i in range(r):
            dh1d1, dh1d2 = self.dh_elp(self.elp_modi_pram[i], x[0, 1], x[1, 1])
            dh1_du = np.ravel(dh1d1*dx_du[3,:] + dh1d2*dx_du[4,:])
            dh2d1, dh2d2 = self.dh_elp(self.elp_modi_pram[i], x[0, 2], x[1, 2])
            dh2_du = np.ravel(dh2d1*dx_du[6,:] + dh2d2*dx_du[7,:])
            dh3d1, dh3d2 = self.dh_elp(self.elp_modi_pram[i], x[0, 3], x[1, 3])
            dh3_du = np.ravel(dh3d1*dx_du[9,:] + dh3d2*dx_du[10,:])
            dCBF1_elp[i, :] = dh1_du + (self.gama - 1)*dh0_du
            dCBF2_elp[i, :] = dh2_du + (self.gama - 1)*dh1_du
            dCBF3_elp[i, :] = dh3_du + (self.gama - 1)*dh2_du
        
        # velo-turn
        df_en1 = np.array([[1, self.s*self.den_du(float(u[1])), 0, 0, 0, 0]])
        df_en2 = np.array([[0, 0, 1, self.s*self.den_du(float(u[3])), 0, 0]])
        df_en3 = np.array([[0, 0, 0, 0, 1, self.s*self.den_du(float(u[5]))]])

        jac = np.concatenate([dCBF1_cir, dCBF1_elp, df_en1, dCBF2_cir,
                              dCBF2_elp, df_en2, dCBF3_cir, dCBF3_elp, df_en3])
        # jac = np.concatenate([dCBF1_cir, dCBF1_elp, dCBF2_cir,
        #                       dCBF2_elp,  dCBF3_cir, dCBF3_elp])
        
        return jac


    def h_cir(self, x1, x2):                        # Cal descition cost in matrix format
        h_v = np.array([])
        for each in self.cir_modi_pram:
            temp = (x1-each[0])**2 + (x2-each[1])**2 - each[2]**2
            h_v = np.append(h_v, float(temp))
        return h_v
    
    def dh_cir(self, cir, x1, x2):
        dx1 = 2*(x1-cir[0])
        dx2 = 2*(x2-cir[1])
        return dx1, dx2
    
    def h_elp(self, x1, x2):                        # Cal descition cost in matrix format
        h_v = np.array([])
        for each in self.elp_modi_pram:
            a = (each[3]*math.cos(each[4]))**2+(each[2]*math.sin(each[4]))**2
            b = 2*math.cos(each[4])*math.sin(each[4])*(each[3]**2-each[2]**2)
            c = (each[3]*math.sin(each[4]))**2+(each[2]*math.cos(each[4]))**2
            v1 = a*(x1-each[0])**2
            v2 = b*(x1-each[0])*(x2-each[1])
            v3 = c*(x2-each[1])**2
            temp = v1+v2+v3-(each[3]*each[2])**2
            h_v = np.append(h_v, float(temp))
        return h_v
    
    def dh_elp(self, elp, x1, x2):
        a = (elp[3]*math.cos(elp[4]))**2+(elp[2]*math.sin(elp[4]))**2
        b = 2*math.cos(elp[4])*math.sin(elp[4])*(elp[3]**2-elp[2]**2)
        c = (elp[3]*math.sin(elp[4]))**2+(elp[2]*math.cos(elp[4]))**2
        dx1 = 2*a*(x1-elp[0])+b*(x2-elp[1])
        dx2 = 2*c*(x2-elp[1])+b*(x1-elp[0])
        return dx1, dx2
    

    def den_du(self, hd):
        if hd == 0:
            df_dhd = 0
        else:
            df_dhd = hd/abs(hd)
        return df_dhd
    
    
    def cal_tar_ang(self, state):
        d_plc = self.goal - state[0:2]
        tar_ang = math.atan2(d_plc[1], d_plc[0])
        return tar_ang, d_plc
    
    def cal_dtar_ang_du(self, state, i, dx_du):
        tar_ang, d_plc = self.cal_tar_ang(state)
        d_tar_du = ((d_plc[0]*(-dx_du[3*(i+1)+1, :])-\
                     d_plc[1]*(-dx_du[3*(i+1), :]))/(d_plc[0]**2+d_plc[1]**2))
        d_ang = (state[2]-tar_ang)*(dx_du[3*(i+1)+2,:] - d_tar_du)
        return d_ang
    
    def cal_dx_du(self, x, u):
        zero_m = np.zeros((3, 2))
        dx1_du1 = np.array([[self.dt*math.cos(x[2,0]), 0],
                            [self.dt*math.sin(x[2,0]), 0],
                            [0, 1]])
        
        dx2_du1 = np.array([[self.dt*math.cos(x[2,0]), -u[2,0]*self.dt*math.sin(x[2,1])],
                            [self.dt*math.sin(x[2,0]), u[2,0]*self.dt*math.cos(x[2,1])],
                            [0, 1]])
        
        dx2_du2 = np.array([[self.dt*math.cos(x[2,1]), 0],
                            [self.dt*math.sin(x[2,1]), 0],
                            [0, 1]])
        
        dx3_du1 = np.array([[self.dt*math.cos(x[2,0]), -u[2,0]*self.dt*math.sin(x[2,1])-u[4,0]*self.dt*math.sin(x[2,2])],
                            [self.dt*math.sin(x[2,0]), u[2,0]*self.dt*math.cos(x[2,1])+u[4,0]*self.dt*math.cos(x[2,2])],
                            [0, 1]])
        
        dx3_du2 = np.array([[self.dt*math.cos(x[2,1]), -u[4,0]*self.dt*math.sin(x[2,2])],
                            [self.dt*math.sin(x[2,1]), u[4,0]*self.dt*math.cos(x[2,2])],
                            [0, 1]])
        
        dx3_du3 = np.array([[self.dt*math.cos(x[2,2]), 0],
                            [self.dt*math.sin(x[2,2]), 0],
                            [0, 1]])
        
        dx0_du = np.concatenate([zero_m, zero_m, zero_m], axis=1)
        dx1_du = np.concatenate([dx1_du1, zero_m, zero_m], axis=1)
        dx2_du = np.concatenate([dx2_du1, dx2_du2, zero_m], axis=1)
        dx3_du = np.concatenate([dx3_du1, dx3_du2, dx3_du3], axis=1)
        dx_du = np.concatenate([dx0_du, dx1_du, dx2_du, dx3_du], axis=0)

        return dx_du
    
    def cal_du_du(self, u):
        n = len(u)
        du_du = np.eye(n)

        return du_du
    

if __name__ == "__main__":  
    start = [0, 0]
    goal = [[10, 10]]
    heading = 0
    glo_v = [0.6, -0.3]
    body_v = [-0.6, -0.3]
    margin = [-0.5, 10.5]
    safe_dis = 0.32

    obs_cir_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [6, 4, 0.8], [6.4, 7.2, 1.0], [4.8, 0.8, 0.4],
                            [2, 6, 0.3]])
    # obs_cir_list = np.array([[1, 1, 0.5], [2, 2, 0.5], [4, 3.2, 0.25]])
    obs_cir_safe = obs_cir_list
    obs_cir_safe = obs_cir_safe + [0, 0, safe_dis]
    obs_elp_list = []
    obs_elp_safe = []
    
    mpc_cbf1 = MPCCBF(goal, obs_cir_list, obs_cir_safe, obs_elp_list, obs_elp_safe, margin)
    init_guess = np.ravel([[0.8, 0], [0.8, 0], [0.8, 0]])
    state = np.concatenate([start, [heading]])
    state, heading, control, close2goal= mpc_cbf1.gen_dd_control(state, init_guess, True)
    print(control)