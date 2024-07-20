import numpy as np
import scipy as sp
from scipy import integrate
import fromFROST as fromFROST
import forwardKinematics as forwardKinematics
import math
#from digit.tools import idqp_wrapper
import helper as helper
from scipy.optimize import NonlinearConstraint
import pdb
import time

# Based on Yuan planner and consider both planes of motion inside the class
class ALIP:
    def __init__(self,params):
        self.nJoints = 20
        self.nSteps = 0
        self.oldStance = -1
        ##########################################
        ######   set the walking parameters ######
        self.H = params.H
        self.T = params.T
        self.m = params.m


        self.t_abs = 0
        self.t_begining_current_Step = 0
        ##########################################

        self.g = 9.81
        self.W = 0.2

        self.q0 = np.zeros(30)

        self.alpha = np.zeros([2, 70])
        self.A_lateral = np.array([[0, -1 / (self.m * self.H), 0],
                               [-self.m * self.g, 0, 0],
                               [0, 0, 0]])


        self.A_sagittal = np.array([[0, 1 / (self.m * self.H), 0],
                               [self.m * self.g, 0, 0],
                               [0, 0, 0]])

        self.lambda_lip = math.sqrt(self.g/self.H)
        self.mhl = self.m * self.H * self.lambda_lip
        self.DRS_T_x = 10
        self.amp_x = 0
        self.DRS_T_y = 10
        self.amp_y = 0
        self.ux_prev = 0
        self.uy_prev = 0
        self.setFullbodyWalkingParas()
        self.hc_yuan_seq = []
        self.hd_yuan_seq = []
        self.domain_yuan_seq = []
        self.dq_seq = []
        self.yt_seq = []
        self.xt_seq = []
        self.t_abs_seq = []

        #self.idqp = idqp_wrapper.IDQP_wrapper()
    def setParam(self,params):
        ######   set the walking parameters ######
        self.H = params.H
        self.T = params.T
        self.m = params.m

        self.K_sagittal = params.K_sagittal
        self.xStar_sagittal = params.xStar_sagittal
        self.uStar_sagittal = params.uStar_sagittal
        self.x0_sagittal = params.x0_sagittal

        self.K_lateral = params.K_lateral
        self.xStar_lateral = params.xStar_lateral
        self.uStar_lateral = params.uStar_lateral
        self.x0_lateral = params.x0_lateral
    def setFullbodyWalkingParas(self):
        Alpha1_R_FD = np.array([1, 1, 1, 1, 1, 1, 1]) * self.H
        Alpha2_R_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha3_R_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha4_R_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha5_R_FD = np.array([-0.25, -0.08, -0.06, 0, 0.06, 0.08, 0.25])
        Alpha6_R_FD = 0.3* np.array([1, 1, 1, 1, 1, 1, 1])
        Alpha7_R_FD = np.array([0, 0.011, 0.077, 0.1, 0.077, 0.011, - 0.003]) / 1.3
        Alpha8_R_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha9_R_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha10_R_FD = np.array([0, 0, 0, 0, 0, 0, 0])

        Alpha1_L_FD = np.array([1, 1, 1, 1, 1, 1, 1]) * self.H
        Alpha2_L_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha3_L_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha4_L_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha5_L_FD = np.array([-0.25, -0.08, -0.06, 0, 0.06, 0.08, 0.25])
        Alpha6_L_FD = -0.3 * np.array([1, 1, 1, 1, 1, 1, 1])
        Alpha7_L_FD = np.array([0, 0.011, 0.077, 0.1, 0.077, 0.011, - 0.003]) / 1.3
        Alpha8_L_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha9_L_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha10_L_FD = np.array([0, 0, 0, 0, 0, 0, 0])
        Alpha_R_FD = np.concatenate((Alpha1_R_FD, Alpha2_R_FD, Alpha3_R_FD, Alpha4_R_FD, Alpha5_R_FD, Alpha6_R_FD, Alpha7_R_FD, Alpha8_R_FD, Alpha9_R_FD, Alpha10_R_FD), axis=None)
        Alpha_L_FD = np.concatenate((Alpha1_L_FD, Alpha2_L_FD, Alpha3_L_FD, Alpha4_L_FD, Alpha5_L_FD, Alpha6_L_FD,
                                    Alpha7_L_FD, Alpha8_L_FD, Alpha9_L_FD, Alpha10_L_FD), axis=None)

        Alpha = np.array([Alpha_R_FD, Alpha_L_FD])

        self.alpha = Alpha
    def getRefs(self):
        return np.zeros((20))

    def getStates(self):
        return self.xState
    def setDRSPara(self,DRS_T_x, amp_x,DRS_T_y, amp_y):
        self.DRS_T_x = DRS_T_x
        self.amp_x = amp_x
        self.DRS_T_y = DRS_T_y
        self.amp_y = amp_y
    def platformMotion(self, t):

        omega_x = 2 * np.pi / self.DRS_T_x
        omega_y = 2 * np.pi / self.DRS_T_y


        xDRS_lat = self.amp_y * np.cos(omega_y * t)
        vDRS_lat = -self.amp_y * omega_y * np.sin(omega_y * t)
        aDRS_lat = -self.amp_y * omega_y * omega_y * np.cos(omega_y * t)

        xDRS_sag = self.amp_x * np.cos(omega_x * t)
        vDRS_sag = -self.amp_x * omega_x * np.sin(omega_x * t)
        aDRS_sag = -self.amp_x * omega_x * omega_x * np.cos(omega_x * t)

        #matDRS = self.idqp.DRS_motion(t, self.amp_x, self.amp_y, self.DRS_T_x, self.DRS_T_y)

        xDRS = np.array([xDRS_lat, xDRS_sag])
        vDRS = np.array([vDRS_lat, vDRS_sag])
        aDRS = np.array([aDRS_lat, aDRS_sag])

        return xDRS, vDRS, aDRS
    def DRS_motion_int(self,T_low,T_high):

        DRS_int = self.idqp.DRS_motion_int(T_low, T_high, self.amp_x, self.amp_y, self.DRS_T_x, self.DRS_T_y, self.m, self.H, self.g)
        print("*************")

        sum_lateral = DRS_int[:,0]
        sum_sagittal = DRS_int[:,1]

        '''
        print(sum_lateral)
        print(sum_sagittal)

        n_int = 500
        d_tau = (T_high-T_low)/n_int

        sum_lateral = np.zeros(2)
        sum_sagittal = np.zeros(2)

        for i in range(n_int):
            tau = T_low+(i+1)*d_tau
            xDRS, vDRS, aDRS = self.platformMotion(tau)

            u_lateral = np.array([-vDRS[0],0])
            u_sagittal = np.array([-vDRS[1],0])

            A_lateral_t = self.A_lateral[0:2,0:2] * (T_high - tau)
            A_sagittal_t = self.A_sagittal[0:2,0:2] * (T_high - tau)

            sum_lateral = sum_lateral + sp.linalg.expm(A_lateral_t)@u_lateral*d_tau
            sum_sagittal = sum_sagittal + sp.linalg.expm(A_sagittal_t)@u_sagittal*d_tau
        print(sum_lateral)
        print(sum_sagittal)
        pdb.set_trace()
        print("<<<<<<<<<<<<<<<<<<<")
        '''
        return sum_lateral, sum_sagittal

    def dynamics_sagittal(self, x, t):
        xDRS, vDRS, aDRS = self.platformMotion(self.t_abs)
        dxdt = self.A_sagittal @ x #+ np.array([vDRS, 0, 1])
        #dydt = self.Ay @ x
        return dxdt
    def dynamics_lateral(self, x, t):
        #print("t: ")
        #print(t)
        xDRS, vDRS, aDRS = self.platformMotion(self.t_abs)
        dxdt = self.A_lateral @ x# + np.array([-vDRS, 0, 1])
        #dydt = self.Ay @ x
        return dxdt

    def getTimedState(self, x0, y0, t):
        xt = np.zeros(2)
        yt = np.zeros(2)

        l = self.lambda_lip

        T = self.T

        t = t if t <= self.T else self.T
        mhl = self.mhl

        Mx = np.array( [[math.cosh(l*(t)), 1/(mhl) * math.sinh(l * (t))],
        [mhl * math.sinh(l * (t)), math.cosh(l * (t)) ]] )        

        My = np.array( [[math.cosh(l*(t)), -1/(mhl) * math.sinh(l * (t))],
        [-mhl * math.sinh(l * (t)), math.cosh(l * (t)) ]] )    

        xt = Mx @ x0
        yt = My @ y0

        return xt, yt

    def AMprediction(self, xt, yt, t):
        l = self.lambda_lip
        px_t = xt[0]
        Ly_t = xt[1]
        py_t = yt[0]
        Lx_t = yt[1]
        T = self.T
        t = t if t <= self.T else self.T      

        #print("t: ", t)
        #print("t_abs", self.t_abs)
        #print("H: ", self.H)
        #print("m: ", self.m)
        #print("px_t: ", px_t)
        #print("Ly_t: ", Ly_t)
        DRS_motion_int_lateral, DRS_motion_int_sagittal = self.DRS_motion_int(self.t_abs,self.t_begining_current_Step+T)
        #print("DRS_motion_int_lateral: ", DRS_motion_int_lateral)
        #print("DRS_motion_int_sagittal: ", DRS_motion_int_sagittal)
        Ly_est = self.mhl * math.sinh(l * (T-t)) * px_t + math.cosh(l * (T-t)) * Ly_t + DRS_motion_int_sagittal[1]
        Lx_est = -self.mhl * math.sinh(l * (T-t)) * py_t + math.cosh(l * (T-t)) * Lx_t + DRS_motion_int_lateral[1]
        return Ly_est, Lx_est

    def computeSw2CoM(self, Ly_est, Lx_est, Ly_des, support):
        l = self.lambda_lip
        T = self.T
        alpha = 0.0
        DRS_motion_int_lateral, DRS_motion_int_sagittal = self.DRS_motion_int(self.t_begining_current_Step + T,
                                                                              self.t_begining_current_Step + 2*T)
        #px_sw2CoM = (Ly_des - math.cosh(l * T) * Ly_est) / (self.mhl * math.sinh(l * T))
        px_sw2CoM = (1-alpha) * (Ly_des-DRS_motion_int_sagittal[1])/(self.mhl * math.sinh(l * T)) + (alpha - math.cosh(l*T))/(self.mhl * math.sinh(l*T)) * Ly_est
        Lx_des_base = 0.5 * self.m * self.H * self.W * (l * math.sinh(l * T)) / (1 + math.cosh(l * T))
        if support == -1: #Left Support
            Lx_des = -Lx_des_base 
        elif support == 1:
            Lx_des = Lx_des_base  
        else:
            Lx_des = -Lx_des_base
        py_sw2CoM = -(1-alpha) * (Lx_des-DRS_motion_int_lateral[1])/(self.mhl * math.sinh(l * T)) - (alpha - math.cosh(l*T))/(self.mhl * math.sinh(l * T)) * Lx_est
        return px_sw2CoM, py_sw2CoM

    def computeStepping(self, p_sp2CoM, Ly_est, Lx_est, v_des, support):

        Ly_des = self.m * self.H * v_des
        px_sw2CoM, py_sw2CoM = self.computeSw2CoM(Ly_est, Lx_est, Ly_des, support)
        px_sp2CoM = p_sp2CoM[0]
        py_sp2CoM = p_sp2CoM[1]

        px_sp2sw = px_sp2CoM - px_sw2CoM
        py_sp2sw = py_sp2CoM - py_sw2CoM
        py_sp2sw = self.regulate_lateral_step(support,py_sp2sw)

        return px_sp2sw, py_sp2sw

    def getFootPlacement(self, state):
        com = state['com']
        pos_cAnkle = state['pos_cAnkle']
        c_AngMom = state['cAngMom']
        speed = state['speed']
        support = state['support']
        time = state['time']

        xt = np.zeros(2)
        yt = np.zeros(2)

        x = state['states']

        yt, xt = self.FOM2LIP(x, time, support)

        #print("approx vx: ", xt[1]/(self.m * self.H))
        

        p_sp2com = np.array([xt[0], yt[0]])
        
        # xt[0] = p_sp2com[0]
        # xt[1] = c_AngMom[1]

        # yt[0] = p_sp2com[1]
        # yt[1] = c_AngMom[0]

        if support == 1 and self.oldStance != sum:
            self.nSteps = self.nSteps + 1
            self.oldStance = support

        Ly_est, Lx_est = self.AMprediction(xt, yt, time)
        self.yt_seq.append(yt)
        self.xt_seq.append(xt)
        self.t_abs_seq.append(self.t_abs)
        speed = 0.1
        ux, uy = self.computeStepping(p_sp2com, Ly_est, Lx_est, speed, support)

        if self.nSteps <= 2:
            uy = uy * 1.0
            ux = ux * 1.0
        else:
            uy = uy * 1.0
            ux = ux * 1.0

        # if support == -1: #Left
        #     uy = -0.18 if uy > -0.18 else uy
        #     # uy = -0.2
        #     ux = 0.5 if ux > 0.5  else ux
        # elif support == 1: # Right
        #     uy = 0.18 if uy < 0.18 else uy
        #     # uy = 0.2
        #     ux = 0.5 if ux > 0.5  else ux

        if time <= self.T:
            self.ux_prev = ux
            self.uy_prev = uy

        return self.ux_prev, self.uy_prev, Ly_est, Lx_est

    def FOM2LIP(self, x, t, foot_index):
        q = x[0:30]
        dq = x[30:60]
        if foot_index == -1:  # left support
            supportFoot = forwardKinematics.digitLeftFootPose(q)
        elif foot_index == 1:  # right support
            supportFoot = forwardKinematics.digitRightFootPose(q)
            #pdb.set_trace()
        elif foot_index == 0:  # double support, let's pretend it's right support
            supportFoot = forwardKinematics.digitRightFootPose(q)
        else:
            supportFoot = forwardKinematics.digitRightFootPose(q)

        supportFootPos = supportFoot[0:3]
        #print(supportFootPos)
        COM = fromFROST.p_CoM(q)
        x_sc = COM[0] - supportFoot[0]
        y_sc = COM[1] - supportFoot[1]
        AM = fromFROST.AMworld_about_pA(q, dq, supportFootPos)
        AM_lateral = AM[0]
        x_LIP_lateral = np.array([y_sc, AM_lateral, t])
        AM_sagittal = AM[1]
        x_LIP_sagittal = np.array([x_sc, AM_sagittal, t])
        return x_LIP_lateral, x_LIP_sagittal
    def regulate_lateral_step(self,foot_index,u_lateral):
        is_regulated = False
        if foot_index == 1: #% right support
            if u_lateral < 0.1:
                u_lateral_regulated = 0.1
                is_regulated = True
            elif u_lateral > 0.45:
                u_lateral_regulated = 0.45
                is_regulated = True
            else:
                u_lateral_regulated = u_lateral
        elif foot_index == -1: # left support
            if u_lateral > -0.1:
                u_lateral_regulated = -0.1
                is_regulated = True
            elif u_lateral < -0.45:
                u_lateral_regulated = -0.45
                is_regulated = True
            else:
                u_lateral_regulated = u_lateral
            if is_regulated == True:
                print("step length is regulated")
        else:
            u_lateral_regulated = u_lateral
        return u_lateral_regulated



    def plan(self, x, t0, foot_index):
        ALIPx0_lateral,ALIPx0_sagittal = self.FOM2LIP(x, t0, foot_index)
        hc_yuan = helper.hcOutput(x[0:30],foot_index)
        s = t0 / self.T
        #print(x[0:30])
        #print(x[30:60])

        #print(s)
        #print(ALIPx0_lateral)
        #print(ALIPx0_sagittal)
        #pdb.set_trace()
        if foot_index == -1:  # letf support
            alpha = self.alpha[1, :]
            hd_yuan,xixi,haha = helper.desiredOutput(alpha, s)
            self.hc_yuan_seq.append(hc_yuan)
            self.domain_yuan_seq.append(foot_index)
            #print(hd_yuan)
            self.hd_yuan_seq.append(hd_yuan)
            self.dq_seq.append(x[30:60])
        elif foot_index == 1:  # right support
            alpha = self.alpha[0, :]
            hd_yuan,xixi,haha = helper.desiredOutput(alpha, s)
            self.hc_yuan_seq.append(hc_yuan)
            self.hd_yuan_seq.append(hd_yuan)
            self.domain_yuan_seq.append(foot_index)
            self.dq_seq.append(x[30:60])
            #print(hd_yuan)
        # ph, dph, ddph = helper.Bezier(self.alpha, s)
        # hd = np.concatenate((ph, np.zeros(8)), axis=None)


        if t0 < self.T + 0.2 and foot_index is not None:
            int_point = 1000

            t_vec = np.linspace(t0, self.T, int_point)
            x_sol_lateral_vec = sp.integrate.odeint(self.dynamics_lateral, ALIPx0_lateral, t_vec, args=(), Dfun=None, col_deriv=0,
                                            full_output=0,
                                            ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0, hmax=0.0,
                                            hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0,
                                            tfirst=False)
            x_minus_lateral = x_sol_lateral_vec[int_point - 1, :]

            x_sol_sagittal_vec = sp.integrate.odeint(self.dynamics_sagittal, ALIPx0_sagittal, t_vec, args=(), Dfun=None,
                                                    col_deriv=0,
                                                    full_output=0,
                                                    ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
                                                    hmax=0.0,
                                                    hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5,
                                                    printmessg=0,
                                                    tfirst=False)
            x_minus_sagittal = x_sol_sagittal_vec[int_point - 1, :]

            if foot_index == -1: # left foot
                u_lateral = self.uStar_lateral[0] + self.K_lateral[0, :] @ (x_minus_lateral[0:2] - self.xStar_lateral[0, :])

            elif foot_index == 1: # right foot
                u_lateral = self.uStar_lateral[1] + self.K_lateral[1, :] @ (
                            x_minus_lateral[0:2] - self.xStar_lateral[1, :])
            elif foot_index == 0:  #  double support, let's pretend it's right support
                u_lateral = self.uStar_lateral[1] + self.K_lateral[1, :] @ (
                        x_minus_lateral[0:2] - self.xStar_lateral[1, :])
            u_lateral = self.regulate_lateral_step(foot_index, u_lateral)
            u_sagittal = self.uStar_sagittal + self.K_sagittal @ (x_minus_sagittal[0:2] - self.xStar_sagittal)
        else:
            u_lateral = 0
            u_sagittal = 0
        #pdb.set_trace()
        u_lateral = 1.0*u_lateral
        self.updateBezier(u_lateral, u_sagittal, foot_index)
        return u_lateral, u_sagittal


    def updateBezier(self, u_lateral, u_sagittal,foot_index):
        if foot_index == -1: # left suppport
            alpha_row_index = 1
        elif foot_index == 1:  # right support
            alpha_row_index = 0
        else:
            alpha_row_index = 0
        self.alpha[alpha_row_index, 34] = u_sagittal
        self.alpha[alpha_row_index, 33] = u_sagittal
        self.alpha[alpha_row_index, 41] = u_lateral
        self.alpha[alpha_row_index, 40] = u_lateral
        self.alpha[alpha_row_index, 29: 34] = np.linspace(self.alpha[alpha_row_index, 29], self.alpha[alpha_row_index, 33], 5)
        self.alpha[alpha_row_index, 36: 41] = np.linspace(self.alpha[alpha_row_index, 36], self.alpha[alpha_row_index, 40], 5)

    def updateBezierForNewWalkingStep(self, q, dq, foot_index):
        # update here
        if foot_index == -1: # left suppport
            alpha_row_index = 1
        elif foot_index == 1:  # right support
            alpha_row_index = 0
        else:
            alpha_row_index = 0
        hc = helper.hcOutput(q, foot_index)
        self.alpha[alpha_row_index, 7] = hc[1]  # update torso roll for new step
        self.alpha[alpha_row_index, 14] = hc[2] # update torso pitch for new step
        self.alpha[alpha_row_index, 28] = hc[4] # update swing foot x
        self.alpha[alpha_row_index, 29] = hc[4] # update swing foot x
        self.alpha[alpha_row_index, 35] = hc[5] # update swing foot y
        self.alpha[alpha_row_index, 36] = hc[5] # update swing foot y

    '''
    def joint2motor(self, joint):
        motor_position_ref = np.zeros(20)
        motor_position_ref[0] = joint[6]
        motor_position_ref[0] = joint[7]
        motor_position_ref[0] = joint[8]
        motor_position_ref[0] = joint[9]
        motor_position_ref[0] = joint[12]
        motor_position_ref[0] = joint[13]
        motor_position_ref[0] = joint[14]
        motor_position_ref[0] = joint[15]
        motor_position_ref[0] = joint[16]
        motor_position_ref[0] = joint[17]
        motor_position_ref[0] = joint[18]
        motor_position_ref[0] = joint[19]
        motor_position_ref[0] = joint[20]
        motor_position_ref[0] = joint[21]
        motor_position_ref[0] = joint[24]
        motor_position_ref[0] = joint[25]
        motor_position_ref[0] = joint[26]
        motor_position_ref[0] = joint[27]
        motor_position_ref[0] = joint[28]
        motor_position_ref[0] = joint[29]
        return motor_position_ref

    def fullbodyReferenceTra(self, x, t0, foot_index):
        u_lateral, u_sagittal = self.plan(x, t0, foot_index)
        self.updateBezier(u_lateral, u_sagittal, foot_index)
        q0 = x[0:30]
        s = t0 / self.T
        
        if foot_index == -1:  # letf support
            alpha = self.alpha[1, :]
        elif foot_index == 1:  # right support
            alpha = self.alpha[0, :]
        # ph, dph, ddph = helper.Bezier(self.alpha, s)
        # hd = np.concatenate((ph, np.zeros(8)), axis=None)
        ALIPx0_lateral, ALIPx0_sagittal = self.FOM2LIP(x, t0, foot_index)
        hd = helper.desiredOutput(alpha, s, q0, foot_index, ALIPx0_lateral, ALIPx0_sagittal)
        #print(ALIPx0_lateral)
        #print(ALIPx0_sagittal)
        paras = (foot_index, hd)
        #action_joint = np.zeros(30)
        action_joint = sp.optimize.fsolve(helper.virtualConstraint, self.q0, args=paras, xtol=1e-06, maxfev=500)
        #print("residual:")
        #print(helper.virtualConstraint(action_joint, foot_index, hd))
        #pdb.set_trace()
        self.q0 = action_joint
        action_motor = self.joint2motor(action_joint)

        return action_motor

    def fullbodyReferenceTra2(self, x, t0, foot_index):
        u_lateral, u_sagittal = self.plan(x, t0, foot_index)
        self.updateBezier(u_lateral, u_sagittal, foot_index)
        q0 = x[0:30]
        s = t0 / self.T
        
        if foot_index == -1:  # letf support
            alpha = self.alpha[1, :]
        elif foot_index == 1:  # right support
            alpha = self.alpha[0, :]
        # ph, dph, ddph = helper.Bezier(self.alpha, s)
        # hd = np.concatenate((ph, np.zeros(8)), axis=None)
        ALIPx0_lateral, ALIPx0_sagittal = self.FOM2LIP(x, t0, foot_index)
        hd = helper.desiredOutput(alpha, s, q0, foot_index, ALIPx0_lateral, ALIPx0_sagittal)

        return hd
'''
class ALIPParam:
    def __int__(self):
        self.H
        self.T
        self.m

        self.K_sagittal
        self.xStar_sagittal
        self.uStar_sagittal
        self.x0_sagittal

        self.K_lateral
        self.xStar_lateral
        self.uStar_lateral
        self.x0_lateral
    def setParam(self, params):
        #H, T, m, K, xStar, uStar, x0):
        H = params[0]
        T = params[1]
        m = params[2]



        self.H = H
        self.T = T
        self.m = m



