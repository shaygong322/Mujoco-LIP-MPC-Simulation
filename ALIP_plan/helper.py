import numpy as np
import digit.planner.forwardKinematics as forwardKinematics
import digit.planner.fromFROST as fromFROST

def hcOutput(q, foot_index):
    if foot_index == -1:  # left support
        support_foot = forwardKinematics.digitLeftFootPose(q)
        swing_foot = forwardKinematics.digitRightFootPose(q)
    elif foot_index == 1:  # right support
        support_foot = forwardKinematics.digitRightFootPose(q)
        swing_foot = forwardKinematics.digitLeftFootPose(q)
    else:
        support_foot = forwardKinematics.digitRightFootPose(q)
        swing_foot = forwardKinematics.digitLeftFootPose(q)        
    CoM = fromFROST.p_CoM(q)
    support_foot[2:6] = 0
    hc = np.concatenate((np.array([CoM[2], q[3], q[4], q[5]]), swing_foot - support_foot, q[14:18], q[26:30]),
                        axis=None)
    return hc
def desiredOutput(alpha, s):
    ph, dph, ddph = Bezier(alpha, s)
    # if foot_index == -1:  # left support
    #     support_foot = forwardKinematics.digitLeftFootPose(q)
    # elif foot_index == 1:  # right support
    #     support_foot = forwardKinematics.digitRightFootPose(q)
    # support_foot[2] = 0
    # support_foot[3] = 0
    # support_foot[4] = 0
    # support_foot[5] = 0
    # desiredHolonomic = np.concatenate((support_foot,np.array([0, 0, 0, 0])), axis=None)
    # ph = np.array([0.71, 0, 0, 0, 0.01, 0.01, 0, 0, 0, 0])
    # hd = np.concatenate((ALIPx0_sagittal[0]+support_foot[0], ALIPx0_lateral[0]+support_foot[1], ph, np.zeros(8), desiredHolonomic), axis=None)
    hd = np.concatenate((0,0, ph, np.zeros(8)), axis=None)   
    Dhd =  np.concatenate((0,0, dph, np.zeros(8)), axis=None) * 1/0.25
    DDhd = np.concatenate((0,0, ddph, np.zeros(8)), axis=None) * 1/0.25 * 1/0.25
    return hd, Dhd, DDhd
def virtualConstraint(q,foot_index,hd):
    #foot_index = param[0]
    #hd = param[1]
    return hcOutput(q, foot_index)-hd
def Bezier(Alpha,s):
    S = np.array([
            (1 - s)** 6,
            6 * s * (1 - s) ** 5,
            15 * s ** 2 * (1 - s) ** 4,
            20 * s ** 3 * (1 - s) ** 3,
            15 * s ** 4 * (1 - s) ** 2,
            6 * s ** 5 * (1 - s) ** 1,
            s ** 6])

    dS = np.array([
            6 * (s - 1) ** 5,
            - 30 * s * (s - 1) ** 4 - 6 * (s - 1) ** 5,
            30 * s * (s - 1) ** 4 + 60 * s ** 2 * (s - 1) ** 3,
            - 60 * s ** 2 * (s - 1) ** 3 - 60 * s ** 3 * (s - 1) ** 2,
            15 * s ** 4 * (2 * s - 2) + 60 * s ** 3 * (s - 1) ** 2,
            - 30 * s ** 4 * (s - 1) - 6 * s ** 5,
            6 * s ** 5])

    ddS = np.array([
            30 * (s - 1) ** 4,
            - 120 * s * (s - 1) ** 3 - 60 * (s - 1) ** 4,
            240 * s * (s - 1) ** 3 + 30 * (s - 1) ** 4 + 180 * s ** 2 * (s - 1) ** 2,
            - 120 * s * (s - 1) ** 3 - 60 * s ** 3 * (2 * s - 2) - 360 * s ** 2 * (s - 1) ** 2,
            120 * s ** 3 * (2 * s - 2) + 180 * s ** 2 * (s - 1) ** 2 + 30 * s ** 4,
            - 120 * s ** 3 * (s - 1) - 60 * s ** 4,
            30 * s ** 4,
        ])
    
    a1 = Alpha[0:7]         #COM z
    a2 = Alpha[7:14]        #torso roll
    a3 = Alpha[14:21]       #torso pitch
    a4 = Alpha[21:28]       #torso yaw
    a5 = Alpha[28:35]       #swing foot x
    a6 = Alpha[35:42]       #swing foot y
    a7 = Alpha[42:49]       #swing foot z
    a8 = Alpha[49:56]       #swing roll
    a9 = Alpha[56:63]       #swing pitch
    a10 = Alpha[63:70]      #swing yaw
    a = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])

    ph = np.dot(a, S)
    dph = np.dot(a, dS)
    ddph = np.dot(a, ddS)
    return ph, dph, ddph