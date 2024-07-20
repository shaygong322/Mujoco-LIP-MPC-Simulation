import numpy as np


def digitLeftFootPose(q):
    # world frame to base frame
    Aw2bp = np.array([[1, 0, 0, q[0]],
                      [0, 1, 0, q[1]],
                      [0, 0, 1, q[2]],
                      [0, 0, 0, 1]])
    Rb_roll = np.array([[1, 0, 0, 0],
                        [0, np.cos(q[3]), -np.sin(q[3]), 0],
                        [0, np.sin(q[3]), np.cos(q[3]), 0],
                        [0, 0, 0, 1]])
    Rb_pitch = np.array([[np.cos(q[4]), 0, np.sin(q[4]), 0],
                         [0, 1, 0, 0],
                         [-np.sin(q[4]), 0, np.cos(q[4]), 0],
                         [0, 0, 0, 1]])
    Rb_yaw = np.array([[np.cos(q[5]), -np.sin(q[5]), 0, 0],
                       [np.sin(q[5]), np.cos(q[5]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    Rb_rpy = Rb_yaw @ Rb_pitch @ Rb_roll
    Aw2b = Aw2bp @ Rb_rpy

    # base to left leg
    Rb2hip_abduction_left1 = np.array([[0, 0, -1, -1e-03],
                                       [-0.366501000000000, 0.930418000000000, 0, 0.091],
                                       [0.930418000000000, 0.366501000000000, 0, 0],
                                       [0, 0, 0, 1]])
    Rb2hip_abduction_left2 = np.array([[np.cos(q[6]), - np.sin(q[6]), 0, 0],
                                        [np.sin(q[6]), np.cos(q[6]), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
    Rb2hip_abduction_left = Rb2hip_abduction_left1 @ Rb2hip_abduction_left2

    Rhip_abduction_left2hip_rotation_left1 = np.array([[0, 0, - 1, - 0.0505],
                                                        [0, 1, 0, 0],
                                                        [1, 0, 0, 0.0440],
                                                        [0, 0, 0, 1]])
    Rhip_abduction_left2hip_rotation_left2 = np.array([[np.cos(q[7]), - np.sin(q[7]), 0, 0],
                                                    [np.sin(q[7]), np.cos(q[7]), 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]])
    Rhip_abduction_left2hip_rotation_left = Rhip_abduction_left2hip_rotation_left1 @ Rhip_abduction_left2hip_rotation_left2

    Rhip_rotation_left2hip_flexion_left1 = np.array([[-0.707107000000000, -0.707107000000000, 0, 0],
                                                     [0, 0, -1, 0.004],
                                                     [0.707107000000000, -0.707107000000000, 0, 0.068],
                                                     [0, 0, 0, 1]])
    Rhip_rotation_left2hip_flexion_left2 = np.array([[np.cos(-q[8]), - np.sin(-q[8]), 0, 0],
                                                     [np.sin(-q[8]), np.cos(-q[8]), 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]])
    Rhip_rotation_left2hip_flexion_left = Rhip_rotation_left2hip_flexion_left1 @ Rhip_rotation_left2hip_flexion_left2

    Rhip_flexion_left2knee_joint_left1 = np.array([[0, 1, 0, 0.12],
                                                   [-1, 0, 0, 0],
                                                   [0, 0, 1, 0.0045],
                                                   [0, 0, 0, 1]])
    Rhip_flexion_left2knee_joint_left2 = np.array([[np.cos(q[9]), - np.sin(q[9]), 0, 0],
                                                 [np.sin(q[9]), np.cos(q[9]), 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]])
    Rhip_flexion_left2knee_joint_left = Rhip_flexion_left2knee_joint_left1 @ Rhip_flexion_left2knee_joint_left2

    Rknee_joint_left2knee_to_shin_left1 = np.array([[1, 0, 0, 0.0607],
                                                    [0, 1, 0, 0.0474],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]])
    Rknee_joint_left2knee_to_shin_left2 = np.array([[np.cos(q[10]), - np.sin(q[10]), 0, 0],
                                                     [np.sin(q[10]), np.cos(q[10]), 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]])
    Rknee_joint_left2knee_to_shin_left = Rknee_joint_left2knee_to_shin_left1 @ Rknee_joint_left2knee_to_shin_left2

    Rknee_to_shin_left2shin_to_tarsus_left1 = np.array([[-0.224951000000000, -0.974370000000000, 0, 0.4348],
                                                        [0.974370000000000, -0.224951000000000, 0, 0.02],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]])
    Rknee_to_shin_left2shin_to_tarsus_left2 = np.array([[np.cos(q[11]), - np.sin(q[11]), 0, 0],
                                                        [np.sin(q[11]), np.cos(q[11]), 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]])
    Rknee_to_shin_left2shin_to_tarsus_left = Rknee_to_shin_left2shin_to_tarsus_left1 @ Rknee_to_shin_left2shin_to_tarsus_left2

    Rshin_to_tarsus_left2toe_pitch_joint_left1 = np.array([[0.366455000000000, -0.930436000000000, 0, 0.408],
                                                           [0.930436000000000, 0.366455000000000, 0, -0.04],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]])
    Rshin_to_tarsus_left2toe_pitch_joint_left2 = np.array([[np.cos(q[12]), - np.sin(q[12]), 0, 0],
                                                           [np.sin(q[12]), np.cos(q[12]), 0, 0],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]])
    Rshin_to_tarsus_left2toe_pitch_joint_left = Rshin_to_tarsus_left2toe_pitch_joint_left1 @ Rshin_to_tarsus_left2toe_pitch_joint_left2

    Rtoe_pitch_joint_left2toe_roll_joint_left1 = np.array([[0, 0, 1, 0],
                                                           [0, 1, 0, 0],
                                                           [-1, 0, 0, 0],
                                                           [0, 0, 0, 1]])
    Rtoe_pitch_joint_left2toe_roll_joint_left2 = np.array([[np.cos(q[13]), - np.sin(q[13]), 0, 0],
                                                           [np.sin(q[13]), np.cos(q[13]), 0, 0],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]])
    Rtoe_pitch_joint_left2toe_roll_joint_left = Rtoe_pitch_joint_left2toe_roll_joint_left1 @ Rtoe_pitch_joint_left2toe_roll_joint_left2;

    Rtoe_roll_joint_left2bottom_feet = np.array([[0.0085, 0.9990, - 0.0443, 0],
                                                 [-0.4347, 0.0436, 0.8995, 0],
                                                 [0.9005, 0.0116, 0.4346, -0],
                                                 [0, 0, 0, 1]])
    Tb2left_toe = Aw2b @ Rb2hip_abduction_left @ Rhip_abduction_left2hip_rotation_left @ Rhip_rotation_left2hip_flexion_left @ Rhip_flexion_left2knee_joint_left @ Rknee_joint_left2knee_to_shin_left @ Rknee_to_shin_left2shin_to_tarsus_left @ Rshin_to_tarsus_left2toe_pitch_joint_left @ Rtoe_pitch_joint_left2toe_roll_joint_left @ Rtoe_roll_joint_left2bottom_feet
    X_L_foot = Tb2left_toe[0, 3]
    Y_L_foot = Tb2left_toe[1, 3]
    Z_L_foot = Tb2left_toe[2, 3]
    Lr21 = Tb2left_toe[1, 0]
    Lr11 = Tb2left_toe[0, 0]
    Lr31 = Tb2left_toe[2, 0]
    Lr32 = Tb2left_toe[2, 1]
    Lr33 = Tb2left_toe[2, 2]
    LPitch = np.arctan2(-Lr31, np.sqrt(Lr32 * Lr32 + Lr33 * Lr33))
    LYaw = np.arctan2(Lr21, Lr11)
    LRoll = np.arctan2(Lr32, Lr33)
    L_foot_pose = np.array([X_L_foot, Y_L_foot, Z_L_foot, LRoll, LPitch, LYaw])
    return L_foot_pose


def digitRightFootPose(q):
    # world frame to base frame
    Aw2bp = np.array([[1, 0, 0, q[0]],
                      [0, 1, 0, q[1]],
                      [0, 0, 1, q[2]],
                      [0, 0, 0, 1]])
    Rb_roll = np.array([[1, 0, 0, 0],
                        [0, np.cos(q[3]), -np.sin(q[3]), 0],
                        [0, np.sin(q[3]), np.cos(q[3]), 0],
                        [0, 0, 0, 1]])
    Rb_pitch = np.array([[np.cos(q[4]), 0, np.sin(q[4]), 0],
                         [0, 1, 0, 0],
                         [-np.sin(q[4]), 0, np.cos(q[4]), 0],
                         [0, 0, 0, 1]])
    Rb_yaw = np.array([[np.cos(q[5]), -np.sin(q[5]), 0, 0],
                       [np.sin(q[5]), np.cos(q[5]), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    Rb_rpy = Rb_yaw @ Rb_pitch @ Rb_roll
    Aw2b = Aw2bp @ Rb_rpy
    # base to right leg
    Rb2hip_abduction_right1 = np.array([[0, 0, -1, -1e-3],
                                        [0.366501000000000, 0.930418000000000, 0, -0.091],
                                        [0.930418000000000, -0.366501000000000, 0, 0],
                                        [0, 0, 0, 1]])
    Rb2hip_abduction_right2 = np.array([[np.cos(q[18]), - np.sin(q[18]), 0, 0],
                                        [np.sin(q[18]), np.cos(q[18]), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
    Rb2hip_abduction_right = Rb2hip_abduction_right1 @ Rb2hip_abduction_right2

    Rhip_abduction_right2hip_rotation_right1 = np.array([[0, 0, -1, -0.0505],
                                                         [0, 1, 0, 0],
                                                         [1, 0, 0, 0.044],
                                                         [0, 0, 0, 1]])
    Rhip_abduction_right2hip_rotation_right2 = np.array([[np.cos(q[19]), - np.sin(q[19]), 0, 0],
                                                         [np.sin(q[19]), np.cos(q[19]), 0, 0],
                                                         [0, 0, 1, 0],
                                                         [0, 0, 0, 1]])
    Rhip_abduction_right2hip_rotation_right = Rhip_abduction_right2hip_rotation_right1 @ Rhip_abduction_right2hip_rotation_right2

    Rhip_rotation_right2hip_flexion_right1 = np.array([[-0.707107000000000, 0.707107000000000, 0, 0],
                                                       [0, 0, 1, -0.004],
                                                       [0.707107000000000, 0.707107000000000, 0, 0.068],
                                                       [0, 0, 0, 1]])
    Rhip_rotation_right2hip_flexion_right2 = np.array([[np.cos(-q[20]), -np.sin(-q[20]), 0, 0],
                                                       [np.sin(-q[20]), np.cos(-q[20]), 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]])
    Rhip_rotation_right2hip_flexion_right = Rhip_rotation_right2hip_flexion_right1 @ Rhip_rotation_right2hip_flexion_right2

    Rhip_flexion_right2knee_joint_right1 = np.array([[0, -1, 0, 0.12],
                                                     [1, 0, 0, 0],
                                                     [0, 0, 1, 0.0045],
                                                     [0, 0, 0, 1]])
    Rhip_flexion_right2knee_joint_right2 = np.array([[np.cos(q[21]), - np.sin(q[21]), 0, 0],
                                                     [np.sin(q[21]), np.cos(q[21]), 0, 0],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]])
    Rhip_flexion_right2knee_joint_right = Rhip_flexion_right2knee_joint_right1 @ Rhip_flexion_right2knee_joint_right2

    Rknee_joint_right2knee_to_shin_right1 = np.array([[1, 0, 0, 0.0607],
                                                      [0, 1, 0, -0.0474],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]])
    Rknee_joint_right2knee_to_shin_right2 = np.array([[np.cos(q[22]), - np.sin(q[22]), 0, 0],
                                                      [np.sin(q[22]), np.cos(q[22]), 0, 0],
                                                      [0, 0, 1, 0],
                                                      [0, 0, 0, 1]])
    Rknee_joint_right2knee_to_shin_right = Rknee_joint_right2knee_to_shin_right1 @ Rknee_joint_right2knee_to_shin_right2

    Rknee_to_shin_right2shin_to_tarsus_right1 = np.array([[-0.224951000000000, 0.974370000000000, 0, 0.4348],
                                                          [-0.974370000000000, -0.224951000000000, 0, -0.02],
                                                          [0, 0, 1, 0],
                                                          [0, 0, 0, 1]])
    Rknee_to_shin_right2shin_to_tarsus_right2 = np.array([[np.cos(q[23]), - np.sin(q[23]), 0, 0],
                                                          [np.sin(q[23]), np.cos(q[23]), 0, 0],
                                                          [0, 0, 1, 0],
                                                          [0, 0, 0, 1]])
    Rknee_to_shin_right2shin_to_tarsus_right = Rknee_to_shin_right2shin_to_tarsus_right1 @ Rknee_to_shin_right2shin_to_tarsus_right2

    Rshin_to_tarsus_right2toe_pitchi_joint_right1 = np.array([[0.366455000000000, 0.930436000000000, 0, 0.408],
                                                              [-0.930436000000000, 0.366455000000000, 0, 0.04],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 0, 1]])
    Rshin_to_tarsus_right2toe_pitchi_joint_right2 = np.array([[np.cos(q[24]), - np.sin(q[24]), 0, 0],
                                                              [np.sin(q[24]), np.cos(q[24]), 0, 0],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 0, 1]])
    Rshin_to_tarsus_right2toe_pitchi_joint_right = Rshin_to_tarsus_right2toe_pitchi_joint_right1 @ Rshin_to_tarsus_right2toe_pitchi_joint_right2

    Rtoe_pitchi_joint_right2toe_roll_joint_right1 = np.array([[0, 0, 1, 0],
                                                              [0, 1, 0, 0],
                                                              [-1, 0, 0, 0],
                                                              [0, 0, 0, 1]])
    Rtoe_pitchi_joint_right2toe_roll_joint_right2 = np.array([[np.cos(q[25]), - np.sin(q[25]), 0, 0],
                                                              [np.sin(q[25]), np.cos(q[25]), 0, 0],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 0, 1]])
    Rtoe_pitchi_joint_right2toe_roll_joint_right = Rtoe_pitchi_joint_right2toe_roll_joint_right1 @ Rtoe_pitchi_joint_right2toe_roll_joint_right2

    Rtoe_roll_joint_right2bottom_feet = np.array([[0.0115, - 0.9974, - 0.0716, 0],
                                                  [0.4451, 0.0692, - 0.8928, 0],
                                                  [0.8954, - 0.0216, 0.4448, -0],
                                                  [0, 0, 0, 1]])
    Tb2right_toe = Aw2b @ Rb2hip_abduction_right @ Rhip_abduction_right2hip_rotation_right @ Rhip_rotation_right2hip_flexion_right @ Rhip_flexion_right2knee_joint_right @ Rknee_joint_right2knee_to_shin_right @ Rknee_to_shin_right2shin_to_tarsus_right @ Rshin_to_tarsus_right2toe_pitchi_joint_right @ Rtoe_pitchi_joint_right2toe_roll_joint_right @ Rtoe_roll_joint_right2bottom_feet
    X_R_foot = Tb2right_toe[0, 3]
    Y_R_foot = Tb2right_toe[1, 3]
    Z_R_foot = Tb2right_toe[2, 3]
    Rr21 = Tb2right_toe[1, 0]
    Rr11 = Tb2right_toe[0, 0]
    Rr31 = Tb2right_toe[2, 0]
    Rr32 = Tb2right_toe[2, 1]
    Rr33 = Tb2right_toe[2, 2]
    RPitch = np.arctan2(-Rr31, np.sqrt(Rr32 *Rr32 + Rr33 * Rr33))
    RYaw = np.arctan2(Rr21, Rr11)
    RRoll = np.arctan2(Rr32, Rr33)
    R_foot_pose = np.array([X_R_foot, Y_R_foot, Z_R_foot, RRoll, RPitch, RYaw])
    return R_foot_pose
