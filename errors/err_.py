import torch
import numpy as np
import cv2

def get_pose_err(pose_gt, pose_est):
    transl_err = np.linalg.norm(pose_gt[0:3,3]-pose_est[0:3,3])
    rot_err = pose_est[0:3,0:3].T.dot(pose_gt[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return transl_err, rot_err[0]

def relative_pose(first_RT, second_RT):
    first_RT = torch.unsqueeze(first_RT, 0).cuda()
    second_RT = torch.unsqueeze(torch.unsqueeze(second_RT, 0),0).cuda()
    R = first_RT[:, :3, :3]
    T = first_RT[:, :3, 3:] 
    R_inv = R.permute(0, 2, 1)
    T_inv = torch.bmm(-R_inv, T)
    L = second_RT.shape[1]
    R_inv = R_inv.unsqueeze(1).repeat(1, L, 1, 1)
    T_inv = T_inv.unsqueeze(1).repeat(1, L, 1, 1)
    second_R = second_RT[:, :, :3, :3]
    second_T = second_RT[:, :, :3, 3:]
    second_T = second_T.reshape(-1, 3, 1) + torch.bmm(
        second_R.reshape(-1, 3, 3), T_inv.reshape(-1, 3, 1)
    )
    second_R = torch.bmm(second_R.reshape(-1, 3, 3), R_inv.reshape(-1, 3, 3))
    new_RT = torch.cat([second_R, second_T], dim=2).reshape(-1, L, 3, 4)
    return new_RT 