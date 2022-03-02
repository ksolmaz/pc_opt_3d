import torch
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion,quaternion_to_matrix
from pytorch3d.structures import Pointclouds

def projection_matrix_to_RT(pose_path):
  _RT = torch.Tensor(np.loadtxt(pose_path)).float().cuda()
  _R = _RT[:3,:3].cuda()
  _R_h = torch.eye(4).cuda()
  _R_h[:3,:3] = _R
  _R_inverse = torch.inverse(_R).cuda()
  _R_h_inverse = torch.inverse(_R_h).cuda()
  _C = torch.matmul(_R_h_inverse,_RT)
  _T = _C[:3,3:].T
  return _R,_R_h,_T,_C

def six_param_opt(x,new_world,rgb):
  x = torch.tensor(x).cuda()
  qua_w = torch.sqrt(abs(1-((x[0]**2)+(x[1]**2)+(x[2]**2))))
  new_qua = torch.ones(4)
  new_qua[0] = qua_w
  new_qua[1:] = x[:3]
  _R  = quaternion_to_matrix(new_qua)
  _R_h = torch.eye(4).cuda()
  _R_h[:3,:3] = _R
  _R_inverse = torch.inverse(_R).cuda()
  _R_transpose = _R_h.T
  ones_for_T = torch.unsqueeze(torch.ones(4),1).cuda()
  ones_for_T[:3] = torch.unsqueeze(x[3:],1)      
  _T = ones_for_T
  _C = torch.matmul(_R_transpose,_T.float())
  rotation_result = torch.matmul(_R_transpose,new_world) -_C
  point_cloud = Pointclouds(points=[rotation_result[:3,:].T], features=[rgb]).cuda()
  return point_cloud