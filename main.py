from point_cloud import projection_matrix_to_RT,pc_extract,world_coor,six_param_opt
from renderer import img_for_loss,render_settings
from diffevo import de_simple
import torch
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion,quaternion_to_matrix
from pytorch3d.structures import Pointclouds

img_color = "/other_files/frame-000000.color.png"
img_depth = '/other_files/frame-000000_depth.tiff'
ref_img   = "/other_files/frame-000150.color.png"

verts,rgb = pc_extract(img_color,img_depth)
new_world_coor = world_coor(img_color,img_depth)
ref_img = img_for_loss(ref_img)

def de_optim(x):
        point_cloud = six_param_opt(x,new_world_coor,rgb)
        images = render_settings(point_cloud)
        no_zero_points = images[0, ..., :3].float().detach().squeeze().cpu().numpy() != 0
        zeros_np = images[0, ..., :3].float().detach().cpu().numpy()

        if np.count_nonzero(zeros_np == 0) < 300000:
          colored_points_indexes = np.where(images[0, ..., :3].float().detach().squeeze().cpu().numpy() != 0)
          colored_points_val = int((np.asarray(np.where(images[0, ..., :3].float().detach().squeeze().cpu().numpy() != 0))[1]).shape[0])
          input = images[0, ..., :3][colored_points_indexes]
          target = torch.from_numpy(ref_img)[colored_points_indexes].cuda()
          loss = torch.sum((input - target) ** 2)/colored_points_val
        else:
          input = torch.ones(480,640,3).cuda()
          target = torch.zeros(480,640,3).cuda()
          colored_points_val = 921600
          loss = torch.sum((input - target) ** 2)/colored_points_val 
                
        return loss.detach().cpu().numpy()

bounds = [(-1,1),(-1,1),(-1,1),(-1.3,1.3),(-1.3,1.3),(-1.3,1.3)]            
popsize = 20                        
mutate = 0.4                        
recombination = 1                
maxiter = 20 

de_simple.minimize(de_optim, bounds, popsize, mutate, recombination, maxiter)
