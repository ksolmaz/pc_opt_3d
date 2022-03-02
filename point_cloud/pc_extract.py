import open3d as o3d
from skimage import io
import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from PIL import Image

def pc_extract(img_path,depht_path):
  color = o3d.io.read_image(img_path)
  im = io.imread(depht_path)

  img = o3d.geometry.Image(im)
  pinhole_camera_intrinsic =o3d.camera.PinholeCameraIntrinsic(640,480,525,525,320,240)

  rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, img,convert_rgb_to_intensity=False,depth_trunc=100.0)
  pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,pinhole_camera_intrinsic)


  img = Image.fromarray(np.uint8(im/im.max()*255))
  pcd.transform([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

  points_3d = np.asarray(pcd.points)
  colors_3d = np.asarray(pcd.colors)

  return torch.Tensor(points_3d).cuda(),torch.Tensor(colors_3d).cuda()

def world_coor(img_path,depht_path):
  verts_,rgb = pc_extract(img_path,depht_path)
  verts_h = torch.ones(((verts_.T[0].shape)[0], 4)).cuda()
  verts_h[:,:3] = verts_[:,:]
  verts_world = ((torch.matmul(torch.eye(4).cuda(),verts_h.T)))
  return verts_world