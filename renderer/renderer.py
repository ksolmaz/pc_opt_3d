import torch
import cv2
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
     PerspectiveCameras,
     PointsRasterizationSettings,
     PointsRasterizer,
     AlphaCompositor,
     PointsRenderer
)

def img_for_loss(img_path):
  img_path = cv2.imread(img_path)[...,::-1]
  takla = cv2.flip(img_path, 0)
  ters_ref_img = cv2.flip(takla, 1)
  ters_ref_img = ters_ref_img/255
  return ters_ref_img

def render_settings(point_cloud):
  if torch.cuda.is_available():
      device = torch.device("cuda:0")
      torch.cuda.set_device(device)
  else:
      device = torch.device("cpu")

  image_size = ((640, 480),)
  fcl_screen = ((525,525),)        
  prp_screen = ((240, 320),)
  cameras_ndc = PerspectiveCameras(device=device,focal_length=fcl_screen,principal_point=prp_screen,in_ndc=False,image_size=image_size)


  raster_settings = PointsRasterizationSettings(
      image_size=(480, 640),
      radius = 0.01,
      points_per_pixel = 10
  )

  rasterizer = PointsRasterizer(cameras=cameras_ndc, raster_settings=raster_settings).cuda()
  renderer = PointsRenderer(
      rasterizer=rasterizer,
      compositor=AlphaCompositor()
  ).cuda()

  images = renderer(point_cloud)
  return images