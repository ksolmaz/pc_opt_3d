# ----
For simple run, change paths and run main.py or run demo.ipynb

#####################################

pc_extract() = RGB path ve DEPTH path alıp point cloud noktalarını ve noktaların rengini döndürüyor.
verts,rgb = pc_extract(img_color,img_depth)

#####################################

six_param_opt() = parametreleri optimizasyona hazılıyarak point cloud çıktısı veriyor.
point_cloud = six_param_opt(x,new_world_coor,rgb)

#####################################

render_settings() = point cloud'u istenen ayarlarda render ediyor ve 2D görsel döndürüyor.
images = render_settings(point_cloud)

#####################################
