<h3 align="center">For simple run, change paths and run main.py or run demo.ipynb</h3>

pc_extract() = RGB path ve DEPTH path alıp point cloud noktalarını ve noktaların rengini döndürüyor.
verts,rgb = pc_extract(img_color,img_depth)


six_param_opt() = parametreleri optimizasyona hazılıyarak point cloud çıktısı veriyor.
point_cloud = six_param_opt(x,new_world_coor,rgb)


render_settings() = point cloud'u istenen ayarlarda render ediyor ve 2D görsel döndürüyor.
images = render_settings(point_cloud)


world_coor() = RGB path ve DEPTH path alıp point cloud noktalarını ve noktaların rengini birim matris ile dünya kordinatlarında döndürüyor değiştirmek için içerisindeki birim matris yerine sahnenin kendi pozunu vermemiz gerekiyor.
new_world_coor = world_coor(img_color,img_depth)


img_for_loss() = hedef görselin path'i ile onu transforma uygun hale getirip o görseli döndürüyor.
ref_img = img_for_loss(ref_img)
#pc_extract
#H2 header
#H3 header
#H4 header
#H5 header
#H6 header
#Heading 1 link [Heading link](https://github.com/ksolmaz/pc_opt_3d/blob/86fd997a1cedc617818ab625df716e50b5901947/point_cloud/pc_extract.py "Heading link")
##Heading 2 link [Heading link](https://github.com/pandao/editor.md "Heading link")
###Heading 3 link [Heading link](https://github.com/pandao/editor.md "Heading link")
####Heading 4 link [Heading link](https://github.com/pandao/editor.md "Heading link") Heading link [Heading link](https://github.com/pandao/editor.md "Heading link")
#####Heading 5 link [Heading link](https://github.com/pandao/editor.md "Heading link")
######Heading 6 link [Heading link](https://github.com/pandao/editor.md "Heading link")

