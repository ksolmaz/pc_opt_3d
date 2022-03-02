<h3 align="center">For simple run, change paths and run main.py or run demo.ipynb</h3>

    pc_extract() = RGB path ve DEPTH path alıp point cloud noktalarını ve noktaların rengini döndürüyor.
    verts,rgb = pc_extract(img_color,img_depth)


    six_param_opt() = parametreleri optimizasyona hazırlayarak point cloud çıktısı veriyor.
    point_cloud = six_param_opt(x,new_world_coor,rgb)


    render_settings() = point cloud'u istenen ayarlarda render ediyor ve 2D görsel döndürüyor.
    images = render_settings(point_cloud)


    world_coor() = RGB path ve DEPTH path alıp point cloud noktalarını ve noktaların rengini birim
    matris ile dünya kordinatlarında döndürüyor. Değiştirmek için içerisindeki birim matris yerine sahnenin
    kendi RT'sini vermemiz gerekiyor.
    new_world_coor = world_coor(img_color,img_depth)


    img_for_loss() = hedef görselin path'i ile onu transforma uygun hale getirip o görseli döndürüyor.
    ref_img = img_for_loss(ref_img)


Python 3.7.12
