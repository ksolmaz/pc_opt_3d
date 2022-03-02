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

<h3 align="center">Requirements</h3> 

    Python 3.7.12
    
    pytorch3d==0.6.1            pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu111_pyt1100/download.html
    torch==1.10.0+cu111         pip install torch==1.10.0+cu111
    open3d==0.15.2              pip install open3d==0.15.2
    imagecodecs==2021.11.20     pip install imagecodecs==2021.11.20
    differentiable_evo          pip install git+https://github.com/nathanrooy/differential-evolution-optimization
    opencv-python==4.1.2.30     pip install opencv-python==4.1.2.30
    numpy==1.21.5               pip install numpy==1.21.5
