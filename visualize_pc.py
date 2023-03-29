import os
import open3d as o3d
import numpy as np


def transform(files, source_dir, target_dir):
    for f in files:
        frame = f.split('.')[0]
        data = np.load(source_dir + f)
        renew_data = [list(raw) for raw in data]
        renew_data = np.asarray(renew_data)  # 可以正常索引的数据
        # print(renew_data[:][:3])
        # print(renew_data.shape)

        txt_file_name = frame + ".txt"
        np.savetxt(target_dir + txt_file_name, renew_data)



def visualize_pc(files,TargetDir):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    vis.get_render_option().point_size = 2                              # 点云大小
    vis.get_render_option().background_color = np.asarray([0, 0, 0])    # 背景颜色
    # colors_0 = np.random.randint(255, size=(23, 3)) / 255.  
    # colors_0 = np.array([[ i, 255-i, i] for i in range(255)])
    # colors_1 = np.array([[255-i, 0,  i] for i in range(255)])
    # colors_0 = np.concatenate((colors_0, colors_1),axis=0)
    colors_0 = np.load('color.npy')
    pcd = o3d.geometry.PointCloud()
    pcd.paint_uniform_color
    to_reset = True
    vis.add_geometry(pcd)
    for f in files:
        
        data = np.load(TargetDir + "/" + f)['arr_0']
        # renew_data = [list(raw) for raw in points]
        # points = np.asarray(renew_data)
        points = data[:, :] # 读取4D点云
        # pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # -10 ~ +10 | 30 -- 255  40/255 =  0.157
        # -1 + 20 = 19 * 8.48
        # 为各个真实标签指定颜色
        colors = colors_0[points[:, -1].astype(np.uint8)]
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # color_int = np.array([abs(points[:,-1])*15]).astype(np.uint8).reshape(-1)
        # colors = colors_0[color_int]
        # # colors = 
        # pcd.colors = o3d.utility.Vector3dVector(colors[:, :])

        
        vis.update_geometry(pcd)
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
        vis.poll_events()
        vis.update_renderer()


if __name__ == '__main__':
    
    TargetDir = "./data/carla_scene_02_unnorm"
    
    # TFiles = os.listdir(TargetDir)
    TFiles = np.load('valid_file.npy')
    
    visualize_pc(TFiles, TargetDir)

