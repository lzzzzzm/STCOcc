import os

import numpy as np
import cv2 as cv
import argparse
from visualizer import OccupancyVisualizer
import open3d as o3d
import colorsys
import mmengine
from nuscenes.nuscenes import NuScenes

occ3d_colors_map = np.array(
    [
        [0, 0, 0],          # others               black
        [255, 120, 50],     # barrier              orange
        [255, 192, 203],    # bicycle              pink         √
        [255, 255, 0],      # bus                  yellow       √
        [0, 150, 245],      # car                  blue         √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [135, 60, 0],       # trailer              brown        √
        [160, 32, 240],     # truck                purple       √
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ]
)
# change to BGR
occ3d_colors_map = occ3d_colors_map[:, ::-1]

openocc_colors_map = np.array(
    [
        [0, 150, 245],      # car                  blue         √
        [160, 32, 240],     # truck                purple       √
        [135, 60, 0],       # trailer              brown        √
        [255, 255, 0],      # bus                  yellow       √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 192, 203],    # bicycle              pink         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [255, 120, 50],     # barrier              orange
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ]
)

foreground_idx = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_args():
    parse = argparse.ArgumentParser('')
    parse.add_argument('--pkl-file', type=str, default='data/nuscenes/stcocc-nuscenes_infos_val.pkl', help='path of pkl for the nuScenes dataset')
    parse.add_argument('--data-path', type=str, default='data/nuscenes', help='path of the nuScenes dataset')
    parse.add_argument('--data-version', type=str, default='v1.0-trainval', help='version of the nuScenes dataset')
    parse.add_argument('--dataset-type', type=str, default='occ3d', help='dataset type, occ3d or openocc')
    parse.add_argument('--pred-path', type=str, default='scene-0274', help='path of the prediction data, dir')
    parse.add_argument('--vis-scene', type=list, default=['scene-0274'], help='visualize scene list')
    parse.add_argument('--vis-path', type=str, default='demo_out', help='path of saving the visualization images')
    parse.add_argument('--car-model', type=str, default='visualizer/3d_model.obj', help='car_model path')
    parse.add_argument('--vis-single-data', type=str, default='scene-0107/84c24cd1d7914f72bb2fa17a6c5c41e5.npz',help='single path of the visualization data')
    args = parse.parse_args()
    return args

def arange_according_to_scene(infos, nusc, vis_scene):
    scenes = dict()

    for i, info in enumerate(infos):
        scene_token = nusc.get('sample', info['token'])['scene_token']
        scene_meta = nusc.get('scene', scene_token)
        scene_name = scene_meta['name']
        if not scene_name in scenes:
            scenes[scene_name] = [info]
        else:
            scenes[scene_name].append(info)

    vis_scenes = dict()
    if len(vis_scene) == 0:
        vis_scenes = scenes
    else:
        for scene_name in vis_scene:
            vis_scenes[scene_name] = scenes[scene_name]

    return vis_scenes

def vis_occ_scene_on_3d(vis_scenes_infos,
                        vis_scene,
                        vis_path,
                        pred_path,
                        dataset_type='occ3d',
                        load_camera_mask=False,
                        voxel_size=(0.4, 0.4, 0.4),
                        vis_gt=True,
                        vis_flow=False,
                        car_model=None,
                        background_color=(255, 255, 255),
                        ):
    # define free_cls
    free_cls = 16 if dataset_type == 'openocc' else 17

    # load car model
    if car_model is not None:
        car_model_mesh = o3d.io.read_triangle_mesh(car_model)
        angle = np.pi / 2  # 90 度
        R = car_model_mesh.get_rotation_matrix_from_axis_angle(np.array([angle, 0, 0]))
        car_model_mesh.rotate(R, center=car_model_mesh.get_center())
        car_model_mesh.scale(0.25, center=car_model_mesh.get_center())
        current_center = car_model_mesh.get_center()
        new_center = np.array([0, 0, 0.5])
        translation = new_center - current_center
        car_model_mesh.translate(translation)
        car_model_mesh.compute_vertex_normals()
    else:
        car_model_mesh = None

    # check vis path
    mmengine.mkdir_or_exist(vis_path)
    for scene_name in vis_scene:
        scene_infos = vis_scenes_infos[scene_name]
        vis_occ_semantics = []
        buffer_vis_path = '{}/{}'.format(vis_path, scene_name)
        # check vis path
        mmengine.mkdir_or_exist(buffer_vis_path)

        for index, info in enumerate(scene_infos):

            save_path = os.path.join(buffer_vis_path, str(index))
            # visualize the scene data
            if vis_gt:
                occ_path = info['occ_path']
                if dataset_type == 'openocc':
                    occ_path = occ_path.replace('gts', 'openocc_v2')
                occ_label_path = os.path.join(occ_path, 'labels.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']

                if load_camera_mask:
                    assert 'mask_camera' in occ_label.keys()
                    mask_camera = occ_label['mask_camera']
                    occ_semantics[mask_camera == 0] = 255
                if vis_flow:
                    occ_flow = occ_label['flow']
                else:
                    occ_flow = None

            else:
                token = info['token']
                occ_label_path = os.path.join(pred_path, token + '.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']
                if vis_flow:
                    # check if flow exists
                    if 'flow' in occ_label.keys():
                        occ_flow = occ_label['flow']
                    if 'flows' in occ_label.keys():
                        occ_flow = occ_label['flows']
                else:
                    occ_flow = None

            # if view json exits
            occ_visualizer = OccupancyVisualizer(color_map=occ3d_colors_map if dataset_type == 'occ3d' else openocc_colors_map,
                                                 background_color=background_color)
            if os.path.exists('view.json'):
                param = o3d.io.read_pinhole_camera_parameters('view.json')
            else:
                param = None

            occ_visualizer.vis_occ(
                occ_semantics,
                occ_flow=occ_flow,
                ignore_labels=[free_cls, 255],
                voxelSize=voxel_size,
                range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                save_path=save_path,
                wait_time=-1,  # 1s, -1 means wait until press q
                view_json=param,
                car_model_mesh=car_model_mesh,
            )

            # press top-right x to close the windows
            param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)

            occ_visualizer.o3d_vis.destroy_window()

        # write video
        for i in range(index):
            img_path = os.path.join(buffer_vis_path, str(i) + '.png')
            img = cv.imread(img_path)
            vis_occ_semantics.append(img)
            os.remove(img_path)

        # save video
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        if vis_gt:
            if vis_flow:
                video_path = vis_path + '/' + 'gt-flow_' + scene_name + '.avi'
            else:
                video_path = vis_path + '/' + 'gt-occ_' + scene_name + '.avi'
        else:
            if vis_flow:
                video_path = vis_path + '/' + 'pred-flow_' + scene_name + '.avi'
            else:
                video_path = vis_path + '/' + 'pred-occ_' + scene_name + '.avi'

        video = cv.VideoWriter(video_path, fourcc, 5, (img.shape[1], img.shape[0]))
        for img in vis_occ_semantics:
            video.write(img)
        video.release()
        print('Save video to {}'.format(video_path))

def vis_occ_single_on_3d(data_path,
                        dataset_type='occ3d',
                        voxel_size=(0.4, 0.4, 0.4),
                        car_model=None,
                        vis_flow=False,
                        background_color=(255, 255, 255),
                        ):
    # define free_cls
    free_cls = 16 if dataset_type == 'openocc' else 17

    # load car model
    if car_model is not None:
        car_model_mesh = o3d.io.read_triangle_mesh(car_model)
        angle = np.pi / 2  # 90 度
        R = car_model_mesh.get_rotation_matrix_from_axis_angle(np.array([angle, 0, 0]))
        car_model_mesh.rotate(R, center=car_model_mesh.get_center())
        car_model_mesh.scale(0.25, center=car_model_mesh.get_center())
        current_center = car_model_mesh.get_center()
        new_center = np.array([0, 0, 0.5])
        translation = new_center - current_center
        car_model_mesh.translate(translation)
        car_model_mesh.compute_vertex_normals()
    else:
        car_model_mesh = None

    # visualize the scene data
    occ_label = np.load(data_path)
    # occ_semantics = occ_label['semantics']
    occ_semantics = occ_label['pred']
    if vis_flow:
        # check if flow exists
        if 'flow' in occ_label.keys():
            occ_flow = occ_label['flow']
        if 'flows' in occ_label.keys():
            occ_flow = occ_label['flows']
    else:
        occ_flow = None

    # if view json exits
    occ_visualizer = OccupancyVisualizer(color_map=occ3d_colors_map if dataset_type=='occ3d' else openocc_colors_map,
                                         background_color=background_color)

    if os.path.exists('view.json'):
        param = o3d.io.read_pinhole_camera_parameters('view.json')
    else:
        param = None

    occ_visualizer.vis_occ(
        occ_semantics,
        occ_flow=occ_flow,
        ignore_labels=[free_cls, 255],
        voxelSize=voxel_size,
        range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        save_path=None,
        wait_time=-1,  # 1s, -1 means wait until press q
        view_json=param,
        car_model_mesh=car_model_mesh,

    )
    # press top-right x to close the windows
    param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('view.json', param)

    occ_visualizer.o3d_vis.destroy_window()

def flow_to_color(vx, vy, max_magnitude=None):
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    angle = np.arctan2(vy, vx)

    hue = (angle + np.pi) / (2 * np.pi)

    if max_magnitude is None:
        max_magnitude = np.max(magnitude)

    saturation = np.clip(magnitude / max_magnitude, 0, 1)
    value = np.ones_like(saturation)

    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

def create_legend_circle(radius=1, resolution=500):
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y)
    vx = X
    vy = Y
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    mask = magnitude <= radius

    vx = vx[mask]
    vy = vy[mask]

    colors = flow_to_color(vx, vy, max_magnitude=radius)

    legend_image = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
    legend_image[mask.reshape(resolution, resolution)] = colors

    return legend_image

if __name__ == '__main__':
    print('open3d version: {}, if you want to use viewcontrol, make sure using 0.16.0 version!!'.format(o3d.__version__))
    args = parse_args()
    # check vis path
    mmengine.mkdir_or_exist(args.vis_path)
    pkl_data = mmengine.load(args.pkl_file)
    nusc = NuScenes(args.data_version, args.data_path)
    vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)
    # GT visualization
    # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, dataset_type=args.dataset_type, vis_gt=True, car_model=args.car_model)
    # Pred visualization
    # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, dataset_type=args.dataset_type, vis_gt=False, car_model=args.car_model)
    # # Single data visualization
    vis_occ_single_on_3d(args.vis_single_data, dataset_type=args.dataset_type, car_model=args.car_model, vis_flow=False)
    # # GT Flow visualization
    # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, dataset_type=args.dataset_type, vis_gt=True, vis_flow=True, car_model=args.car_model)
    # # Pred Flow visualization
    # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, dataset_type=args.dataset_type, vis_gt=False, vis_flow=True, car_model=args.car_model)
    # # Single Flow data visualization
    # vis_occ_single_on_3d(args.vis_single_data, dataset_type=args.dataset_type, car_model=args.car_model, vis_flow=True)

