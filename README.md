# STCOcc: Sparse Spatial-Temporal Cascade Renovation for 3D Occupancy and Scene Flow Prediction

This is the official PyTorch implementation for our paper:

## Environment

Install Pytorch 1.13 + CUDA 11.6

```setup
conda create --name stcocc python=3.8
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

Install mmdet3d (v1.0.0rc4) related packages and build this project
```setup
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
pip install -v -e .
```

Install other dependencies
```setup
pip install numpy==1.23.4
pip install yapf==0.40.1
pip install setuptools==59.5.0
pip install ninja
pip install einops
```

Due to the version of the dependencies, you may rise follow error, this [blog](https://blog.csdn.net/lzzzzzzm/article/details/133890916?spm=1001.2014.3001.5501) may help you
```error
error: too few arguments for template template parameter "Tuple" detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"  (721): here
```

## Prepare Dataset

1. Download nuScenes from [nuScenes](https://www.nuscenes.org/nuscenes) 

2. Download Occ3D-nus from [Occ3D-nus](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link)

3. Download OpenOcc from [OpenOcc-OpenDataLab](https://opendatalab.com/OpenDriveLab/CVPR24-Occ-Flow-Challenge/tree/main) 
or [OpenOcc-Google Drive](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq)

4. Download the generated info file from [Google Drive](https://drive.google.com/file/d/1KP25b3excY4N-3rqfkijuUmJLZeMwxZw/view?usp=sharing)
and unzip it to the data/nuscenes folder.

5. Download the pretrained weights from [Google Drive](https://drive.google.com/file/d/18Mxghwok1mlD1Pu2b16jjE13tszaxJUr/view?usp=drive_link).
The pretrained weights is drived from [BEVDet](https://github.com/HuangJunJie2017/BEVDet), we just rename the weights to fit our model.

5. Organize your folder structure as below:

```
├── project
├── data/
│   ├── nuscenes/
│   │   ├── samples/ 
│   │   ├── v1.0-trainval/
│   │   ├── gts/ (Occ3D-nus)
│   │   ├── openocc_v2/
│   │   ├── stcocc-nuscenes_infos_train.pkl
│   │   ├── stcocc-nuscenes_infos_val.pkl
```

6. Generate the multi-scale ground truth for Occ3D-nus or OpenOcc dataset:
```generate_multi-scale-gt
python tools/generate_ms_occ.py --dataset occ3d --pkl_path data/nuscenes/stcocc-nuscenes_infos_train.pkl
```

Finally the folder structure:

```
Project
├── mmdet3d/
├── tools/
├── pretrained/
│   ├── forward_projection-r50-4d-stereo-pretrained.pth
├── data/
│   ├── nuscenes/
│   │   ├── samples/     # You can download our imgs.tar.gz or using the original sample files of the nuScenes dataset
│   │   ├── v1.0-trainval/
│   │   ├── gts/
│   │   │   ├── scene_01/
│   │   │   │   ├── scene_token/
│   │   │   │   │   ├── lables.npz
│   │   │   │   │   ├── lables_1_2.npz
│   │   │   │   │   ├── lables_1_4.npz
│   │   │   │   │   ├── lables_1_8.npz
│   │   ├── stcocc-nuscenes_infos_train.pkl
│   │   ├── stcocc-nuscenes_infos_val.pkl
```

## Training

Train STCOcc with 8GPUs:

```train
bash tools/dist_train.sh config/stcocc/stcocc-r50-16f-openocc-12e.py 8
```

## Evaluation

Evaluate STCOcc with 8GPUs:

```eval
bash tools/dist_test.sh config/stcocc/stcocc-r50-16f-openocc-12e.py path/to/ckpts 8
```

## Results

Our model achieves the following performance on :


| Model name | Backbone | Input Size | RayIoU@1 | RayIoU@2 | RayIoU@4 | RayIoU |GPU(BS=1) |
|------------|----------|------------|----------|----------|----------|--------|-----------|
| CascadeOcc | R50      | 256x704    | 35.1     | 41.4     | 45.3     | 40.6   |5.0       |