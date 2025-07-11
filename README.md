<div align="center">

# STCOcc: Sparse Spatial-Temporal Cascade Renovation for 3D Occupancy and Scene Flow Prediction

</div>
This is the official PyTorch implementation for our paper:

> [**STCOcc: Sparse Spatial-Temporal Cascade Renovation for 3D Occupancy and Scene Flow Prediction**](https://arxiv.org/abs/2504.19749)\
> Zhimin Liao, Ping Wei*, Shuaijia Chen, Haoxuan Wang, Ziyang Ren                                     
> *CVPR2025 ([arXiv 2506.03079](https://arxiv.org/abs/2504.19749))*

![demo](https://github.com/lzzzzzm/STCOcc/blob/main/asserts/demo_video_.gif)

## 🚀 News

* **[2025-07]** Check out our new work [**II-World**](https://github.com/lzzzzzm/II-World)

* **[2025-03]** STCOcc is accepted to CVPR 2025.

## 🤗 Model Zoo

We utilize 8 RTX4090 GPUs to train our model.

|         Setting         | Epochs | Training Cost | RayIoU | MAVE |                                                Weights                                                | 
|:-----------------------:|:------:|:-------------:|:------:|:----:|-------------------------------------------------------------------------------------------------------|
| r50_704x256_16f_openocc |  ~48   |  32h,~8.7GB   |  40.8  | 0.44 | [Google-drive](https://drive.google.com/file/d/1_Ici4fsOk30Eqtc-nqUMcsj8NGi_dcxe/view?usp=drive_link) |
|  r50_704_256_16f_occ3d  |  ~36   |  21h,~7,7GB   |  41.7  |  -   | [Google-drive](https://drive.google.com/file/d/1ZbjYlzq9B7b_ac8lLXP_1gV7TRzL1TvX/view?usp=drive_link)                                            |

## 🛠️Environment

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
pip install mmengine
pip install -v -e .
```

Install other dependencies
```setup
pip install numpy==1.23.4
pip install yapf==0.40.1
pip install setuptools==59.5.0
pip install ninja
pip install einops
pip install open3d==0.16.0
```

Due to the version of the dependencies, you may rise follow error, this [blog](https://blog.csdn.net/lzzzzzzm/article/details/133890916?spm=1001.2014.3001.5501) may help you
```error
error: too few arguments for template template parameter "Tuple" detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"  (721): here
```

## 📦 Prepare Dataset

1. Download nuScenes from [nuScenes](https://www.nuscenes.org/nuscenes) 

2. Download Occ3D-nus from [Occ3D-nus](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link)

3. Download OpenOcc from [OpenOcc-OpenDataLab](https://opendatalab.com/OpenDriveLab/CVPR24-Occ-Flow-Challenge/tree/main) 
or [OpenOcc-Google Drive](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq)

4. Download the generated info file from [Google Drive](https://drive.google.com/file/d/1KP25b3excY4N-3rqfkijuUmJLZeMwxZw/view?usp=sharing)
and unzip it to the `data/nuscenes` folder. These `*pkl` files can be generated by running the `tools/create_data_bevdet.py`

5. Download the pretrained weights from [Google Drive](https://drive.google.com/file/d/18Mxghwok1mlD1Pu2b16jjE13tszaxJUr/view?usp=drive_link).
The pretrained weights is drived from [BEVDet](https://github.com/HuangJunJie2017/BEVDet), we just rename the weights to fit our model.

6. (Optional) Download the visualization car model [Google Drive](https://drive.google.com/file/d/1Uds-14smeKPYJkLC_DhH9ajap_zawfdi/view?usp=drive_link)

7. Organize your folder structure as below:

```
├── project
├── visualizer/
│   ├── 3d_model.obj/ (optional)
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

## 🎇 Training and Evaluation

Train STCOcc with 8GPUs:

```train
bash tools/dist_train.sh config/stcocc/stcocc_r50_704x256_16f_openocc_12e.py 8
```

Evaluate STCOcc with 6GPUs:

```eval
bash tools/dist_test.sh config/stcocc/stcocc_r50_704x256_16f_openocc_12e.py path/to/ckpts 6
```

## 🎥 Visualization
If you want to visualize the results, change the config setting `save_results` to `True` and run the evaluation script.

To visualize the single occ results, you can run the following command:
```visualize
python tools/vis_results.py --vis-single-data path/to/results
```
More visualization options can be found in the `tools/vis_results.py` script.

## 📄 Citation
if you find our work useful, please consider citing:

```bibtex
@inproceedings{liao2025stcocc,
  title={Stcocc: Sparse spatial-temporal cascade renovation for 3d occupancy and scene flow prediction},
  author={Liao, Zhimin and Wei, Ping and Chen, Shuaijia and Wang, Haoxuan and Ren, Ziyang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```

## Acknowledgement

Thanks to the following excellent projects:

- [SparseOcc](https://github.com/MCG-NJU/SparseOcc)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [FB-Occ](https://github.com/NVlabs/FB-BEV)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
