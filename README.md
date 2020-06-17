# Relightable Neural Renderer
### [Paper](https://arxiv.org/abs/1911.11530) | [Video](https://drive.google.com/file/d/1_uqgmkfQCjItk3sT6ye247Tjl_SWnQLD/view?usp=sharing) | [Supplementary Video](https://drive.google.com/file/d/1mCBHOCJ4h6dlh3NETotpWBbWVv0COPZj/view?usp=sharing) | [Data](https://drive.google.com/drive/folders/11-YKY9e3aPhSYm7k9i9sA4clAYMvc4gM?usp=sharing)
This repository contains a pytorch implementation for the paper: [A Neural Rendering Framework for Free-Viewpoint Relighting (CVPR 2020)](https://arxiv.org/abs/1911.11530). Our work takes multi-view images of an object under an unknown illumination as input, and produces a neural representation that can be rendered for both novel viewpoint and novel lighting.<br><br>

![teasergif](https://github.com/LansburyCH/relightable-nr/blob/master/other/teaser.gif)

## Installation

#### Tested on Ubuntu 16.04 + CUDA 9.0 + gcc 4.9.2 + Anaconda 3

Add CUDA to paths (modify based on your CUDA location):
```
export PATH=/usr/local/cuda/bin:$PATH \
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Install conda environment:
```
sudo apt install libopenexr-dev
conda env create -f environment.yml
conda activate relightable-nr
```
Install PyTorch Geometric:
```
pip --no-cache-dir install torch-scatter==1.3.2 -f https://pytorch-geometric.com/whl/torch-1.1.0.html
pip --no-cache-dir install torch-sparse==0.4.3 -f https://pytorch-geometric.com/whl/torch-1.1.0.html 
pip --no-cache-dir install torch-cluster==1.4.5 -f https://pytorch-geometric.com/whl/torch-1.1.0.html 
pip --no-cache-dir install torch-geometric==1.3.2 -f https://pytorch-geometric.com/whl/torch-1.1.0.html
```

Install our modified version of daniilidis-group's [neural_renderer](https://github.com/daniilidis-group/neural_renderer):
```
cd neural_renderer
python setup.py install
```

## Data Preparation
#### Data Layout
An example data is provided [here](https://drive.google.com/file/d/1jlQFeQnZy7jW87-_exv3LZ_cKlkZxVst/view?usp=sharing). After downloading, create a ```data/``` directory under this project and extract the downloaded .zip to this directory. The final layout will look like:
```
data/
    material_sphere/         # root directory for a scene
        light_probe/         # contains environment maps for lighting
        rgb0/                # images under the first lighting
        rgb1/                # (optional) images under the second lighting
        test_seq/            # viewpoint sequences for inference
            spiral_step720/  # an example sequence
        calib.mat            # camera intrinsics and extrinsics for the images in rgb0/
	mesh.obj             # proxy mesh with texture coordinates
	mesh.obj.mtl
	mesh_7500v.obj       # downsampled proxy mesh used for GCN
	mesh_7500v.obj.mtl
	tex.png              # texture image, can be just all white
```
For your own scenes, you need to prepare the data layout as listed above. For example, you may need to use structure-from-motion and multi-view stereo tools to generate the required camera calibration and proxy mesh.

#### Preprocessing
Run the following script to generate rasterization data and initial lighting information that will be used during training:
```
bash preproc.sh
```

## Training
After data preparation completes, run:
```
bash train_rnr.sh
```
For tweaking the arguments in the script, please refer to ```train_rnr.py``` for available options and their meaning. 

During training, various logging information will be recorded to a tensorboard summary, located at ```[scene_dir]/logs/rnr/[experiment_name]/```. Checkpoints (```model*.pth```), training settings (```params.txt```) and validation outputs (```val_*/```) are also saved to this directory.

## Inference (Free-Viewpoint Relighting)
Two examples are included in ```test_rnr.sh```, one for **novel view synthesis** (synthesizes novel views but keeps the lighting condition the same as training data) and one for **free-viewpoint relighting** (synthesizes images under both novel views and novel lighting). The synthesized images will be saved in ```[seq_dir]/resol_512/rnr/[model_id]/```. Please refer to ```test_rnr.py``` for more options during inference. 

The above two examples use the viewpoint sequence determined by ```spiral_step720/calib.mat```. To generate new viewpoint sequence, you need to create camera intrinsics and extrinsics for each frame, just like the provided ```calib.mat```.



## Citation
If you find our code or paper useful, please consider citing:
```
@inproceedings{chen2020neural,
  title={A Neural Rendering Framework for Free-Viewpoint Relighting},
  author={Chen, Zhang and Chen, Anpei and Zhang, Guli and Wang, Chengyuan and Ji, Yu and Kutulakos, Kiriakos N and Yu, Jingyi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5599--5610},
  year={2020}
}
```

## Relevant Works
[**Deferred Neural Rendering: Image Synthesis using Neural Textures (SIGGRAPH 2019)**](https://niessnerlab.org/projects/thies2019neural.html)<br>
Justus Thies, Michael Zollhöfer, Matthias Nießner

[**DeepVoxels: Learning Persistent 3D Feature Embeddings (CVPR 2019)**](https://vsitzmann.github.io/deepvoxels/)<br>
Vincent Sitzmann, Justus Thies, Felix Heide, Matthias Nießner, Gordon Wetzstein, Michael Zollhöfer

[**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (arXiv 2020)**](http://www.matthewtancik.com/nerf)<br>
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng

[**Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool (CVPR 2020)**](http://www.krematas.com/nvr/index.html)<br>
Konstantinos Rematas, Vittorio Ferrari

[**Learning Implicit Surface Light Fields (arXiv 2020)**](https://arxiv.org/abs/2003.12406)<br>
Michael Oechsle, Michael Niemeyer, Lars Mescheder, Thilo Strauss, Andreas Geiger
