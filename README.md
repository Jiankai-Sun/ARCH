# ARCH
ARCH: Hierarchical Hybrid Learning for Long-Horizon Contact-Rich Robotic Assembly

## Requirements
- Ubuntu 20.04
- Nvidia-docker == 2.1.1
- IsaacLab == 4.0

### DiT Training
```bash
# pip install huggingface_hub==0.23.0 timm
mamba create -f conda_environment.yaml
cd diffusion_policy(_v1)
bash train.sh
```

### DiT Evaluation
```bash
cd diffusion_policy
python eval.py --checkpoint data/outputs/2024.06.18/23.35.17_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt --output_dir data/fmb_eval_output --device cuda:0
```

## IsaacGym

run `pip install "numpy<1.24"`

## IsaacLab
Dependencies:
```bash
cd IsaacLab
./isaaclab.sh -c
./isaaclab.sh -i
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
pip install gym pybullet rl_games zuko wandb
```

```bash
cd IsaacLab/
./isaaclab.sh -p source/standalone/environments/random_agent.py --task Assembly-v0 --num_envs 32
```
Convert `.urdf` to `.usd`
```bash
cd IsaacLab/source/standalone/tools/
python convert_urdf.py ../../../../IsaacGymEnvs/assets/assembly/urdf/factory_franka.urdf ../source/extensions/omni.isaac.lab_assets/data/factory_franka.usda
```

#### quaternion
```bash
conda install -c conda-forge quaternion
```

#### Segment-Anything
```bash
mamba env create -n sam python=3.10
pip install git+https://github.com/facebookresearch/segment-anything.git
cd tools
python dataset_stats.py # run npy2video()
cd FoundationPose
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
python run_segment_mask.py
```

## CPPF2 (SHOT descriptor extraction)
```
conda install pybind11 -c conda-forge
conda install pcl=1.9 -c conda-forge
cd src_shot
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Relase
cmake --build . --config Release
# /var/opt/araas/miniconda3/envs/araas/include/boost/thread/pthread/thread_data.hpp:60:5: error: missing binary operator before token "("
#   60 | #if PTHREAD_STACK_MIN > 0
# refer to https://github.com/boostorg/thread/issues/364
..............................................................................
conda install qhull=2019.1
# https://github.com/rusty1s/pytorch_scatter/tree/master
conda install pytorch-scatter -c pyg or pip install git+https://github.com/rusty1s/pytorch_scatter.git  # 2.1.2
# install cuda==12.4 and set ~/.bashrc
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install icecream==2.1.3
```

#### Pipeline
```bash
pip install hydra-core dill
# Collect Data
cd packages/multi-robot-assembly/multi-robot-assembly/
python run_rl_araas_assembly_high_level.py
mkdir -p logs/demos_isaaclab/0/
mv logs/demos_isaaclab/*.pkl logs/demos_isaaclab/0/
python construct_hl_dataset.py
cp -r logs/assembly_dataset/ 
cd ../packages/diffusion_policy_v1
bash train.sh
```

## Reference
**[ARCH: Hierarchical Hybrid Learning for Long-Horizon Contact-Rich Robotic Assembly
](https://arxiv.org/pdf/2409.16451)**
<br />
[Jiankai Sun](https://scholar.google.com/citations?user=726MCb8AAAAJ&hl=en), 
[Aidan Curtis](https://scholar.google.com/citations?user=tRJf4Q8AAAAJ&hl=en), 
[Yang You](https://scholar.google.com/citations?user=1YV1_KUAAAAJ&hl=en),
[Yan Xu](https://scholar.google.com/citations?user=OmySDsMAAAAJ&hl=zh-CN), 
[Michael Koehle](),
[Qianzhong Chen](https://scholar.google.com/citations?user=MqU82XsAAAAJ&hl=en),
[Suning Huang](https://scholar.google.com/citations?user=wX_5YMIAAAAJ&hl=zh-CN),
[Leonidas Guibas](https://scholar.google.com/citations?user=5JlEyTAAAAAJ&hl=en),
[Sachin Chitta](https://scholar.google.com/citations?user=S2vB4tAAAAAJ&hl=en), 
[Mac Schwager](https://scholar.google.com/citations?user=-EqbTXoAAAAJ&hl=en),and
[Hui Li](https://scholar.google.com/citations?user=e_Npqt4AAAAJ&hl=en)
<br />
**In Proceedings of the Conference on Robot Learning (CoRL) 2025**
<br />
[[Paper]](https://arxiv.org/abs/2409.16451)
[[Project Page]](https://long-horizon-assembly.github.io/)

```
@inproceedings{sun2025arch,
  title={ARCH: Hierarchical Hybrid Learning for Long-Horizon Contact-Rich Robotic Assembly},
  author={Sun, Jiankai and Curtis, Aidan and You, Yang and Xu, Yan and Koehle, Michael and Chen, Qianzhong and Huang, Suning and Guibas, Leonidas and Chitta, Sachin and Schwager, Mac and others},
  booktitle={9th Annual Conference on Robot Learning}
}
```