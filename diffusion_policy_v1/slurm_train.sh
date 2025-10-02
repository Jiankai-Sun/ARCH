#!/bin/bash
#SBATCH --job-name=assemb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-cpu=64G
# export LD_LIBRARY_PATH=/home/groups/schwager/jksun/Apps/miniforge3/envs/isaacgym/lib:$LD_LIBRARY_PATH
HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_fmb_multi_assemb1_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' # hydra.launcher=debug
#python eval.py --checkpoint data/outputs/2024.03.13/15.44.23_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt --output_dir data/pusht_eval_output --device cuda:0