# export LD_LIBRARY_PATH=/home/groups/schwager/jksun/Apps/miniforge3/envs/isaacgym/lib:$LD_LIBRARY_PATH
#python train.py --config-dir=. --config-name=image_fmb_multi_assemb1_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
#HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_fmb_multi_assemb1_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' # hydra.launcher=debug
#python eval.py --checkpoint data/outputs/2024.03.13/15.44.23_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt --output_dir data/pusht_eval_output --device cuda:0
# Train Diffusion Policy 
HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_fmb_multi_assemb2_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' # hydra.launcher=debug
# Train High-level policy
# HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=image_fmb_meta_diffusion_policy_dit.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' # hydra.launcher=debug
# Eval
#python eval_high_level_policy.py --checkpoint data/outputs/2024.07.11/00.06.03_train_dit_hybrid_fmb/checkpoints/latest.ckpt --output_dir data/hl --device cuda:0