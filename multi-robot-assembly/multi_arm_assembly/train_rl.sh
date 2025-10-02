#python train_rl.py --task=AssemblyInsert-v0 --enable_cameras
#python train_rl.py --task=AssemblyGrasp-v0 --enable_cameras
#python train_rl.py --task=AssemblyPlace-v0 --enable_cameras
python train_rl.py --task=AssemblyMove-v0 --enable_cameras
python train_rl.py --task=AssemblyInsert-v0 --play --enable_cameras --checkpoint /home/jiankai/Programs/long-horizon-assembly/packages/multi-robot-assembly/multi_arm_assembly/logs/rl_games/apa_impedance/2024-07-27_00-33-59/nn/last_apa_impedance_ep_1700_rew_-20.040802.pth --video --num_envs 2