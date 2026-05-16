#!/bin/bash

# Start a new tmux session
tmux new-session -d -s MySession

# Split the window vertically twice to create three rows
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# In the last pane (third row), split it horizontally to create two columns
tmux select-pane -t 2
tmux split-window -h

# Activate Conda environment in each pane and set up commands as needed
# First pane
tmux select-pane -t 0
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'cd multi_arm_assembly' C-m
tmux send-keys 'export PYTHONPATH=""' C-m
tmux send-keys 'export CHECKPOINT=""' C-m
tmux send-keys 'python run_rl_araas.py --checkpoint=$CHECKPOINT --enable-hardware'

# Second pane
tmux select-pane -t 1
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'python -m pyaraas.server -a /home/jiankai/Programs/long-horizon-assembly/packages/apa_workcells/'

# Third pane
tmux select-pane -t 2
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'sh test_ur_gripper_socketA.sh'

# Fourth pane
tmux select-pane -t 3
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'sh test_ur_gripper_socketB.sh'

# Attach to the tmux session
tmux attach-session -t MySession