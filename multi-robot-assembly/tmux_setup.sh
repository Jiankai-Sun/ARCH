#!/bin/bash

# Start a new tmux session
tmux new-session -d -s MySession

# Split the window vertically to create two columns
tmux split-window -v

# Split the left column horizontally to create top and bottom rows
tmux select-pane -t 0
tmux split-window -h

# Split the bottom row horizontally to create two columns in the bottom row
tmux select-pane -t 2
tmux split-window -h

tmux select-pane -t 1
tmux split-window -v

# Resize the top-left pane (pane 0)
tmux resize-pane -t 0 -y 70  # Approximate 4/5 height
tmux resize-pane -t 0 -x 80  # Approximate 3/4 width

# Activate Conda environment in each pane and set up commands as needed
# First pane (top-left)
tmux select-pane -t 0
tmux send-keys 'conda activate araas_cppf2' C-m
tmux send-keys 'cd multi_arm_assembly' C-m
tmux send-keys 'export PYTHONPATH=""' C-m
tmux send-keys 'export CHECKPOINT=""' C-m
tmux send-keys 'python run_rl_araas_assembly_high_level.py --enable-hardware' C-m

# Second pane (top-right)
tmux select-pane -t 1
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'cd ${HOME}/Documents/Github/araas/src/robot_server/build' C-m
tmux send-keys './robot_server -a ${HOME}/Programs/long-horizon-assembly/packages/nick_apa_workcells/' C-m

# Third pane (bottom-left)
tmux select-pane -t 2
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'cd ${HOME}/Documents/Github/araas/src/front_end' C-m
tmux send-keys 'npm start' C-m

tmux select-pane -t 3
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'sh test_ur_gripper_socketA.sh' C-m

# Fourth pane (bottom-right)
tmux select-pane -t 4
tmux send-keys 'conda activate araas' C-m
tmux send-keys 'sh test_ur_gripper_socketB.sh' C-m

# Attach to the tmux session
tmux attach-session -t MySession
