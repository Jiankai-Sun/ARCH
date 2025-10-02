# Setup your robot IP here:
export ROBOT_IP=192.168.1.23
# Setup the name for the device you wish to create. Make sure that your user can write to that location.
export LOCAL_DEVICE_NAME=/dev/ttyS12

# This command was tested and working in araas and with the standalone gripper test script
sudo socat -d -d -d -d pty,link=${LOCAL_DEVICE_NAME},raw,ignoreeof,group-late=dialout,mode=666,waitslave tcp:${ROBOT_IP}:54321