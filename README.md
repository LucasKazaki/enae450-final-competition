# enae450-final-competition

enae450 final competition using ros2 and turtlebot4 to solve a maze

SLAM_TOOLBOX tutorial: https://roboticsbackend.com/ros2-nav2-generate-a-map-with-slam_toolbox/

# QUICK START

## Teleop Run:
Terminal 1:
```
  cd ~/final-comp

  colcon build

  source install/setup.bash

  ros2 run final-comp bag_cmd_vel_player record --bag my_teleop_run --namespace /tb4_4 --center --lidar-offset-deg -90

```
Terminal 2:
```
  cd ~/final-comp

  colcon build

  source install/setup.bash

  ros2 run final-comp teleop --ros-args -p cmd_topic:=/tb4_4/cmd_vel
```
In terminal 2, use the wasd keys to move and the space key to stop.
Solve the maze using terminal 2 and while terminal 1 is recording.
Once the maze is solved, ctrl+c both terminals.
Terminal 1:
```
  ros2 run final-comp bag_cmd_vel_player follow --bag my_teleop_run --namespace /tb4_4 --center --lidar-offset-deg -90
```
Teleop run should be done

## Blind Run
```
  cd ~/final-comp

  colcon build

  source install/setup.bash

  ros2 run final-comp right_hand_solver --ros-args -p scan_topic:=/tb4_4/scan -p cmd_topic:=/tb4_4/cmd_vel
```
## Build the package

From the package folder:
```
  cd ~/final-comp
  
  colcon build --symlink-install
  ```
Source the package
```
  source install/setup.bash
```
Check that the package built correctly
```
  ros2 pkg list | grep final_comp
```
If nothing appears, rebuild and source again:
```
  cd ~/final-comp
  
  colcon build --symlink-install
  
  source install/setup.bash
```
## Run the Python nodes

Use these commands after sourcing:
```
  ros2 run final-comp maze_solver --ros-args -r __ns:=/tbX
  
  ros2 run final-comp move_robot --ros-args -r __ns:=/tbX
  
  ros2 run final-comp slam_map_viewer --ros-args -r __ns:=/tbX
  
  ros2 run final-comp view_map --ros-args -r __ns:=/tbX
  
  ros2 run final-comp bag_cmd_vel_player record --bag my_teleop_run --cmd-topic /tb4_4/cmd_vel

  ros2 run final-comp teleop --ros-args -p cmd_topic:=/tb4_4/cmd_vel
```
# Common commands

## Rebuild after editing files:
```
  cd ~/final-comp
  
  colcon build --symlink-install
  
  source install/setup.bash
```
## Run a file directly for debugging:
```
  python3 src/maze_solver.py
```
## Check executable names:
```
  ros2 pkg executables final_comp
```
## Run gazebo simulator:
```
  ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py
```
