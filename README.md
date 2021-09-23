# Training
## Dependencies
Ubuntu 18.04, ROS Melodic, python 2.7.17, cuda 10.1, 10.0, torch 1.4.0

## How to run the code
(1) Clone the repository

(2) Set up the environments
* Make up your ROS catkin space. And
we were using a Turtlebot to train the policy.
For next step, we'll use UUVSimulator to train 
it instead.
* Place the designed world `empty.world` into the worlds directory.
* Install the dependencies for training.

(3) Run the code
* Launch Gazebo in one terminal:
        
        $ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

* In another terminal, run the code:
        
         $ python DDDQN.py 
         
(4) After training, you could use `DDDQN_test.py`
and `DDDQN_uwsim.py` to test the performance in Gazebo
worlds and UWSim worlds respectively. For real-world
tests, refer to [this](https://github.com/pengzhi1998/underwater_navigation_test)
 repository.
