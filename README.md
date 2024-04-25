# MPC-Path-follow-and-Obs-Avoidance

1. For quick prototyping scripts/tracking.py it plots everything -path, waypoints, LQR,MPC, Pure pursuit added. 
2. I have also added path creation to JSON script, in the code there is an option to take it directly from global planner as well 
3. I added an option to have obstacles from costmap itself if we want to pursue that 
4. I also documented a lot of these paths which are in Data folder 
5. Mainly you need to work with scripts/tracking_r.py and controllers/mpc2/controller_r.py (Has major code for visualising, adding obstacle, scipy.minimize() for opti 

To run:

1. source
2. export TURTLEBOT3_MODEL='burger/waffle'
3. roslaunch path_tracking_py run_simulation.launch
4. roslaunch path_tracking_py tracking.launch
