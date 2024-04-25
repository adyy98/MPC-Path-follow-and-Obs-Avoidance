import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import scripts.pt_scripts.utils as utils 


class Controller:
	def __init__(self, ref_traj, robot, plotting, dt = 0.2):
		
		# input data (robot & reference trajectory)
		self.robot = robot
		self.ref_traj = ref_traj
		self.goal_x = ref_traj.x[-1]
		self.goal_y = ref_traj.y[-1]
		
		# plotting
		self.plotting = plotting

		# controller
		self.from_beggins = True
		self.controller_name = "MPC"	# PID or MPC

		# # settings
		self.dt = dt
		self.dist_thresh = 0.3
		self.goal_dist_thresh = 0.3
		self.dist_thresh_horizon = 1.2
		# set the next target if velocity is less than these values:
		self.stop_v_thresh = 0.06 
		self.stop_w_thresh = np.deg2rad(4)

		# MPC settings
		self.horizon = 5

		# PID settings
		pid_linear = {'kp': 0.5, 'kd': 0.1, 'ki': 0}
		pid_angular = {'kp': 3.0, 'kd': 0.1, 'ki': 0}
		pid_params = {'linear':pid_linear, 'angular':pid_angular}

		# create controller
		if self.controller_name == "PID":
			self.controller = PID(pid_params)
		elif self.controller_name == "MPC":			
			self.controller = MPC(self.dt, horizon = self.horizon)

		# recorded trajectory
		self.rec_traj_x = []
		self.rec_traj_y = []
		self.rec_traj_yaw = []
		self.rec_l = 0
		self.rec_t = 0
		self.rec_w = 0
		self.prev_w = 0

    # --------------------------------------- control ---------------------------------------

	def control(self):
		k = self.horizon
		goal_dist = utils.distance(self.goal_x, self.goal_y, self.robot.x, self.robot.y)

		# starting index
		if self.from_beggins:
			current_idx = 0
		else:
			current_idx, dists = self.find_nearest_ind(self.robot.pose)

		while (current_idx < self.ref_traj.count):
			if goal_dist<self.goal_dist_thresh and current_idx>self.ref_traj.count*0.8:
				print("Goal dist")

			# update index and horizon
			_, dists = self.find_nearest_ind(self.robot.pose)
			ind = np.argmin(dists[current_idx:current_idx+self.horizon])
			current_idx = current_idx+ind 
			min_dist = dists[current_idx]

			while min_dist<self.dist_thresh and current_idx!= len(dists)-1:
				current_idx+= 1
				k = min(self.horizon, len(dists)-current_idx)
				min_dist = min(dists[current_idx:current_idx+k])

			# # m4
			current_v, _ = self.robot.get_velocity_vec()
			current_l = self.dist_thresh #current_v * self.dt
			dists_horizon = dists[current_idx:current_idx+self.horizon]
			i = 0
			dm = 0
			ref_ind = [current_idx]
			while len(ref_ind)<self.horizon and i<len(dists_horizon):
				dm += current_l
				if dm<dists_horizon[i]:
					pass
				else:
					i+=1
				ii = min(current_idx+i, self.ref_traj.count-1)
				ref_ind.append(ii)
			ref_path = [self.ref_traj.xy_poses[i] for i in ref_ind]
			ref_vel = [self.ref_traj.v[i] for i in ref_ind]
			
			print("current_idx: ", current_idx)
			print("ref_path len: ", len(ref_path))
			print("ref_ind: ", ref_ind)

			# robot pose and lookahead point
			lookahead_point = self.ref_traj.get_pose_vec(current_idx)
			self.lookahead_point = lookahead_point

			# calculate velocity
			if self.controller_name == "PID":
				cmd_v, cmd_w = self.controller.get_control_inputs(self.robot.pose, lookahead_point) # self.robot.get_points()[2]
			
			if self.controller_name == "MPC":
				cmd_v, cmd_w = self.controller.optimize(robot = self.robot, points = ref_path, vels = ref_vel)

			# check vel
			if cmd_v<self.stop_v_thresh and abs(cmd_w)<self.stop_w_thresh:
				current_idx += 1
				# continue

			cmd_v = min(max(cmd_v, self.robot.v_min), self.robot.v_max)
			cmd_w = min(max(cmd_w, self.robot.w_min), self.robot.w_max)

			# update
			self.robot.set_robot_velocity(cmd_v, cmd_w)
			self.robot.update_sim(self.dt)
			self.robot.update_robot(self.robot.get_model_pose())
			goal_dist = self.update(cmd_v, cmd_w)
			
	# -------------------------------------- update --------------------------------------

	def update(self, cmd_v, cmd_w):
		# goal distance
		goal_dist = utils.distance(self.goal_x, self.goal_y, self.robot.x, self.robot.y)

		# record trajectory
		self.rec_traj_x.append(self.robot.x)
		self.rec_traj_y.append(self.robot.y)
		self.rec_traj_yaw.append(self.robot.yaw)
		self.rec_l += cmd_v * self.dt
		self.rec_t += self.dt
		self.rec_w += abs(cmd_w-self.prev_w)
		self.prev_w = cmd_w

		# direction vector [dx_th, dy_th]
		dx_th = np.cos(self.robot.yaw)
		dy_th = np.sin(self.robot.yaw)
		
		# update plot
		self.plotting.update_plot(dx_th, dy_th, self.robot.pose, self.lookahead_point)
		
		# result
		print("cmd_v:", round(cmd_v, 2), "cmd_w:", round(cmd_w, 2))
		print(" -------------------------------------- ")
		
		return goal_dist
	
	def find_nearest_ind(self, pose):
		dists = [utils.distance(pose[0], pose[1], p[0], p[1]) for p in self.ref_traj.xy_poses]
		dists = np.array(dists)
		idx = np.argmin(dists)
		return idx, dists

# -------------------------------------- PID --------------------------------------

class PID:
	def __init__(self, pid_params):
		self.kp_linear = pid_params['linear']['kp']
		self.kd_linear = pid_params['linear']['kd']
		self.ki_linear = pid_params['linear']['ki']

		self.kp_angular = pid_params['angular']['kp']
		self.kd_angular = pid_params['angular']['kd']
		self.ki_angular = pid_params['angular']['ki']

		self.error_ang_last = 0
		self.error_lin_last = 0

		# self.prev_body_to_goal = 0
		# self.prev_waypoint_idx = -1


	def get_control_inputs(self, current_pose, goal_x):
		error_position = utils.distance(current_pose[0], current_pose[1], goal_x[0], goal_x[1])
		
		body_to_goal = np.arctan2(goal_x[1]- current_pose[1], goal_x[0] - current_pose[0])
		error_angle = utils.angle_diff(body_to_goal, current_pose[2])

		linear_velocity_control = self.kp_linear*error_position + self.kd_linear*(error_position - self.error_lin_last)
		angular_velocity_control = self.kp_angular*error_angle + self.kd_angular*(error_angle - self.error_ang_last)

		self.error_ang_last = error_angle
		self.error_lin_last = error_position

		# self.prev_waypoint_idx = waypoint_idx
		# self.prev_body_to_goal = body_to_goal

		if linear_velocity_control>5:
			linear_velocity_control = 5

		return linear_velocity_control, angular_velocity_control

# -------------------------------------- MPC --------------------------------------

class MPC:
	def __init__(self, dt, horizon):
		self.dt = dt
		self.horizon = horizon
		self.R = np.diag([0.01, 0.01])		# input cost matrix
		self.Rd = np.diag([0.01, 0.01])		# input difference cost matrix
		self.Q = np.diag([1.0, 1.0])		# state cost matrix
		self.Qf = self.Q					# state final matrix
		self.H = 0.5						# heading cost matrix
		self.CVW = 0.1
		self.CV = 0.01

	def cost(self, u_k, robot, path, vels):
		path = np.array(path)
		controller_robot = deepcopy(robot)
		u_k = u_k.reshape(self.horizon, 2).T
		z_k = np.zeros((2, self.horizon+1))

		desired_state = path.T

		cost = 0.0
		C = 1
		for i in range(self.horizon):
			controller_robot.set_robot_velocity(u_k[0,i], u_k[1,i])
			controller_robot.update_sim(self.dt)
			current_pose, _ = controller_robot.get_state()
			z_k[:,i] = [current_pose[0, 0], current_pose[1, 0]]
			h = controller_robot.get_los(path[i])

			if i ==0:
				C = 1
			cost += self.H*utils.angle_diff(h, current_pose[2, 0])**2
			cost += self.CVW * (u_k[0,i]*u_k[1,i])**2
			cost += self.CV*(u_k[0,i]-vels[i])**2

			cost += np.sum(np.dot(self.R, u_k[:,i]**2))  			   			#	np.sum(self.R@(u_k[:,i]**2))
			cost += C*np.sum(np.dot(self.Q, desired_state[:,i]-z_k[:,i])**2) 	#	np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
			if i < (self.horizon-1):     
				cost += np.sum(np.dot(self.Rd, u_k[:,i+1] - u_k[:,i])**2)  		#	np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))

		return cost

	def optimize(self, robot, points, vels):
		self.horizon = len(points)
		bnd = [(0.0, 0.5),(np.deg2rad(-60), np.deg2rad(60))]*self.horizon
		result = minimize(self.cost, args=(robot, points, vels), x0 = np.zeros((2*self.horizon)), method='SLSQP', bounds = bnd)
		return result.x[0],  result.x[1]
