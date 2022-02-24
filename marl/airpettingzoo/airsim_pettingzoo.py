from gym.spaces import Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import from_parallel
from gym import spaces
import numpy as np
import airsim
import time
import threading
from multiprocessing import Process
from torch import _euclidean_dist

class MultiDroneEnv(ParallelEnv):
    metadata = {'render.modes':  ["rgb_array"]}

    def __init__(self, step_length, image_shape, no_of_agents, target_position):

        self.step_length = step_length # todo: re-modify this
        self.image_shape = image_shape
        self.target_position = target_position
        self.no_of_agents = no_of_agents
        self.possible_agents = ["Drone" + str(i) for i in range(self.no_of_agents)]
        self.start_postions = [airsim.Vector3r(4*i, 0, 0) for i in range(1,self.no_of_agents+1)] # todo: automate this
        self.similarity_window = 5
        self.similarity_metrices = [None for i in range(self.similarity_window+1)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.state_space = {
            agent: {
                # "depth_image":spaces.Box(0, 255, shape=image_shape, dtype=np.uint8),
                "position": np.zeros(3),
                "velocity": np.zeros(3) 
                } for agent in self.possible_agents}
        self.drone = airsim.MultirotorClient()
        self.action_spaces = {agent: Discrete(7) for agent in self.possible_agents}

        # reward configs
        self.target_radius_inner = 0.5
        self.target_radius_outer = 2.0
        self.inter_drone_proximity_threshold = 2.0
        self.max_velocity = airsim.Vector3r(5, 5, 5) 
 

    def get_pos_NED(self, agent, abs_pos):
        """
        returns the position vector of the agent w.r.t to world coordinate frame in NED coordinate system
        """
        start_pos = self.start_postions[self.agent_name_mapping[agent]]
        return start_pos - abs_pos

    def get_position_wrt_target(self,abs_pos, target):
        """
        returns the x,y position vectors and eucledian distance (D) w.r.t to target
        """
        X = np.zeros(3) #[x, y, D]
        X[0] = target.x_val-abs_pos.x_val
        X[1] = target.y_val-abs_pos.y_val
        X[2] = np.linalg.norm(target.to_numpy_array() - abs_pos.to_numpy_array()) # eucledian distance between agent and target
        return X

    def _move_to_init_single_drone(self, agent: str):
        print(f"Moving {agent} to start position...")
        self.drone.moveToPositionAsync(0.0, 0.0, -8, 5, vehicle_name=agent)

    def _setup_flight(self):
        # todo: implement randomized starting positions for drones 
        self.drone.reset()

        self.agents = self.possible_agents[:]

        for agent in self.agents:
            print(f"Setting up {agent}...")
            self.drone.enableApiControl(True, agent)
            self.drone.armDisarm(True, agent)
            print(f"Finished setting up {agent}...")

        threads = []
        for agent in self.agents:
            t = threading.Thread(target=self._move_to_init_single_drone, args=(agent,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()  
    
        for agent in self.agents:
            self.drone_state = self.drone.getMultirotorState(vehicle_name=agent)
            self.state_space[agent]["position"] = self.get_position_wrt_target(self.get_pos_NED(agent, self.drone_state.kinematics_estimated.position), self.target_position)
            
        self.image_request = airsim.ImageRequest(0, airsim.ImageType.DepthVis)
        time.sleep(5)
        print("Finished setting up drones...")


    def transform_obs(self, responses):
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width,3))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_temp = image.resize((self.image_shape[0], self.image_shape[1]))
        im_temp = np.mean(np.array(im_temp), axis=2).astype(np.uint8)
        im_final = np.reshape(im_temp, (self.image_shape[0],self.image_shape[1],1))
        return im_final

    def get_similarity_matrix(self):
        """Returns the similarity matrix S based on euclidean distance metric"""
        S = np.zeros([self.no_of_agents,self.no_of_agents])
        similarity_matrix = np.zeros([self.no_of_agents,self.no_of_agents])
        
        for i in range(self.no_of_agents):
            for j in range(self.no_of_agents):
                pi = self.state_space["Drone" + str(i)]["abs_position_NED"]
                pj = self.state_space["Drone" + str(j)]["abs_position_NED"]
                S[i][j] = np.linalg.norm(pi.to_numpy_array()-pj.to_numpy_array())
        # max_d_ij = S.max()
        # for i in range(self.no_of_agents):
        #     for j in range(self.no_of_agents):
        #         similarity_matrix[i][j] = 1-(S[i][j]/max_d_ij)
        return S


    def _get_obs(self):

        observations = {agent: 'NONE' for agent in self.agents}
        print("test2")
        for agent in self.agents:
            responses = self.drone.simGetImages([self.image_request], vehicle_name=agent)
            # self.state_space[agent]["depth_image"] = responses

            self.drone_state = self.drone.getMultirotorState(vehicle_name=agent)
            self.state_space[agent]["abs_position_NED"] = self.get_pos_NED(agent, self.drone_state.kinematics_estimated.position)
            self.state_space[agent]["prev_position"] = self.state_space[agent]["position"]
            self.state_space[agent]["position"] = self.get_position_wrt_target(self.state_space[agent]["abs_position_NED"], self.target_position)
            self.state_space[agent]["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

            self.state_space[agent]["collision"] = self.drone.simGetCollisionInfo(vehicle_name=agent).has_collided
            observations[agent] = self.state_space[agent]
        observations["similarity_matrix"] = self.get_similarity_matrix()
        _ = self.similarity_metrices.pop(0)
        self.similarity_metrices.append(observations["similarity_matrix"])

        return observations

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def _do_action_single_drone(self, agent, action):

        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState(vehicle_name=agent).kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
            vehicle_name=agent
        )

    # def _move_to_init_single_drone(self, agent: str):
    #     print(f"Moving {agent} to start position...")
    #     self.drone.moveToPositionAsync(self.agent_name_mapping[agent]*1, 0.0, -16, 5, vehicle_name=agent)

    def step(self, actions):

        threads = []
        for agent in self.agents:
            # self._move_to_init_single_drone(agent)
            # self._do_action_single_drone(agent, actions[self.agent_name_mapping[agent]])
            t = threading.Thread(target=self._do_action_single_drone, args=(agent, actions[self.agent_name_mapping[agent]],))
            threads.append(t)
            t.start()

        for t in threads:
            t.join() 

        obs = self._get_obs()
        rewards, dones = self._compute_reward()
        print("action taken")
        return obs, rewards, dones, self.state_space

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset


    def target_reached(self, agent):
        # eucledian distance to the target from agent's position
        euclidean_dist_target = self.state_space[agent]["position"][2]

        # checks whether the drone reached the target or not
        if euclidean_dist_target <= self.target_radius_inner:
            return True
        return False

    def has_collided(self, agent):
        # checks if there is a collision
        if self.state_space[agent]["collision"]:
            return True
        return False

    def inter_drone_proximity(self, agent):
        """Dp is the distance between the agent and other closeby agents which are likely to collide."""
        
        Dp = 0
        # eucledian distances to other drones w.r.t to the given drone
        drone_eucledian_distances = self.observations["similarity_matrix"][self.agent_name_mapping[agent]]
        for i in range(self.no_of_agents):
            if i == self.agent_name_mapping[agent]:
                continue
            elif drone_eucledian_distances[i] <= self.inter_drone_proximity_threshold:
                Dp += drone_eucledian_distances[i]

        return Dp

    def velocity_correction(self, agent):
        """Dc is the velcoity correction which is applied to penalize the agent if it chooses actions which
        speed up the agent away from the target."""
        # eucledian distance to the target from agent's position
        euclidean_dist_target = self.state_space[agent]["position"][2]

        # gamma is a binary variable which is set to ‘1’ if the agent is closer to the target.
        gamma = 1 if euclidean_dist_target <= self.target_radius_outer else 0

        # calculate Velocity difference between V_max and V_now
        V_diff = np.linalg.norm(self.state_space[agent]["velocity"].to_numpy_array()-self.max_velocity.to_numpy_array()) 
        
        Dc = gamma*V_diff

        return Dc

    def getting_closer_to_target(self, agent):
        """Dg is the distance to the goal at any time steps from the agents' current position w.r.t prev. position.
        If the agent is going away from the target, the distance to the target increases thus penalizing
        the agent."""
        # eucledian distance to the target from agent's position
        Dg = self.state_space[agent]["position"][2] - self.state_space[agent]["prev_position"][2]
        return Dg

    def structure_formation(self):
        """Ds is the similarity between the drone's structure w.r.t to previuos time steps.
        If the agents are maintaining theior prescribed structure over time, they will get postive reward."""   

        # similarity matrix at time t=t
        St = self.observations["similarity_matrix"]                                                                                     

        # S_avg is the avg. approx. of similarity matrices of last W sized time window.
        S_avg = sum(self.similarity_metrices[:-1])/self.similarity_window  

        # calculate similarity between the drone's structure w.r.t to previuos time steps
        Ds = np.linalg.norm(St - S_avg)
        return Ds 

    def _compute_reward(self):
        
        rewards = []
        dones = []

        # penalise the agents for maintaining their prescribed structure
        Ds = self.structure_formation()

        for agent in self.agents:
            reward_agent = 0
            done = 0
            # alpha is a binary variable where ‘1’ denotes if the target is reached else it is ‘0’.
            alpha = self.target_reached(agent)

            # beta is a binary variable where ‘1’ denotes if there is a collision with walls, obstacles or ground else it is ‘0’.
            beta = self.has_collided(agent)
        
            # penalize the agent if it get close to other drones
            Dp = self.inter_drone_proximity(agent)
            Dp = 1/Dp if Dp > 0 else 0

            # penalize the agent if it chooses actions which speed up the agent away from the target
            Dc = self.velocity_correction(agent)

            # penalize the agent the distance to the target increases
            Dg = self.getting_closer_to_target(agent) 

            reward_agent = 100*alpha - 100*beta - 10*Dp + Dc - Dg - Ds

            if alpha: done = 1
            if beta : done = -1

            rewards.append(reward_agent)
            dones.append(done)
        
        return rewards, dones
