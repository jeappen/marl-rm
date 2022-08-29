import random
from sklearn.neighbors import NearestNeighbors
from matplotlib.pyplot import gca
from gym.utils import seeding
from gym import spaces
from ray.rllib.env import MultiAgentEnv
import matplotlib.pyplot as plt
import random
import numpy as np
from enum import Enum

from rm_cooperative_marl.src.reward_machines.sparse_reward_machine import SparseRewardMachine

"""
Enum with the actions that the agent can execute
"""


class Actions(Enum):
    up = 0  # move up
    right = 1  # move right
    down = 2  # move down
    left = 3  # move left
    none = 4  # none


class GridWorldEnv:

    def __init__(self, rm_file, agent_id, env_settings):
        """
        Initialize gridworld environment.

        Parameters
        ----------
        rm_file : string
            File path leading to the text file containing the reward machine
            encoding this environment's reward function.
        agent_id : int
            Index {0,1} indicating which agent
        env_settings : dict
            Dictionary of environment settings
        """
        self.env_settings = env_settings
        self.agent_id = agent_id
        self._load_map()
        self.reward_machine = SparseRewardMachine(rm_file)

        self.u = self.reward_machine.get_initial_state()
        self.last_action = -1  # Initialize last action to garbage value

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.Nr = self.env_settings['Nr']
        self.Nc = self.env_settings['Nc']

        initial_states = self.env_settings['initial_states']

        self.s_i = initial_states[self.agent_id - 1]
        self.objects = {}
        # rendezvous location
        self.objects[self.env_settings['rendezvous_loc']] = "w"
        goal_locations = self.env_settings['goal_locations']
        self.objects[goal_locations[self.agent_id - 1]] = "g"  # goal location

        self.p = self.env_settings['p']

        self.num_states = self.Nr * self.Nc

        self.actions = [Actions.up.value, Actions.right.value,
                        Actions.left.value, Actions.down.value, Actions.none.value]

        # Define forbidden transitions corresponding to map edges
        self.forbidden_transitions = set()

        for row in range(self.Nr):
            # If in left-most column, can't move left.
            self.forbidden_transitions.add((row, 0, Actions.left))
            # If in right-most column, can't move right.
            self.forbidden_transitions.add((row, self.Nc - 1, Actions.right))
        for col in range(self.Nc):
            # If in top row, can't move up
            self.forbidden_transitions.add((0, col, Actions.up))
            # If in bottom row, can't move down
            self.forbidden_transitions.add((self.Nr - 1, col, Actions.down))

    def environment_step(self, s, a):
        """
        Execute action a from state s.

        Parameters
        ----------
        s : int
            Index representing the current environment state.
        a : int
            Index representing the action being taken.

        Outputs
        -------
        r : float
            Reward achieved by taking action a from state s.
        l : list
            List of events occuring at this step.
        s_next : int
            Index of next state.
        """
        s_next, last_action = self.get_next_state(s, a)
        self.last_action = last_action

        l = self.get_mdp_label(s, s_next, self.u)  # Can be outside the env
        r = 0

        for e in l:
            # Get the new reward machine state and the reward of this step
            u2 = self.reward_machine.get_next_state(self.u, e)
            r = r + self.reward_machine.get_reward(self.u, u2)
            # Update the reward machine state
            self.u = u2

        return r, l, s_next

    def get_mdp_label(self, s, s_next, u):
        """
        Has 3 stages for start, rendezvous and goal.
        """
        row, col = self.get_state_description(s)
        row_next, col_next = self.get_state_description(s_next)

        l = []

        thresh = 0.3

        if u == 0 and (row_next, col_next) in self.objects:
            if self.objects[(row_next, col_next)] == 'w':
                l.append('r{}'.format(self.agent_id))
        elif u == 1:
            if not((row_next, col_next) in self.objects):
                l.append('l{}'.format(self.agent_id))
            elif self.objects[(row_next, col_next)] == 'w' and (row, col) in self.objects:
                if self.objects[(row, col)] == 'w' and np.random.random() <= thresh:
                    l.append('r')
        elif u == 2:
            if (row_next, col_next) in self.objects:
                if self.objects[(row_next, col_next)] == 'g':
                    l.append('g{}'.format(self.agent_id))
        return l

    def get_next_state(self, s, a):
        """
        Get the next state in the environment given action a is taken from state s.
        Update the last action that was truly taken due to MDP slip.

        Parameters
        ----------
        s : int
            Index of the current state.
        a : int
            Action to be taken from state s.

        Outputs
        -------
        s_next : int
            Index of the next state.
        last_action :int
            Last action taken by agent due to slip proability.
        """
        slip_p = [self.p, (1 - self.p) / 2, (1 - self.p) / 2]
        check = random.random()

        row, col = self.get_state_description(s)

        # up    = 0
        # right = 1
        # down  = 2
        # left  = 3

        if (check <= slip_p[0]) or (a == Actions.none.value):
            a_ = a

        elif (check > slip_p[0]) & (check <= (slip_p[0] + slip_p[1])):
            if a == 0:
                a_ = 3
            elif a == 2:
                a_ = 1
            elif a == 3:
                a_ = 2
            elif a == 1:
                a_ = 0

        else:
            if a == 0:
                a_ = 1
            elif a == 2:
                a_ = 3
            elif a == 3:
                a_ = 0
            elif a == 1:
                a_ = 2

        action_ = Actions(a_)
        if (row, col, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                row -= 1
            if action_ == Actions.down:
                row += 1
            if action_ == Actions.left:
                col -= 1
            if action_ == Actions.right:
                col += 1

        s_next = self.get_state_from_description(row, col)

        last_action = a_
        return s_next, last_action

    def get_state_from_description(self, row, col):
        """
        Given a (row, column) index description of gridworld location, return
        index of corresponding state.

        Parameters
        ----------
        row : int
            Index corresponding to the row location of the state in the gridworld.
        col : int
            Index corresponding to the column location of the state in the gridworld.

        Outputs
        -------
        s : int
            The index of the gridworld state corresponding to location (row, col).
        """
        return self.Nc * row + col

    def get_state_description(self, s):
        """
        Return the row and column indeces of state s in the gridworld.

        Parameters
        ----------
        s : int
            Index of the gridworld state.

        Outputs
        -------
        row : int
            The row index of state s in the gridworld.
        col : int
            The column index of state s in the gridworld.
        """
        row = np.floor_divide(s, self.Nr)
        col = np.mod(s, self.Nc)

        return (row, col)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.last_action

    def get_initial_state(self):
        """
        Outputs
        -------
        s_i : int
            Index of agent's initial state.
        """
        return self.s_i

    def show(self, s):
        """
        Create a visual representation of the current state of the gridworld.

        Parameters
        ----------
        s : int
            Index of the current state
        """
        display = np.zeros((self.Nr, self.Nc))

        # Display the location of key points in world
        for loc in self.objects.keys():
            display[loc] = 1

        # Display the location of the agent in the world
        row, col = self.get_state_description(s)
        display[row, col] = np.nan

        print(display)


# Colors from SSD
DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

# From Arbaaz gym flock, deterministic and 10 agents initially


class NavEnvRMLabelling(MultiAgentEnv):
    """ Multiple agents moving in a Linear System with waypoints provided as observation
        Goal to distribute evenly

        Has a labelling function for use with reward machines. No reward otherwise.
    """

    def __init__(self, n_agents=3, time_horizon=400, end_when_goal_reached=False, end_when_area_exited=True, env_settings=None):

        self.dynamic = True  # if the agents are moving or not
        # normalize the adjacency matrix by the number of neighbors or not
        self.mean_pooling = False
        # self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 0
        self.n_agents = n_agents
        # number of goals, visible to all agents for now
        self.n_goals = 3
        # goal range (agents within this region have reached the goal)
        self.goal_range = 0.2
        # goal distribution : how many goals are meant to be used
        self.goals_used = self.n_goals - 1
        # Constant used in reward (fix this)
        self.goals_over_limit_penalty_const = -1

        # number states per agent
        self.nx_system = 4
        # number of features per agent
        self.n_features = self.nx_system  # *self.n_goals # Assuming one goal for now
        # number of actions per agent
        self.nu = 2

        # Time horizon to set at start
        self.time_horizon = time_horizon

        # To end when the goal is reached (False for custom goals)
        self.end_when_goal_reached = end_when_goal_reached

        # Maximum boundary to end trajectory
        # End run if agent goes beyond this region
        self.end_when_area_exited = end_when_area_exited
        self.x_bound = 1000

        # problem parameters from file

        self.comm_radius = float(2.0)
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(0.01)
        self.v_max = float(2.0)
        self.v_bias = self.v_max
        self.r_max = float(6.0)
        self.std_dev = float(0.1) * self.dt

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))

        self.a_net = np.zeros((self.n_agents, self.n_agents))

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1
        self.gain = 1.0  # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_features,),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.counter = 0
        self.seed()

        self.agents = [f'agent-{_i}' for _i in range(self.n_agents)]

        # RM stuff
        # TODO: maybe move somewhere else?
        self.inf_norm_bound = 1.0
        self.env_settings = env_settings
        self._load_map()

    def _load_map(self):
        """
        Initialize the environment.
        """
        self.objects = {}
        # rendezvous location
        self.objects[self.env_settings['rendezvous_loc']] = "w"
        goal_locations = self.env_settings['goal_locations']
        for a in range(self.n_agents):
            self.objects[goal_locations[a]] = "g"  # goal location
            # TODO : Consider different goals for each agent. Now assuming same goal (only first goal)
            break

        self.p = self.env_settings['p']

    def get_state_description(self, s):
        # Gets x,y of agent
        return s[:2]

    def difference_goal(self, old, new, goal):
        # Rewards heading towards goal (>0 when closer, <0 when further)
        return -(np.linalg.norm(new - np.array(goal), ord=np.Inf) -
                 np.linalg.norm(old - np.array(goal), ord=np.Inf))

    def get_continuous_rw(self, s, s_next, u, agent_id=None, rw_bound=-2, next_u=None, difference_rw=False):
        xy = self.get_state_description(s[agent_id])
        xy_next = self.get_state_description(s_next[agent_id])
        rew = 0
        rw_norm_bound = 0  # can set to self.inf_norm_bound but ruins rw shaping
        if u == 0 or u == 1:
            # reward getting to first rendezvous
            rew = rw_norm_bound - np.linalg.norm(xy_next - np.array(self.env_settings['rendezvous_loc']),
                                                 ord=np.Inf)
            if rew < 0 and difference_rw:
                # not in goal region yet
                rew = self.difference_goal(
                    xy, xy_next, self.env_settings['rendezvous_loc'])
        elif u == 2:
            # reward getting to goal after all agents sync @ rendezvous
            rew = (rw_norm_bound - np.linalg.norm(xy_next - np.array(self.env_settings['goal_locations'][0]),
                                                  ord=np.Inf))
            if rew < 0 and difference_rw:
                # not in goal region yet
                rew = self.difference_goal(
                    xy, xy_next, self.env_settings['goal_locations'][0])
        if next_u == 3:
            # reward reaching end of task
            rew = 20

        # Normalize reward before returning
        rew = max(rw_bound, rew)
        # print("\n navenv : {} getting rw {}".format(agent_id, rew))
        return rew / self.n_agents

    def get_total_continuous_rw(self, s, s_next, u):
        # NOT USED
        raise NotImplementedError

    def get_mdp_label(self, s, s_next, u, agent_id=None, thresh=0.3):
        """
        Has 3 stages for start, rendezvous and goal.

        s: current step state dict
        s_next: next step state dict
        u: monitor state \in {0, 1, 2}
        agent_id: (str) to pick out which state and goal to use eg: 'agent-0'
        thresh: probability of assuming all agents sync rendezvous

        Needs to be able to get the individual labels for each agent's reward machine.
        """
        xy = self.get_state_description(s[agent_id])
        xy_next = self.get_state_description(s_next[agent_id])
        # Some people don't know what 0-indexing is :(
        agent_ind = int(agent_id[-1]) + 1
        # object label is True if the agent is nearby the object.
        object_label = {}
        object_label_next = {}
        for o_key in self.objects:
            if (self.objects[o_key] == "w" or self.objects[o_key] == "g"):
                # Uses inf. norm ball around object to decide closeness.
                object_label[o_key] = self.inf_norm_bound > np.linalg.norm(xy - np.array(o_key),
                                                                           ord=np.Inf)
                object_label_next[o_key] = self.inf_norm_bound > np.linalg.norm(xy_next - np.array(o_key),
                                                                                ord=np.Inf)

        l = []

        if any(object_label_next.values()):
            # Close to some object in next state
            for o_key in object_label_next:
                # Find which object the agent is close to
                if object_label_next[o_key]:
                    if self.objects[o_key] == 'w':
                        if u == 0:
                            # Near rendezvous for the first time
                            l.append('r{}'.format(agent_ind))
                        elif u == 1:
                            if object_label[o_key] and np.random.random() <= thresh:
                                # Implies rendezvous has been 'synced' among all agents.
                                l.append('r')
                    elif self.objects[o_key] == 'g':
                        if u == 2:
                            # Near goal
                            l.append('g{}'.format(agent_ind))
        else:
            # Not near any object
            # TODO: Check if need to do only when u == 1
            l.append('l{}'.format(agent_ind))
        return l

    def get_mdp_label_multi(self, s, s_next, u=None):
        """
        Get the mdp label resulting from transitioning from state s to state s_next for all agents.

        Parameters
        ----------
        s : numpy integer array
            Array of integers representing the environment states of the various agents.
            s[id] represents the state of the agent indexed by index "id".
        s_next : numpy integer array
            Array of integers representing the next environment states of the various agents.
            s_next[id] represents the next state of the agent indexed by index "id".
        u : int (not used)
            Index of the reward machine state

        Outputs
        -------
        l : string
            MDP label resulting from the state transition.
        """
        agent_all_on_wait = True

        l = []

        for i in range(self.n_agents):
            agent_id = self.agents[i]
            xy = self.get_state_description(s[agent_id])
            xy_next = self.get_state_description(s_next[agent_id])

            # object label is True if the agent is nearby the object.
            object_label = {}
            object_label_next = {}
            for o_key in self.objects:
                if (self.objects[o_key] == "w" or self.objects[o_key] == "g"):
                    # Uses inf. norm ball around object to decide closeness.
                    object_label[o_key] = self.inf_norm_bound > np.linalg.norm(xy - np.array(o_key),
                                                                               ord=np.Inf)
                    object_label_next[o_key] = self.inf_norm_bound > np.linalg.norm(xy_next - np.array(o_key),
                                                                                    ord=np.Inf)
            if any(object_label.values()):
                # Check if near an object and that object is not the rendezvous point (over all agents separately)
                for o_key in object_label:
                    if object_label[o_key] and self.objects[o_key] != 'w':
                        agent_all_on_wait = False
            else:
                agent_all_on_wait = False

            if any(object_label_next.values()):
                for o_key in object_label_next:
                    if object_label_next[o_key] and self.objects[o_key] == 'w':
                        l.append('r{}'.format(i + 1))
                    elif object_label_next[o_key] and self.objects[o_key] == 'g':
                        l.append('g{}'.format(i + 1))
            else:
                # Not near any object in the next step
                l.append("l{}".format(i + 1))

        if agent_all_on_wait:
            # If all agents are on wait space, only return r
            l = []
            l.append('r')
        return l

    def get_a_net(self):
        return self.a_net

    def get_agent_index(self, agent):
        return self.agents.index(agent)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):

        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = action  # self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        action = np.hstack(list(agent_actions.values()))
        self.u = np.reshape(action, (self.n_agents, self.nu))

        self.x[:, 0] = self.x[:, 0] + self.u[:, 0] * 0.1
        # update  y position
        self.x[:, 1] = self.x[:, 1] + self.u[:, 1] * 0.1

        done = False
        self.counter += 1

        # diffs_x = np.abs(self.x[:,0] - self.goal_xpoints)
        # diffs_y = np.abs(self.x[:,1] - self.goal_ypoints)

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        dist2goal = (x_dis ** 2 + y_dis ** 2)**0.5
        near_goal = dist2goal < self.goal_range

        if self.end_when_area_exited:
            # Check if each agent  is within the active area -x_bound,x_bound)
            done = (self.x[:, 0:2] > self.x_bound).any() or (
                self.x[:, 0:2] < -self.x_bound).any()

        if self.counter >= self.time_horizon:
            done = True
        if self.end_when_goal_reached and near_goal.any(axis=0).all():
            # Check if each agent (all) is near some goal (any)
            done = True

        # centralized reward for now (not agent-specific)
        rewards = {a: self.instant_cost() for a in self.agents}
        dones = {a: done for a in self.agents}
        dones['__all__'] = done

        return self._turn_mat_to_MA_dict(self._get_obs()), rewards, dones, {}

    def instant_cost(self):
        # Cost to reach goal
        # for even goal distribution, check environment below

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        robot_xs = self.x[:, 0]
        robot_ys = self.x[:, 1]

        robot_goalxs = self.x[:, 2]
        robot_goalys = self.x[:, 3]

        diff = ((robot_xs - robot_goalxs)**2 +
                (robot_ys - robot_goalys)**2)**0.5
        return -np.sum(diff)

        # self.feats[:,::2] = x_dis.T
        # self.feats[:,1::2] = y_dis.T

        # sum of differences in velocities

        # robot_xs = self.x[:,0]
        # robot_ys = self.x[:,1]

        # robot_goalxs = self.x[:,2]
        # robot_goalys = self.x[:,3]

        # diff = ((robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2)**0.5
        # return -np.sum(diff)

    def reset(self):
        # TODO: Change this to self.nx_system (after using all old checkpoint data)
        # keep this to track position
        x = np.zeros((self.n_agents, self.nx_system))
        # this is the feature we return to the agent
        self.feats = np.zeros((self.n_agents, self.n_features))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25

        self.counter = 0

        # set arbitrary goal
        self.goal_x1 = 0
        self.goal_y1 = np.random.uniform(2, 3)

        self.goal_x2 = -2
        self.goal_y2 = np.random.uniform(2, 3)

        # self.goal_x3 = -4
        # self.goal_y3 = np.random.uniform(2,3)

        # self.goal_x4 = -6
        # self.goal_y4 = np.random.uniform(2,3)

        # self.goal_x5 = -8
        # self.goal_y5 = np.random.uniform(2,3)

        self.goal_x6 = 2
        self.goal_y6 = np.random.uniform(2, 3)

        # self.goal_x7 = 4
        # self.goal_y7 = np.random.uniform(2,3)

        # self.goal_x8 = 6
        # self.goal_y8 = np.random.uniform(2,3)

        # self.goal_x9 = 8
        # self.goal_y9 = np.random.uniform(2,3)

        # self.goal_x10 = 10
        # self.goal_y10 = np.random.uniform(2,3)

        n = self.n_agents
        xpoints = np.linspace(-2 * ((n) // 2), 2 * ((n) // 2),
                              n if n % 2 else n + 1)[:self.n_agents]
        #ypoints = np.array((0,0,0,0,0))

        ypoints = np.array((np.random.uniform(-1, 0, self.n_agents)))

        self.start_xpoints = xpoints
        self.start_ypoints = ypoints

        self.goal_xpoints = np.array((self.goal_x1, self.goal_x2, self.goal_x6))[
            :self.n_goals]
        # self.goal_x2,self.goal_x3,self.goal_x4,self.goal_x5,self.goal_x6,self.goal_x7,self.goal_x8,self.goal_x9,self.goal_x10))
        self.goal_ypoints = np.array((self.goal_y1, self.goal_y2, self.goal_y6))[
            :self.n_goals]
        # self.goal_y2,self.goal_y3,self.goal_y4,self.goal_y5,self.goal_y6,self.goal_y7,self.goal_y8,self.goal_y9,self.goal_y10))

        x[:, 0] = xpoints  # - self.goal_xpoints
        x[:, 1] = ypoints  # - self.goal_ypoints

        #x[:,2] = np.array((self.goal_x1,self.goal_x2,self.goal_x3,self.goal_x4,self.goal_x5))
        #x[:,3] = np.array((self.goal_y1,self.goal_y2,self.goal_y3,self.goal_y4,self.goal_y5))
        num_goals_used = min(self.n_goals, self.n_agents)
        x[:self.n_goals, 2] = self.goal_xpoints[:num_goals_used]
        x[:self.n_goals, 3] = self.goal_ypoints[:num_goals_used]
        # compute distances between agents
        a_net = self.dist2_mat(x)

        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))

        self.x = x

        self.a_net = self.get_connectivity(self.x)

        return self._turn_mat_to_MA_dict(self._get_obs())

    def _turn_mat_to_MA_dict(self, matrix):
        """ Turns a matrix [n_agent* N] to the dict format for rllib Multiagent
        """
        output = {}
        for i, a in enumerate(self.agents):
            output[a] = matrix[i, :]
        return output

    def _get_obs(self):

        # Get matrix of displacements to all goals (row: # goals, col: # agents)
        x_dis = self.x[:, 0] - self.x[:self.n_goals, 2].reshape(1, -1).T
        y_dis = self.x[:, 1] - self.x[:self.n_goals, 3].reshape(1, -1).T
        if self.feats.shape[1] == 2:
            # Means using just one goal as feature
            self.feats[:, 0] = np.diag(x_dis)
            self.feats[:, 1] = np.diag(y_dis)
        if self.feats.shape[1] == self.nx_system:
            self.feats = self.x
        else:
            self.feats[:, ::2] = x_dis.T
            self.feats[:, 1::2] = y_dis.T

        # displacement to just one goal below (from formation flying)
        # self.feats[:,0] = self.x[:,0] - self.x[:,2]
        # self.feats[:,1] = self.x[:,1] - self.x[:,3]

        if self.dynamic:
            state_network = self.get_connectivity(self.x)
        else:
            state_network = self.a_net

        # return (state_values, state_network)
        return self.feats

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.n_agents, 2, 1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0, 2, 1)) -
                       np.transpose(x_loc, (2, 0, 1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net

    def get_connectivity(self, x):

        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:, 2:4])
            a_net = np.array(neigh.kneighbors_graph(
                mode='connectivity').todense())

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            # TODO or axis=0? Is the mean in the correct direction?
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors

        return a_net

    def render(self, filename=None, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """

        if self.fig is None:
            # plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Returns a tuple of line objects, thus the comma
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')

            #ax.plot([0], [0], 'kx')
            ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            ax.plot(self.goal_xpoints, self.goal_ypoints, 'rx')

            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('2D Navigation')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
