from rm_cooperative_marl.src.reward_machines.sparse_reward_machine import SparseRewardMachine
from rm_cooperative_marl.src.tester.tester import Tester
import numpy as np
import random, time, os
import matplotlib.pyplot as plt

class Agent:
    """
    Modified Agent for Continuous environments in Ray

    Simply a tool to keep track of rm state for each agent.
    
    The agent also has a representation of its own local reward machine, which it uses
    for learning, and of its state in the world/reward machine.
    
    Note: Users of this class must manually reset the world state and the reward machine
    state when starting a new episode by calling self.initialize_world() and 
    self.initialize_reward_machine().
    """
    def __init__(self, rm_file, agent_id):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file : str
            File path pointing to the reward machine this agent is meant to use for learning.
        agent_id : int
            Index of this agent.
        """
        self.rm_file = rm_file
        self.agent_id = agent_id

        self.rm = SparseRewardMachine(self.rm_file)
        self.u = self.rm.get_initial_state()
        self.local_event_set = self.rm.get_events()
        
        self.total_local_reward = 0
        self.is_task_complete = 0

    def initialize_reward_machine(self):
        """
        Reset the state of the reward machine to the initial state and reset task status.
        """
        self.u = self.rm.get_initial_state()
        self.is_task_complete = 0

    def is_local_event_available(self, label):
        if label: # Only try accessing the first event in label if it exists
            event = label[0]
            return self.rm.is_event_available(self.u, event)
        else:
            return False

    def update_agent(self, reward, label):
        """
        Update the agent's state, q-function, and reward machine after 
        interacting with the environment.

        Parameters
        ----------
        reward : float
            Reward the agent achieved during this step.
        label : string
            Label returned by the MDP this step.
        """

        # Keep track of the RM location at the start of the 
        u_start = self.u

        for event in label: # there really should only be one event in the label provided to an individual agent
            # Update the agent's RM
            u2 = self.rm.get_next_state(self.u, event)
            self.u = u2
            # print("\n Event {} with state change {} from start u {}".format(event, u2, u_start))
        
        self.total_local_reward += reward

        if self.rm.is_terminal_state(self.u):
            # Completed task. Set flag.
            self.is_task_complete = 1