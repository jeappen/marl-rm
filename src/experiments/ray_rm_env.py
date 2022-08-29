from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv
from typing import Dict
import numpy as np
from ray.rllib.env import MultiAgentEnv
from rm_cooperative_marl.src.Agent.agent_ray import Agent


def _filter_dict(dict2filter, keys2keep):
    return {k: dict2filter[k] for k in keys2keep}


def reshape_state_for_spec(flat_state):
    return flat_state[:-1], int(flat_state[-1])


class MA_RM_MDP(MultiAgentEnv):
    """ Reward Machine based environment for RLLib MA
    """

    # system : System MDP (no need for reward function)
    # action_dim: action space dimension for the system
    # res_model : Resource_Model (optional)
    # spec : TaskSpec
    # min_reward (C_l) = Min possible unshaped reward
    # local_reward_bound (C_u) = Max possible absolute value of local reward (quant. sem. value)
    # multi_specs =  list of specifications (one for each agent)
    # multi_spec_map = fn(agent_id) mapping agent id to multi_spec index
    # eval_mode =  If true use multi-agent RM for transition
    def __init__(self, system, action_dim, tester, spec=None, min_reward=None, local_reward_bound=None,
                 res_model=None, use_shaped_rewards=True, multi_specs=None, multi_spec_map=None,
                 eval_mode=False):

        self.system = system
        self.num_agents = system.n_agents

        # RM config

        # Reseting default step values
        tester.restart()

        rm_test_file = tester.rm_test_file
        rm_learning_file_list = tester.rm_learning_file_list

        # Verify that the number of local reward machines matches the number of agents in the experiment.
        assertion_string = "Number of specified local reward machines must match specified number of agents."
        assert (len(tester.rm_learning_file_list) ==
                self.num_agents), assertion_string

        # Create the list of RM agents for this experiment
        self.agent_list = {}
        for i, a in enumerate(self.system.agents):
            self.agent_list[a] = Agent(rm_learning_file_list[i], i)

        self.reward_machine = Agent(rm_test_file, -1).rm
        self.eval_mode = eval_mode
        if eval_mode:
            print('\n SETTING EVAL MODE FOR THE RM ENV\n')
            # Access the reward machine specific to each agent via agent_list.

        # Use the agent list to access the reward machine.
        # The environment should also have a "labelling function". E.g. when an agent is within a region it should have the boolean proposition label true.
        init_system_state = self.system.reset()
        system_state_dim = len(list(init_system_state.values())[0])

        for a in self.system.agents:
            init_system_state[a] = (init_system_state[a], 0)
        self.state = init_system_state

        from gym import spaces
        self.action_space = spaces.Box(low=self.system.action_space.low[0], high=self.system.action_space.high[0],
                                       shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf,
                                            shape=(system_state_dim + 1,), dtype=np.float32)

    def reset(self, **kwargs):
        MA_reset_state = self.system.reset(**kwargs)
        for a in self.system.agents:
            self.agent_list[a].initialize_reward_machine()
            MA_reset_state[a] = np.hstack(
                (MA_reset_state[a], self.agent_list[a].u))
        self.u = self.reward_machine.get_initial_state()  # For eval mode
        self.state = MA_reset_state
        return self.state

    def step(self, actions, dbg=False, terminate_early=False):
        """ General Process with baseline reward machine
        1. Step the environment using self.system
        2. Maintain agent-specific reward machine in a wrapper which also keeps track of the rm state (u) 
        3. Get the labels from the self.system.get_mdp_label while training
        4. Calculate reward and update the rm state in the wrapper.
        5. output concatenated state (s,u), r at each env step for all agents.
        6. During evaluation change env settings to use entire system's rm and labelling function self.system.get_mdp_label_multi

        terminate_early: Set done[agent-id] to terminate for each agent as soon as their task is complete
                         Not used now since this means termination early for some agents, screws up self.system.step()

        """
        system_actions = {a: actions[a] for a in self.system.agents}
        # print(system_actions,self.spec.action_dim)
        next_state, rew, done, render = self.system.step(system_actions)

        if not self.eval_mode:
            # Here use probablistic threshold for synchronized tasks
            for i, a in enumerate(self.system.agents):
                # Now handle reward machine specific tasks like u and reward.
                current_u = self.agent_list[a].u
                l = self.system.get_mdp_label(
                    self.state, next_state, current_u, agent_id=a)
                # Now calculate reward
                r = 0
                u = current_u
                for e in l:
                    # Should be just one label.
                    # Get the new reward machine state and the reward of this step
                    u2 = self.agent_list[a].rm.get_next_state(u, e)
                    # Choose between continuous reward (coded as part of env) and rm reward
                    # r +=  self.agent_list[a].rm.get_reward(u, u2)
                    r += self.system.get_continuous_rw(
                        self.state, next_state, u, agent_id=a, next_u=u2)
                    # print("\n {} with state {} had event {} rw change to {}".format(a, u, e, r))
                    # Update the reward machine state
                    u = u2

                # update rm state if task incomplete
                if not self.agent_list[a].is_task_complete:
                    self.agent_list[a].update_agent(r, l)
                    rew[a] = r
                else:
                    # Don't give any reward since task already done
                    rew[a] = 0.0
                    if terminate_early:
                        done[a] = True
                self.state[a] = np.hstack(
                    (next_state[a], self.agent_list[a].u))
        else:
            # In eval mode check properly for synchronized tasks
            l = self.system.get_mdp_label_multi(self.state, next_state)
            r = 0

            for e in l:
                # Get the new reward machine state and the reward of this step
                u2 = self.reward_machine.get_next_state(self.u, e)
                r = r + self.reward_machine.get_reward(self.u, u2)
                # print("\n multi RM {} had event {} rw change to {}".format(self.u,e, r))
                # Update the reward machine state
                self.u = u2

            projected_l_dict = {}
            for i, a in enumerate(self.system.agents):
                # Agent i's projected label is the intersection of l with its local event set
                projected_l_dict[i] = list(
                    set(self.agent_list[a].local_event_set) & set(l))
                # Check if the event causes a transition from the agent's current RM state
                if not(self.agent_list[a].is_local_event_available(projected_l_dict[i])):
                    projected_l_dict[i] = []

            for i, a in enumerate(self.system.agents):
                # Enforce synchronization requirement on shared events
                if projected_l_dict[i]:
                    for event in projected_l_dict[i]:
                        for j, a2 in enumerate(self.system.agents):
                            if (event in set(self.agent_list[a2].local_event_set)) and (not (projected_l_dict[j] == projected_l_dict[i])):
                                # if a's event is in a2's local event set but not happened for a2, don't assume event happened for a
                                projected_l_dict[i] = []

                # update the agent's internal representation
                # a = testing_env.get_last_action(i)
                # print("\n Evalmode {} had projected_l_dict {}".format(a,projected_l_dict))
                self.agent_list[a].update_agent(r, projected_l_dict[i])

            for a in self.agent_list:
                if terminate_early and self.agent_list[a].is_task_complete:
                    done[a] = True
                self.state[a] = np.hstack(
                    (next_state[a], self.agent_list[a].u))
                rew[a] = r / self.num_agents  # Normalize

        if all(self.agent_list[a].is_task_complete for a in self.agent_list):
            done["__all__"] = True
        return self.state, rew, done, render

    def render(self):
        self.system.render()


# Returns the RM wrapped environment here
def rm_env_creator(config):
    from rm_cooperative_marl.src.Environments.rendezvous_continuous.nav_env_rm import NavEnvRMLabelling
    spec_id = config['spec_id'] if 'spec_id' in config else 1
    err_arg = config['err_arg'] if 'err_arg' in config else 0
    max_mode_arg = config['max_mode'] if 'max_mode' in config else False
    num_agents_arg = config['num_agents'] if 'num_agents' in config else 3
    horizon_arg = config['horizon'] if 'horizon' in config else 400
    DS_mode_arg = config['distributed_spectrl_mode'] if 'distributed_spectrl_mode' in config else False
    limit_pred_val_arg = config['limit_pred_val'] if 'limit_pred_val' in config else False
    eval_mode = config['marl_rm_eval_mode'] if 'marl_rm_eval_mode' in config else False

    print('RM SPEC spec_id {}, err arg{}'.format(spec_id, err_arg))
    # Config file with rm locations
    from rm_cooperative_marl.src.rendezvous_config import rendezvous_config
    # from experiments.dqprm import run_multi_agent_experiment
    num_times = 1
    # Get test object from config script
    tester = rendezvous_config(num_times, num_agents_arg)
    base_env = NavEnvRMLabelling(
        n_agents=num_agents_arg, time_horizon=horizon_arg, env_settings=tester.env_settings)

    num_agents = base_env.n_agents
    state_dim = base_env.nx_system
    x_dim = 2  # For 2-D state

    # Goals and obstacles
    gtop = np.array([5.0, 10.0])
    gbot = np.array([5.0, 0.0])
    gright = np.array([10.0, 0.0])
    gcorner = np.array([10.0, 10.0])
    gcorner2 = np.array([0.0, 10.0])
    origin = np.array([0.0, 0.0])
    obs = np.array([4.0, 4.0, 6.0, 6.0])
    gbot_closer = np.array([3.0, 0.0])
    gtop_origin_closer = np.array([0, 3])
    gtop_closer = np.array([5.0, 3.0])
    gcenter = np.array([5.0, 5.0])
    err = 1.0 if err_arg <= 0 \
        else err_arg

    final_env = MA_RM_MDP(
        base_env, base_env.action_space.shape[0], tester, eval_mode=eval_mode)

    return final_env


class CustomCallbacks2(DefaultCallbacks):
    pass


custom_metric_name = 'max_monitor_state'
custom_metric_name2 = 'max_depth'
# To check for satisfaction of spec wrt task monitor
custom_metric_name3 = 'final_state_reached'
# To check for satisfaction of spec wrt task monitor
custom_metric_name4 = 'stage_reached'


class CustomCallbacks(DefaultCallbacks):
    # Uses monitor state to determine if Spec is satisfied
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        # print("episode {}  started.".format(
        #     episode.episode_id))
        episode.user_data[custom_metric_name] = []
        episode.hist_data[custom_metric_name] = []
        if 'spectrl_logn' in worker.policy_config['env_config'] and worker.policy_config['env_config']['spectrl_logn']:
            # Include stage and extract monitor_state correctly
            episode.user_data[custom_metric_name4] = []
            episode.hist_data[custom_metric_name4] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):

        log_n_scale_mode = custom_metric_name4 in episode.user_data
        if log_n_scale_mode:
            # Last dim is actually current stage
            current_stages = []
            for aid in episode._agent_to_index:
                current_stage = episode.last_raw_obs_for(aid)[-1]
                current_stages.append(current_stage)
            episode.user_data[custom_metric_name4].append(current_stages)

        monitor_states = []
        for aid in episode._agent_to_index:
            # iterate over agents
            monitor_state = episode.last_raw_obs_for(
                aid)[-2] if log_n_scale_mode else episode.last_raw_obs_for(aid)[-1]
            monitor_states.append(monitor_state)
        episode.user_data[custom_metric_name].append(monitor_states)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        final_monitor_state = np.min(episode.user_data[custom_metric_name][-1])
        episode.custom_metrics[custom_metric_name] = final_monitor_state
        log_n_scale_mode = custom_metric_name4 in episode.user_data
        if log_n_scale_mode:
            # Last dim is actually current stage
            current_stage = np.min(episode.user_data[custom_metric_name4][-1])
            episode.custom_metrics[custom_metric_name4] = current_stage

        final_depth = int(final_monitor_state)
        episode.custom_metrics[custom_metric_name2] = final_depth

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
