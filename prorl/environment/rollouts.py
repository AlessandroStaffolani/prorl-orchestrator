import time
from typing import List, Tuple, Dict, Union, Optional

import numpy as np

from prorl import SingleRunConfig
from prorl.common.stats_tracker import Tracker
from prorl.core.state import State
from prorl.core.step import Step
from prorl.core.step_data import StepData
from prorl.common.data_structure import RunMode
from prorl.environment.action_space import Action, ActionType
from prorl.environment.agent import AgentType
from prorl.environment.agent.abstract import AgentAbstract, AgentLoss, AgentQEstimate
from prorl.environment.node import Node
from prorl.environment.state_builder import StateType
from prorl.environment.wrapper import EnvWrapper


class Rollout:

    def __init__(
            self,
            obs: List[Dict[StateType, np.ndarray]],
            acs: List[np.ndarray],
            rewards: List[np.ndarray],
            next_obs: List[Dict[StateType, np.ndarray]],
            dones: List[bool]
    ):
        self.states: List[Dict[StateType, np.ndarray]] = obs
        self.rewards: np.ndarray = np.array(rewards, dtype=np.float32)
        self.actions: np.ndarray = np.array(acs, dtype=np.float32)
        self.next_states: List[Dict[StateType, np.ndarray]] = next_obs
        self.dones: np.ndarray = np.array(dones, dtype=np.float32)

    def get_states(self, action_type: ActionType) -> np.ndarray:
        if action_type == ActionType.Add:
            filter_type = StateType.Add
        elif action_type == ActionType.Remove:
            filter_type = StateType.Remove
        elif action_type == ActionType.Quantity:
            filter_type = StateType.Quantity
        else:
            raise Exception(f'ActionType {action_type} not supported')
        states = [s[filter_type] for s in self.states]
        return np.array(states, dtype=np.float32)

    def get_actions(self, action_type: ActionType) -> np.ndarray:
        return self.actions

    def get_rewards(self, action_type: ActionType) -> np.ndarray:
        return self.rewards

    def get_next_states(self, action_type: ActionType) -> np.ndarray:
        if action_type == ActionType.Add:
            filter_type = StateType.Add
        elif action_type == ActionType.Remove:
            filter_type = StateType.Remove
        elif action_type == ActionType.Quantity:
            filter_type = StateType.Quantity
        else:
            raise Exception(f'ActionType {action_type} not supported')
        states = [s[filter_type] for s in self.next_states]
        return np.array(states, dtype=np.float32)

    def get_dones(self, action_type: ActionType) -> np.ndarray:
        return self.dones

    def to_dict(self):
        return {
            'states': self.states,
            'rewards': self.rewards,
            'actions': self.actions,
            'next_states': self.next_states,
            'dones': self.dones
        }

    def __str__(self):
        return f'<Rollout size={len(self)}>'

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.rewards)

    def __getattr__(self, item: str):
        attr = getattr(self, item, None)
        if attr is not None:
            return attr


def convert_list_of_rollouts(
        paths: List[Rollout]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    states = np.concatenate([path.states for path in paths])
    actions = np.concatenate([path.actions for path in paths])
    next_states = np.concatenate([path.next_states for path in paths])
    concatenated_rewards = np.concatenate([path.rewards for path in paths])
    return states, actions, next_states, concatenated_rewards


def init_nodes_stats_rollout(config: SingleRunConfig, nodes: List[Node], mode: RunMode) -> Optional[Dict[str, List[float]]]:
    if not config.saver.stats_condensed and mode == RunMode.Eval:
        nodes_stats: Dict[str, List[float]] = {}
        for i, node in enumerate(nodes):
            nodes_stats[f'nodes/n_{i}/demand_history'] = []
            nodes_stats[f'nodes/n_{i}/resource_history'] = []
        return nodes_stats
    return None


def store_nodes_step_data(nodes_stats: Optional[Dict[str, List[float]]], nodes: List[Node],
                          demand: StepData, resource: str):
    if nodes_stats is not None:
        current_demand = demand.get_resource_values(resource, as_array=True)
        for i, node in enumerate(nodes):
            nodes_stats[f'nodes/n_{i}/demand_history'].append(current_demand[i])
            nodes_stats[f'nodes/n_{i}/resource_history'].append(node.get_current_allocated(resource))


def sample_trajectory(
        env: EnvWrapper,
        agent: AgentAbstract,
        batch_size: int,
        resource_name: str,
        iteration: int,
        stats_tracker: Tracker,
        config: SingleRunConfig,
        initial_state: Optional[Dict[str, State]] = None,
        mode: RunMode = RunMode.Train,
        previous_state=None,
        previous_action=None,
        reward=None,
        previous_done=False,
        seed: Optional[int] = None
) -> Tuple[Rollout, Dict[str, State], Action, Dict[str, State], tuple, bool]:
    if AgentType.is_mc_method(agent.name):
        return sample_n_trajectories(env, agent, batch_size, resource_name, iteration,
                                     stats_tracker, config, initial_state, mode, previous_state,
                                     previous_action, reward, previous_done, seed)
    else:
        return sample_n_steps(env, agent, batch_size, resource_name, iteration,
                              stats_tracker, config, initial_state, mode, previous_state,
                              previous_action, reward, previous_done, seed)


def sample_n_steps(
        env: EnvWrapper,
        agent: AgentAbstract,
        batch_size: int,
        resource_name: str,
        iteration: int,
        stats_tracker: Tracker,
        config: SingleRunConfig,
        initial_state: Optional[Dict[str, State]] = None,
        mode: RunMode = RunMode.Train,
        previous_state: Optional[Dict[StateType, np.ndarray]] = None,
        previous_action=None,
        reward=None,
        previous_done=False,
        seed: Optional[int] = None
) -> Tuple[Rollout, Dict[str, State], Action, Dict[str, State], tuple, bool]:
    # init vars
    states: List[Dict[StateType, np.ndarray]] = []
    actions: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    rewards_not_scalarized: List[Tuple[float, float]] = []
    surpluses: List[float] = []
    dones: List[bool] = []
    # pre_gaps: List[float] = []
    next_states: List[Dict[StateType, np.ndarray]] = []
    nodes_satisfied: List[int] = []
    global_steps: List[int] = []
    choose_time: List[float] = []
    nodes_stats: Optional[Dict[str, List[float]]] = init_nodes_stats_rollout(config, env.nodes, mode)
    hour_satisfied_list: List[bool] = []
    resource_utilization: List[bool] = []
    state = initial_state
    done = previous_done
    steps = 0
    stop_step = batch_size
    if iteration == 0 or mode != RunMode.Train:
        stop_step += 1
    while steps < stop_step:
        if initial_state is None or done:
            if iteration >= 0 or mode == RunMode.Train:
                add_plus_one = False
            else:
                add_plus_one = True
            if done:
                state = env.reset(show_log=False, add_plus_one=add_plus_one)
                agent.reset()
            else:
                state = env.reset(show_log=False, add_plus_one=add_plus_one)
            done = False
        while not done and steps < stop_step:
            action, state, action_info = agent.choose(state, nodes=env.nodes, resource=resource_name,
                                                      node_groups=env.node_groups,
                                                      demand=env.current_demand, pool_node=env.pool_node)
            choose_time.append(action_info['action_time'])
            if previous_state is not None:
                states.append(previous_state)
                actions.append(np.array([list(a) for a in previous_action]))
                next_states.append(state)
                rewards.append(np.array([reward]))
                dones.append(previous_done)
            next_state_wrapper, reward, step_info = env.step(action, resource_name)
            done = step_info['done']
            previous_state = state
            previous_action = action
            previous_done = done
            state = next_state_wrapper
            nodes_satisfied.append(step_info['nodes_satisfied'])
            not_scalarized = (step_info['reward_info']['remaining_gap'], step_info['reward_info']['cost'])
            hour_satisfied_list.append(step_info['hour_satisfied'])
            resource_utilization.append(step_info['resource_utilization'])
            if 'surplus' in step_info['reward_info']:
                surpluses.append(step_info['reward_info']['surplus'])
            # pre_gaps.append(step_info['pre_gap'])
            rewards_not_scalarized.append(not_scalarized)
            store_nodes_step_data(nodes_stats, nodes=env.nodes, demand=env.current_demand, resource=resource_name)
            global_steps.append(agent.choose_action_calls)
            steps += 1
    if done:
        states.append(previous_state)
        actions.append(np.array([list(a) for a in previous_action]))
        next_states.append(state)
        rewards.append(np.array([reward]))
        dones.append(previous_done)
    rollout = Rollout(states, actions, rewards, next_states, dones)
    if len(surpluses) == 0:
        surpluses = None
    track_rollout(stats_tracker, mode, iteration, rollout, nodes_satisfied, global_steps,
                  choose_time, len(env.nodes), rewards_not_scalarized, nodes_stats, seed, pre_gaps=None,
                  hour_satisfied_list=hour_satisfied_list, surpluses=surpluses,
                  resource_utilization=resource_utilization)
    if done:
        state = None
    return rollout, previous_state, previous_action, state, reward, previous_done


def sample_n_trajectories(
        env: EnvWrapper,
        agent: AgentAbstract,
        batch_size: int,
        resource_name: str,
        iteration: int,
        stats_tracker: Tracker,
        config: SingleRunConfig,
        initial_state: Optional[Dict[str, State]] = None,
        mode: RunMode = RunMode.Train,
        previous_state=None,
        previous_action=None,
        reward=None,
        previous_done=False,
        seed: Optional[int] = None,
        n_trajectories: int = 1,
) -> Tuple[Rollout, Dict[str, State], Action, Dict[str, State], tuple, bool]:
    # init vars
    states: List[Dict[StateType, np.ndarray]] = []
    actions: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    rewards_not_scalarized: List[Tuple[float, float]] = []
    dones: List[bool] = []
    # pre_gaps: List[float] = []
    next_states: List[Dict[StateType, np.ndarray]] = []
    nodes_satisfied: List[int] = []
    global_steps: List[int] = []
    choose_time: List[float] = []
    nodes_stats: Optional[Dict[str, List[float]]] = init_nodes_stats_rollout(config, env.nodes)
    hour_satisfied_list: List[bool] = []
    state = initial_state
    done = previous_done
    steps = 0
    add_plus_one = False

    for i in range(n_trajectories):
        state = env.reset(add_plus_one=add_plus_one, show_log=False)
        agent.reset()
        done = False
        while not done:
            t: Step = env.current_time_step
            action, state, action_info = agent.choose(state, nodes=env.nodes, resource=resource_name,
                                                      node_groups=env.node_groups,
                                                      demand=env.current_demand, pool_node=env.pool_node)
            choose_time.append(action_info['action_time'])
            if previous_state is not None:
                states.append(previous_state)
                actions.append(np.array(list(previous_action)))
                next_states.append(state)
                rewards.append(np.array([reward]))
                dones.append(previous_done)
            next_state_wrapper, reward, step_info = env.step(action, resource_name)
            done = step_info['done']
            previous_state = state
            previous_action = action
            previous_done = done
            state = next_state_wrapper
            nodes_satisfied.append(step_info['nodes_satisfied'])
            not_scalarized = (step_info['reward_info']['remaining_gap'], step_info['reward_info']['cost'])
            hour_satisfied_list.append(step_info['hour_satisfied'])
            # pre_gaps.append(step_info['pre_gap'])
            rewards_not_scalarized.append(not_scalarized)
            store_nodes_step_data(nodes_stats, nodes=env.nodes, demand=env.current_demand, resource=resource_name)
            global_steps.append(t.total_steps // env.time_step.step_size)
            steps += 1
    if done:
        states.append(previous_state)
        actions.append(np.array(list(previous_action)))
        next_states.append(state)
        rewards.append(np.array([reward]))
        dones.append(previous_done)
        previous_state = None
        previous_action = None
        reward = None

    rollout = Rollout(states, actions, rewards, next_states, dones)
    track_rollout(stats_tracker, mode, iteration, rollout, nodes_satisfied, global_steps,
                  choose_time, len(env.nodes), rewards_not_scalarized, nodes_stats, seed, pre_gaps=None,
                  hour_satisfied_list=hour_satisfied_list)
    if done:
        state = None
    return rollout, previous_state, previous_action, state, reward, previous_done


def track_training_rollout(
        tracker: Tracker,
        iteration: int,
        rollout: Rollout,
        nodes_satisfied: List[int],
        global_steps: List[int],
        n_nodes: int,
        rewards_not_scalarized: List[Tuple[float, float]],
        nodes_stats: Optional[Dict[str, List[float]]] = None,
        pre_gaps: Optional[List[float]] = None,
        hour_satisfied_list: Optional[List[bool]] = None,
        surpluses: Optional[List[float]] = None,
):
    prefix = RunMode.Train.value
    problem_solved_count = 0
    remaining_gap: List[float] = []
    cost: List[float] = []
    for i in range(len(rollout)):
        nodes_satisfied_step = nodes_satisfied[i]
        remaining_gap.append(rewards_not_scalarized[i][0])
        cost.append(rewards_not_scalarized[i][1])
        tracker.track(f'{prefix}/reward/utility/total', rollout.rewards[i][0], global_steps[i])
        tracker.track(f'{prefix}/reward/remaining_gap/total', rewards_not_scalarized[i][0], global_steps[i])
        tracker.track(f'{prefix}/reward/cost/total', rewards_not_scalarized[i][1], global_steps[i])
        # tracker.track(f'{prefix}/reward/pre_gap/total', pre_gaps[i], global_steps[i])
        tracker.track(f'{prefix}/done', rollout.dones[i], global_steps[i])
        tracker.track(f'{prefix}/action_history/add_node', rollout.actions[i][0], global_steps[i])
        tracker.track(f'{prefix}/action_history/remove_node', rollout.actions[i][1], global_steps[i])
        tracker.track(f'{prefix}/action_history/resource_class', rollout.actions[i][2], global_steps[i])
        tracker.track(f'{prefix}/action_history/quantity', rollout.actions[i][3], global_steps[i])
        tracker.track(f'{prefix}/problem_solved/nodes_satisfied', nodes_satisfied_step, global_steps[i])
        tracker.track(f'{prefix}/problem_solved/count', 1 if nodes_satisfied_step == n_nodes else 0,
                      global_steps[i])
        tracker.track(f'{prefix}/steps', 1, global_steps[i])
        if nodes_stats is not None:
            for key, array in nodes_stats.items():
                tracker.track(f'{prefix}/{key}', array[i], global_steps[i])
        problem_solved_count += 1 if nodes_satisfied_step == n_nodes else 0
    remaining_gap_np = np.array(remaining_gap)
    cost_np = np.array(cost)
    # pre_gaps_np = np.array(pre_gaps)
    n_dones = len(np.where(rollout.dones == 1)[0])
    tracker.track(f'{prefix}/batch/reward/utility', rollout.rewards, iteration)
    tracker.track(f'{prefix}/batch/reward/remaining_gap', remaining_gap_np, iteration)
    tracker.track(f'{prefix}/batch/reward/cost', cost_np, iteration)
    # tracker.track(f'{prefix}/batch/reward/pre_gap', pre_gaps_np, iteration)
    tracker.track(f'{prefix}/batch/problem_solved/count', problem_solved_count, iteration)
    tracker.track(f'{prefix}/batch/dones', n_dones, iteration)
    tracker.track(f'{prefix}/episodes', n_dones, iteration)
    tracker.track(f'{prefix}/episode/length', len(rollout), iteration)


def track_eval_and_val_rollout(
        tracker: Tracker,
        mode: RunMode,
        iteration: int,
        rollout: Rollout,
        nodes_satisfied: List[int],
        global_steps: List[int],
        choose_time: List[float],
        n_nodes: int,
        rewards_not_scalarized: List[Tuple[float, float]],
        nodes_stats: Optional[Dict[str, List[float]]] = None,
        seed: Optional[int] = None,
        pre_gaps: Optional[List[float]] = None,
        hour_satisfied_list: Optional[List[bool]] = None,
        surpluses: Optional[List[float]] = None,
        resource_utilization: Optional[List[float]] = None
):
    if seed is None:
        seed = 0
    prefix = f'{mode.value}-{seed}'
    problem_solved_count = 0
    hours_satisfied = 0
    remaining_gap: List[float] = []
    cost: List[float] = []
    step_per_hour = 3600/tracker.config.run.step_size
    for i in range(len(rollout)):
        nodes_satisfied_step = nodes_satisfied[i]
        if nodes_satisfied_step is not None:
            problem_solved_count += 1 if nodes_satisfied_step == n_nodes else 0
        if hour_satisfied_list[i] is not None and hour_satisfied_list[i] is True:
            hours_satisfied += 1
        append_not_none(remaining_gap, rewards_not_scalarized[i][0])
        append_not_none(cost, rewards_not_scalarized[i][1])
        # remaining_gap.append(rewards_not_scalarized[i][0])
        # cost.append(rewards_not_scalarized[i][1])
        if len(rollout.actions.shape) == 3:
            tracker.track(f'{prefix}/action_history/add', rollout.actions[i][0][0], global_steps[i])
            tracker.track(f'{prefix}/action_history/remove', rollout.actions[i][1][1], global_steps[i])
        else:
            tracker.track(f'{prefix}/action_history/add_node', rollout.actions[i][0], global_steps[i])
            tracker.track(f'{prefix}/action_history/remove_node', rollout.actions[i][1], global_steps[i])
            tracker.track(f'{prefix}/action_history/resource_class', rollout.actions[i][2], global_steps[i])
            tracker.track(f'{prefix}/action_history/quantity', rollout.actions[i][3], global_steps[i])
        if nodes_stats is not None:
            for key, array in nodes_stats.items():
                tracker.track(f'{prefix}/{key}', array[i], global_steps[i])
        if resource_utilization is not None and resource_utilization[i] is not None:
            tracker.track(f'{prefix}/resource_utilization', resource_utilization[i], global_steps[i])
        if mode == RunMode.Eval:
            tracker.track(f'{prefix}/times/choose_action', choose_time[i], global_steps[i])
            tracker.track(f'{prefix}/reward_history/utility', rollout.rewards[i].item(), global_steps[i])
            if rewards_not_scalarized[i][0] is not None:
                tracker.track(f'{prefix}/reward_history/remaining_gap', rewards_not_scalarized[i][0], global_steps[i])
            if surpluses is not None and surpluses[i] is not None:
                tracker.track(f'{prefix}/reward_history/surplus', surpluses[i], global_steps[i])
            if rewards_not_scalarized[i][1] is not None:
                tracker.track(f'{prefix}/reward_history/cost', rewards_not_scalarized[i][1], global_steps[i])
            if nodes_satisfied_step is not None:
                tracker.track(f'{prefix}/problem_solved_history/count', 1 if nodes_satisfied_step == n_nodes else 0,
                              global_steps[i])

            if hour_satisfied_list[i] is not None:
                tracker.track(f'{prefix}/hour_satisfied_history', 1 if hour_satisfied_list[i] else 0,
                              int(global_steps[i] // step_per_hour))

    remaining_gap_np = np.array(remaining_gap)
    cost_np = np.array(cost)
    # pre_gaps_np = np.array(pre_gaps)
    tracker.track(f'{prefix}/reward/utility', rollout.rewards, iteration)
    tracker.track(f'{prefix}/reward/remaining_gap', remaining_gap_np, iteration)
    if surpluses is not None:
        surpluses_np = np.array([s for s in surpluses if s is not None])
        tracker.track(f'{prefix}/reward/surplus', surpluses_np, iteration)
    tracker.track(f'{prefix}/reward/cost', cost_np, iteration)
    # tracker.track(f'{prefix}/reward/pre_gap', pre_gaps_np, iteration)
    tracker.track(f'{prefix}/problem_solved/count', problem_solved_count, iteration)
    tracker.track(f'{prefix}/hour_satisfied', hours_satisfied, iteration)

    tracker.track(f'{prefix}/episode/length', len(rollout), iteration)


def _get_single_average_value(tracker: Tracker, prefix: str, key: str, iteration: int, seeds: List[int]) -> float:
    values = []
    for seed in seeds:
        values.append(tracker.get(f'{prefix}-{seed}/{key}')[iteration])
    return np.mean(values).item()


def track_average_values(
        tracker: Tracker,
        mode: RunMode,
        iteration: int,
        seeds: List[int]
):
    prefix = mode.value
    tracker.track(
        f'{prefix}/avg/reward/utility',
        _get_single_average_value(tracker, prefix, 'reward/utility/total', iteration, seeds),
        iteration,
    )
    tracker.track(
        f'{prefix}/avg/reward/remaining_gap',
        _get_single_average_value(tracker, prefix, 'reward/remaining_gap/total', iteration, seeds),
        iteration,
    )
    if len(tracker.get(f'{prefix}-{seeds[0]}/reward/surplus/total')) > 0:
        tracker.track(
            f'{prefix}/avg/reward/surplus',
            _get_single_average_value(tracker, prefix, 'reward/surplus/total', iteration, seeds),
            iteration,
        )
    tracker.track(
        f'{prefix}/avg/reward/cost',
        _get_single_average_value(tracker, prefix, 'reward/cost/total', iteration, seeds),
        iteration,
    )
    tracker.track(
        f'{prefix}/avg/problem_solved/count',
        _get_single_average_value(tracker, prefix, 'problem_solved/count', iteration, seeds),
        iteration,
    )
    tracker.track(
        f'{prefix}/avg/hour_satisfied',
        _get_single_average_value(tracker, prefix, 'hour_satisfied', iteration, seeds),
        iteration,
    )
    values = []
    for seed in seeds:
        np_resource_utilization = np.array(tracker.get(f'{prefix}-{seed}/resource_utilization'))
        values.append(np_resource_utilization.mean())
    final_value = np.mean(values).item()
    tracker.track(f'{prefix}/avg/resource_utilization', final_value, iteration)
    tracker.track(
        f'{prefix}/avg/episode/length',
        _get_single_average_value(tracker, prefix, 'episode/length', iteration, seeds),
        iteration,
    )
    # tracker.track(
    #     f'{prefix}/avg/reward/pre_gap',
    #     _get_single_average_value(tracker, prefix, 'reward/pre_gap/total', iteration, seeds),
    #     iteration,
    # )
    if mode == RunMode.Eval:
        choose_times = []
        for s in seeds:
            choose_times += tracker.get(f'{prefix}-{s}/times/choose_action')
        tracker.track(f'{prefix}/avg/times/choose_action', np.mean(choose_times), iteration)


def track_loss(tracker: Tracker, actor_loss: AgentLoss, critic_loss: Optional[AgentLoss], iteration):
    if tracker.config.environment.agent.type == AgentType.TD_AC \
            or tracker.config.environment.agent.type == AgentType.MC_AC:
        tracker.track('loss/add_node/actor', actor_loss.add_net, iteration)
        tracker.track('loss/add_node/critic', critic_loss.add_net, iteration)
        tracker.track('loss/movement/actor', actor_loss.remove_net, iteration)
        tracker.track('loss/movement/critic', critic_loss.remove_net, iteration)
    elif tracker.config.environment.agent.type == AgentType.Reinforce:
        tracker.track('loss/add_node/policy', actor_loss.add_net, iteration)
        tracker.track('loss/movement/policy', actor_loss.remove_net, iteration)


def track_rollout(
        tracker: Tracker,
        mode: RunMode,
        iteration: int,
        rollout: Rollout,
        nodes_satisfied: List[int],
        global_steps: List[int],
        choose_time: List[float],
        n_nodes: int,
        rewards_not_scalarized: List[Tuple[float, float]],
        nodes_stats: Optional[Dict[str, List[float]]] = None,
        seed: Optional[int] = None,
        pre_gaps: Optional[List[float]] = None,
        hour_satisfied_list: Optional[List[bool]] = None,
        surpluses: Optional[List[float]] = None,
        resource_utilization: Optional[List[float]] = None
):
    if mode == RunMode.Train:
        track_training_rollout(tracker, iteration, rollout, nodes_satisfied,
                               global_steps, n_nodes, rewards_not_scalarized, nodes_stats,
                               pre_gaps, hour_satisfied_list, surpluses)
    else:
        track_eval_and_val_rollout(tracker, mode, iteration, rollout, nodes_satisfied, global_steps, choose_time,
                                   n_nodes, rewards_not_scalarized, nodes_stats, seed,
                                   pre_gaps, hour_satisfied_list, surpluses, resource_utilization)


def track_off_policy_training_iteration(
        iteration: int,
        episode_stats: Dict[str, Union[list, int]],
        tracker: Tracker,
        actions: List[Action],
        reward: float,
        done: bool,
        step_info: Dict[str, Union[int, float]],
):
    if len(episode_stats) == 0:
        episode_stats['n_episodes'] = 0
        episode_stats['len'] = 0
        episode_stats['utility'] = []
        episode_stats['remaining_gap'] = []
        episode_stats['surplus'] = []
        episode_stats['cost'] = []
        episode_stats['problem_solved'] = 0
        episode_stats['hour_satisfied'] = 0
        episode_stats['resource_utilization'] = []
    prefix = RunMode.Train.value
    satisfied_nodes = step_info['nodes_satisfied']
    n_nodes = tracker.config.environment.nodes.get_n_nodes()
    if len(actions) > 1:
        tracker.track(f'{prefix}/action_history/add', actions[0].add_node, iteration)
        tracker.track(f'{prefix}/action_history/remove', actions[1].remove_node, iteration)
    else:
        tracker.track(f'{prefix}/action_history/add_node', actions[0].add_node, iteration)
        tracker.track(f'{prefix}/action_history/remove_node', actions[0].remove_node, iteration)
        tracker.track(f'{prefix}/action_history/resource_class', actions[0].resource_class, iteration)
        tracker.track(f'{prefix}/action_history/quantity', actions[0].quantity, iteration)
    if satisfied_nodes is not None:
        tracker.track(f'{prefix}/problem_solved/nodes_satisfied', satisfied_nodes, iteration)
    tracker.track(f'{prefix}/total_steps', 1, iteration)

    # accumulate the episode stats
    episode_stats['len'] += 1
    episode_stats['utility'].append(reward)
    append_not_none(episode_stats['remaining_gap'], step_info['reward_info']['remaining_gap'])
    append_not_none(episode_stats['cost'], step_info['reward_info']['cost'])

    # episode_stats['remaining_gap'].append(step_info['reward_info']['remaining_gap'])
    # episode_stats['cost'].append(step_info['reward_info']['cost'])
    if 'surplus' in step_info['reward_info']:
        # episode_stats['surplus'].append(step_info['reward_info']['surplus'])
        append_not_none(episode_stats['surplus'], step_info['reward_info']['surplus'])
    if satisfied_nodes is not None:
        episode_stats['problem_solved'] += 1 if n_nodes == satisfied_nodes else 0
    if step_info['hour_satisfied'] is not None and step_info['hour_satisfied'] is True:
        episode_stats['hour_satisfied'] += 1
    if step_info['resource_utilization'] is not None:
        episode_stats['resource_utilization'].append(step_info['resource_utilization'])
    if done:
        episode_stats['n_episodes'] += 1
        step = episode_stats['n_episodes']
        # we track the episode values
        tracker.track(f'{prefix}/episode/reward/utility', np.array(episode_stats['utility']), step)
        tracker.track(f'{prefix}/episode/reward/remaining_gap', np.array(episode_stats['remaining_gap']), step)
        if len(episode_stats['surplus']) > 0:
            tracker.track(f'{prefix}/episode/reward/surplus', np.array(episode_stats['surplus']), step)
        tracker.track(f'{prefix}/episode/reward/cost', np.array(episode_stats['cost']), step)
        tracker.track(f'{prefix}/episode/problem_solved/count', episode_stats['problem_solved'], step)
        tracker.track(f'{prefix}/episode/hour_satisfied', episode_stats['hour_satisfied'], step)
        tracker.track(f'{prefix}/episode/resource_utilization',
                      np.mean(episode_stats['resource_utilization']).item(), step)
        tracker.track(f'{prefix}/episode/length', episode_stats['len'], step)
        tracker.track(f'{prefix}/episode/total', 1, step)
        # we reset episode stats except for n_episodes
        episode_stats['len'] = 0
        episode_stats['utility'] = []
        episode_stats['remaining_gap'] = []
        episode_stats['surplus'] = []
        episode_stats['cost'] = []
        episode_stats['problem_solved'] = 0
        episode_stats['hour_satisfied'] = 0
        episode_stats['resource_utilization'] = []


def append_not_none(array: list, value):
    if value is not None:
        array.append(value)
