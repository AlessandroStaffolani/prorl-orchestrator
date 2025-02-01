import copy
from logging import Logger
from typing import Optional, Dict, Union, Tuple, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from prorl import SingleRunConfig
from prorl.common.data_structure import RunMode
from prorl.common.stats_tracker import Tracker
from prorl.core.state import State
from prorl.core.step_data import StepData
from prorl.environment.action_space import ActionSpaceWrapper, Action, ActionType, ActionSpace, \
    CombinedActionSpaceWrapper
from prorl.environment.agent import AgentType
from prorl.environment.agent.abstract import AgentAbstract, AgentLoss, AgentQEstimate
from prorl.environment.agent.dqn.q_network import QNet, QNetworkFactory
from prorl.environment.agent.experience_replay import ExperienceEntry, ReplayBuffer, \
    replay_buffer_factory
from prorl.environment.agent.parameter_schedulers import EpsilonType, get_epsilon_scheduler, Scheduler, \
    LinearScheduler
from prorl.environment.agent.policy import random_policy, greedy_policy, \
    local_optimal_add_node_policy, local_optimal_movement_policy, new_movement_local_optimal_policy, \
    local_optimal_add_node_policy_with_pool, movement_local_optimal_policy_with_pool, \
    local_add_or_remove_action_heuristic, stochastic_policy
from prorl.environment.node import Node
from prorl.environment.node_groups import NodeGroups
from prorl.environment.state_builder import StateType


def init_qnet(input_size: int, output_size: int, config: SingleRunConfig, device) -> QNet:
    additional_parameters = copy.deepcopy(config.environment.agent.double_dqn.q_net_addition_parameters)
    net_type = config.environment.agent.double_dqn.q_net_type
    return QNetworkFactory.get_net(
        type_name=net_type,
        device=device,
        input_size=input_size,
        output_size=output_size,
        **additional_parameters
    )


def init_replay_buffer(config: SingleRunConfig,
                       random_state: np.random.RandomState
                       ) -> ReplayBuffer:
    capacity = config.environment.agent.replay_buffer.capacity
    batch_size = config.environment.agent.double_dqn.batch_size
    if batch_size > capacity:
        capacity = batch_size
    return replay_buffer_factory(
        buffer_type=config.environment.agent.replay_buffer.type,
        capacity=capacity,
        random_state=random_state,
        alpha=config.environment.agent.replay_buffer.alpha,
    )


class SubAgent:

    def __init__(
            self,
            action_space: ActionSpace,
            action_space_dim: int,
            state_space_dim: int,
            learning_rate: float,
            gamma: float,
            batch_size: int,
            replay_buffer: ReplayBuffer,
            random_state: np.random.RandomState,
            device: torch.device,
            config: SingleRunConfig,
            is_prioritized_buffer: bool,
            log: Logger,
            is_add_node_agent: bool = False,
            prioritized_beta: float = 0.4,
            prioritized_epsilon: float = 1e-6,
            run_mode: RunMode = RunMode.Train
    ):
        self.logger: Logger = log
        self.is_add_node_agent: bool = is_add_node_agent
        self.config: SingleRunConfig = config
        self.action_space: ActionSpace = action_space
        self.action_space_dim: int = action_space_dim
        self.state_space_dim: int = state_space_dim
        self.learning_rate: float = learning_rate
        self.gamma: float = gamma
        self.run_mode: RunMode = run_mode
        self.bootstrap_steps: int = self.config.environment.agent.double_dqn.bootstrap_steps
        self.is_prioritized_buffer: bool = is_prioritized_buffer
        self.prioritized_beta_scheduler: Scheduler = LinearScheduler(
            total=self.config.environment.agent.replay_buffer.beta_annealing_steps,
            end=1,
            start=prioritized_beta
        )
        self.train_with_expert: bool = self.config.environment.agent.double_dqn.train_with_expert
        self.use_stochastic_policy: bool = self.config.environment.agent.double_dqn.use_stochastic_policy
        self.use_full_stochastic_policy: bool = self.config.environment.agent.double_dqn.use_full_stochastic_policy
        self.theta_scheduler: Scheduler = LinearScheduler(**self.config.environment.agent.double_dqn.theta_parameters)
        self.current_theta = self.theta_scheduler.value(0)
        self.current_prioritized_beta = self.prioritized_beta_scheduler.value(0)
        self.prioritized_epsilon: float = prioritized_epsilon
        self.batch_size: int = batch_size
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.random: np.random.RandomState = random_state
        self.use_pool_node: bool = self.config.environment.nodes.use_pool_node
        self.stats_tracker: Optional[Tracker] = None
        self.device: torch.device = device
        self.net: QNet = init_qnet(
            input_size=self.state_space_dim,
            output_size=self.action_space_dim,
            config=self.config,
            device=self.device
        ).to(self.device)
        self.max_cost: Dict[str, float] = {}
        self.remaining_gap_max_factor: Dict[str, float] = {}
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.current_combined_action = None
        if self.run_mode == RunMode.Train:
            self.logger.debug(f'Agent online network device is {str(next(self.net.online.parameters()).device)}')
            self.logger.debug(f'Agent target network device is {str(next(self.net.target.parameters()).device)}')

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker

    def set_max_factors(self, max_cost: Dict[str, float], max_remaining_gap: Dict[str, float]):
        self.max_cost = max_cost
        self.remaining_gap_max_factor = max_remaining_gap

    @torch.no_grad()
    def choose(
            self,
            state: State,
            epsilon: float,
            random: float,
            expert_train_random: float,
            choose_action_calls: int,
            demand: StepData,
            resource: str,
            nodes: List[Node],
            add_node_index: Optional[int] = None,
            action_space_wrapper: Optional[ActionSpaceWrapper] = None,
            is_remove_node_sub_action: bool = False,
            combined_action: Optional[int] = None,
            pool_node: Optional[Node] = None,
    ) -> int:
        if random < epsilon:
            # random action
            return random_policy(
                action_space=self.action_space, random_state=self.random)
        else:
            # if enabled with probability theta we choose an action using the local optimal policy (the expert)
            expert_action = self._train_with_expert(choose_action_calls, expert_train_random,
                                                    state, demand, resource, nodes,
                                                    add_node_index, action_space_wrapper, is_remove_node_sub_action,
                                                    combined_action, pool_node)
            # expert_action is None if train_with_expert is disabled or the probability is greater than theta
            if expert_action is not None:
                return expert_action
            # greedy action
            if self.run_mode == RunMode.Train:
                self.net.online.eval()  # necessary for batchNorm layers and similar
            q_values = self.net(state.to_tensor(device=self.device).unsqueeze(0), model='online')
            if self.run_mode == RunMode.Train:
                self.net.online.train()  # necessary for batchNorm layers and similar
            if (self.use_stochastic_policy and self.run_mode == RunMode.Train) or self.use_full_stochastic_policy:
                return stochastic_policy(q_values, action_space=self.action_space, device=self.device)
            else:
                return greedy_policy(q_values, action_space=self.action_space, device=self.device)

    def _train_with_expert(
            self,
            choose_action_calls: int,
            expert_train_random: float,
            state: State,
            demand: StepData,
            resource: str,
            nodes: List[Node],
            add_node_index: Optional[int] = None,
            action_space_wrapper: Optional[CombinedActionSpaceWrapper] = None,
            is_remove_node_sub_action: bool = False,
            combined_action: Optional[int] = None,
            pool_node: Optional[Node] = None,
    ) -> Optional[int]:
        action: Optional[int] = None
        if self.train_with_expert and self.run_mode == RunMode.Train:
            self.current_theta = self.theta_scheduler.value(choose_action_calls,
                                                            bootstrapped_steps=self.bootstrap_steps)
            if expert_train_random < self.current_theta:
                if self.is_add_node_agent:
                    if self.use_pool_node:
                        action = local_optimal_add_node_policy_with_pool(
                            action_space_wrapper.add_node_space, demand, resource, nodes,
                            self.config.environment.get_env_resources(),
                            self.config.environment.reward.parameters['delta_target'],
                            self.config.environment.reward.parameters['gap_with_units']
                        )
                    else:
                        action = local_optimal_add_node_policy(
                            state=state,
                            action_space=self.action_space,
                            demand=demand,
                            resource=resource,
                            nodes=nodes,
                            add_node_with_node_groups=False
                        )
                else:
                    if is_remove_node_sub_action:
                        if self.use_pool_node:
                            action = local_add_or_remove_action_heuristic(
                                resource=resource, nodes=nodes, demand=demand, pool_node=pool_node,
                                action_space_wrapper=action_space_wrapper,
                                config=self.config,
                                resources=self.config.environment.get_env_resources(),
                                is_add_action=False
                            )
                        else:
                            combined_action = new_movement_local_optimal_policy(
                                add_node_index=add_node_index,
                                resource=resource,
                                nodes=nodes,
                                demand=demand,
                                action_space_wrapper=action_space_wrapper,
                                quantity_movements=self.config.environment.action_space.bucket_move,
                                resource_classes=self.config.environment.resources[0].classes,
                                config=self.config,
                                remaining_gap_max_factor=self.remaining_gap_max_factor[resource],
                                max_cost=self.max_cost[resource]
                            )
                            self.current_combined_action = combined_action
                            action, _, _ = action_space_wrapper.combined_mapping[combined_action]
                    else:
                        if self.use_pool_node:
                            action = local_add_or_remove_action_heuristic(
                                resource=resource, nodes=nodes, demand=demand, pool_node=pool_node,
                                action_space_wrapper=action_space_wrapper,
                                config=self.config,
                                resources=self.config.environment.get_env_resources(),
                                is_add_action=True
                            )
                        else:
                            _, _, action = action_space_wrapper.combined_mapping[combined_action]
            else:
                action = None
        return action

    def td_estimate(self, state: Tensor, action: Tensor):
        current_Q = self.net(state, model='online').gather(-1, action)  # Q(s, a) of the action-value function
        return current_Q

    @torch.no_grad()
    def td_target(self, reward: Tensor, next_state: Tensor, done: Tensor):
        next_state_Q = self.net(next_state, model="online")
        best_action = next_state_Q.argmax(-1).unsqueeze(-1)
        next_Q = self.net(next_state, model="target").gather(-1, best_action)

        # (1 - done) because the value of terminal states is only the reward
        return (reward + (1 - done) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate: Tensor, td_target: Tensor, weights: Tensor):
        if self.is_prioritized_buffer:
            losses = F.smooth_l1_loss(td_estimate, td_target, reduction='none')
            losses.to(self.device)
            # weighted mean loss
            loss = torch.mean(losses * weights).to(self.device)
        else:
            loss = F.smooth_l1_loss(td_estimate, td_target, reduction='mean')
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10)
        self.optimizer.step()
        return loss.item()

    def update_priorities(self, indexes, td_estimate, td_target) -> float:
        if self.is_prioritized_buffer:
            td_error = td_estimate - td_target
            new_priorities = torch.abs(td_error) + self.prioritized_epsilon
            self.replay_buffer.update_priorities(indexes=indexes, priorities=new_priorities.detach().cpu().numpy())
            return torch.mean(td_error).item()
        else:
            return 0

    def update_prioritized_beta_parameter(self, t: int):
        self.current_prioritized_beta = self.prioritized_beta_scheduler.value(t)

    def learn(self, t: int, samples: Optional[ExperienceEntry] = None, track_key='agent') -> Tuple[float, float]:
        if samples is None:
            samples = self.replay_buffer.sample(batch_size=self.batch_size,
                                                device=self.device,
                                                beta=self.current_prioritized_beta)
        state, action, reward, next_state, done, weights, indexes = samples
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        if t == 0:
            message = f'Samples entities device: state {str(state.device)} | action {str(action.device)} '
            message += f' reward {str(reward.device)} | next_state {str(next_state.device)} | done {str(done.device)}'
            if weights is not None:
                message += f'| weights {str(weights.device)}'
            self.logger.debug(message)
        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        # update experience priorities
        td_error = self.update_priorities(indexes, td_est, td_tgt)
        # Backpropagation of the loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt, weights)

        self._update_stats(t, loss, td_est.mean().item(), td_error, track_key)

        return loss, td_est.mean().item()

    def _update_stats(self, t: int, loss: float, td_est: float, td_error: float, track_key: str):
        self.stats_tracker.track(f'{RunMode.Train.value}/loss/{track_key}', loss, t)
        self.stats_tracker.track(f'{RunMode.Train.value}/q_estimate_avg/{track_key}', td_est, t)
        self.stats_tracker.track(f'{RunMode.Train.value}/td_error_avg/{track_key}', td_error, t)

    def sync_with_target_net(self):
        self.net.copy_online_on_target()

    def get_agent_state(self) -> Dict[str, Any]:
        return {
            'net_state': self.net.export_state(model='online'),
            'optimizer_state': self.optimizer.state_dict()
        }

    def load_agent_state(self, agent_state: Dict[str, Any]):
        self.net.load_model(agent_state['net_state'], model='online', copy_online_on_target=True)
        self.optimizer.load_state_dict(copy.deepcopy(agent_state['optimizer_state']))

    def set_mode(self, mode: RunMode):
        if mode == RunMode.Train:
            self.net.set_train()
        if mode == RunMode.Eval or mode == RunMode.Validation:
            self.net.set_eval()


def init_sub_agent(
        agent: 'DoubleDQNAgent',
        action_space: ActionSpace,
        action_space_dim: int,
        state_space_dim: int,
        is_add_node_agent: bool,
) -> SubAgent:
    buffer = init_replay_buffer(
        config=agent.config,
        random_state=agent.random,
    )
    return SubAgent(
        action_space=action_space,
        action_space_dim=action_space_dim,
        state_space_dim=state_space_dim,
        random_state=agent.random,
        learning_rate=agent.learning_rate,
        gamma=agent.gamma,
        batch_size=agent.batch_size,
        replay_buffer=buffer,
        device=agent.device,
        config=agent.config,
        log=agent.logger,
        is_add_node_agent=is_add_node_agent,
        is_prioritized_buffer=agent.is_prioritized_buffer,
        prioritized_beta=agent.prior_beta,
        prioritized_epsilon=agent.prior_epsilon,
        run_mode=agent.mode
    )


class DoubleDQNAgent(AgentAbstract):

    def __init__(
            self,
            action_space_wrapper: ActionSpaceWrapper,
            random_state: np.random.RandomState,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            log: Logger,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            init_seed: bool = True,
            **kwargs
    ):
        super(DoubleDQNAgent, self).__init__(
            action_space_wrapper=action_space_wrapper,
            random_state=random_state,
            name=AgentType.DoubleDQN,
            action_spaces=action_spaces,
            state_spaces=state_spaces,
            log=log,
            config=config,
            mode=mode,
            **kwargs
        )
        if init_seed:
            torch.manual_seed(self.config.random_seeds.training)
        # hyperparameters
        self.learning_rate: float = self.config.environment.agent.double_dqn.learning_rate
        self.gamma: float = self.config.environment.agent.double_dqn.gamma
        self.batch_size: int = self.config.environment.agent.double_dqn.batch_size
        self.target_net_update_frequency: int = self.config.environment.agent.double_dqn.target_net_update_frequency
        self.updates_per_step: int = self.config.environment.agent.double_dqn.updates_per_step
        self.bootstrap_steps: int = self.config.environment.agent.double_dqn.bootstrap_steps
        self.train_with_expert: bool = self.config.environment.agent.double_dqn.train_with_expert
        self.current_expert_train_random: Optional[float] = None

        self.epsilon_type: EpsilonType = self.config.environment.agent.double_dqn.epsilon_type
        self.epsilon_scheduler: Scheduler = get_epsilon_scheduler(
            epsilon_type=self.epsilon_type,
            **self.config.environment.agent.double_dqn.epsilon_parameters
        )
        # prioritized experience replay hyperparameters
        self.prior_beta = self.config.environment.agent.replay_buffer.beta
        self.prior_epsilon = self.config.environment.agent.replay_buffer.prioritized_epsilon
        # internal props
        self.requires_validation = True
        self.save_agent_state = True
        self.is_full_action_split = self.config.environment.sub_agents_setup.full_action_split
        self.use_pool_node = self.config.environment.nodes.use_pool_node
        # sub agents
        self._init_sub_agents()

    def _init_sub_agents(self):
        if self.use_pool_node:
            self.add_node_agent: SubAgent = init_sub_agent(
                agent=self,
                action_space=self.action_space_wrapper.add_action_space,
                action_space_dim=self.action_spaces[ActionType.AddAction],
                state_space_dim=self.state_spaces[StateType.Add],
                is_add_node_agent=False
            )
            self.remove_node_agent: SubAgent = init_sub_agent(
                agent=self,
                action_space=self.action_space_wrapper.remove_action_space,
                action_space_dim=self.action_spaces[ActionType.RemoveAction],
                state_space_dim=self.state_spaces[StateType.Remove],
                is_add_node_agent=False
            )
        elif self.is_full_action_split:
            self.add_node_agent: SubAgent = init_sub_agent(
                agent=self,
                action_space=self.action_space_wrapper.add_node_space,
                action_space_dim=self.action_spaces[ActionType.Add],
                state_space_dim=self.state_spaces[StateType.Add],
                is_add_node_agent=True
            )
            self.remove_node_agent: SubAgent = init_sub_agent(
                agent=self,
                action_space=self.action_space_wrapper.remove_node_space,
                action_space_dim=self.action_spaces[ActionType.Remove],
                state_space_dim=self.state_spaces[StateType.Remove],
                is_add_node_agent=False
            )
            self.quantity_agent: SubAgent = init_sub_agent(
                agent=self,
                action_space=self.action_space_wrapper.quantity_space,
                action_space_dim=self.action_spaces[ActionType.Quantity],
                state_space_dim=self.state_spaces[StateType.Quantity],
                is_add_node_agent=False
            )
        else:
            self.add_node_agent: SubAgent = init_sub_agent(
                agent=self,
                action_space=self.action_space_wrapper.add_node_space,
                action_space_dim=self.action_spaces[ActionType.Add],
                state_space_dim=self.state_spaces[StateType.Add],
                is_add_node_agent=True
            )
            self.movement_agent: SubAgent = init_sub_agent(
                agent=self,
                action_space=self.action_space_wrapper.combined_space,
                action_space_dim=self.action_spaces[ActionType.Combined],
                state_space_dim=self.state_spaces[StateType.Combined],
                is_add_node_agent=False
            )

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker
        self.add_node_agent.set_stats_tracker(tracker)
        if self.is_full_action_split:
            self.remove_node_agent.set_stats_tracker(tracker)
            if not self.use_pool_node:
                self.quantity_agent.set_stats_tracker(tracker)
        else:
            self.movement_agent.set_stats_tracker(tracker)

    def set_max_factors(self, max_cost: Dict[str, float], max_remaining_gap: Dict[str, float]):
        super(DoubleDQNAgent, self).set_max_factors(max_cost, max_remaining_gap)
        self.add_node_agent.set_max_factors(max_cost, max_remaining_gap)
        if self.is_full_action_split:
            self.remove_node_agent.set_max_factors(max_cost, max_remaining_gap)
            if not self.use_pool_node:
                self.quantity_agent.set_max_factors(max_cost, max_remaining_gap)
        else:
            self.movement_agent.set_max_factors(max_cost, max_remaining_gap)

    @property
    def is_bootstrapping(self) -> bool:
        return self.choose_action_calls < self.bootstrap_steps and self.mode == RunMode.Train

    def get_action_epsilon(self) -> Optional[float]:
        if self.is_bootstrapping:
            eps = 1
        else:
            eps = self.epsilon_scheduler.value(self.choose_action_calls, bootstrapped_steps=self.bootstrap_steps)
        return eps

    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        self.current_expert_train_random = self.random.random()
        add_node = self.add_node_agent.choose(state, epsilon, random,
                                              expert_train_random=self.current_expert_train_random,
                                              choose_action_calls=self.choose_action_calls,
                                              demand=demand, resource=resource, nodes=nodes,
                                              add_node_index=None,
                                              action_space_wrapper=self.action_space_wrapper,
                                              )
        self.current_add_node = add_node
        return add_node

    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        self.current_expert_train_random = self.random.random()
        if self.epsilon_type == EpsilonType.AlternateLinearDecay and self.mode == RunMode.Train:
            if self.is_bootstrapping:
                epsilon = 1
            else:
                epsilon = self.epsilon_scheduler.value(self.choose_action_calls, self.bootstrap_steps, 0)
        add_node = self.add_node_agent.choose(state, epsilon, random,
                                              expert_train_random=self.current_expert_train_random,
                                              choose_action_calls=self.choose_action_calls,
                                              demand=demand, resource=resource, nodes=nodes,
                                              add_node_index=self.current_add_node,
                                              action_space_wrapper=self.action_space_wrapper,
                                              is_remove_node_sub_action=False, pool_node=pool_node)
        self.current_add_node = add_node
        return add_node

    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        if self.epsilon_type == EpsilonType.AlternateLinearDecay and self.mode == RunMode.Train:
            if self.is_bootstrapping:
                epsilon = 1
            else:
                epsilon = self.epsilon_scheduler.value(self.choose_action_calls, self.bootstrap_steps, 1)
        return self.remove_node_agent.choose(state, epsilon, random,
                                             expert_train_random=self.current_expert_train_random,
                                             choose_action_calls=self.choose_action_calls,
                                             demand=demand, resource=resource, nodes=nodes,
                                             add_node_index=self.current_add_node,
                                             action_space_wrapper=self.action_space_wrapper,
                                             is_remove_node_sub_action=True, pool_node=pool_node)

    def _choose_node_remove(self, state: State,
                            epsilon: Optional[float] = None, random: Optional[float] = None,
                            resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None) -> int:
        if self.is_full_action_split:
            return self.remove_node_agent.choose(state, epsilon, random,
                                                 expert_train_random=self.current_expert_train_random,
                                                 choose_action_calls=self.choose_action_calls,
                                                 demand=demand, resource=resource, nodes=nodes,
                                                 add_node_index=self.current_add_node,
                                                 action_space_wrapper=self.action_space_wrapper,
                                                 is_remove_node_sub_action=True)
        else:
            return self.current_remove_node

    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        combined_action = self.movement_agent.choose(state, epsilon, random,
                                                     expert_train_random=self.current_expert_train_random,
                                                     choose_action_calls=self.choose_action_calls,
                                                     demand=demand, resource=resource, nodes=nodes,
                                                     add_node_index=add_node,
                                                     action_space_wrapper=self.action_space_wrapper)
        return self._handle_combined_action(combined_action)

    def _choose_quantity(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None) -> int:
        if self.is_full_action_split:
            return self.quantity_agent.choose(state, epsilon, random,
                                              expert_train_random=self.current_expert_train_random,
                                              choose_action_calls=self.choose_action_calls,
                                              demand=demand, resource=resource, nodes=nodes,
                                              add_node_index=self.current_add_node,
                                              action_space_wrapper=self.action_space_wrapper,
                                              is_remove_node_sub_action=False,
                                              combined_action=self.remove_node_agent.current_combined_action)
        else:
            return self.current_quantity

    def _update_prioritized_beta_parameter(self, t: int):
        self.add_node_agent.update_prioritized_beta_parameter(t)
        if self.is_full_action_split:
            self.remove_node_agent.update_prioritized_beta_parameter(t)
            if not self.use_pool_node:
                self.quantity_agent.update_prioritized_beta_parameter(t)
        else:
            self.movement_agent.update_prioritized_beta_parameter(t)

    def push_experience(
            self,
            state_wrapper: Dict[StateType, State],
            actions: List[Action],
            next_state_wrapper: Dict[StateType, State],
            reward: float,
            done: bool
    ):
        if len(actions) == 1:
            action = actions[0]
            self.add_node_agent.replay_buffer.push(
                state_wrapper[StateType.Add], action.add_node, reward,
                next_state_wrapper[StateType.Add], done, self.device)
            if self.is_full_action_split:
                self.remove_node_agent.replay_buffer.push(
                    state_wrapper[StateType.Remove], action.remove_node, reward,
                    next_state_wrapper[StateType.Remove], done, self.device)
                self.quantity_agent.replay_buffer.push(
                    state_wrapper[StateType.Quantity], action.quantity, reward,
                    next_state_wrapper[StateType.Quantity], done, self.device)
            else:
                combined_action = self.action_space_wrapper.combined_inverted_mapping[
                    (action.remove_node, action.resource_class, action.quantity)]
                self.movement_agent.replay_buffer.push(state_wrapper[StateType.Combined], combined_action, reward,
                                                       next_state_wrapper[StateType.Combined], done, self.device)
        else:
            self.add_node_agent.replay_buffer.push(
                state_wrapper[StateType.Add], actions[0].add_node, reward,
                next_state_wrapper[StateType.Add], done, self.device)
            self.remove_node_agent.replay_buffer.push(
                state_wrapper[StateType.Remove], actions[1].remove_node, reward,
                next_state_wrapper[StateType.Remove], done, self.device)

    def _learn(self) -> Tuple[Optional[Union[AgentLoss, float]], Optional[Union[AgentQEstimate, float]]]:
        self._update_prioritized_beta_parameter(self.learn_calls)
        buffer = self.add_node_agent.replay_buffer
        if self.is_bootstrapping or self.mode != RunMode.Train or len(buffer) < self.batch_size:
            # experience buffer has not a sufficient number of entries,
            #  we are in eval mode, so no need to do learn,
            #  or we in during the initial bootstrapping
            return None, None
        else:
            add_losses, add_td_ests = [], []
            remove_losses, remove_td_ests = [], []
            quantity_losses, quantity_td_ests = [], []
            movement_losses, movement_td_ests = [], []
            for _ in range(self.updates_per_step):
                add_loss, add_td_est = self.add_node_agent.learn(t=self.learn_calls, track_key='add_node')
                if self.is_full_action_split:
                    remove_loss, remove_td_est = self.remove_node_agent.learn(t=self.learn_calls,
                                                                              track_key='remove_node')
                    remove_losses.append(remove_loss)
                    remove_td_ests.append(remove_td_est)
                    if not self.use_pool_node:
                        quantity_loss, quantity_td_est = self.quantity_agent.learn(t=self.learn_calls,
                                                                                   track_key='quantity')
                        quantity_losses.append(quantity_loss)
                        quantity_td_ests.append(quantity_td_est)
                else:
                    movement_loss, movement_td_est = self.movement_agent.learn(t=self.learn_calls, track_key='movement')
                    movement_losses.append(movement_loss)
                    movement_td_ests.append(movement_td_est)
                add_losses.append(add_loss)
                add_td_ests.append(add_td_est)
            if self.learn_calls % self.target_net_update_frequency == 0:
                # copy online nets weights on target network
                self._sync_target_net()

            self.learn_calls += 1
            if self.is_full_action_split:
                if self.use_pool_node:
                    return (
                        AgentLoss(add_net=np.mean(add_losses).item(), remove_net=np.mean(remove_losses).item(),
                                  resource_class_net=0, quantity_net=0),
                        AgentQEstimate(add_net=np.mean(add_td_ests).item(), remove_net=np.mean(remove_td_ests).item(),
                                       resource_class_net=0, quantity_net=0)
                    )
                else:
                    return (
                        AgentLoss(add_net=np.mean(add_losses).item(), remove_net=np.mean(remove_losses).item(),
                                  resource_class_net=0, quantity_net=np.mean(quantity_losses).item()),
                        AgentQEstimate(add_net=np.mean(add_td_ests).item(), remove_net=np.mean(remove_td_ests).item(),
                                       resource_class_net=0, quantity_net=np.mean(quantity_td_ests).item())
                    )
            else:
                return (
                    AgentLoss(add_net=np.mean(add_losses).item(), remove_net=np.mean(add_td_ests).item(),
                              resource_class_net=0, quantity_net=0),
                    AgentQEstimate(add_net=np.mean(movement_losses).item(), remove_net=np.mean(movement_td_ests).item(),
                                   resource_class_net=0, quantity_net=0)
                )

    def _sync_target_net(self):
        self.add_node_agent.sync_with_target_net()
        if self.is_full_action_split:
            self.remove_node_agent.sync_with_target_net()
            if not self.use_pool_node:
                self.quantity_agent.sync_with_target_net()
        else:
            self.movement_agent.sync_with_target_net()

    def get_model(self, net_type: StateType = None, policy_net=True) -> Optional[torch.nn.Module]:
        if net_type == StateType.Add:
            return self.add_node_agent.net.online
        elif net_type == StateType.Remove:
            return self.remove_node_agent.net.online
        elif net_type == StateType.Quantity:
            return self.quantity_agent.net.online
        else:
            return self.movement_agent.net.online

    def get_agent_state(self) -> Dict[str, Any]:
        agent_state = super(DoubleDQNAgent, self).get_agent_state()
        agent_state['add_net'] = self.add_node_agent.get_agent_state()
        if self.is_full_action_split:
            agent_state['remove_net'] = self.remove_node_agent.get_agent_state()
            if not self.use_pool_node:
                agent_state['quantity_net'] = self.quantity_agent.get_agent_state()
        else:
            agent_state['movement_net'] = self.movement_agent.get_agent_state()
        return agent_state

    def load_agent_state(self, agent_state: Dict[str, Any]):
        super(DoubleDQNAgent, self).load_agent_state(agent_state)
        self.add_node_agent.load_agent_state(agent_state['add_net'])
        if self.is_full_action_split:
            self.remove_node_agent.load_agent_state(agent_state['remove_net'])
            if not self.use_pool_node:
                self.quantity_agent.load_agent_state(agent_state['quantity_net'])
        else:
            self.movement_agent.load_agent_state(agent_state['movement_net'])

    def set_mode(self, mode: RunMode):
        super(DoubleDQNAgent, self).set_mode(mode)
        self.add_node_agent.set_mode(mode)
        if self.is_full_action_split:
            self.remove_node_agent.set_mode(mode)
            if not self.use_pool_node:
                self.quantity_agent.set_mode(mode)
        else:
            self.movement_agent.set_mode(mode)

    def get_prioritized_beta_parameter(self) -> Optional[float]:
        return self.add_node_agent.current_prioritized_beta

    def get_expert_train_theta(self) -> Optional[float]:
        return self.add_node_agent.current_theta
