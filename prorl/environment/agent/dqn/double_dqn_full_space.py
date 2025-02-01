from logging import Logger
from typing import Optional, Dict, Union, Tuple, Any, List

import numpy as np
import torch

from prorl import SingleRunConfig
from prorl.common.data_structure import RunMode
from prorl.common.stats_tracker import Tracker
from prorl.core.state import State
from prorl.core.step_data import StepData
from prorl.environment.action_space import ActionSpaceWrapper, Action, ActionType, CombinedActionSpaceWrapper, \
    WAIT_ACTION
from prorl.environment.agent import AgentType
from prorl.environment.agent.abstract import AgentAbstract, AgentLoss, AgentQEstimate, SubActionUtils
from prorl.environment.agent.dqn.double_dqn import SubAgent, init_sub_agent
from prorl.environment.agent.parameter_schedulers import EpsilonType, get_epsilon_scheduler, Scheduler
from prorl.environment.node import Node
from prorl.environment.node_groups import NodeGroups
from prorl.environment.state_builder import StateType


class SpecialSubActionUtils(SubActionUtils):

    def __init__(self,
                 action_space_wrapper: Union[ActionSpaceWrapper, CombinedActionSpaceWrapper],
                 config: SingleRunConfig,
                 are_combined_actions: bool = True,
                 is_multi_reward: bool = True
                 ):
        super().__init__(action_space_wrapper, config, are_combined_actions, is_multi_reward)

    def add_action_shrink(self, pool_node: Node, resource: str, nodes: List[Node]):
        self.action_space_wrapper.full_space.unmask_all()
        for action_index in self.action_space_wrapper.full_space.get_available_actions():
            add_node, remove_node = self.action_space_wrapper.full_space.actions_mapping[action_index]
            if pool_node.get_current_allocated(resource) <= 0 and add_node != WAIT_ACTION:
                # pool empty we can only do actions where add node is WAIT
                self.action_space_wrapper.full_space.disable_action(action_index)
            elif pool_node.get_current_allocated(resource) > 0 and add_node == WAIT_ACTION:
                # if pool not empty, we can't wait for add node
                self.action_space_wrapper.full_space.disable_action(action_index)
            if remove_node != WAIT_ACTION:
                node, resource_class, quantity = self.action_space_wrapper.remove_action_space.actions_mapping[remove_node]
                if nodes[node].get_current_resource_class_units(resource, resource_class) < quantity:
                    self.action_space_wrapper.full_space.disable_action(action_index)

    def remove_action_shrink(self, nodes: List[Node], resource: str, add_action_index: int):
        pass


class DoubleDQNFullSpaceAgent(AgentAbstract):

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
        super(DoubleDQNFullSpaceAgent, self).__init__(
            action_space_wrapper=action_space_wrapper,
            random_state=random_state,
            name=AgentType.DoubleDQNFullSpace,
            action_spaces=action_spaces,
            state_spaces=state_spaces,
            log=log,
            config=config,
            mode=mode,
            **kwargs
        )
        if init_seed:
            torch.manual_seed(self.config.random_seeds.training)
        # sub action utils override
        self.sub_actions_utils: SpecialSubActionUtils = SpecialSubActionUtils(
            action_space_wrapper=self.action_space_wrapper,
            config=self.config,
            are_combined_actions=self.combined_sub_actions,
            is_multi_reward=self.is_multi_reward
        )
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
        self.sub_agent: SubAgent = init_sub_agent(
            agent=self,
            action_space=self.action_space_wrapper.full_space,
            action_space_dim=self.action_spaces[ActionType.Combined],
            state_space_dim=self.state_spaces[StateType.Add],
            is_add_node_agent=False
        )

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker
        self.sub_agent.set_stats_tracker(tracker)

    def set_max_factors(self, max_cost: Dict[str, float], max_remaining_gap: Dict[str, float]):
        super().set_max_factors(max_cost, max_remaining_gap)
        self.sub_agent.set_max_factors(max_cost, max_remaining_gap)

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
        raise NotImplemented()

    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        # TODO
        self.current_expert_train_random = self.random.random()
        if self.epsilon_type == EpsilonType.AlternateLinearDecay and self.mode == RunMode.Train:
            if self.is_bootstrapping:
                epsilon = 1
            else:
                epsilon = self.epsilon_scheduler.value(self.choose_action_calls, self.bootstrap_steps, 0)
        full_action = self.sub_agent.choose(state, epsilon, random,
                                            expert_train_random=self.current_expert_train_random,
                                            choose_action_calls=self.choose_action_calls,
                                            demand=demand, resource=resource, nodes=nodes,
                                            add_node_index=self.current_add_node,
                                            action_space_wrapper=self.action_space_wrapper,
                                            is_remove_node_sub_action=False, pool_node=pool_node)
        add_node, remove_node = self.action_space_wrapper.full_space.actions_mapping[full_action]
        if add_node == WAIT_ACTION:
            add_node = self.action_space_wrapper.add_action_space.wait_action_index
        if remove_node == WAIT_ACTION:
            remove_node = self.action_space_wrapper.remove_action_space.wait_action_index
        self.current_add_node = add_node
        self.current_remove_node = remove_node
        return add_node

    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        return self.current_remove_node

    def _choose_node_remove(self, state: State,
                            epsilon: Optional[float] = None, random: Optional[float] = None,
                            resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None) -> int:
        raise NotImplemented()

    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        raise NotImplemented()

    def _choose_quantity(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None) -> int:
        raise NotImplemented()

    def _update_prioritized_beta_parameter(self, t: int):
        self.sub_agent.update_prioritized_beta_parameter(t)

    def push_experience(
            self,
            state_wrapper: Dict[StateType, State],
            actions: List[Action],
            next_state_wrapper: Dict[StateType, State],
            reward: float,
            done: bool
    ):
        add_node = actions[0].add_node \
            if not self.action_space_wrapper.add_action_space.is_wait_action(actions[0].add_node) else WAIT_ACTION
        remove_node = actions[1].remove_node \
            if not self.action_space_wrapper.remove_action_space.is_wait_action(actions[1].remove_node) else WAIT_ACTION
        action = self.action_space_wrapper.full_space.inverted_actions_mapping[(add_node, remove_node)]
        self.sub_agent.replay_buffer.push(
            state_wrapper[StateType.Add], action, reward,
            next_state_wrapper[StateType.Add], done, self.device)

    def _learn(self) -> Tuple[Optional[Union[AgentLoss, float]], Optional[Union[AgentQEstimate, float]]]:
        self._update_prioritized_beta_parameter(self.learn_calls)
        buffer = self.sub_agent.replay_buffer
        if self.is_bootstrapping or self.mode != RunMode.Train or len(buffer) < self.batch_size:
            # experience buffer has not a sufficient number of entries,
            #  we are in eval mode, so no need to do learn,
            #  or we in during the initial bootstrapping
            return None, None
        else:
            losses, td_ests = [], []
            for _ in range(self.updates_per_step):
                loss, td_est = self.sub_agent.learn(t=self.learn_calls, track_key='sub_agent')
                losses.append(loss)
                td_ests.append(td_est)
            if self.learn_calls % self.target_net_update_frequency == 0:
                # copy online nets weights on target network
                self._sync_target_net()

            self.learn_calls += 1
            return (
                AgentLoss(add_net=np.mean(losses).item(), remove_net=np.mean(td_ests).item(),
                          resource_class_net=0, quantity_net=0),
                None
            )

    def _sync_target_net(self):
        self.sub_agent.sync_with_target_net()

    def get_model(self, net_type: StateType = None, policy_net=True) -> Optional[torch.nn.Module]:
        return self.sub_agent.net.online

    def get_agent_state(self) -> Dict[str, Any]:
        agent_state = super().get_agent_state()
        agent_state['sub_net'] = self.sub_agent.get_agent_state()
        return agent_state

    def load_agent_state(self, agent_state: Dict[str, Any]):
        super().load_agent_state(agent_state)
        self.sub_agent.load_agent_state(agent_state['sub_net'])

    def set_mode(self, mode: RunMode):
        super().set_mode(mode)
        self.sub_agent.set_mode(mode)

    def get_prioritized_beta_parameter(self) -> Optional[float]:
        return self.sub_agent.current_prioritized_beta

    def get_expert_train_theta(self) -> Optional[float]:
        return self.sub_agent.current_theta
