from abc import abstractmethod
from logging import Logger
from typing import Optional, Dict, Tuple, Union, List

import numpy as np

from prorl import SingleRunConfig
from prorl.core.state import State
from prorl.core.step import Step
from prorl.core.step_data import StepData
from prorl.environment.action_space import ActionSpaceWrapper, Action, ActionType
from prorl.environment.agent import AgentType
from prorl.environment.agent.abstract import AgentAbstract, AgentLoss, AgentQEstimate
from prorl.environment.agent.policy import random_policy
from prorl.environment.node import Node
from prorl.environment.node_groups import NodeGroups
from prorl.environment.state_builder import StateType


class BaselineAgent(AgentAbstract):

    def __init__(self,
                 action_space_wrapper: ActionSpaceWrapper,
                 random_state: np.random.RandomState,
                 action_spaces: Dict[ActionType, int],
                 state_spaces: Dict[StateType, int],
                 log: Logger,
                 name: AgentType,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(BaselineAgent, self).__init__(action_space_wrapper,
                                            random_state=random_state,
                                            name=name,
                                            action_spaces=action_spaces,
                                            state_spaces=state_spaces,
                                            log=log,
                                            config=config,
                                            **kwargs)
        self.is_baseline = True

    @abstractmethod
    def _choose_node_remove(self, state: State,
                            epsilon: Optional[float] = None, random: Optional[float] = None,
                            resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None, **kwargs) -> int:
        pass

    @abstractmethod
    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        pass

    @abstractmethod
    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        pass

    # @abstractmethod
    # def _choose_resource_class(self, state: State,
    #                            epsilon: Optional[float] = None, random: Optional[float] = None,
    #                            resource: Optional[str] = None) -> int:
    #     pass

    @abstractmethod
    def _choose_quantity(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None, **kwargs) -> int:
        pass

    @abstractmethod
    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        pass

    @abstractmethod
    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        pass

    def _learn(self) -> Tuple[Optional[Union[AgentLoss, float]], Optional[Union[AgentQEstimate, float]]]:
        return None, None

    def push_experience(
            self,
            state_wrapper: Union[Dict[str, State], np.ndarray],
            action: Union[Action, np.ndarray],
            next_state_wrapper: Union[Dict[str, State], np.ndarray],
            reward: Union[float, Tuple[float, float], np.ndarray],
            done: bool,
    ):
        pass


class RandomAgent(BaselineAgent):

    def __init__(self,
                 action_space_wrapper: ActionSpaceWrapper,
                 random_state: np.random.RandomState,
                 action_spaces: Dict[ActionType, int],
                 state_spaces: Dict[StateType, int],
                 log: Logger,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(RandomAgent, self).__init__(action_space_wrapper,
                                          random_state=random_state,
                                          name=AgentType.Random,
                                          action_spaces=action_spaces,
                                          state_spaces=state_spaces,
                                          log=log,
                                          config=config,
                                          **kwargs)

    def _choose_node_remove(self, state: State,
                            epsilon: Optional[float] = None, random: Optional[float] = None,
                            resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None, **kwargs) -> int:
        return random_policy(
            action_space=self.action_space_wrapper.remove_node_space,
            random_state=self.random
        )

    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        return random_policy(
            action_space=self.action_space_wrapper.add_action_space,
            random_state=self.random
        )

    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        return random_policy(
            action_space=self.action_space_wrapper.remove_action_space,
            random_state=self.random
        )

    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        return random_policy(
            action_space=self.action_space_wrapper.add_node_space,
            random_state=self.random
        )

    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        combined_action = random_policy(
            action_space=self.action_space_wrapper.combined_space,
            random_state=self.random
        )
        return self._handle_combined_action(combined_action)

    # def _choose_resource_class(self, state: State,
    #                            epsilon: Optional[float] = None, random: Optional[float] = None,
    #                            resource: Optional[str] = None) -> int:
    #     if self.combined_sub_actions:
    #         return self.current_resource_class
    #     return random_policy(
    #         action_space=self.action_space_wrapper.resource_classes_space,
    #         random_state=self.random
    #     )

    def _choose_quantity(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None, **kwargs) -> int:
        if self.combined_sub_actions and not self.config.environment.sub_agents_setup.full_action_split:
            return self.current_quantity
        return random_policy(
            action_space=self.action_space_wrapper.quantity_space,
            random_state=self.random
        )
