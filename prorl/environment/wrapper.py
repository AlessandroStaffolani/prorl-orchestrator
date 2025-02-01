import functools
import math
from copy import deepcopy
from logging import Logger
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from numpy.random import RandomState

from prorl.common.config import ExportMode
from prorl.common.data_structure import RunMode
from prorl.common.stats_tracker import Tracker
from prorl.core.state import State, StateFeature, StateFeatureValue
from prorl.core.step import Step
from prorl.core.step_data import StepData
from prorl.core.timestep import TimeStep, time_step_factory_get, time_step_factory_reset
from prorl.emulator import emulator_config, EmulatorConfig
from prorl.emulator.data_structure import ModelTypes
from prorl.emulator.models import create_model_from_type
from prorl.emulator.models.abstract import AbstractModel
from prorl.environment import logger, env_config, EnvConfig
from prorl.environment.action_space import Action, ActionSpace, ActionSpaceWrapper, ActionType, \
    CombinedActionSpaceWrapper
from prorl.environment.agent import AgentType
from prorl.environment.data_structure import NodeResource, EnvResource, ResourceDistribution, \
    NodeTypeDistribution
from prorl.environment.node import Node, init_nodes
from prorl.environment.node_groups import NodeGroups
from prorl.environment.reward import RewardAbstract, get_reward_class
from prorl.environment.state_builder import StateFeatureName, build_state, get_feature_size, StateType, \
    STACKED_STATE_FEATURES
from prorl.run.config import RunConfig


def env_resource_to_node_resource(env_resources: List[EnvResource]) -> List[NodeResource]:
    return [NodeResource(
        name=res.name,
        bucket_size=res.bucket_size,
        min_buckets=res.min_buckets,
        allocated=res.allocated,
        classes=res.classes
    ) for res in env_resources]


class EnvNotInitializedError(Exception):

    def __init__(self, method, *args):
        super(EnvNotInitializedError, self).__init__(
            f'Trying to call env "{method}" method on a not initialized environment',
            *args
        )


class EnvNotReadyError(Exception):

    def __init__(self, method, *args):
        super(EnvNotReadyError, self).__init__(
            f'Trying to call env "{method}" method on a not ready environment. You must call the reset method before',
            *args
        )


class ActionInvalidError(Exception):

    def __init__(self, action: Action, field: str = None, *args):
        message = f'Action is invalid.'
        if field is not None:
            message += f' {field} value is incorrect.'
        message += f' {action}'
        super(ActionInvalidError, self).__init__(message, *args)


class ActionDisabledError(Exception):

    def __init__(self, action: Action, sub_action: str, value: int, *args):
        message = f'Action {value} for sub action {sub_action} is currently disabled. Action wrapper: {action}'
        super(ActionDisabledError, self).__init__(message, *args)


class ActionMoveSameNodeError(Exception):

    def __init__(self, action: Action, *args):
        message = f'Trying to move quantity from the same node. Action = {action}'
        super(ActionMoveSameNodeError, self).__init__(message, *args)


class ActionGroupEmpty(Exception):

    def __init__(self, group_name: str, action: Action, *args):
        message = f'Action group {group_name} is empty, no nodes belong to the group. Action = {action}'
        super(ActionGroupEmpty, self).__init__(message, *args)


class ResourceInvalidError(Exception):

    def __init__(self, resource: str, *args):
        message = f'Resource {resource} is not available'
        super(ResourceInvalidError, self).__init__(message, *args)


def _validate_single_sub_action(field: str, value: int, action: Action, action_space: ActionSpace):
    try:
        if not action_space.is_action_available(value):
            raise ActionDisabledError(action, sub_action=field, value=value)
    except IndexError:
        raise ActionInvalidError(action, field=field)


def env_function(method_name: str, need_ready: bool = False):
    def outer_wrapper_function(func):
        @functools.wraps(func)
        def inner_wrapper_function(self, *args, **kwargs):
            if self.initialized:
                if need_ready:
                    if not self.ready:
                        raise EnvNotReadyError(method=method_name)
                return func(self, *args, **kwargs)
            else:
                raise EnvNotInitializedError(method=method_name)

        return inner_wrapper_function

    return outer_wrapper_function


class EnvWrapper:

    def __init__(
            self,
            base_stations_mappings: Dict[str, str],
            run_code: str,
            run_config: RunConfig,
            auto_init=True,
            nodes_type_distribution: Optional[NodeTypeDistribution] = None,
            state_features: List[StateFeatureName] = None,
            env_configuration: Optional[EnvConfig] = None,
            emulator_configuration: Optional[EmulatorConfig] = None,
            random_state: Optional[RandomState] = None,
            log: Optional[Logger] = None,
            test_mode: bool = False,
            run_mode: RunMode = RunMode.Train,
            disk_data_path: Optional[str] = None,
            random_seed: int = 42,
            no_init_reset: bool = True
    ):
        self.env_config: EnvConfig = env_configuration if env_configuration is not None else env_config
        self.emulator_config: EmulatorConfig = emulator_configuration if emulator_configuration is not None \
            else emulator_config
        self.logger: Logger = log if log is not None else logger
        self.random: RandomState = random_state if random_state is not None else RandomState()
        self.random_seed: int = random_seed
        self.run_code: str = run_code
        # protected properties
        self._base_stations_mappings: Dict[str, str] = base_stations_mappings
        self._nodes_distribution: NodeTypeDistribution = nodes_type_distribution if nodes_type_distribution is not None \
            else self.env_config.nodes.nodes_type_distribution
        self._state_features: List[StateFeatureName] = state_features if state_features is not None \
            else self.env_config.state.features
        self._stacked_state_features: List[StateFeatureName] = [
            sf for sf in self._state_features if sf in STACKED_STATE_FEATURES]
        self._not_stacked_state_features: List[StateFeatureName] = [
            sf for sf in self._state_features if sf not in STACKED_STATE_FEATURES]
        self._base_state_features: List[StateFeatureName] = self.env_config.state.base_features
        self._initial_seed = self.random.get_state()[1][0]
        self._no_init_reset = no_init_reset
        # public properties
        self.run_mode: RunMode = run_mode
        self.run_config: RunConfig = run_config
        self.nodes: List[Node] = []
        self.pool_node: Optional[Node] = None
        self.resources_info: List[EnvResource] = []
        self.time_step: Optional[TimeStep] = None
        self.load_generator: Optional[AbstractModel] = None
        self.node_groups: Optional[NodeGroups] = None
        self.stats_tracker: Optional[Tracker] = None
        self.action_spaces: Dict[ActionType, int] = {
            ActionType.Add: 0,
            ActionType.Remove: 0,
            ActionType.ResourceClass: 0,
            ActionType.Quantity: 0,
            ActionType.Combined: 0
        }
        self.state_spaces: Dict[StateType, int] = {
            StateType.Add: 0,
            StateType.Remove: 0,
            StateType.ResourceClass: 0,
            StateType.Quantity: 0
        }
        self.is_full_split_sub_action: bool = self.env_config.sub_agents_setup.full_action_split
        self.action_space_wrapper: Optional[Union[ActionSpaceWrapper, CombinedActionSpaceWrapper]] = None
        self.reward_class: Optional[RewardAbstract] = None
        self.debug_info: Dict[str, Union[int, Step]] = {}
        self.initialized = False
        self.initial_budget: float = 0
        self.current_budget: float = 0
        self.ready = False
        self.test_mode = test_mode
        self.disk_data_path = disk_data_path
        self.episode_lives: int = self.env_config.episode.lives
        self.episode_loose_life_threshold: int = self.env_config.episode.loose_life_threshold
        self.use_pool_node = self.env_config.nodes.use_pool_node
        # running props
        self.current_state_wrapper: Optional[Dict[StateType, State]] = None
        self.current_demand: Optional[StepData] = None
        self.previous_demand: Optional[StepData] = None
        self.current_state_feature_values: Optional[Dict[StateFeatureName, List[StateFeatureValue]]] = None
        self.current_lives: int = self.episode_lives
        self.time_step_plus_one: bool = False
        self.episode_ended_lives = False
        self.step_action_cost = 0
        if auto_init:
            self.init()

    def __str__(self):
        return f'<EnvWrapper nodes={len(self.nodes)} time_steps={self.time_step.stop_step} seed={self._initial_seed} >'

    @property
    def n_nodes(self) -> int:
        return self.env_config.nodes.n_nodes

    @property
    def is_a_null_step(self) -> bool:
        if self.current_time_step.total_steps % self.env_config.reward.null_between_interval != 0:
            return True
        else:
            return False

    def init(self):
        """
        Init all the internal components of the environment
        """
        self._init_resource_info()
        node_resources: List[NodeResource] = env_resource_to_node_resource(self.resources_info)
        self._init_nodes(node_resources)
        self._init_load_generator()
        self._init_time_step()
        self._init_node_groups()
        self._init_action_space()
        self._init_reward_class()
        self._init_spaces()
        self._init_budget_feature()
        self._init_debug_info()
        self.initialized = True
        if self.run_mode != RunMode.Validation and not self._no_init_reset:
            model_step_size = self.load_generator.model_step_size
            time_step_size = self.time_step.step_size
            step_size = model_step_size / time_step_size
            self.logger.debug(f'Environment {self.run_code} initialized with {self.n_nodes} nodes,'
                              f' {self.episode_lives} lives, reward function {self.reward_class.name}, '
                              f'step size {step_size}, {len(self._base_state_features)} base state features'
                              f' {len(self.env_config.resources)} '
                              f'resources and {self.env_config.providers.n_providers} providers')

    def _is_add_node_with_node_groups(self) -> bool:
        add_node_features = self.env_config.state.base_features
        return StateFeatureName.NodeGrouped in add_node_features

    def _init_resource_info(self):
        self.resources_info = self.env_config.get_env_resources()

    def _init_debug_info(self):
        if self.logger.level <= 10:
            self.debug_info = {
                'days_generated': 0,
                'steps_generated': 0,
                'steps_remaining': 0,
                'stop_step': self.time_step.stop_step
            }

    def _init_nodes(self, node_resources: List[NodeResource]):
        self.nodes, self.pool_node = init_nodes(
            node_resources=node_resources,
            env_config=self.env_config,
            resources_info=self.resources_info,
            base_stations_mappings=self._base_stations_mappings,
            nodes_distribution=self._nodes_distribution,
            random=self.random
        )
        assert len(self.nodes) == self.n_nodes
        if self.use_pool_node:
            assert self.pool_node is not None
        else:
            assert self.pool_node is None

    def _init_load_generator(self):
        base_station_names: List[str] = [node.base_station_type for node in self.nodes]
        resource_names: List[str] = [res.name for res in self.resources_info]
        self.load_generator: AbstractModel = create_model_from_type(
            model_type=self.emulator_config.model.type,
            base_station_names=base_station_names,
            resource_names=resource_names,
            em_config=self.emulator_config,
            random_state=self.random,
            log=self.logger,
            disk_data_folder=self.disk_data_path if not self.test_mode else None,
            run_mode=self.run_mode,
            random_seed=self.random_seed,
        )

    def _init_time_step(self):
        config = self.run_config
        model_start_step: Optional[Step] = self.load_generator.get_start_step(config.step_size)
        model_stop_step: Union[None, int, Step] = self.load_generator.get_stop_step(config.step_size)
        config_initial_date = config.initial_date
        config_stop_step = config.stop_step
        config_stop_date = config.stop_date
        if model_start_step is not None:
            config_initial_date = model_start_step
        if model_stop_step is not None:
            if isinstance(model_stop_step, int):
                config_stop_step = model_stop_step
            elif isinstance(model_stop_step, Step):
                config_stop_date = model_stop_step
        self.time_step: TimeStep = time_step_factory_get(
            run_code=self.run_code,
            step_per_second=config.step_per_second,
            step_size=config.step_size,
            stop_step=config_stop_step,
            stop_date=config_stop_date,
            initial_date=config_initial_date,
            logger=self.logger
        )
        if AgentType.is_policy_gradient(self.env_config.agent.type) or self.run_config.use_on_policy_agent:
            pass
            # self.time_step.stop_step += self.time_step.step_size
            # self.time_step_plus_one = True
        # multiply the time steps for the number of state stacked
        if self.env_config.state.stack_n_states > 1:
            self.time_step.stop_step += self.time_step.step_size
        self.time_step.stop_step *= self.env_config.state.stack_n_states

    def _init_node_groups(self):
        if self.env_config.node_groups.enabled:
            self.node_groups: NodeGroups = NodeGroups(
                quantity_actions=self.env_config.action_space.bucket_move,
                resources=self.resources_info,
                random_state=self.random,
                node_groups_with_resource_classes=self.env_config.state.node_groups_with_resource_classes
            )

    def _init_action_space(self):
        resource_classes = self.env_config.resources[0].classes
        node_groups = self.node_groups if self._is_add_node_with_node_groups() else None
        if self.env_config.agent.combine_last_sub_actions:
            self.action_space_wrapper: Union[
                ActionSpaceWrapper, CombinedActionSpaceWrapper] = CombinedActionSpaceWrapper(
                n_nodes=len(self.nodes),
                n_quantities=len(self.env_config.action_space.bucket_move),
                n_resource_classes=len(resource_classes),
                quantities_mapping={i: a for i, a in enumerate(self.env_config.action_space.bucket_move)},
                resource_classes_mapping={i: a for i, a in enumerate(resource_classes.keys())},
                add_wait_action=True,
                node_groups=node_groups,
                add_full_space=self.env_config.sub_agents_setup.add_full_space,
            )
        else:
            self.action_space_wrapper: Union[ActionSpaceWrapper, CombinedActionSpaceWrapper] = ActionSpaceWrapper(
                n_nodes=len(self.nodes),
                n_quantities=len(self.env_config.action_space.bucket_move),
                n_resource_classes=len(resource_classes),
                quantities_mapping={i: a for i, a in enumerate(self.env_config.action_space.bucket_move)},
                resource_classes_mapping={i: a for i, a in enumerate(resource_classes.keys())},
                add_wait_action=True,
                node_groups=node_groups,
                add_full_space=self.env_config.sub_agents_setup.add_full_space,
            )

    def _init_reward_class(self):
        self.reward_class: RewardAbstract = get_reward_class(
            reward_type=self.env_config.reward.type,
            env_config=self.env_config,
            resources=self.resources_info,
            n_nodes=self.n_nodes,
            disable_cost=self.env_config.reward.disable_cost,
            run_mode=self.run_mode,
            action_per_step=self.load_generator.model_step_size // self.time_step.step_size,
            **self.env_config.reward.parameters
        )
        for res in self.resources_info:
            self.reward_class.set_remaining_gap_max_factor(res.name, -res.total_available / 2)

    def _init_spaces(self):
        # action space
        self.action_spaces[ActionType.Add] = self.action_space_wrapper.add_node_space.size()
        self.action_spaces[ActionType.Remove] = self.action_space_wrapper.remove_node_space.size()
        self.action_spaces[ActionType.ResourceClass] = self.action_space_wrapper.resource_classes_space.size()
        self.action_spaces[ActionType.Quantity] = self.action_space_wrapper.quantity_space.size()
        if isinstance(self.action_space_wrapper, CombinedActionSpaceWrapper):
            self.action_spaces[ActionType.Combined] = self.action_space_wrapper.combined_space.size()
            self.action_spaces[ActionType.AddAction] = self.action_space_wrapper.add_action_space.size()
            self.action_spaces[ActionType.RemoveAction] = self.action_space_wrapper.remove_action_space.size()
        if self.env_config.sub_agents_setup.add_full_space:
            self.action_spaces[ActionType.Combined] = self.action_space_wrapper.full_space.size()
        # states space
        add_node_size = get_feature_size(StateFeatureName.NodeAdd, self.nodes, self.resources_info,
                                         node_groups=None,
                                         additional_properties=self.env_config.state.additional_properties,
                                         stacked_states=self.env_config.state.stack_n_states)
        remove_node_size = get_feature_size(StateFeatureName.NodeRemove, self.nodes, self.resources_info,
                                            node_groups=None,
                                            additional_properties=self.env_config.state.additional_properties,
                                            stacked_states=self.env_config.state.stack_n_states)
        resource_class_size = get_feature_size(StateFeatureName.ResourceClass, self.nodes, self.resources_info,
                                               node_groups=None,
                                               additional_properties=self.env_config.state.additional_properties,
                                               stacked_states=self.env_config.state.stack_n_states)
        base_features_size = 0
        for feature in self._base_state_features:
            if feature != StateFeatureName.NodeRemove and feature != StateFeatureName.NodeAdd \
                    and feature != StateFeatureName.ResourceClass and feature != StateFeatureName.PreviousAddAction \
                    and feature != StateFeatureName.PreviousRemoveAction:
                base_features_size += get_feature_size(
                    feature, self.nodes, self.resources_info,
                    node_groups=self.node_groups,
                    additional_properties=self.env_config.state.additional_properties,
                    stacked_states=self.env_config.state.stack_n_states)
        self.state_spaces[StateType.Add] = base_features_size
        if StateFeatureName.PreviousAddAction in self._base_state_features:
            self.state_spaces[StateType.Add] += get_feature_size(
                StateFeatureName.PreviousAddAction, self.nodes, self.resources_info,
                additional_properties=self.env_config.state.additional_properties)
        if self.use_pool_node and StateFeatureName.NodeAdd not in self._base_state_features:
            add_node_size = 0
        self.state_spaces[StateType.Remove] = base_features_size + add_node_size
        if StateFeatureName.PreviousRemoveAction in self._base_state_features:
            self.state_spaces[StateType.Remove] += get_feature_size(
                StateFeatureName.PreviousRemoveAction, self.nodes, self.resources_info,
                additional_properties=self.env_config.state.additional_properties)
        self.state_spaces[StateType.ResourceClass] = base_features_size + add_node_size + remove_node_size
        self.state_spaces[StateType.Quantity] = base_features_size + add_node_size + remove_node_size
        # self.state_spaces[StateType.Quantity] = other_features_size + add_node_size \
        #                                         + remove_node_size + resource_class_size
        self.state_spaces[StateType.Combined] = base_features_size + add_node_size

    def _init_budget_feature(self):
        if self.env_config.budget_feature.enabled:
            self.initial_budget = self.env_config.budget_feature.budget
            self.current_budget = self.env_config.budget_feature.budget

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker

    def _populate_node_groups(self, current_demand: StepData):
        if self.node_groups is not None:
            self.node_groups.reset()
            self.node_groups.add_all_nodes(nodes=self.nodes, current_demand=current_demand)

    def _end_bootstrapping_phase(self):
        self._reset_env(reset_time_step=True, reset_generator=True, reset_node_groups=True, reset_debug_info=True,
                        reset_nodes=True, reset_reward=True)
        self.logger.info(f'Completed bootstrapping phase after {self.run_config.bootstrapping_steps} steps. '
                         f'Environment has been reset')

    def _build_state(self, features: List[StateFeatureName], current_demand: StepData,
                     current_step: Step, node_groups: Optional[NodeGroups] = None,
                     previous_actions: Optional[List[Action]] = None) -> State:
        previous_add_action = None
        previous_remove_action = None
        if previous_actions is not None:
            previous_add_action = previous_actions[0].add_node
            previous_remove_action = previous_actions[1].remove_node
        state, state_features = build_state(
            features=features,
            resources=self.resources_info,
            nodes=self.nodes,
            current_load=current_demand,
            dtype=np.float64,
            normalized=self.env_config.state.normalized,
            floor_deltas=self.env_config.state.floor_deltas,
            node_groups=node_groups,
            additional_properties=self.env_config.state.additional_properties,
            current_step=current_step,
            current_budget=self.current_budget,
            initial_budget=self.initial_budget,
            previous_demand=self.previous_demand,
            state_feature_values=self.current_state_feature_values,
            system_load=self.load_generator.system_load,
            pool_node=self.pool_node,
            previous_add_action=previous_add_action,
            previous_remove_action=previous_remove_action,
            lives=self.current_lives,
            initial_lives=self.episode_lives,
        )
        self.current_state_feature_values = state_features
        return state

    def _next_step(self, increase_step: bool) -> Step:
        if increase_step:
            return self.time_step.next()
        else:
            return self.time_step.current_step

    def _merge_stacked_and_not_stacked_state(self, stacked: np.ndarray, not_stacked: State) -> State:
        features: List[StateFeature] = []
        values = stacked.tolist()
        features.append(
            StateFeature(name=self._stacked_state_features[0].name, size=len(stacked), start=0)
        )
        previous_start = len(stacked)
        for f in not_stacked.features():
            f_values = not_stacked.get_feature_value(f.name)
            size = len(f_values)
            values += f_values.tolist()
            features.append(
                StateFeature(name=f.name, size=size, start=previous_start)
            )
            previous_start += size
        return State(values, features=tuple(features))

    def _get_full_action_split_state(
            self,
            current_demand: StepData,
            state_step: Step,
            previous_actions: Optional[List[Action]] = None
    ) -> Dict[StateType, State]:
        add_node_state_features: List[StateFeatureName] = []
        remove_node_state_features: List[StateFeatureName] = []
        quantity_state_features: List[StateFeatureName] = []

        for feature in self._base_state_features:
            if feature != StateFeatureName.NodeAdd and feature != StateFeatureName.PreviousAddAction \
                    and feature != StateFeatureName.PreviousRemoveAction:
                add_node_state_features.append(feature)
                remove_node_state_features.append(feature)
                quantity_state_features.append(feature)

        if self.use_pool_node and StateFeatureName.NodeAdd not in self._base_state_features:
            quantity_state_features.append(StateFeatureName.NodeRemove)
        else:
            remove_node_state_features.append(StateFeatureName.NodeAdd)
            quantity_state_features.append(StateFeatureName.NodeAdd)
            quantity_state_features.append(StateFeatureName.NodeRemove)

        if StateFeatureName.PreviousAddAction in self._base_state_features:
            add_node_state_features.append(StateFeatureName.PreviousRemoveAction)

        if StateFeatureName.PreviousRemoveAction in self._base_state_features:
            remove_node_state_features.append(StateFeatureName.PreviousAddAction)

        state_wrapper: Dict[StateType, State] = {
            StateType.Add: self._build_state(
                add_node_state_features, current_demand, state_step, self.node_groups, previous_actions),
            StateType.Remove: self._build_state(
                remove_node_state_features, current_demand, state_step, self.node_groups, previous_actions),
            StateType.Quantity: self._build_state(
                quantity_state_features, current_demand, state_step, self.node_groups, previous_actions)
        }
        assert state_wrapper[StateType.Add].size == self.state_spaces[StateType.Add]
        assert state_wrapper[StateType.Remove].size == self.state_spaces[StateType.Remove]
        assert state_wrapper[StateType.Quantity].size == self.state_spaces[StateType.Quantity]
        return state_wrapper

    def _get_state(
            self, increase_step: bool = True, previous_actions: Optional[List[Action]] = None
    ) -> Dict[StateType, State]:
        if self.current_demand is not None:
            self.previous_demand = self.current_demand
        self.current_state_feature_values = None
        state_step: Step = self._next_step(increase_step)
        current_demand: StepData = self.load_generator.generate(step=state_step, is_last_step=self.time_step.is_last)

        self._populate_node_groups(current_demand)

        if self.is_full_split_sub_action and self.env_config.sub_agents_setup.same_state_features:
            state_wrapper = self._get_full_action_split_state(current_demand, state_step, previous_actions)
        else:
            add_node_state: State = self._build_state(features=self._base_state_features,
                                                      current_demand=current_demand, current_step=state_step,
                                                      node_groups=self.node_groups, previous_actions=previous_actions)
            not_stacked_state = self._build_state(self._not_stacked_state_features, current_demand, state_step,
                                                  node_groups=None, previous_actions=previous_actions)
            if len(self._stacked_state_features):
                stacked_state = self._build_state(self._stacked_state_features, current_demand, state_step,
                                                  node_groups=None, previous_actions=previous_actions)
                # repeat for stack_n_states -1 the stacked_state
                for i in range(self.env_config.state.stack_n_states - 1):
                    state_step = self._next_step(increase_step=True)
                    current_demand = self.load_generator.generate(step=state_step, is_last_step=self.time_step.is_last)
                    tmp = self._build_state(self._stacked_state_features, current_demand, state_step,
                                            node_groups=None, previous_actions=previous_actions)
                    # merge stacked_state and tmp
                    stacked_state = np.concatenate((stacked_state, tmp))
                # merge stacked and not stacked
                state: State = self._merge_stacked_and_not_stacked_state(stacked_state, not_stacked_state)
            else:
                state: State = not_stacked_state
            state_wrapper = {
                StateType.Add: add_node_state,
                StateType.Combined: state
            }
        self.current_demand: StepData = current_demand
        self.current_state_wrapper = state_wrapper
        # self.debug()
        return state_wrapper

    def _reset_time_step(self, show_log=True, add_plus_one=True):
        config = self.run_config
        model_start_step: Optional[Step] = self.load_generator.get_start_step(config.step_size)
        model_stop_step: Union[None, int, Step] = self.load_generator.get_stop_step(config.step_size)
        config_initial_date = config.initial_date
        config_stop_step = config.stop_step
        config_stop_date = config.stop_date
        if model_start_step is not None:
            config_initial_date = model_start_step
        if model_stop_step is not None:
            if isinstance(model_stop_step, int):
                config_stop_step = model_stop_step
            elif isinstance(model_stop_step, Step):
                config_stop_date = model_stop_step
        self.time_step: TimeStep = time_step_factory_reset(
            run_code=self.run_code,
            recreate=True,
            step_per_second=config.step_per_second,
            step_size=config.step_size,
            stop_step=config_stop_step,
            stop_date=config_stop_date,
            initial_date=config_initial_date,
            logger=self.logger,
            show_log=show_log
        )
        if add_plus_one:
            if AgentType.is_policy_gradient(self.env_config.agent.type) or self.run_config.use_on_policy_agent:
                pass
                # self.time_step.stop_step += self.time_step.step_size
                # self.time_step_plus_one = True
        else:
            self.time_step_plus_one = False
        if self.env_config.state.stack_n_states > 1:
            self.time_step.stop_step += self.time_step.step_size
        self.time_step.stop_step *= self.env_config.state.stack_n_states

    def _reset_env(self, reset_time_step, reset_generator, reset_debug_info,
                   reset_node_groups, reset_nodes, reset_reward, show_log,
                   add_plus_one, reset_lives, reset_pool):
        if reset_time_step:
            self._reset_time_step(show_log, add_plus_one)
        is_full_reset = self.load_generator.is_full_reset()
        if reset_generator:
            if self.episode_ended_lives:
                is_full_reset = True
                self.episode_ended_lives = False
            self.load_generator.reset(show_log, full_reset=is_full_reset)
        if reset_debug_info:
            self._init_debug_info()
        if reset_node_groups:
            if self.node_groups is not None:
                self.node_groups.reset()
        if reset_nodes and is_full_reset:
            for node in self.nodes:
                node.reset()
        if reset_pool:
            self.pool_node.reset()
        if reset_reward:
            self.reward_class.reset()
        if reset_lives:
            self.current_lives = self.episode_lives
        self.current_budget = self.initial_budget
        if show_log:
            self.logger.debug(f'Environment {self.run_code} has been reset')

    @env_function(method_name='reset')
    def reset(self, reset_time_step=True, reset_generator=True, reset_debug_info=False,
              reset_node_groups=True, reset_nodes=True, reset_reward=False, show_log=True,
              add_plus_one=True, reset_lives=True, reset_pool=True) -> Dict[StateType, State]:
        self._reset_env(reset_time_step, reset_generator, reset_debug_info,
                        reset_node_groups, reset_nodes, reset_reward, show_log, add_plus_one, reset_lives, reset_pool)
        state: Dict[StateType, State] = self._get_state(increase_step=False)
        self.ready = True
        return state

    def reset_done(self, add_plus_one=True):
        if self.time_step.is_last:
            # full reset
            self._reset_env(reset_time_step=True, reset_generator=True, reset_node_groups=True, reset_nodes=True,
                            reset_reward=False, reset_debug_info=False,
                            show_log=False, add_plus_one=add_plus_one, reset_lives=True, reset_pool=True)
        else:
            # reset only nodes and node groups
            self._reset_env(reset_time_step=False, reset_generator=False, reset_node_groups=True, reset_nodes=True,
                            reset_reward=False, reset_debug_info=False,
                            show_log=False, add_plus_one=add_plus_one, reset_lives=True, reset_pool=True)
            # self.time_step_plus_one = add_plus_one
        state: Dict[StateType, State] = self._get_state(increase_step=False)
        return state

    def _compute_action_cost(self, action: Action, resource: str):
        resource_classes = self.nodes[0].initial_resources[resource].classes
        action_resource_class = self.action_space_wrapper.resource_classes_space.actions_mapping[action.resource_class]
        if self.action_space_wrapper.is_wait_action(action):
            units_moved = 0
            resource_unit_cost = 0
        else:
            units_moved = self.action_space_wrapper.quantity_space.actions_mapping[action.quantity]
            resource_unit_cost = resource_classes[action_resource_class]['cost']
        action_cost = units_moved * resource_unit_cost
        self.current_budget -= action_cost
        current_step = self.time_step.current_step.total_steps // self.run_config.step_size
        self.stats_tracker.track('movement_cost', action_cost, current_step)
        return action_cost

    def check_hour_satisfaction(self, resource: str) -> Optional[bool]:
        next_step = self.time_step.get_simulated_next_step()
        if next_step is None or (next_step is not None and self.current_time_step.hour != next_step.hour):
            # the next step is the end or the hour changed
            nodes_satisfied = self.n_nodes_satisfied(resource=resource)
            return nodes_satisfied == len(self.nodes)
        else:
            return None

    def compute_resource_utilization(self, resource: str, surplus: int) -> float:
        res_class = 'resource'
        for res_info in self.resources_info:
            if res_info.name == resource:
                for res_class_name, _ in res_info.classes.items():
                    res_class = res_class_name
        nodes_resources = sum([node.get_current_allocated(resource, res_class) for node in self.nodes])
        if nodes_resources == 0:
            resource_utilization = 1
        else:
            resources_needed = nodes_resources - surplus
            resource_utilization = resources_needed / nodes_resources
        return resource_utilization

    def _step_return(
            self,
            actions: List[Action],
            resource: str,
            penalty: Union[int, float] = 0,
    ) -> Tuple[Dict[StateType, State], Union[float, Tuple[float, float, float]], Dict[str, Union[int, float]]]:
        hour_satisfied = self.check_hour_satisfaction(resource)
        next_state = self._get_state(increase_step=True, previous_actions=actions)
        reward = self.reward_class.compute(
            state_wrapper=self.current_state_wrapper,
            actions=actions,
            step=self.time_step.current_step,
            demand=self.current_demand,
            nodes=self.nodes,
            resource=resource,
            penalty=penalty,
            action_space_wrapper=self.action_space_wrapper,
            current_budget=self.current_budget,
            node_groups=self.node_groups,
            hour_satisfied=hour_satisfied
        )
        nodes_satisfied = self.n_nodes_satisfied(resource=resource)
        self._update_current_lives(nodes_satisfied)
        if self.current_budget <= 0:
            # before we construct the new state, we check if the budget is finished
            self.current_budget = self.initial_budget
        step_info = {
            'penalty': penalty,
            'nodes_satisfied': nodes_satisfied,
            'hour_satisfied': hour_satisfied
        }
        reward_info = self.reward_class.reward_info()
        if reward_info is not None:
            step_info['reward_info'] = reward_info
            # if 'cost' in reward_info:
            #     self.step_action_cost += reward_info['cost']
            #     if not self.is_a_null_step:
            #         reward_info['cost'] = self.step_action_cost
            #         self.step_action_cost = 0

        if self.is_a_null_step:
            step_info['nodes_satisfied'] = None
            step_info['hour_satisfied'] = None
            step_info['reward_info']['cost'] = None
            step_info['reward_info']['remaining_gap'] = None
            step_info['reward_info']['surplus'] = None
            step_info['resource_utilization'] = None
        else:
            resource_utilization = self.compute_resource_utilization(resource, step_info['reward_info']['surplus'])
            step_info['resource_utilization'] = resource_utilization

        done = self._episode_is_done()
        step_info['done'] = done

        return next_state, reward, step_info

    def _update_current_lives(self, satisfied_nodes: int):
        if self.env_config.episode.enabled and self.run_mode == RunMode.Train:
            if self.reward_class.null_between_interval is not None \
                    and self.current_time_step.total_steps % self.reward_class.null_between_interval != 0:
                return
            unsatisfied_nodes = len(self.nodes) - satisfied_nodes
            if unsatisfied_nodes > self.episode_loose_life_threshold:
                self.current_lives -= 1

    def _episode_is_done(self):
        self.episode_ended_lives = False
        if self.env_config.episode.enabled:
            if self.current_lives <= 0:
                self.episode_ended_lives = True
                return True
        return self.time_step.is_last
        # if self.emulator_config.model.type == ModelTypes.TimDatasetModel and False:
        #     step_size = self.time_step.step_size
        #     week_steps = Step.from_str('1W').total_steps // step_size
        #     initial_total_steps = self.time_step.initial_date.total_steps
        #     if self.time_step_plus_one:
        #         initial_total_steps += step_size
        #     now_total_steps = self.current_time_step.total_steps
        #     difference = now_total_steps - initial_total_steps
        #     difference = difference // step_size
        #     if (difference > 0 and difference % week_steps == 0) or self.time_step.is_last:
        #         return True
        #     else:
        #         return False
        # else:
        #     return self.time_step.is_last

    def _validate_action(self, action: Action, resource: str):
        add_node, remove_node, resource_class, quantity = action
        _validate_single_sub_action(ActionType.Add.value, add_node, action, self.action_space_wrapper.add_node_space)
        if self.env_config.agent.combine_last_sub_actions and not self.env_config.sub_agents_setup.full_action_split:
            combined_action_val = (action.remove_node, action.resource_class, action.quantity)
            combined_action_index = self.action_space_wrapper.combined_inverted_mapping[combined_action_val]
            _validate_single_sub_action(ActionType.Combined.value, combined_action_index, action,
                                        self.action_space_wrapper.combined_space)
        else:
            _validate_single_sub_action(ActionType.Remove.value, remove_node, action,
                                        self.action_space_wrapper.remove_node_space)
            _validate_single_sub_action(ActionType.ResourceClass.value, resource_class, action,
                                        self.action_space_wrapper.resource_classes_space)
            _validate_single_sub_action(ActionType.Quantity.value, quantity, action,
                                        self.action_space_wrapper.quantity_space)
        if not self.action_space_wrapper.is_wait_action(action):
            if self._is_add_node_with_node_groups():
                add_node_group = self.action_space_wrapper.add_node_space.actions_mapping[add_node]
                if len(self.node_groups.get_group_by_name(add_node_group, resource)) == 0:
                    raise ActionGroupEmpty(add_node_group, action)
                add_node_index = self.node_groups.sample_group(group_name=add_node_group, resource_name=resource)
                if add_node_index == remove_node:
                    raise ActionMoveSameNodeError(action)
            else:
                if add_node == remove_node:
                    raise ActionMoveSameNodeError(action)

    def _validate_resource(self, resource_name: str):
        for resource in self.resources_info:
            if resource_name == resource.name:
                return True
        raise ResourceInvalidError(resource_name)

    def _unmask_actions(self):
        self.action_space_wrapper.unmask_all()

    def n_nodes_satisfied(self, resource: str) -> int:
        count = 0
        for i, node in enumerate(self.nodes):
            node_demand = self.current_demand[resource][i][node.base_station_type]
            if node.allocation_sufficient(node_demand, resource):
                count += 1
        return count

    def _get_nodes_indexes(self, action: Action, resource: str) -> Tuple[int, int]:
        add_node, remove_node, _, _ = action
        if remove_node == -1:
            remove_node_index = -1
        else:
            remove_node_index = self.action_space_wrapper.remove_node_space.actions_mapping[remove_node]
        if self.node_groups is None or not self._is_add_node_with_node_groups():
            if add_node == -1:
                add_node_index = -1
            else:
                add_node_index = self.action_space_wrapper.add_node_space.actions_mapping[add_node]
            return remove_node_index, add_node_index
        else:
            add_node_group = self.action_space_wrapper.add_node_space.actions_mapping[add_node]
            add_node_index = self.node_groups.sample_group(group_name=add_node_group, resource_name=resource)
            return remove_node_index, add_node_index

    def _is_action_possible(self, remove_node_index: int, quantity_val: int,
                            resource_class_val: str, resource: str) -> bool:
        allocated = self.nodes[remove_node_index].get_current_allocated(resource, res_class=resource_class_val)
        return allocated - quantity_val >= 0

    def _is_node_satisfied(self, demand: StepData, node_index: int, resource: str) -> bool:
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        node = self.nodes[node_index]
        node_demand: float = resource_demand[node_index][node.base_station_type]
        difference = math.floor(node.get_current_allocated(resource) - node_demand)
        return difference >= 0

    @property
    def current_time_step(self) -> Step:
        return self.time_step.current_step

    def apply_action(
            self,
            action: Action,
            resource: str,
            penalty: float
    ) -> float:
        add_node, remove_node, resource_class, quantity = action
        self._validate_resource(resource)
        self._validate_action(action, resource)
        if not self.action_space_wrapper.is_wait_action(action):
            # we need to move some resource buckets
            resource_class_val = self.action_space_wrapper.resource_classes_space.actions_mapping[resource_class]
            quantity_val = self.action_space_wrapper.quantity_space.actions_mapping[quantity]
            remove_node_index, add_node_index = self._get_nodes_indexes(action, resource)
            if remove_node_index == -1:
                node_remove = self.pool_node
            else:
                node_remove = self.nodes[remove_node_index]
            if add_node_index == -1:
                node_add = self.pool_node
            else:
                node_add = self.nodes[add_node_index]
            if node_remove.get_current_allocated(resource, resource_class_val) >= quantity_val:
                node_remove.remove(quantity_val, resource, res_class=resource_class_val)
                node_add.allocate(quantity_val, resource, res_class=resource_class_val)
                penalty += 0
            else:
                penalty += self.env_config.reward.invalid_action_penalty * self.n_nodes
                self.logger.warning('Agent did an action that caused the penalty')
        return penalty

    @env_function(method_name='step', need_ready=True)
    def step(
            self,
            actions: List[Action],
            resource: str
    ) -> Tuple[Dict[str, State], Union[float, Tuple[float, float, float]], Dict[str, Union[int, float]]]:
        total_penalty = 0
        for action in actions:
            total_penalty = self.apply_action(action, resource, total_penalty)
        self._unmask_actions()
        nodes_units = [n.get_current_allocated(resource, 'resource') for n in self.nodes]
        pool_units = self.pool_node.get_current_allocated(resource, 'resource')
        assert sum(nodes_units) + pool_units == self.resources_info[0].total_units
        return self._step_return(actions=actions, resource=resource, penalty=total_penalty)

    def is_last_step(self) -> bool:
        if self.time_step.is_last:
            return True
        else:
            return False


class DummyObject(object):

    def __init__(self):
        self.level = 40

    def __getattr__(self, name):
        return lambda *x: None


def get_max_initial_gap(env: EnvWrapper, resource, n_seeds: int = 100) -> float:
    main_state = np.random.RandomState(seed=1)
    values = []
    emu_config = EmulatorConfig(env.emulator_config.root_dir,
                                **deepcopy(env.emulator_config.export(mode=ExportMode.DICT)))
    e_config = EnvConfig(env.env_config.root_dir, **deepcopy(env.env_config.export(mode=ExportMode.DICT)))
    r_config = RunConfig(**deepcopy(env.run_config.export(mode=ExportMode.DICT)))
    for i in range(n_seeds):
        seed = main_state.randint(2, 10000)
        random_state = np.random.RandomState(seed)
        wrapper = EnvWrapper(
            base_stations_mappings=emu_config.model.base_station_name_mappings,
            run_code=f'{env.run_code}_{i}',
            auto_init=True,
            nodes_type_distribution=e_config.nodes.nodes_type_distribution,
            state_features=e_config.state.features,
            run_config=r_config,
            env_configuration=e_config,
            emulator_configuration=emu_config,
            random_state=random_state,
            log=DummyObject(),
            test_mode=True
        )
        wrapper.reset()
        values.append(wrapper.reward_class.remaining_gap_max_factor[resource])
    return min(values)
