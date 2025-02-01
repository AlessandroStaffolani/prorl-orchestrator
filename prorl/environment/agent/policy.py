import math
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from prorl import SingleRunConfig
from prorl.common.enum_utils import ExtendedEnum
from prorl.common.math_util import normalize_scalar, l1_distance
from prorl.core.state import State
from prorl.core.step_data import StepData
from prorl.environment.action_space import ActionSpace, Action, CombinedActionSpaceWrapper, ActionSpaceWrapper
from prorl.environment.data_structure import ResourceClass, EnvResource
from prorl.environment.node import Node, POOL_NODE
from prorl.environment.node_groups import NodeGroups
from prorl.environment.reward import RewardFunctionType
from prorl.environment.scalarization_function import scalarize_rewards, ScalarizationFunction


class ExplorationPolicy(str, ExtendedEnum):
    Random = 'random'
    Greedy = 'epsilon'


@dataclass
class MovementInfo:
    remove_node_index: int
    resource_class: str
    quantity: int
    movement_cost: float
    add_node_delta_simulated: float
    remove_node_delta_simulated: float


def random_policy(action_space: ActionSpace, random_state: np.random.RandomState) -> int:
    return int(action_space.sample(random_state=random_state))


def greedy_policy(
        q_values: Union[torch.Tensor, np.ndarray],
        action_space: ActionSpace,
        device=torch.device('cpu')
) -> int:
    if isinstance(q_values, torch.Tensor):
        min_value = torch.tensor(float(np.finfo(np.float32).min), device=device)
        disabled_actions = torch.tensor(action_space.get_disabled_actions(), dtype=torch.long, device=device)
        q_values.index_fill_(1, disabled_actions, min_value.item())
        return int(q_values.max(1)[1].item())
    else:
        min_value = float(np.finfo(np.float32).min)
        disabled_actions = action_space.get_disabled_actions()
        q_values[disabled_actions] = min_value
        return int(q_values.argmax())


def stochastic_policy(
        q_values: torch.Tensor,
        action_space: ActionSpace,
        device: torch.device = torch.device('cpu')
) -> int:
    min_value = torch.tensor(float(np.finfo(np.float32).min), device=device)
    disabled_actions = torch.tensor(action_space.get_disabled_actions(), dtype=torch.long, device=device)
    q_values.index_fill_(1, disabled_actions, min_value.item())
    dist = Categorical(probs=F.softmax(q_values, -1))
    return dist.sample().item()


def local_optimal_add_node_policy(
        state: State,
        action_space: ActionSpace,
        demand: StepData,
        resource: str,
        nodes: List[Node],
        add_node_with_node_groups: bool = False,
) -> int:
    # we suppose the state is ordered from the most negative group to the most positive
    if add_node_with_node_groups:
        negative_groups = state[: len(state) // 2]
        for group_index, group_count in enumerate(negative_groups):
            if group_count > 0:
                return group_index
    else:
        min_delta = 0
        action_index = action_space.wait_action_index
        resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
        for i, node in enumerate(nodes):
            node_demand: float = resource_demand[i][node.base_station_type]
            delta = math.floor(node.get_current_allocated(resource) - node_demand)
            if delta < min_delta:
                min_delta = delta
                action_index = i

        return action_index
    return action_space.wait_action_index


def local_optimal_add_node_policy_with_pool(
        action_space: ActionSpace,
        demand: StepData,
        resource: str,
        nodes: List[Node],
        resources: List[EnvResource],
        delta_target: float = 0,
        gap_with_units: bool = True,
) -> int:
    pool_node_index = _get_pool_node_index(nodes)
    deltas = _get_nodes_deltas(nodes, demand, resource, gap_with_units, resources)
    deltas.insert(pool_node_index, delta_target)
    deltas = np.array(deltas)
    distances = l1_distance(deltas, delta_target)
    max_index = distances.argmax()
    target_delta = deltas[max_index]
    if target_delta < delta_target:
        # add node is target delta, it needs for resources
        return max_index
    elif target_delta > delta_target:
        # we remove resources
        return pool_node_index
    else:
        return action_space.wait_action_index


def _get_pool_node_index(nodes: List[Node]) -> Optional[int]:
    for i, node in enumerate(nodes):
        if node.base_station_type == POOL_NODE:
            return i
    return None


def _get_nodes_deltas(
        nodes: List[Node],
        demand: StepData,
        resource: str,
        gap_with_units: bool,
        resources: List[EnvResource],
) -> List[float]:
    deltas: List[float] = []
    res_capacity = 1
    if gap_with_units:
        for env_res in resources:
            if env_res.name == resource:
                for _, res_info in env_res.classes.items():
                    res_capacity = res_info['capacity']
    resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
    for i, node in enumerate(nodes):
        if node.base_station_type != POOL_NODE:
            node_demand = resource_demand[i][node.base_station_type]
            current_allocated = node.get_current_allocated(resource)
            delta = current_allocated - node_demand
            delta = delta / res_capacity
            delta = math.floor(delta)
            deltas.append(delta)
    return deltas


def local_optimal_movement_policy(
        state: State,
        add_node_index: int,
        resource: str,
        nodes: List[Node],
        demand: StepData,
        movement_action_space: ActionSpace,
        add_node_action_space: ActionSpace,
        quantity_movements: List[int],
        resource_classes: Dict[str, ResourceClass],
        is_conservative: bool = True
) -> int:
    # default action is wait
    action = movement_action_space.wait_action_index
    if add_node_index is None or add_node_action_space.is_wait_action(add_node_index):
        # if add_node_index is None the add node sub-action is wait, so we wait as well
        return action
    add_node = nodes[add_node_index]
    # get all the nodes with non negative delta
    # non_negative_nodes: List[int] = _get_non_negative_nodes_indexes(state, add_node_index, len(nodes))
    non_negative_nodes: List[int] = _get_non_negative_nodes_indexes(demand, nodes, resource, add_node_index)
    possible_movements_solved: List[MovementInfo] = []
    possible_movement_not_solved: List[MovementInfo] = []
    # check if at least one node as a positive delta
    if len(non_negative_nodes) > 0:
        add_node_delta: float = _get_node_delta(nodes, add_node_index, demand, resource)
        if add_node_delta < 0:
            # iterate through all the non-negative nodes
            for index in non_negative_nodes:
                node = nodes[index]
                # iterate through all the resource classes
                for res_class_name, res_class in resource_classes.items():
                    node_availability = node.get_current_resource_class_units(resource, res_class_name)
                    # iterate through all the quantity movements available
                    for quantity in quantity_movements:
                        if node_availability >= quantity:
                            # simulate the effect of the movement, only if the quantity is available on the node
                            add_node_delta_simulated, remove_node_delta_simulated = _simulate_movement(
                                add_node, add_node_index, node, index, res_class_name,
                                quantity, resource, demand, resource_classes
                            )
                            movement_cost = _compute_movement_cost(res_class_name, quantity, resource_classes)
                            if remove_node_delta_simulated >= 0 or not is_conservative:
                                # keep the movement only if the remove node does not become negative
                                #  or if we are not conservative
                                movement_info = MovementInfo(
                                    index, res_class_name, quantity, movement_cost, add_node_delta_simulated,
                                    remove_node_delta_simulated
                                )
                                if add_node_delta_simulated >= 0:
                                    possible_movements_solved.append(movement_info)
                                else:
                                    possible_movement_not_solved.append(movement_info)
    if len(possible_movements_solved) + len(possible_movement_not_solved) > 0:
        # pick the best movement
        action = _pick_best_movement(possible_movements_solved, possible_movement_not_solved, movement_action_space)
    if movement_action_space.is_wait_action(action):
        movement_action_space.enable_wait_action()
    return action


def _get_non_negative_nodes_indexes(
        demand: StepData,
        nodes: List[Node],
        resource: str,
        add_node_index: int
) -> List[int]:
    indexes = []
    for index, node in enumerate(nodes):
        node_delta = _get_node_delta(nodes, index, demand, resource)
        if node_delta > 0 and index != add_node_index:
            indexes.append(index)
    return indexes


def _get_node_delta(nodes: List[Node], index: int, demand: StepData, resource: str,
                    gap_with_units: bool = False, resource_classes: Dict[str, ResourceClass] = None) -> float:
    resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
    node = nodes[index]
    node_demand = resource_demand[index][node.base_station_type]
    current_allocated = node.get_current_allocated(resource)
    delta = current_allocated - node_demand
    if gap_with_units:
        res_capacity = 1
        for _, res_info in resource_classes.items():
            res_capacity = res_info['capacity']
        delta = math.floor(delta / res_capacity)
    return delta


def _simulate_movement(
        add_node: Node,
        add_node_index: int,
        remove_node: Node,
        remove_node_index: int,
        resource_class: str,
        quantity: int,
        resource: str,
        demand: StepData,
        resource_classes: Dict[str, ResourceClass],
        gap_with_units: bool = False
) -> Tuple[float, float]:
    resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
    add_node_current_allocated = add_node.get_current_allocated(resource)
    add_node_demand = resource_demand[add_node_index][add_node.base_station_type]
    remove_node_current_allocated = remove_node.get_current_allocated(resource)
    remove_node_demand = resource_demand[remove_node_index][remove_node.base_station_type]
    resource_capacity = resource_classes[resource_class]['capacity']
    capacity_moved = resource_capacity * quantity
    add_node_current_allocated += capacity_moved
    remove_node_current_allocated -= capacity_moved
    new_add_node_delta = add_node_current_allocated - add_node_demand
    new_remove_node_delta = remove_node_current_allocated - remove_node_demand
    if gap_with_units:
        new_add_node_delta = math.floor(new_add_node_delta / resource_capacity)
        new_remove_node_delta = math.floor(new_remove_node_delta / resource_capacity)
    return new_add_node_delta, new_remove_node_delta


def _compute_movement_cost(resource_class: str, quantity: int, resource_classes: Dict[str, ResourceClass]) -> float:
    resource_cost = resource_classes[resource_class]['cost']
    return resource_cost * quantity


def _get_action_from_movement_info(movement_info: MovementInfo, movement_action_space: ActionSpace) -> int:
    remove_node = movement_info.remove_node_index
    resource_class = movement_info.resource_class
    quantity = movement_info.quantity
    return movement_action_space.inverted_actions_mapping[
        (remove_node, resource_class, quantity)]


def _pick_best_movement(possible_movement_solved: List[MovementInfo],
                        possible_movement_not_solved: List[MovementInfo],
                        movement_action_space: ActionSpace) -> int:
    picked_movement_info: Optional[MovementInfo] = None
    if len(possible_movement_solved) > 0:
        # if at least one of the solution solved the problem for add node then we pick the one with the lowest cost
        min_cost: Optional[float] = None
        for movement in possible_movement_solved:
            if min_cost is None or min_cost > movement.movement_cost:
                min_cost = movement.movement_cost
                picked_movement_info = movement
    else:
        # no possible movements solve add node, so we pick the one that bring us the closest to solve it
        max_add_node_delta: Optional[float] = None
        for movement in possible_movement_not_solved:
            if max_add_node_delta is None or max_add_node_delta < movement.add_node_delta_simulated:
                max_add_node_delta = movement.add_node_delta_simulated
                picked_movement_info = movement
    return _get_action_from_movement_info(picked_movement_info, movement_action_space)


def _compute_remaining_gap(
        nodes: List[Node],
        demand: StepData,
        resource: str,
        resource_classes: Dict[str, ResourceClass],
        gap_with_units: bool = True,
) -> float:
    resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
    gap = 0.0
    for i, node in enumerate(nodes):
        node_demand = resource_demand[i][node.base_station_type]
        current_allocated = node.get_current_allocated(resource)
        difference = current_allocated - node_demand
        difference = math.floor(difference)
        if difference < 0:
            gap += difference
    if gap_with_units:
        res_capacity = 1
        for _, res_info in resource_classes.items():
            res_capacity = res_info['capacity']
        gap = np.floor(gap / res_capacity)
    return gap


def _compute_satisfied_nodes(
        nodes: List[Node],
        demand: StepData,
        resource: str
) -> int:
    satisfied = 0
    resource_demand: List[Dict[str, float]] = demand.get_resource_values(resource)
    for i in range(len(nodes)):
        node_demand: float = resource_demand[i][nodes[i].base_station_type]
        node_current_allocation = nodes[i].get_current_allocated(resource)
        difference = node_current_allocation - node_demand
        if difference >= 0:
            satisfied += 1
    return satisfied


def _estimate_utility(
        config: SingleRunConfig,
        gap: float,
        action_cost: float,
        remaining_gap_max_factor: float,
        max_cost: float,
        pre_gap: float = 0,
        satisfied_nodes: int = 0,
        n_nodes: int = 0
):
    alpha: float = config.environment.reward.parameters['alpha']
    weights: List[float] = config.environment.reward.parameters['weights']
    scalarization_function: ScalarizationFunction = config.environment.reward.parameters['scalarization_function']
    normalization_range: Tuple[int, int] = config.environment.reward.parameters['val_eval_normalization_range']
    values = [gap, action_cost, pre_gap]
    only_apply_scalarization = False
    if config.environment.reward.type == RewardFunctionType.GlobalRemainingGap:
        if config.environment.reward.parameters['normalize_objectives']:
            normalized_post_gap = normalize_scalar(gap, 0,
                                                   remaining_gap_max_factor,
                                                   normalization_range[0], normalization_range[1])
            normalized_cost = normalize_scalar(action_cost, 0,
                                               max_cost,
                                               normalization_range[0], normalization_range[1])
            normalized_pre_gap = normalize_scalar(pre_gap, 0,
                                                  remaining_gap_max_factor,
                                                  normalization_range[0], normalization_range[1])
            values = [normalized_post_gap, normalized_cost, normalized_pre_gap]
            only_apply_scalarization = False
    elif config.environment.reward.type == RewardFunctionType.MultiObjectivesSimple:
        action_cost = normalize_scalar(-action_cost, 0, -max_cost, a=0, b=1)
        if gap == 0:
            # remaining gap is 0
            gap = 1
        elif gap <= pre_gap:
            # we have more gap than before, or we waited with a gap
            gap = 0
        else:
            # we improved but it is still not solved
            gap = 0.5
        only_apply_scalarization = True
    elif config.environment.reward.type == RewardFunctionType.GlobalSatisfiedNodes:
        action_cost = normalize_scalar(-action_cost, 0, -max_cost, a=normalization_range[0], b=normalization_range[1])
        gap = normalize_scalar(satisfied_nodes, max_val=n_nodes, min_val=0,
                               a=normalization_range[0], b=normalization_range[1])
        only_apply_scalarization = True
    utility = scalarize_rewards(
        func_type=scalarization_function,
        remaining_gap=gap,
        cost=action_cost,
        initial_gap=remaining_gap_max_factor,
        max_cost=max_cost,
        values=values,
        alpha=alpha,
        weights=weights,
        normalization_range=normalization_range,
        normalize_objectives=config.environment.reward.parameters['normalize_objectives'],
        only_apply_scalarization=only_apply_scalarization
    )
    return utility


def new_movement_local_optimal_policy(
        add_node_index: int,
        resource: str,
        nodes: List[Node],
        demand: StepData,
        action_space_wrapper: CombinedActionSpaceWrapper,
        quantity_movements: List[int],
        resource_classes: Dict[str, ResourceClass],
        config: SingleRunConfig,
        remaining_gap_max_factor: float,
        max_cost: float,
) -> int:
    add_node_action_space = action_space_wrapper.add_node_space
    movement_action_space = action_space_wrapper.combined_space
    # default action is wait
    gap_with_units = config.environment.reward.parameters['gap_with_units']
    action = movement_action_space.wait_action_index
    if add_node_index is None or add_node_action_space.is_wait_action(add_node_index):
        # if add_node_index is None the add node sub-action is wait, so we wait as well
        return action
    add_node = nodes[add_node_index]
    # get all the nodes with non-negative delta
    non_negative_nodes: List[int] = _get_non_negative_nodes_indexes(demand, nodes, resource, add_node_index)
    # get the initial remaining gap
    initial_remaining_gap = _compute_remaining_gap(nodes, demand, resource, resource_classes, gap_with_units)
    initially_satisfied_nodes = _compute_satisfied_nodes(nodes, demand, resource)

    # compute the wait action utility
    max_utility = _estimate_utility(config, initial_remaining_gap, 0, remaining_gap_max_factor, max_cost,
                                    pre_gap=initial_remaining_gap,
                                    satisfied_nodes=initially_satisfied_nodes, n_nodes=len(nodes))

    # check if at least one node as a positive delta
    if len(non_negative_nodes) > 0:
        add_node_delta: float = _get_node_delta(nodes, add_node_index, demand, resource,
                                                gap_with_units, resource_classes)
        if add_node_delta < 0:
            # iterate through all the non-negative nodes
            for index in non_negative_nodes:
                node = nodes[index]
                # iterate through all the resource classes
                for res_class_name, res_class in resource_classes.items():
                    node_availability = node.get_current_resource_class_units(resource, res_class_name)
                    # iterate through all the quantity movements available
                    for quantity in quantity_movements:
                        if node_availability >= quantity:
                            pre_move_remove_node_delta = _get_node_delta(nodes, index, demand, resource,
                                                                         gap_with_units, resource_classes)
                            # simulate the effect of the movement, only if the quantity is available on the node
                            add_node_delta_simulated, remove_node_delta_simulated = _simulate_movement(
                                add_node, add_node_index, node, index, res_class_name,
                                quantity, resource, demand, resource_classes, gap_with_units=gap_with_units
                            )
                            movement_cost = _compute_movement_cost(res_class_name, quantity, resource_classes)

                            after_movement_remaining_gap = initial_remaining_gap
                            after_movement_satisfied_nodes = initially_satisfied_nodes
                            # remove the initial delta to the gap
                            after_movement_remaining_gap -= add_node_delta
                            # add the new delta to the gap only if new delta is negative
                            if add_node_delta_simulated < 0:
                                after_movement_remaining_gap += add_node_delta_simulated

                            if remove_node_delta_simulated < 0:
                                # if remove node delta is negative we add it to the remaining gap
                                after_movement_remaining_gap += remove_node_delta_simulated

                            if pre_move_remove_node_delta >= 0 and remove_node_delta_simulated < 0:
                                after_movement_satisfied_nodes -= 1

                            if add_node_delta_simulated >= 0:
                                after_movement_satisfied_nodes += 1

                            # compute the utility
                            utility = _estimate_utility(config, after_movement_remaining_gap, movement_cost,
                                                        remaining_gap_max_factor, max_cost,
                                                        pre_gap=initial_remaining_gap,
                                                        satisfied_nodes=after_movement_satisfied_nodes,
                                                        n_nodes=len(nodes))

                            # keep track of the max utility
                            if utility > max_utility:
                                max_utility = utility
                                max_utility_movement_info = MovementInfo(
                                    index, res_class_name, quantity, movement_cost, add_node_delta_simulated,
                                    remove_node_delta_simulated
                                )
                                action = _get_action_from_movement_info(
                                    max_utility_movement_info, movement_action_space)

    if movement_action_space.is_wait_action(action):
        movement_action_space.enable_wait_action()
    # return the action with the max utility
    return action


def _emulate_movement_from_pool_to_node(
        add_node_delta: float,
        quantity: int,
        delta_target: float
) -> float:
    add_node_delta += quantity
    new_distance = l1_distance(add_node_delta, delta_target)
    return new_distance


def _emulate_movement_from_node_to_pool(
        node_delta: float,
        quantity: int,
        delta_target: float
) -> float:
    node_delta -= quantity
    new_distance = l1_distance(node_delta, delta_target)
    return new_distance


def movement_local_optimal_policy_with_pool(
        add_node_index: int,
        resource: str,
        nodes: List[Node],
        demand: StepData,
        action_space_wrapper: CombinedActionSpaceWrapper,
        quantity_movements: List[int],
        resource_classes: Dict[str, ResourceClass],
        config: SingleRunConfig,
        resources: List[EnvResource],
        delta_target: float = 0
) -> int:
    pool_node_index = _get_pool_node_index(nodes)
    add_node_action_space = action_space_wrapper.add_node_space
    movement_action_space = action_space_wrapper.combined_space
    # default action is wait
    gap_with_units = config.environment.reward.parameters['gap_with_units']
    deltas = _get_nodes_deltas(nodes, demand, resource, gap_with_units, resources)
    deltas.insert(pool_node_index, delta_target)
    deltas = np.array(deltas)
    initial_distance = l1_distance(deltas, delta_target).sum()
    best_distance = initial_distance
    best_action: int = movement_action_space.wait_action_index
    if add_node_index is None or add_node_action_space.is_wait_action(add_node_index):
        # if add_node_index is None the add node sub-action is wait, so we wait as well
        return best_action
    if add_node_index != pool_node_index:
        add_node_delta = deltas[add_node_index]
        add_node_distance = l1_distance(add_node_delta, delta_target)
        # remove node is pool_node
        remove_node = nodes[pool_node_index]
        for res_class_name, res_class in resource_classes.items():
            node_availability = remove_node.get_current_resource_class_units(resource, res_class_name)
            # iterate through all the quantity movements available
            for quantity in quantity_movements:
                if node_availability >= quantity:
                    # emulate the movement
                    movement_distance = _emulate_movement_from_pool_to_node(add_node_delta, quantity, delta_target)
                    new_distance = initial_distance - add_node_distance + movement_distance
                    if new_distance < best_distance:
                        best_distance = new_distance
                        movement = MovementInfo(pool_node_index, res_class_name, quantity, 0, 0, 0)
                        best_action = _get_action_from_movement_info(movement, movement_action_space)
    else:
        # remove node is a regular node
        for remove_index, remove_node in enumerate(nodes):
            if remove_index != add_node_index:
                node_delta = deltas[remove_index]
                node_distance = l1_distance(node_delta, delta_target)
                for res_class_name, res_class in resource_classes.items():
                    node_availability = remove_node.get_current_resource_class_units(resource, res_class_name)
                    # iterate through all the quantity movements available
                    for quantity in quantity_movements:
                        if node_availability >= quantity:
                            # emulate the movement
                            movement_distance = _emulate_movement_from_node_to_pool(node_delta, quantity, delta_target)
                            new_distance = initial_distance - node_distance + movement_distance
                            if new_distance < best_distance:
                                best_distance = new_distance
                                movement = MovementInfo(remove_index, res_class_name, quantity, 0, 0, 0)
                                best_action = _get_action_from_movement_info(movement, movement_action_space)

    if movement_action_space.is_wait_action(best_action):
        movement_action_space.enable_wait_action()
    # return the action with the max utility
    return best_action


def new_full_action_greedy_policy(
        resource: str,
        nodes: List[Node],
        demand: StepData,
        action_space_wrapper: CombinedActionSpaceWrapper,
        quantity_movements: List[int],
        resource_classes: Dict[str, ResourceClass],
        config: SingleRunConfig,
        remaining_gap_max_factor: float,
        max_cost: float,
) -> Action:
    action = Action(
        action_space_wrapper.add_node_space.wait_action_index,
        action_space_wrapper.remove_node_space.wait_action_index,
        action_space_wrapper.resource_classes_space.wait_action_index,
        action_space_wrapper.quantity_space.wait_action_index
    )
    gap_with_units = config.environment.reward.parameters['gap_with_units']
    initial_remaining_gap = _compute_remaining_gap(nodes, demand, resource, resource_classes, gap_with_units)

    max_utility = _estimate_utility(config, initial_remaining_gap, 0, remaining_gap_max_factor, max_cost,
                                    pre_gap=initial_remaining_gap)

    nodes_gap: Dict[str, float] = {}
    for i, node in enumerate(nodes):
        node_delta = _get_node_delta(nodes, i, demand, resource, gap_with_units, resource_classes)
        if node_delta >= 0:
            node_delta = 0
        nodes_gap[node.base_station_type] = node_delta

    for i, add_node in enumerate(nodes):
        for j, remove_node in enumerate(nodes):
            if i != j:
                for res_class_name, res_class in resource_classes.items():
                    node_availability = remove_node.get_current_resource_class_units(resource, res_class_name)
                    # iterate through all the quantity movements available
                    for quantity in quantity_movements:
                        if node_availability >= quantity:
                            add_node_delta_simulated, remove_node_delta_simulated = _simulate_movement(
                                add_node, i, remove_node, j, res_class_name,
                                quantity, resource, demand, resource_classes, gap_with_units=gap_with_units
                            )
                            movement_cost = _compute_movement_cost(res_class_name, quantity, resource_classes)

                            after_movement_remaining_gap = initial_remaining_gap
                            # remove the initial delta to the gap
                            after_movement_remaining_gap -= nodes_gap[add_node.base_station_type]
                            # add the new delta to the gap only if new delta is negative
                            if add_node_delta_simulated < 0:
                                after_movement_remaining_gap += add_node_delta_simulated

                            if remove_node_delta_simulated < 0:
                                # if remove node delta is negative we add it to the remaining gap
                                after_movement_remaining_gap += remove_node_delta_simulated

                            # compute the utility
                            utility = _estimate_utility(config, after_movement_remaining_gap, movement_cost,
                                                        remaining_gap_max_factor, max_cost,
                                                        pre_gap=initial_remaining_gap)

                            # keep track of the max utility
                            if utility > max_utility:
                                max_utility = utility
                                action = Action(
                                    add_node=i, remove_node=j,
                                    resource_class=action_space_wrapper.resource_classes_space.inverted_actions_mapping[
                                        res_class_name],
                                    quantity=action_space_wrapper.quantity_space.inverted_actions_mapping[quantity]
                                )

    return action


def local_add_or_remove_action_heuristic(
        nodes: List[Node],
        pool_node: Node,
        resource: str,
        demand: StepData,
        resources: List[EnvResource],
        config: SingleRunConfig,
        is_add_action: bool,
        action_space_wrapper: CombinedActionSpaceWrapper,
) -> int:
    action = action_space_wrapper.add_action_space.wait_action_index if is_add_action else \
        action_space_wrapper.remove_action_space.wait_action_index
    gap_with_units = config.environment.reward.parameters['gap_with_units']
    deltas = _get_nodes_deltas(nodes, demand, resource, gap_with_units, resources)
    deltas = np.array(deltas)
    if is_add_action:
        if pool_node.get_current_allocated(resource) > 0:
            add_node_index = deltas.argmin()
            if deltas[add_node_index] < 0:
                action = add_node_index
    else:
        remove_node_index = deltas.argmax()
        if deltas[remove_node_index] > 0 and nodes[remove_node_index].get_current_allocated(resource) > 0:
            action = remove_node_index
    return action
