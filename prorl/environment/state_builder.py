from typing import List, Optional, Dict, Any, Union, Tuple

import numpy as np

from prorl.core.state import State, StateFeatureValue
from prorl.core.step import Step, STEP_UNITS_NORMALIZATION_FACTOR
from prorl.core.step_data import StepData
from prorl.common.enum_utils import ExtendedEnum
from prorl.environment.node import Node, POOL_NODE
from prorl.environment.data_structure import EnvResource
from prorl.environment.node_groups import NodeGroups


class StateFeatureName(str, ExtendedEnum):
    PoolCapacity = 'pool-capacity'
    NodeCapacity = 'node-capacity'
    NodeDemand = 'node-demand'
    NodeDelta = 'node-delta'
    SatisfiedNodes = 'satisfied-nodes'
    CostBudget = 'cost-budget'
    NodeRemove = 'node-remove'
    NodeAdd = 'node-add'
    PreviousAddAction = 'previous-add-action'
    PreviousRemoveAction = 'previous-remove-action'
    ResourceClass = 'resource-class'
    NodeGrouped = 'node-grouped'
    CurrentTime = 'current-time'
    TimeEncoded = 'time-encoded'
    NodeResourceUnits = 'node-resource-units'
    NodeResourceCapacity = 'node-resource-capacity'
    NodeResourceCost = 'node-resource-cost'
    SystemLoad = 'system-load'
    SystemSatisfaction = 'system-satisfaction'
    CurrentLives = 'current-lives'


STACKED_STATE_FEATURES = [
    StateFeatureName.NodeDelta
]


FEATURE_RESOURCE_MAPPING = {
    StateFeatureName.PoolCapacity: True,
    StateFeatureName.NodeCapacity: True,
    StateFeatureName.NodeDemand: True,
    StateFeatureName.NodeDelta: True,
    StateFeatureName.CostBudget: False,
    StateFeatureName.SatisfiedNodes: True,
    StateFeatureName.NodeResourceUnits: True,
    StateFeatureName.NodeResourceCapacity: True,
    StateFeatureName.NodeRemove: False,
    StateFeatureName.NodeAdd: False,
    StateFeatureName.PreviousAddAction: False,
    StateFeatureName.PreviousRemoveAction: False,
    StateFeatureName.ResourceClass: False,
    StateFeatureName.NodeGrouped: False,
    StateFeatureName.CurrentTime: False,
    StateFeatureName.TimeEncoded: False,
    StateFeatureName.SystemLoad: False,
    StateFeatureName.SystemSatisfaction: False,
    StateFeatureName.CurrentLives: False,
}


class StateType(str, ExtendedEnum):
    Add = 'add_state'
    Remove = 'remove_state'
    ResourceClass = 'resource_class'
    Quantity = 'quantity_state'
    Combined = 'combined'


def normalize_values(values: List[float],
                     max_val: float, min_val: float = 0, a: float = 0, b: float = 1) -> List[float]:
    np_values = np.array(values, dtype=np.float64)
    np_result = (b - a) * ((np_values - min_val) / (max_val - min_val)) + a
    return np_result.tolist()


def _build_feature_pool_capacity(
        resources: List[EnvResource],
        normalized: bool,
        pool_node: Optional[Node] = None,
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    if pool_node is not None:
        delta_with_units = False
        if 'delta_with_units' in kwargs['additional_properties']:
            delta_with_units = kwargs['additional_properties']['delta_with_units']

        for resource in resources:
            max_val = resource.total_available
            min_val = resource.min_buckets * resource.bucket_size
            divider = 1
            if delta_with_units:
                max_val = resource.total_units
                min_val = resource.min_buckets
                divider = resource.allocated / resource.units_allocated

            allocated = pool_node.get_current_allocated(resource.name)
            feature_values: List[float] = [allocated / divider]
            if normalized:
                feature_values = normalize_values(feature_values,
                                                  max_val=max_val,
                                                  min_val=min_val,
                                                  a=0, b=1)
            features.append(
                StateFeatureValue(name=f'{StateFeatureName.PoolCapacity.value}_{resource.name}', value=feature_values))
    return features


def _build_feature_node_capacity(resources: List[EnvResource],
                                 nodes: List[Node],
                                 normalized: bool,
                                 **kwargs) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    delta_with_units = False
    if 'delta_with_units' in kwargs['additional_properties']:
        delta_with_units = kwargs['additional_properties']['delta_with_units']
    for resource in resources:
        max_val = resource.total_available
        min_val = resource.min_buckets * resource.bucket_size
        divider = 1
        if delta_with_units:
            max_val = resource.total_units
            min_val = resource.min_buckets
            divider = resource.allocated / resource.units_allocated

        feature_values: List[float] = []
        for node in nodes:
            allocated = node.get_current_allocated(resource.name)
            feature_values.append(allocated / divider)
        if normalized:
            feature_values = normalize_values(feature_values,
                                              max_val=max_val,
                                              min_val=min_val,
                                              a=0, b=1)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.NodeCapacity.value}_{resource.name}', value=feature_values))
    return features


def _build_feature_node_demand(resources: List[EnvResource],
                               current_load: StepData,
                               normalized: bool,
                               **kwargs) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    delta_with_units = False
    if 'delta_with_units' in kwargs['additional_properties']:
        delta_with_units = kwargs['additional_properties']['delta_with_units']
    for resource in resources:
        max_val = resource.total_available
        min_val = resource.min_buckets * resource.bucket_size
        divider = 1
        if delta_with_units:
            max_val = resource.total_units
            min_val = resource.min_buckets
            divider = resource.allocated / resource.units_allocated
        feature_values = current_load.get_resource_values(resource.name, as_array=True)
        feature_values = np.array(feature_values) / divider
        if delta_with_units:
            feature_values = np.ceil(feature_values)
        if normalized:
            feature_values = normalize_values(feature_values,
                                              max_val=max_val,
                                              min_val=min_val,
                                              a=0, b=1)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.NodeDemand.value}_{resource.name}', value=feature_values))
    return features


def _build_feature_node_delta(resources: List[EnvResource],
                              nodes: List[Node],
                              current_load: StepData,
                              normalized: bool,
                              floor_deltas: bool,
                              **kwargs) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    delta_with_units = False
    if 'delta_with_units' in kwargs['additional_properties']:
        delta_with_units = kwargs['additional_properties']['delta_with_units']
    for resource in resources:
        res_units = 1
        units_allocated = 1
        for _, res_class_info in resource.classes.items():
            res_units = res_class_info['capacity']
            units_allocated = resource.total_units
        resource_demand: List[float] = current_load.get_resource_values(resource.name, as_array=True)
        nodes_current_allocation: List[float] = []
        for node in nodes:
            nodes_current_allocation.append(node.get_current_allocated(resource.name))
        resource_demand_np = np.array(resource_demand, dtype=np.float64)
        nodes_current_np = np.array(nodes_current_allocation, dtype=np.float64)
        if delta_with_units:
            resource_demand_np = np.ceil(resource_demand_np / res_units)
            nodes_current_np = nodes_current_np / res_units
        feature_values = nodes_current_np - resource_demand_np
        if floor_deltas and not delta_with_units:
            feature_values = np.floor(feature_values)
        if normalized:
            max_val = resource.allocated * 2
            if delta_with_units:
                max_val = units_allocated
            min_val = -max_val
            feature_values = normalize_values(feature_values,
                                              max_val=max_val,
                                              min_val=min_val,
                                              a=0, b=1)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.NodeDelta.value}_{resource.name}', value=feature_values))
    return features


def _build_feature_satisfied_nodes(
        resources: List[EnvResource],
        nodes: List[Node],
        current_load: StepData,
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    for resource in resources:
        if current_load is not None:
            values = []
            resource_demand: List[Dict[str, float]] = current_load.get_resource_values(resource.name)
            for i, node in enumerate(nodes):
                node_demand: float = resource_demand[i][node.base_station_type]
                node_current_allocation = node.get_current_allocated(resource.name)
                difference = node_current_allocation - node_demand
                if difference >= 0:
                    # satisfied
                    values.append(1)
                else:
                    values.append(0)
        else:
            values = np.ones(len(nodes))
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.SatisfiedNodes.value}_{resource.name}', value=values))
    return features


def _build_feature_hour_satisfaction(
        resources: List[EnvResource],
        nodes: List[Node],
        current_load: StepData,
        previous_demand: StepData,
        normalized: bool,
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    for resource in resources:
        if previous_demand is not None:
            demand = previous_demand.get_resource_values(resource.name)
        elif current_load is not None:
            demand = current_load.get_resource_values(resource.name)
        else:
            demand = None
        if demand is not None:
            satisfied_nodes = 0
            for i, node in enumerate(nodes):
                node_demand: float = demand[i][node.base_station_type]
                node_current_allocation = node.get_current_allocated(resource.name)
                difference = node_current_allocation - node_demand
                if difference >= 0:
                    # satisfied
                    satisfied_nodes += 1
        else:
            satisfied_nodes = 0
        if normalized:
            satisfied_nodes = satisfied_nodes / len(nodes)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.SystemSatisfaction.value}_{resource.name}', value=[satisfied_nodes]))
    return features


def _build_feature_node_resource_units(
        resources: List[EnvResource],
        nodes: List[Node],
        normalized: bool,
        additional_properties: Dict[str, Any],
        **kwargs
) -> List[StateFeatureValue]:
    normalization_type = 'total'
    if additional_properties is not None and 'resource_units_normalization' in additional_properties:
        normalization_type = additional_properties['resource_units_normalization']
    features: List[StateFeatureValue] = []
    for resource in resources:
        resource_classes = nodes[0].initial_resources[resource.name].classes
        if resource_classes is None:
            raise Exception(f'Feature {StateFeatureName.NodeResourceUnits.value} requires resource classes configured')
        values: List[float] = []
        total_units_classes = {}
        for class_name, info in resource_classes.items():
            total_units_classes[class_name] = info['allocated'] * len(nodes)
        total_units = sum(total_units_classes.values())
        for i, node in enumerate(nodes):
            for class_name, _ in resource_classes.items():
                class_units = node.get_current_resource_class_units(resource.name, res_class=class_name)
                if normalization_type == 'by_class':
                    if normalized:
                        class_units = class_units / total_units_classes[class_name]
                values.append(class_units)
        if normalization_type == 'total' and normalized:
            values = normalize_values(values, max_val=total_units)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.NodeResourceUnits.value}_{resource.name}', value=values))
    return features


def _build_feature_node_resource_capacity(
        resources: List[EnvResource],
        nodes: List[Node],
        normalized: bool,
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    for resource in resources:
        resource_classes = nodes[0].initial_resources[resource.name].classes
        resource_total_capacity = resource.total_available
        if resource_classes is None:
            raise Exception(f'Feature {StateFeatureName.NodeResourceUnits.value} requires resource classes configured')
        values: List[float] = []
        for i, node in enumerate(nodes):
            for class_name, class_info in resource_classes.items():
                class_units = node.get_current_resource_class_units(resource.name, res_class=class_name)
                class_capacity = class_info['capacity']
                values.append(class_units * class_capacity)
        if normalized:
            values = normalize_values(values, max_val=resource_total_capacity, min_val=0)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.NodeResourceUnits.value}_{resource.name}', value=values))
    return features


def _build_feature_node_resource_cost(
        resources: List[EnvResource],
        nodes: List[Node],
        normalized: bool,
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    for resource in resources:
        resource_classes = nodes[0].initial_resources[resource.name].classes
        max_cost: Optional[float] = None
        min_cost: Optional[float] = None
        if resource_classes is None:
            raise Exception(f'Feature {StateFeatureName.NodeResourceUnits.value} requires resource classes configured')
        values: List[float] = []
        for i, node in enumerate(nodes):
            for class_name, _ in resource_classes.items():
                node_classes = node.initial_resources[resource.name].classes
                class_cost = node_classes[class_name]['cost']
                if max_cost is None or max_cost < class_cost:
                    max_cost = class_cost
                if min_cost is None or min_cost > class_cost:
                    min_cost = class_cost
                values.append(class_cost)
        if normalized:
            values = normalize_values(values, max_val=max_cost, min_val=min_cost)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.NodeResourceUnits.value}_{resource.name}', value=values))
    return features


def _build_cost_budget_feature(
    current_budget: float,
    initial_budget: float,
    normalized: bool,
    **kwargs
) -> List[StateFeatureValue]:
    budget = current_budget
    if normalized:
        budget = normalize_values([budget], max_val=initial_budget)[0]
    return [StateFeatureValue(name=StateFeatureName.CostBudget.value, value=[budget])]


def _build_node_grouped_feature(
        resources: List[EnvResource],
        nodes: List[Node],
        node_groups: NodeGroups,
        normalized: bool,
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    n_nodes = len(nodes)
    for resource in resources:
        group = node_groups.get_groups(resource.name)
        feature_values = []
        for g_name, node_indexes in group.items():
            n_indexes = len(node_indexes)
            if normalized:
                n_indexes = n_indexes / n_nodes
            feature_values.append(n_indexes)
        features.append(
            StateFeatureValue(name=f'{StateFeatureName.NodeGrouped.value}_{resource.name}', value=feature_values)
        )
    return features


def _build_one_hot_encoding_feature(size: int, feature_name: str, index: Optional[int] = None) -> StateFeatureValue:
    values = [0] * size
    if index is not None:
        values[index] = 1
    return StateFeatureValue(name=feature_name, value=values)


def _build_feature_node_remove(
        nodes: List[Node], node_remove_index: Optional[int] = None,
        node_groups: Optional[NodeGroups] = None, **kwargs) -> List[StateFeatureValue]:
    if node_groups is None:
        feature: StateFeatureValue = _build_one_hot_encoding_feature(
            size=len(nodes),
            feature_name=StateFeatureName.NodeRemove.value,
            index=node_remove_index
        )
    else:
        res_name = node_groups.resources[0].name
        feature: StateFeatureValue = _build_one_hot_encoding_feature(
            size=len(node_groups.get_groups(res_name)),
            feature_name=StateFeatureName.NodeRemove.value,
            index=node_remove_index
        )
    return [feature]


def _build_feature_node_add(
        nodes: List[Node], node_add_index: Optional[int] = None,
        node_groups: Optional[NodeGroups] = None, **kwargs) -> List[StateFeatureValue]:
    if node_groups is None:
        feature: StateFeatureValue = _build_one_hot_encoding_feature(
            size=len(nodes),
            feature_name=StateFeatureName.NodeAdd.value,
            index=node_add_index
        )
    else:
        res_name = node_groups.resources[0].name
        feature: StateFeatureValue = _build_one_hot_encoding_feature(
            size=len(node_groups.get_groups(res_name)),
            feature_name=StateFeatureName.NodeAdd.value,
            index=node_add_index
        )
    return [feature]


def _build_feature_previous_add_action(
        nodes: List[Node], previous_add_action: Optional[int] = None, **kwargs) -> List[StateFeatureValue]:
    if previous_add_action is not None and previous_add_action == len(nodes):
        # was selected to wait
        previous_add_action = None
    feature: StateFeatureValue = _build_one_hot_encoding_feature(
        size=len(nodes),
        feature_name=StateFeatureName.PreviousAddAction.value,
        index=previous_add_action
    )
    return [feature]


def _build_feature_previous_remove_action(
        nodes: List[Node], previous_remove_action: Optional[int] = None, **kwargs) -> List[StateFeatureValue]:
    if previous_remove_action is not None and previous_remove_action == len(nodes):
        # was selected to wait
        previous_remove_action = None
    feature: StateFeatureValue = _build_one_hot_encoding_feature(
        size=len(nodes),
        feature_name=StateFeatureName.PreviousRemoveAction.value,
        index=previous_remove_action
    )
    return [feature]


def _build_feature_resource_class(
        nodes: List[Node],
        resources: List[EnvResource],
        resource_class_index: Optional[int] = None,
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    for resource in resources:
        resource_classes = nodes[0].initial_resources[resource.name].classes
        features.append(
            _build_one_hot_encoding_feature(
                size=len(resource_classes),
                feature_name=f'{StateFeatureName.ResourceClass.value}_{resource.name}',
                index=resource_class_index)
        )
    return features


def _build_feature_current_time(current_step: Step,
                                additional_properties: Dict[str, Any],
                                normalized: bool,
                                **kwargs) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.CurrentTime} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    values = current_step.to_array(normalized=normalized, units_to_skip=units_to_skip)
    features.append(
        StateFeatureValue(name=f'{StateFeatureName.CurrentTime.value}', value=values)
    )
    return features


def _build_feature_time_encoded(current_step: Step,
                                additional_properties: Dict[str, Any],
                                normalized: bool,
                                **kwargs) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.TimeEncoded} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    values = []
    for unit, value in current_step.to_dict().items():
        if unit not in units_to_skip:
            unit_values = np.zeros(STEP_UNITS_NORMALIZATION_FACTOR[unit] + 1)
            unit_values[value] = 1
            values += unit_values.tolist()
    features.append(
        StateFeatureValue(name=f'{StateFeatureName.TimeEncoded.value}', value=values)
    )
    return features


def _build_system_load_feature(system_load: float, **kwargs) -> List[StateFeatureValue]:
    return [
        StateFeatureValue(name=f'{StateFeatureName.SystemLoad.value}', value=[system_load])
    ]


def _build_current_lives_feature(lives: int, initial_lives: int, normalized: bool, **kwargs) -> List[StateFeatureValue]:
    value = [lives]
    if normalized:
        value = normalize_values(value, max_val=initial_lives, min_val=0, a=0, b=1)
    return [
        StateFeatureValue(name=StateFeatureName.CurrentLives.value, value=value)
    ]


FEATURE_BUILD_MAPPING = {
    StateFeatureName.PoolCapacity: _build_feature_pool_capacity,
    StateFeatureName.NodeCapacity: _build_feature_node_capacity,
    StateFeatureName.NodeDemand: _build_feature_node_demand,
    StateFeatureName.NodeDelta: _build_feature_node_delta,
    StateFeatureName.SatisfiedNodes: _build_feature_satisfied_nodes,
    StateFeatureName.CostBudget: _build_cost_budget_feature,
    StateFeatureName.NodeResourceUnits: _build_feature_node_resource_units,
    StateFeatureName.NodeResourceCapacity: _build_feature_node_resource_capacity,
    StateFeatureName.NodeResourceCost: _build_feature_node_resource_cost,
    StateFeatureName.NodeRemove: _build_feature_node_remove,
    StateFeatureName.NodeAdd: _build_feature_node_add,
    StateFeatureName.PreviousAddAction: _build_feature_previous_add_action,
    StateFeatureName.PreviousRemoveAction: _build_feature_previous_remove_action,
    StateFeatureName.ResourceClass: _build_feature_resource_class,
    StateFeatureName.NodeGrouped: _build_node_grouped_feature,
    StateFeatureName.CurrentTime: _build_feature_current_time,
    StateFeatureName.TimeEncoded: _build_feature_time_encoded,
    StateFeatureName.SystemLoad: _build_system_load_feature,
    StateFeatureName.SystemSatisfaction: _build_feature_hour_satisfaction,
    StateFeatureName.CurrentLives: _build_current_lives_feature,
}


def _feature_node_resources_size(resources: List[EnvResource], nodes: List[Node], **kwargs) -> int:
    return len(resources) * len(nodes)


def _feature_pool_capacity_size(resources: List[EnvResource], **kwargs) -> int:
    return len(resources)


def _feature_node_resource_units_size(resources: List[EnvResource], nodes: List[Node], **kwargs) -> int:
    size = 0
    for resource in resources:
        resource_classes = nodes[0].initial_resources[resource.name].classes
        if resource_classes is None:
            raise Exception(f'Feature {StateFeatureName.NodeResourceUnits.value} requires resource classes configured')
        size += len(resource_classes) * len(nodes)
    return size


def _feature_node_size(nodes: List[Node], node_groups: Optional[NodeGroups] = None, **kwargs) -> int:
    if node_groups is None:
        return len(nodes)
    else:
        return _feature_node_grouped_size(node_groups)


def _feature_cost_budget_size(**kwargs) -> int:
    return 1


def _feature_node_grouped_size(node_groups: NodeGroups, **kwargs) -> int:
    size = 0
    for res in node_groups.resources:
        size += len(node_groups.get_groups(res.name))
    return size


def _feature_resource_class_size(resources: List[EnvResource], nodes: List[Node], **kwargs) -> int:
    size = 0
    for res in resources:
        res_classes = nodes[0].initial_resources[res.name].classes
        size += len(res_classes)
    return size


def _feature_current_time_size(additional_properties: Dict[str, Any], **kwargs) -> int:
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.CurrentTime} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    return len(STEP_UNITS_NORMALIZATION_FACTOR) - len(units_to_skip)


def _feature_time_encoded_size(additional_properties: Dict[str, Any], **kwargs) -> int:
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.TimeEncoded} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    size = 0
    for unit, max_value in STEP_UNITS_NORMALIZATION_FACTOR.items():
        if unit not in units_to_skip:
            size += max_value + 1
    return size


def _feature_system_load_size(**kwargs) -> int:
    return 1


FEATURE_SIZE_MAPPING = {
    StateFeatureName.PoolCapacity: _feature_pool_capacity_size,
    StateFeatureName.NodeCapacity: _feature_node_resources_size,
    StateFeatureName.NodeDemand: _feature_node_resources_size,
    StateFeatureName.NodeDelta: _feature_node_resources_size,
    StateFeatureName.SatisfiedNodes: _feature_node_resources_size,
    StateFeatureName.CostBudget: _feature_cost_budget_size,
    StateFeatureName.NodeResourceUnits: _feature_node_resource_units_size,
    StateFeatureName.NodeResourceCapacity: _feature_node_resource_units_size,
    StateFeatureName.NodeResourceCost: _feature_node_resource_units_size,
    StateFeatureName.NodeRemove: _feature_node_size,
    StateFeatureName.NodeAdd: _feature_node_size,
    StateFeatureName.PreviousAddAction: _feature_node_size,
    StateFeatureName.PreviousRemoveAction: _feature_node_size,
    StateFeatureName.ResourceClass: _feature_resource_class_size,
    StateFeatureName.NodeGrouped: _feature_node_grouped_size,
    StateFeatureName.CurrentTime: _feature_current_time_size,
    StateFeatureName.TimeEncoded: _feature_time_encoded_size,
    StateFeatureName.SystemLoad: _feature_system_load_size,
    StateFeatureName.SystemSatisfaction: _feature_system_load_size,
    StateFeatureName.CurrentLives: _feature_system_load_size
}


class FeatureMissingError(Exception):

    def __init__(self, feature, all_features=tuple(FEATURE_BUILD_MAPPING.keys()), *args):
        message = f'State feature "{feature}" missing in features mapping: {all_features}'
        super(FeatureMissingError, self).__init__(message, *args)


def build_state(
        features: List[StateFeatureName],
        resources: List[EnvResource],
        nodes: List[Node],
        current_load: StepData,
        additional_properties: Dict[str, Any],
        dtype=np.float64,
        normalized=True,
        floor_deltas=True,
        node_remove_index: Optional[int] = None,
        node_add_index: Optional[int] = None,
        node_groups: Optional[NodeGroups] = None,
        current_step: Optional[Step] = None,
        previous_demand: Optional[StepData] = None,
        current_budget: float = 0,
        initial_budget: float = 0,
        system_load: float = 0,
        state_feature_values: Optional[Dict[StateFeatureName, List[StateFeatureValue]]] = None,
        pool_node: Optional[Node] = None,
        previous_add_action: Optional[int] = None,
        previous_remove_action: Optional[int] = None,
        lives: int = 0,
        initial_lives: int = 0,
        **kwargs
) -> Tuple[State, Dict[StateFeatureName, List[StateFeatureValue]]]:
    state_features: List[StateFeatureValue] = []
    state_features_dict: Dict[
        StateFeatureName, List[StateFeatureValue]] = {} if state_feature_values is None else state_feature_values
    for feature in features:
        if feature in FEATURE_BUILD_MAPPING:
            if feature in state_features_dict:
                state_features += state_features_dict[feature]
            else:
                feature_build_fn = FEATURE_BUILD_MAPPING[feature]
                feature_features = feature_build_fn(resources=resources,
                                                    nodes=nodes,
                                                    current_load=current_load,
                                                    normalized=normalized,
                                                    floor_deltas=floor_deltas,
                                                    node_remove_index=node_remove_index,
                                                    node_add_index=node_add_index,
                                                    node_groups=node_groups,
                                                    additional_properties=additional_properties,
                                                    current_step=current_step,
                                                    current_budget=current_budget,
                                                    initial_budget=initial_budget,
                                                    previous_demand=previous_demand,
                                                    system_load=system_load,
                                                    pool_node=pool_node,
                                                    previous_add_action=previous_add_action,
                                                    previous_remove_action=previous_remove_action,
                                                    lives=lives,
                                                    initial_lives=initial_lives,
                                                    **kwargs,
                                                    )
                state_features += feature_features
                state_features_dict[feature] = feature_features
        else:
            raise FeatureMissingError(feature)
    if len(state_features) > 0:

        return State(feature_values=tuple(state_features), dtype=dtype), state_features_dict
    else:
        raise Exception('State features array is empty. Impossible to build state')


def get_feature_size(
        feature: StateFeatureName,
        nodes: List[Node],
        resources: List[EnvResource],
        additional_properties: Dict[str, Any],
        node_groups: Optional[NodeGroups] = None,
        stacked_states: int = 1
) -> int:
    if feature in FEATURE_SIZE_MAPPING:
        size = FEATURE_SIZE_MAPPING[feature](resources=resources, nodes=nodes,
                                             node_groups=node_groups, additional_properties=additional_properties)
        if feature in STACKED_STATE_FEATURES:
            size *= stacked_states
        return size
    else:
        raise FeatureMissingError(feature)


def update_state_one_hot_encoding_feature(state: State, feature_name: Union[StateFeatureName, str],
                                          index: int, reset_value: int = 0):
    state.set_feature_value(feature_name, 1, index=index, reset_value=reset_value)

