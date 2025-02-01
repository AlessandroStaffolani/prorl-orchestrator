from copy import deepcopy
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union

from numpy.random import RandomState

from prorl.environment.data_structure import NodeResource, NodeResourceValue, ResourceClass, \
    ResourceDistribution, EnvResource, NodeTypeDistribution


class Permutations(str, Enum):
    Allocate = 'allocate'
    Remove = 'remove'


POOL_NODE = 'pool'


class NotAllowedPermutationError(Exception):

    def __init__(self, resource: str, res_class: Optional[str] = None, *args):
        message = f'Trying to remove more resource then available in the node for resource: {resource}'
        if res_class is not None:
            message += f' and resource class: {res_class}'
        super(NotAllowedPermutationError, self).__init__(message, *args)


def node_resources_to_dict(resources: List[NodeResource]) -> Dict[str, NodeResource]:
    return {r.name: r for r in resources}


def node_resources_to_node_resource_values_dict(resources: List[NodeResource]) -> Dict[str, NodeResourceValue]:
    return {r.name: NodeResourceValue(name=r.name,
                                      allocated=r.allocated,
                                      classes=deepcopy(r.classes)) for r in resources}


class Node:

    def __init__(
            self,
            resources: Optional[List[NodeResource]],
            base_station_type: str
    ):
        self._name = 'BaseNode'
        self.base_station_type = base_station_type
        if resources is not None:
            self.initial_resources: Dict[str, NodeResource] = node_resources_to_dict(resources)
            self.current_resources: Dict[str, NodeResourceValue] = node_resources_to_node_resource_values_dict(resources)
            self.buckets_permutations: Dict[str, Dict[Permutations, Dict[str, int]]] = self._reset_buckets_permutations()

    def __str__(self):
        res_allocated = {}
        res_classes = ''
        for resource, node_resource in self.current_resources.items():
            res_allocated[resource] = self.get_current_allocated(resource)
            for key, res_class in node_resource.classes.items():
                res_classes += f'{key}-units={res_class["allocated"]}'
        return f'<Node capacity={res_allocated} {res_classes} base_station_type={self.base_station_type}>'

    def _reset_buckets_permutations(self) -> Dict[str, Dict[Permutations, Dict[str, int]]]:
        return {
            r: {
                Permutations.Allocate: {'operations': 0, 'buckets': 0},
                Permutations.Remove: {'operations': 0, 'buckets': 0}
            } for r, _ in self.initial_resources.items()
        }

    def _get_resource_info(
            self,
            resource: str,
            resource_class: Optional[str] = None
    ) -> Tuple[int, int, int, Optional[Dict[str, ResourceClass]]]:
        node_resource_class: NodeResource = self.initial_resources[resource]
        resource_value_class: NodeResourceValue = self.current_resources[resource]
        bucket_size = node_resource_class.bucket_size
        min_buckets = node_resource_class.min_buckets
        allocated = resource_value_class.allocated
        classes = resource_value_class.classes
        if classes is not None and resource_class is not None:
            bucket_size = classes[resource_class]['capacity']
            allocated = classes[resource_class]['allocated']
        return bucket_size, min_buckets, allocated, classes

    def _total_allocated_with_classes(self, classes: Dict[str, ResourceClass]) -> int:
        allocated = 0
        for name, res_class in classes.items():
            allocated += res_class['capacity'] * res_class['allocated']
        return allocated

    def _permute(self, resource: str, permutation: Permutations, buckets: int, res_class: Optional[str] = None):
        bucket_size, min_buckets, allocated, classes = self._get_resource_info(resource, res_class)
        new_allocated = allocated
        if permutation == Permutations.Allocate:
            if classes is None and res_class is None:
                new_allocated += (buckets * bucket_size)
            else:
                new_allocated += buckets
        elif permutation == Permutations.Remove:
            if classes is None and res_class is None:
                new_allocated -= (buckets * bucket_size)
            else:
                new_allocated -= buckets
            if new_allocated < 0:
                raise NotAllowedPermutationError(resource, res_class)
        if res_class is not None and classes is not None:
            classes[res_class]['allocated'] = max(0, new_allocated)
            self.current_resources[resource] = NodeResourceValue(
                name=resource,
                allocated=self._total_allocated_with_classes(classes),
                classes=classes
            )
        else:
            self.current_resources[resource] = NodeResourceValue(
                name=resource,
                allocated=max(0, new_allocated),
                classes=classes
            )

    def _update_buckets_permutations(self, resource: str, permutation: Permutations, n_buckets: int):
        self.buckets_permutations[resource][permutation]['operations'] += 1
        self.buckets_permutations[resource][permutation]['buckets'] += n_buckets

    def allocate(self, buckets: int, resource: str, res_class: Optional[str] = None):
        self._permute(resource, Permutations.Allocate, buckets, res_class=res_class)
        self._update_buckets_permutations(resource, Permutations.Allocate, buckets)

    def remove(self, buckets: int, resource: str, res_class: Optional[str] = None):
        self._permute(resource, Permutations.Remove, buckets, res_class=res_class)
        self._update_buckets_permutations(resource, Permutations.Remove, buckets)

    def buckets_allocated(self, resource: str, res_class: Optional[str] = None) -> int:
        bucket_size, _, allocated, _ = self._get_resource_info(resource, res_class)
        return allocated // bucket_size

    def allocation_sufficient(self, demand: float, resource: str, res_class: Optional[str] = None) -> bool:
        _, _, current_allocation, _ = self._get_resource_info(resource, res_class)
        return demand <= current_allocation

    def allocation_lower_min(self, resource: str, res_class: Optional[str] = None) -> bool:
        bucket_size, min_buckets, _, _ = self._get_resource_info(resource, res_class)
        return self.buckets_allocated(resource, res_class) < min_buckets

    def reset_buckets_permutations(self):
        self.buckets_permutations = self._reset_buckets_permutations()

    def get_permutations(self, resource: str) -> Dict[Permutations, Dict[str, int]]:
        return self.buckets_permutations[resource]

    def get_current_allocated(self, resource: str, res_class: Optional[str] = None) -> float:
        _, _, current_allocation, _ = self._get_resource_info(resource, res_class)
        return current_allocation

    def get_current_resource_class_units(self, resource: str, res_class: str) -> int:
        return self.current_resources[resource].classes[res_class]['allocated']

    def get_total_resource_classes_units(self, resource: str) -> int:
        res_classes = self.current_resources[resource].classes
        total = 0
        for _, res_class in res_classes.items():
            total += res_class['allocated']
        return total

    def get_resource_min_requested(self, resource: str, res_class: Optional[str] = None) -> float:
        bucket_size, min_buckets, _, _ = self._get_resource_info(resource, res_class)
        return min_buckets * bucket_size

    def get_resource_cost(self, resource: str, res_class: Optional[str] = None) -> float:
        if res_class is None:
            return 1.0
        node_res_class = self.current_resources[resource].classes[res_class]
        return node_res_class['cost']

    def get_permutation_cost(self, buckets: int, resource: str, res_class: Optional[str] = None) -> float:
        cost = self.get_resource_cost(resource, res_class)
        return cost * buckets

    def reset(self):
        self.reset_buckets_permutations()
        self.current_resources = node_resources_to_node_resource_values_dict(
            resources=list(self.initial_resources.values()))

    def copy(self) -> 'Node':
        node = Node(None, self.base_station_type)
        initial_resources: Dict[str, NodeResource] = {}
        current_resources: Dict[str, NodeResourceValue] = {}
        for res, node_res in self.initial_resources.items():
            res_classes = None
            if node_res.classes is not None:
                res_classes = {}
                for res_class_name, res_class in node_res.classes.items():
                    res_classes[res_class_name] = {
                        'cost': res_class['cost'],
                        'capacity': res_class['capacity'],
                        'allocated': res_class['allocated']
                    }
            initial_resources[res] = NodeResource(
                node_res.name, node_res.bucket_size, node_res.allocated, node_res.min_buckets, res_classes
            )

        for res, node_res_val in self.current_resources.items():
            res_classes = None
            if node_res_val.classes is not None:
                res_classes = {}
                for res_class_name, res_class in node_res_val.classes.items():
                    res_classes[res_class_name] = {
                        'cost': res_class['cost'],
                        'capacity': res_class['capacity'],
                        'allocated': res_class['allocated']
                    }
            current_resources[res] = NodeResourceValue(
                node_res_val.name, node_res_val.allocated, res_classes
            )
        node.initial_resources = initial_resources
        node.current_resources = current_resources
        node.buckets_permutations = node._reset_buckets_permutations()
        return node


def _random_distributed_nodes(
        bs_names_list: List[str],
        n_nodes: int,
        random: RandomState,
        node_resources_dict: Dict[str, List[NodeResource]],
        use_pool: bool,
        **kwargs
) -> Tuple[List[Node], Optional[Node]]:
    nodes: List[Node] = []
    pool_node: Optional[Node] = None
    bs_names = random.choice(bs_names_list, n_nodes)
    if use_pool:
        pool_node = Node(resources=node_resources_dict['pool'], base_station_type=POOL_NODE)
    for i, name in enumerate(bs_names):
        if i == 0:
            res = node_resources_dict['first']
        else:
            res = node_resources_dict['all']
        nodes.append(Node(res, base_station_type=name))
    return nodes, pool_node


def _equally_distributed_nodes(
        bs_names_list: List[str],
        n_nodes: int,
        node_resources_dict: Dict[str, List[NodeResource]],
        use_pool: bool,
        **kwargs
) -> Tuple[List[Node], Optional[Node]]:
    nodes: List[Node] = []
    pool_node: Optional[Node] = None
    bs_index = 0
    if use_pool:
        pool_node = Node(resources=node_resources_dict['pool'], base_station_type=POOL_NODE)
    for node_index in range(n_nodes):
        if node_index == 0:
            res = node_resources_dict['first']
        else:
            res = node_resources_dict['all']
        nodes.append(Node(res, base_station_type=bs_names_list[bs_index]))
        bs_index += 1
        if bs_index == len(bs_names_list):
            bs_index = 0
    return nodes, pool_node


def _equally_distributed_resources(
        node_resources: List[NodeResource],
        resources_info: List[EnvResource],
        **kwargs
) -> Dict[str, List[NodeResource]]:
    node_resources_dict: Dict[str, List[NodeResource]] = {
        'first': node_resources,
        'all': node_resources,
    }
    pool_res = []
    for res in resources_info:
        pool_classes = {}
        for res_class_name, res_class in res.classes.items():
            pool_classes[res_class_name] = {
                'cost': res_class['cost'],
                'capacity': res_class['capacity'],
                'allocated': 0
            }
        pool_res.append(NodeResource(
            name=res.name,
            bucket_size=res.bucket_size,
            min_buckets=res.min_buckets,
            allocated=0,
            classes=pool_classes
        ))
    node_resources_dict['pool'] = pool_res
    return node_resources_dict


def _all_and_nothing_distributed_resources(
        resources_info: List[EnvResource],
        **kwargs
) -> Dict[str, List[NodeResource]]:
    node_resources_dict: Dict[str, List[NodeResource]] = {}
    first_res = []
    all_res = []
    pool_res = []
    for res in resources_info:
        first_classes = {}
        all_classes = {}
        for res_class_name, res_class in res.classes.items():
            res_class['allocated'] = res.total_units
            first_classes[res_class_name] = res_class
            all_classes[res_class_name] = {
                'cost': res_class['cost'],
                'capacity': res_class['capacity'],
                'allocated': 0
            }
        first_res.append(NodeResource(
            name=res.name,
            bucket_size=res.bucket_size,
            min_buckets=res.min_buckets,
            allocated=res.total_available,
            classes=first_classes
        ))
        all_res.append(NodeResource(
            name=res.name,
            bucket_size=res.bucket_size,
            min_buckets=res.min_buckets,
            allocated=0,
            classes=all_classes
        ))
        pool_res.append(NodeResource(
            name=res.name,
            bucket_size=res.bucket_size,
            min_buckets=res.min_buckets,
            allocated=0,
            classes=all_classes
        ))
    node_resources_dict['first'] = first_res
    node_resources_dict['all'] = all_res
    node_resources_dict['pool'] = pool_res
    return node_resources_dict


def _pool_distributed_resources(
        resources_info: List[EnvResource],
        resource_distribution_parameters: Dict[str, Union[int, float, bool, str]],
        n_nodes: int,
        **kwargs
) -> Dict[str, List[NodeResource]]:
    node_resources_dict: Dict[str, List[NodeResource]] = {}
    first_res = []
    all_res = []
    pool_res = []
    initial_nodes_units: int = resource_distribution_parameters['initial_node_units']
    allocated_units = initial_nodes_units * n_nodes
    for res in resources_info:
        first_classes = {}
        first_allocated = 0
        all_classes = {}
        all_allocated = 0
        pool_classes = {}
        pool_allocated = 0
        for res_class_name, res_class in res.classes.items():
            res_class['allocated'] = res.total_units - allocated_units
            pool_classes[res_class_name] = res_class
            pool_allocated += (res_class['allocated'] * res_class['capacity'])
            first_classes[res_class_name] = {
                'cost': res_class['cost'],
                'capacity': res_class['capacity'],
                'allocated': initial_nodes_units
            }
            first_allocated += (initial_nodes_units * res_class['capacity'])
            all_classes[res_class_name] = {
                'cost': res_class['cost'],
                'capacity': res_class['capacity'],
                'allocated': initial_nodes_units
            }
            all_allocated += (initial_nodes_units * res_class['capacity'])
        first_res.append(NodeResource(
            name=res.name,
            bucket_size=res.bucket_size,
            min_buckets=res.min_buckets,
            allocated=first_allocated,
            classes=first_classes
        ))
        all_res.append(NodeResource(
            name=res.name,
            bucket_size=res.bucket_size,
            min_buckets=res.min_buckets,
            allocated=all_allocated,
            classes=all_classes
        ))
        pool_res.append(NodeResource(
            name=res.name,
            bucket_size=res.bucket_size,
            min_buckets=res.min_buckets,
            allocated=pool_allocated,
            classes=pool_classes
        ))
    node_resources_dict['first'] = first_res
    node_resources_dict['all'] = all_res
    node_resources_dict['pool'] = pool_res
    return node_resources_dict


RESOURCE_DISTRIBUTION_MAPPINGS = {
    ResourceDistribution.Equally: _equally_distributed_resources,
    ResourceDistribution.AllAndNothing: _all_and_nothing_distributed_resources,
    ResourceDistribution.Pool: _pool_distributed_resources
}


NODES_DISTRIBUTION_MAPPINGS = {
    NodeTypeDistribution.Random: _random_distributed_nodes,
    NodeTypeDistribution.Equally: _equally_distributed_nodes
}


def init_nodes(
        node_resources: List[NodeResource],
        env_config,
        resources_info: List[EnvResource],
        base_stations_mappings: Dict[str, str],
        nodes_distribution: NodeTypeDistribution,
        random: RandomState
) -> Tuple[List[Node], Optional[Node]]:
    node_resource_distribution = env_config.nodes.node_resource_distribution
    if node_resource_distribution in RESOURCE_DISTRIBUTION_MAPPINGS:
        node_resources_dict: Dict[str, List[NodeResource]] = RESOURCE_DISTRIBUTION_MAPPINGS[node_resource_distribution](
            node_resources=node_resources,
            resources_info=resources_info,
            resource_distribution_parameters=env_config.nodes.resource_distribution_parameters,
            n_nodes=env_config.nodes.n_nodes,
        )
    else:
        raise AttributeError('Nodes node_resource_distribution must have one of the following values: {}'
                             .format(ResourceDistribution.list()))

    bs_names_list = [key for key, _ in base_stations_mappings.items()]
    use_pool = env_config.nodes.use_pool_node
    if nodes_distribution in NODES_DISTRIBUTION_MAPPINGS:
        nodes, pool_node = NODES_DISTRIBUTION_MAPPINGS[nodes_distribution](
            bs_names_list=bs_names_list, n_nodes=env_config.nodes.n_nodes,
            random=random, node_resources_dict=node_resources_dict, use_pool=use_pool
        )
    else:
        raise AttributeError('Nodes nodes_type_distribution must have one of the following values: {}'
                             .format(NodeTypeDistribution.list()))
    return nodes, pool_node

