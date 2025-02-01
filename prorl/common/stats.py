import json
from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple
from uuid import uuid4

from prorl import SingleRunConfig, ROOT_DIR
from prorl.common.config import ExportMode
from prorl.core.step import Step
from prorl.common.data_structure import RunMode
from prorl.environment.action_space import Action
from prorl.environment.agent.abstract import AgentLoss, AgentQEstimate
from prorl.environment.node import Node


class RunStats:

    def __init__(self, nodes: List[Node], config: SingleRunConfig,
                 run_mode: RunMode, code: str = None, from_dict: bool = False):
        self.run_code: str = code if code is not None else str(uuid4())
        self.config: SingleRunConfig = config
        self.run_mode: RunMode = run_mode
        self.reward_history: List[float] = []
        self.add_node_reward_history: List[float] = []
        self.reward_info_history: List[Dict[str, float]] = []
        self.reward_penalty_history: List[float] = []
        self.action_cost_history: List[float] = []
        self.action_total_cost: float = 0
        self.validation_reward_history: List[float] = []
        self.action_history: Dict[str, List[Union[int, bool, None]]] = {
            'add_node': [],
            'remove_node': [],
            'resource_class': [],
            'quantity': [],
            'is_random': []
        }
        self.loss_history: Dict[str, List[float]] = {
            'add_net': [],
            'remove_net': [],
            'resource_class_net': [],
            'quantity_net': []
        }
        self.q_estimates_history: Dict[str, List[float]] = {
            'add_net': [],
            'remove_net': [],
            'resource_class_net': [],
            'quantity_net': []
        }
        self.epsilon_history: List[float] = []
        self.resources: List[str] = self.config.emulator.model.resource_names()
        self.nodes_satisfied_history: List[int] = []
        self.nodes_resources_history: Dict[str, Dict[str, List[float]]] = {}
        self.nodes_demand_history: Dict[str, Dict[str, List[float]]] = {}
        self.nodes_base_station_mapping: Dict[str, str] = {}
        self.choose_action_time_history: List[float] = []
        self.learn_time_history: List[float] = []
        self.start_time_step: Optional[Step] = None
        self.last_time_step: Optional[Step] = None
        if not from_dict:
            self._init_nodes_history(nodes)

    def _init_nodes_history(self, nodes: List[Node]):
        for res_name in self.resources:
            tmp_resources = {}
            tmp_demand = {}
            for i in range(self.config.environment.nodes.get_n_nodes()):
                node_name = f'n_{i}'
                tmp_resources[node_name] = []
                tmp_demand[node_name] = []
                self.nodes_base_station_mapping[node_name] = nodes[i].base_station_type
            self.nodes_resources_history[res_name] = tmp_resources
            self.nodes_demand_history[res_name] = tmp_demand

    def __str__(self):
        return f'<RunStats code={self.run_code} >'

    def append(
            self,
            step: Step,
            reward: Union[float, Tuple[float, float]],
            action: Action,
            nodes_satisfied: int,
            nodes_resource: List[float],
            nodes_demand: List[float],
            loss: Optional[AgentLoss] = None,
            q_estimate: Optional[AgentQEstimate] = None,
            epsilon: Optional[float] = None,
            resource_name: str = None,
            choose_action_time: float = None,
            learn_time: float = None,
            action_info: Optional[Dict[str, Union[str, int, float, bool]]] = None,
            penalty: Optional[Union[int, float]] = None,
            step_info: Optional[Dict[str, Any]] = None
    ):
        resource = resource_name if resource_name is not None else self.resources[0]
        if isinstance(reward, tuple):
            self.add_node_reward_history.append(reward[0])
            self.reward_history.append(reward[1])
        else:
            self.reward_history.append(reward)
        if penalty is not None:
            self.reward_penalty_history.append(penalty)
        self.action_history['add_node'].append(action.add_node)
        self.action_history['remove_node'].append(action.remove_node)
        self.action_history['resource_class'].append(action.resource_class)
        self.action_history['quantity'].append(action.quantity)
        if action_info is not None and 'random_action' in action_info:
            self.action_history['is_random'].append(action_info['random_action'])
        else:
            self.action_history['is_random'].append(None)
        self.choose_action_time_history.append(choose_action_time)
        self.learn_time_history.append(learn_time)
        self.nodes_satisfied_history.append(nodes_satisfied)
        if loss is not None:
            self.loss_history['add_net'].append(loss.add_net)
            self.loss_history['remove_net'].append(loss.remove_net)
            if loss.resource_class_net is not None:
                self.loss_history['resource_class_net'].append(loss.resource_class_net)
            if loss.quantity_net is not None:
                self.loss_history['quantity_net'].append(loss.quantity_net)
        if q_estimate is not None:
            self.q_estimates_history['add_net'].append(q_estimate.add_net)
            self.q_estimates_history['remove_net'].append(q_estimate.remove_net)
            if q_estimate.resource_class_net is not None:
                self.q_estimates_history['resource_class_net'].append(q_estimate.resource_class_net)
            if q_estimate.quantity_net is not None:
                self.q_estimates_history['quantity_net'].append(q_estimate.quantity_net)
        if epsilon is not None:
            self.epsilon_history.append(epsilon)
        for i in range(len(nodes_resource)):
            self.nodes_resources_history[resource][f'n_{i}'].append(nodes_resource[i])
            self.nodes_demand_history[resource][f'n_{i}'].append(nodes_demand[i])
        if self.start_time_step is None:
            self.start_time_step: Step = step
        if step_info is not None:
            if 'reward_info' in step_info:
                tmp = {}
                for key, value in step_info['reward_info'].items():
                    tmp[key] = value
                    if key == 'cost_reward':
                        self.action_cost_history.append(value)
                        self.action_total_cost += value
                self.reward_info_history.append(tmp)
        self.last_time_step: Step = step

    def append_validation_history(self, validation_total_reward: float):
        self.validation_reward_history.append(validation_total_reward)

    def to_dict(self) -> Dict[str, Union[Dict, List, str]]:
        return {
            'run_code': self.run_code,
            'run_mode': self.run_mode,
            'steps_interval': {
                'start': self.start_time_step.to_dict(),
                'end': self.last_time_step.to_dict()
            },
            'config': self.config.export(mode=ExportMode.DICT),
            'resources': self.resources,
            'action_history': self.action_history,
            'reward_history': self.reward_history,
            'add_node_reward_history': self.add_node_reward_history,
            'reward_info_history': self.reward_info_history,
            'reward_penalty_history': self.reward_penalty_history,
            'action_cost_history': self.action_cost_history,
            'action_total_cost': self.action_total_cost,
            'validation_reward_history': self.validation_reward_history,
            'loss_history': self.loss_history,
            'q_estimates_history': self.q_estimates_history,
            'epsilon_history': self.epsilon_history,
            'nodes_satisfied_history': self.nodes_satisfied_history,
            'nodes_base_station_mapping': self.nodes_base_station_mapping,
            'nodes_resources_history': self.nodes_resources_history,
            'nodes_demand_history': self.nodes_demand_history,
            'choose_action_time_history': self.choose_action_time_history,
            'learn_time_history': self.learn_time_history,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Union[Dict, List, str]]):
        config = SingleRunConfig(ROOT_DIR, **dict_data['config'])
        stats = cls([], config, code=dict_data['run_code'],
                    run_mode=RunMode(dict_data['run_mode']), from_dict=True)
        stats.resources = dict_data['resources']
        stats.action_history = dict_data['action_history']
        stats.reward_history = dict_data['reward_history']
        stats.add_node_reward_history = dict_data['add_node_reward_history']
        if 'reward_info_history' in dict_data:
            stats.reward_info_history = dict_data['reward_info_history']
        stats.reward_penalty_history = dict_data['reward_penalty_history']
        if 'validation_reward_history' in dict_data:
            stats.validation_reward_history = dict_data['validation_reward_history']
        if 'action_cost_history' in dict_data:
            stats.action_cost_history = dict_data['action_cost_history']
        if 'action_total_cost' in dict_data:
            stats.action_total_cost = dict_data['action_total_cost']
        stats.loss_history = dict_data['loss_history']
        stats.q_estimates_history = dict_data['q_estimates_history']
        stats.epsilon_history = dict_data['epsilon_history']
        stats.nodes_satisfied_history = dict_data['nodes_satisfied_history']
        stats.nodes_base_station_mapping = dict_data['nodes_base_station_mapping']
        stats.nodes_resources_history = dict_data['nodes_resources_history']
        stats.nodes_demand_history = dict_data['nodes_demand_history']
        stats.choose_action_time_history = dict_data['choose_action_time_history']
        if 'learn_time_history' in dict_data:
            stats.learn_time_history = dict_data['learn_time_history']
        elif 'step_time_history' in dict_data:
            stats.learn_time_history = dict_data['step_time_history']
        stats.start_time_step = Step(**dict_data['steps_interval']['start'])
        stats.last_time_step = Step(**dict_data['steps_interval']['end'])
        return stats

    @classmethod
    def from_json(cls, json_data: str):
        return RunStats.from_dict(json.loads(json_data))


class RunStatsCondensed:

    def __init__(self, config: SingleRunConfig,
                 run_mode: RunMode, code: str = None, **kwargs):
        self.run_code: str = code if code is not None else str(uuid4())
        self.config: SingleRunConfig = config
        self.run_mode: RunMode = run_mode
        self.reward_history: List[float] = []
        self.add_node_reward_history: List[float] = []
        self.reward_info_history: List[Dict[str, float]] = []
        self.reward_penalty_history: List[float] = []
        self.action_cost_history: List[float] = []
        self.action_total_cost: float = 0
        self.action_history: Dict[str, List[Union[int, bool, None]]] = {
            'add_node': [],
            'remove_node': [],
            'resource_class': [],
            'quantity': [],
            'is_random': []
        }
        self.loss_history: Dict[str, List[float]] = {
            'add_net': [],
            'remove_net': [],
            'resource_class_net': [],
            'quantity_net': []
        }
        self.q_estimates_history: Dict[str, List[float]] = {
            'add_net': [],
            'remove_net': [],
            'resource_class_net': [],
            'quantity_net': []
        }
        self.epsilon_history: List[float] = []
        self.nodes_satisfied_history: List[int] = []
        self.choose_action_time_history: List[float] = []
        self.learn_time_history: List[float] = []
        self.start_time_step: Optional[Step] = None
        self.last_time_step: Optional[Step] = None

    def __str__(self):
        return f'<RunStats condensed code={self.run_code} >'

    def append(
            self,
            step: Step,
            reward: Union[float, Tuple[float, float]],
            action: Action,
            nodes_satisfied: int,
            loss: Optional[AgentLoss] = None,
            q_estimate: Optional[AgentQEstimate] = None,
            epsilon: Optional[float] = None,
            choose_action_time: float = None,
            learn_time: float = None,
            action_info: Optional[Dict[str, Union[str, int, float, bool]]] = None,
            penalty: Optional[Union[int, float]] = None,
            step_info: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        if isinstance(reward, tuple):
            self.add_node_reward_history.append(reward[0])
            self.reward_history.append(reward[1])
        else:
            self.reward_history.append(reward)
        if penalty is not None:
            self.reward_penalty_history.append(penalty)
        self.action_history['add_node'].append(action.add_node)
        self.action_history['remove_node'].append(action.remove_node)
        self.action_history['resource_class'].append(action.resource_class)
        self.action_history['quantity'].append(action.quantity)
        if action_info is not None and 'random_action' in action_info:
            self.action_history['is_random'].append(action_info['random_action'])
        else:
            self.action_history['is_random'].append(None)
        self.choose_action_time_history.append(choose_action_time)
        self.learn_time_history.append(learn_time)
        self.nodes_satisfied_history.append(nodes_satisfied)
        if loss is not None:
            self.loss_history['add_net'].append(loss.add_net)
            self.loss_history['remove_net'].append(loss.remove_net)
            if loss.resource_class_net is not None:
                self.loss_history['resource_class_net'].append(loss.resource_class_net)
            if loss.quantity_net is not None:
                self.loss_history['quantity_net'].append(loss.quantity_net)
        if q_estimate is not None:
            self.q_estimates_history['add_net'].append(q_estimate.add_net)
            self.q_estimates_history['remove_net'].append(q_estimate.remove_net)
            if q_estimate.resource_class_net is not None:
                self.q_estimates_history['resource_class_net'].append(q_estimate.resource_class_net)
            if q_estimate.quantity_net is not None:
                self.q_estimates_history['quantity_net'].append(q_estimate.quantity_net)
        if epsilon is not None:
            self.epsilon_history.append(epsilon)
        if self.start_time_step is None:
            self.start_time_step: Step = step
        if step_info is not None:
            if 'reward_info' in step_info:
                tmp = {}
                for key, value in step_info['reward_info'].items():
                    tmp[key] = value
                    if key == 'cost_reward':
                        self.action_cost_history.append(value)
                        self.action_total_cost += value
                self.reward_info_history.append(tmp)
        self.last_time_step: Step = step

    def to_dict(self) -> Dict[str, Union[Dict, List, str]]:
        return {
            'run_code': self.run_code,
            'run_mode': self.run_mode,
            'steps_interval': {
                'start': self.start_time_step.to_dict(),
                'end': self.last_time_step.to_dict()
            },
            'config': self.config.export(mode=ExportMode.DICT),
            'action_history': self.action_history,
            'action_total_cost': self.action_total_cost,
            'reward_history': self.reward_history,
            'add_node_reward_history': self.add_node_reward_history,
            'reward_info_history': self.reward_info_history,
            'reward_penalty_history': self.reward_penalty_history,
            'action_cost_history': self.action_cost_history,
            'loss_history': self.loss_history,
            'q_estimates_history': self.q_estimates_history,
            'epsilon_history': self.epsilon_history,
            'nodes_satisfied_history': self.nodes_satisfied_history,
            'choose_action_time_history': self.choose_action_time_history,
            'learn_time_history': self.learn_time_history,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Union[Dict, List, str]]):
        config = SingleRunConfig(ROOT_DIR, **dict_data['config'])
        stats = cls(config, code=dict_data['run_code'],
                    run_mode=RunMode(dict_data['run_mode']))
        stats.action_history = dict_data['action_history']
        stats.reward_history = dict_data['reward_history']
        stats.add_node_reward_history = dict_data['add_node_reward_history']
        if 'reward_info_history' in dict_data:
            stats.reward_info_history = dict_data['reward_info_history']
        stats.reward_penalty_history = dict_data['reward_penalty_history']
        if 'validation_reward_history' in dict_data:
            stats.validation_reward_history = dict_data['validation_reward_history']
        if 'action_cost_history' in dict_data:
            stats.action_cost_history = dict_data['action_cost_history']
        if 'action_total_cost' in dict_data:
            stats.action_total_cost = dict_data['action_total_cost']
        stats.loss_history = dict_data['loss_history']
        stats.q_estimates_history = dict_data['q_estimates_history']
        stats.epsilon_history = dict_data['epsilon_history']
        stats.nodes_satisfied_history = dict_data['nodes_satisfied_history']
        stats.choose_action_time_history = dict_data['choose_action_time_history']
        if 'learn_time_history' in dict_data:
            stats.learn_time_history = dict_data['learn_time_history']
        elif 'step_time_history' in dict_data:
            stats.learn_time_history = dict_data['step_time_history']
        stats.start_time_step = Step(**dict_data['steps_interval']['start'])
        stats.last_time_step = Step(**dict_data['steps_interval']['end'])
        return stats

    @classmethod
    def from_json(cls, json_data: str):
        return RunStats.from_dict(json.loads(json_data))


class EvalStats:

    def __init__(self,
                 nodes: List[Node],
                 resource_names: List[str],
                 n_nodes: int,
                 origin_code: str, code: str = None, from_dict: bool = False):
        self.origin_run_code: str = origin_code
        self.eval_run_code: str = code if code is not None else str(uuid4())
        self.reward_history: List[float] = []
        self.action_history: Dict[str, List[Union[int, bool, None]]] = {
            'add_node': [],
            'remove_node': [],
            'resource_class': [],
            'quantity': [],
            'is_random': []
        }
        self.nodes_satisfied_history: List[int] = []
        self.start_time_step: Optional[Step] = None
        self.last_time_step: Optional[Step] = None
        self.resources: List[str] = resource_names
        self.n_nodes: int = n_nodes
        self.nodes_resources_history: Dict[str, Dict[str, List[float]]] = {}
        self.nodes_demand_history: Dict[str, Dict[str, List[float]]] = {}
        self.nodes_base_station_mapping: Dict[str, str] = {}
        if not from_dict:
            self._init_nodes_history(nodes)

    def _init_nodes_history(self, nodes: List[Node]):
        for res_name in self.resources:
            tmp_resources = {}
            tmp_demand = {}
            for i in range(self.n_nodes):
                node_name = f'n_{i}'
                tmp_resources[node_name] = []
                tmp_demand[node_name] = []
                self.nodes_base_station_mapping[node_name] = nodes[i].base_station_type
            self.nodes_resources_history[res_name] = tmp_resources
            self.nodes_demand_history[res_name] = tmp_demand

    def __str__(self):
        return f'<EvalStats code={self.eval_run_code} >'

    def append(
            self,
            step: Step,
            reward: float,
            action: Action,
            nodes_satisfied: int,
            nodes_resource: List[float],
            nodes_demand: List[float],
            resource_name: Optional[str] = None
    ):
        resource = resource_name if resource_name is not None else self.resources[0]
        self.reward_history.append(reward)
        self.action_history['add_node'].append(action.add_node)
        self.action_history['remove_node'].append(action.remove_node)
        self.action_history['resource_class'].append(action.resource_class)
        self.action_history['quantity'].append(action.quantity)
        self.nodes_satisfied_history.append(nodes_satisfied)
        if self.start_time_step is None:
            self.start_time_step: Step = step
        self.last_time_step: Step = step
        for i in range(len(nodes_resource)):
            self.nodes_resources_history[resource][f'n_{i}'].append(nodes_resource[i])
            self.nodes_demand_history[resource][f'n_{i}'].append(nodes_demand[i])

    def to_dict(self) -> Dict[str, Union[Dict, List, str, int]]:
        return {
            'origin_run_code': self.origin_run_code,
            'eval_run_code': self.eval_run_code,
            'steps_interval': {
                'start': self.start_time_step.to_dict(),
                'end': self.last_time_step.to_dict()
            },
            'action_history': self.action_history,
            'reward_history': self.reward_history,
            'nodes_satisfied_history': self.nodes_satisfied_history,
            'nodes_base_station_mapping': self.nodes_base_station_mapping,
            'nodes_resources_history': self.nodes_resources_history,
            'nodes_demand_history': self.nodes_demand_history,
            'resources': self.resources,
            'n_nodes': self.n_nodes
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Union[Dict, List, str, int]]):
        origin_run_code = dict_data['origin_run_code']
        eval_run_code = dict_data['eval_run_code']
        stats = cls(nodes=[], resource_names=dict_data['resources'], origin_code=origin_run_code, code=eval_run_code,
                    n_nodes=dict_data['n_nodes'], from_dict=True)
        stats.action_history = dict_data['action_history']
        stats.reward_history = dict_data['reward_history']
        stats.nodes_satisfied_history = dict_data['nodes_satisfied_history']
        stats.start_time_step = Step(**dict_data['steps_interval']['start'])
        stats.last_time_step = Step(**dict_data['steps_interval']['end'])
        stats.nodes_satisfied_history = dict_data['nodes_satisfied_history']
        stats.nodes_base_station_mapping = dict_data['nodes_base_station_mapping']
        stats.nodes_resources_history = dict_data['nodes_resources_history']
        stats.nodes_demand_history = dict_data['nodes_demand_history']
        return stats

    @classmethod
    def from_json(cls, json_data: str):
        return RunStats.from_dict(json.loads(json_data))


def convert_run_stats_in_condensed(run_stats: RunStats) -> RunStatsCondensed:
    config = SingleRunConfig(root_dir=run_stats.config.root_dir,
                             **deepcopy(run_stats.config.export(mode=ExportMode.DICT)))
    config.saver.stats_condensed = True
    condensed: RunStatsCondensed = RunStatsCondensed(config, run_mode=run_stats.run_mode, code=run_stats.run_code)
    condensed.reward_history = run_stats.reward_history
    condensed.reward_penalty_history = run_stats.reward_penalty_history
    condensed.action_history = run_stats.action_history
    condensed.loss_history = run_stats.loss_history
    condensed.q_estimates_history = run_stats.q_estimates_history
    condensed.epsilon_history = run_stats.epsilon_history
    condensed.nodes_satisfied_history = run_stats.nodes_satisfied_history
    condensed.choose_action_time_history = run_stats.choose_action_time_history
    condensed.learn_time_history = run_stats.learn_time_history
    condensed.start_time_step = run_stats.start_time_step
    condensed.last_time_step = run_stats.last_time_step
    return condensed
