from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union, Any

import numpy as np

from prorl import SingleRunConfig, ROOT_DIR
from prorl.common.config import ExportMode
from prorl.common.tensorboard_wrapper import TensorboardWrapper
from prorl.common.data_structure import RunMode
from prorl.environment.agent import AgentType
from prorl.environment.node import Node


def last_item(arr):
    return arr[-1]


@dataclass
class VariableInfo:
    tensorboard: bool = False
    is_only_cumulative: bool = False
    is_string: bool = False
    keep_cumulative: bool = False
    disk_save: bool = True
    db_save: bool = False
    redis_save: bool = False
    aggregation_fn: Optional[callable] = None
    single_value_is_array: bool = False
    keep_every_x: Optional[int] = None


class Tracker:

    def __init__(
            self,
            run_code: str = None,
            config: Optional[SingleRunConfig] = None,
            run_mode: Optional[RunMode] = None,
            tensorboard: Optional[TensorboardWrapper] = None
    ):
        self.config: SingleRunConfig = config
        self.run_code: str = run_code
        self.run_mode: RunMode = run_mode
        self.tensorboard: Optional[TensorboardWrapper] = tensorboard
        self.is_condensed: bool = True
        # tracking props
        self.history_suffix = '/history'
        self.cumulative_suffix = '/cumulative'
        self._tracked: Dict[str, Union[List[Union[int, float, str, bool]], Union[int, float, str]]] = {}
        self._keep_tmp: Dict[str, Union[List[Union[int, float, str, bool]], Union[int, float, str]]] = {}
        self._keep_tmp_n_updates: Dict[str, int] = {}
        self._variables_info: Dict[str, VariableInfo] = {}
        self._last_tracked_step: Dict[str, Optional[int]] = {}

    def __str__(self):
        return f'<StatsTracker tracked_variables={len(self._variables_info)}>'

    def __len__(self):
        return len(self._variables_info)

    def _track_value(self, key: str, value: Union[int, float, str, bool, np.ndarray], step: Optional[int] = None):
        variable_info = self._variables_info[key]
        # prevent multiple tracking of the same variable at the same step
        if step is not None and step == self._last_tracked_step[key]:
            return
        self._last_tracked_step[key] = step
        add_value = True
        if variable_info.keep_every_x is not None:
            if variable_info.single_value_is_array:
                self._keep_tmp[key] += value.tolist()
            else:
                self._keep_tmp[key] += value
            self._keep_tmp_n_updates[key] += 1
            if self._keep_tmp_n_updates[key] == variable_info.keep_every_x:
                add_value = True
                if variable_info.single_value_is_array:
                    value = np.array(self._keep_tmp[key])
                    self._keep_tmp[key] = []
                else:
                    value = self._keep_tmp[key]
                    self._keep_tmp[key] = 0
                self._keep_tmp_n_updates[key] = 0
            else:
                add_value = False
        if add_value:
            if variable_info.single_value_is_array:
                self._track_value(f'{key}/total', value.sum(), step)
                self._track_value(f'{key}/avg', value.mean(), step)
                self._track_value(f'{key}/std', value.std(), step)
                self._track_value(f'{key}/min', value.min(), step)
                self._track_value(f'{key}/max', value.max(), step)
            else:
                if variable_info.is_only_cumulative:
                    self._tracked[key] += value
                elif variable_info.is_string:
                    self._tracked[key] = value
                elif variable_info.keep_cumulative:
                    self._tracked[key + self.history_suffix].append(value)
                    self._tracked[key + self.cumulative_suffix] += value
                else:
                    self._tracked[key].append(value)
                if variable_info.tensorboard and step is not None:
                    self._tensorboard_track(key, value, step)

    def _tensorboard_track(self, key: str, value: Union[int, float, str, bool], step: int):
        if self.tensorboard is not None:
            variable_info = self._variables_info[key]
            if variable_info.keep_cumulative:
                self.tensorboard.add_scalar(tag=key + self.history_suffix, value=value, step=step)
                self.tensorboard.add_scalar(tag=key + self.cumulative_suffix,
                                            value=self._tracked[key + self.cumulative_suffix], step=step)
            elif variable_info.is_only_cumulative:
                self.tensorboard.add_scalar(tag=key, value=self._tracked[key], step=step)
            else:
                self.tensorboard.add_scalar(tag=key, value=value, step=step)

    def init_tracking(self, key: str, initial_value: Optional[Any] = None, **variable_info):
        if key not in self._variables_info:
            if key.endswith(self.cumulative_suffix):
                raise KeyError(f'trying to track a new variable using protected suffix "{self.cumulative_suffix}"')
            if key.endswith(self.history_suffix):
                raise KeyError(f'trying to track a new variable using protected suffix "{self.history_suffix}"')
            variable_info = VariableInfo(**variable_info)
            self._variables_info[key] = variable_info
            self._last_tracked_step[key] = None
            if variable_info.single_value_is_array:
                self.init_tracking(f'{key}/total', tensorboard=variable_info.tensorboard,
                                   db_save=variable_info.db_save, redis_save=variable_info.redis_save,
                                   aggregation_fn=np.max)
                self.init_tracking(f'{key}/avg', tensorboard=variable_info.tensorboard,
                                   db_save=variable_info.db_save, redis_save=variable_info.redis_save,
                                   aggregation_fn=np.max)
                self.init_tracking(f'{key}/std', tensorboard=variable_info.tensorboard,
                                   db_save=variable_info.db_save, redis_save=variable_info.redis_save,
                                   aggregation_fn=np.max)
                self.init_tracking(f'{key}/min', tensorboard=variable_info.tensorboard)
                self.init_tracking(f'{key}/max', tensorboard=variable_info.tensorboard)
            if variable_info.keep_every_x is not None:
                if variable_info.single_value_is_array:
                    self._keep_tmp[key] = []
                else:
                    self._keep_tmp[key] = 0
                    self._tracked[key] = []
                self._keep_tmp_n_updates[key] = 0
            else:
                if variable_info.is_only_cumulative:
                    self._tracked[key] = 0
                    if initial_value:
                        self._tracked[key] = initial_value
                elif variable_info.keep_cumulative:
                    self._tracked[key + self.history_suffix] = []
                    self._tracked[key + self.cumulative_suffix] = 0
                    if initial_value:
                        self._tracked[key + self.history_suffix] = initial_value[0]
                        self._tracked[key + self.cumulative_suffix] = initial_value[1]
                else:
                    self._tracked[key] = []
                    if initial_value:
                        self._tracked[key] = initial_value

    def track(
            self,
            key: str,
            value: Union[int, float, str, bool, None, np.ndarray],
            step: Optional[int] = None,
            **variable_info):
        if key not in self._variables_info:
            # init the variable
            self.init_tracking(key, **variable_info)
        # track the value
        self._track_value(key, value, step)

    def disk_stats(self) -> Dict[str, Any]:
        stats = {
            'run_code': self.run_code,
            'run_mode': self.run_mode,
            'is_condensed': self.is_condensed,
            'config': self.config.export(mode=ExportMode.DICT)
        }
        for key, variable_info in self._variables_info.items():
            if variable_info.disk_save:
                if not variable_info.single_value_is_array:
                    if variable_info.keep_cumulative:
                        stats[key + self.history_suffix] = self._tracked[key + self.history_suffix]
                        stats[key + self.cumulative_suffix] = self._tracked[key + self.cumulative_suffix]
                    else:
                        stats[key] = self._tracked[key]
        return stats

    def _get_aggregable_stats(self, key: str, variable_info: VariableInfo) -> Any:
        if variable_info.keep_cumulative:
            return self._tracked[key + self.cumulative_suffix]
        elif variable_info.is_only_cumulative:
            return self._tracked[key]
        else:
            aggregation_fn = variable_info.aggregation_fn
            if aggregation_fn is not None:
                if len(self._tracked[key]) > 0:
                    return aggregation_fn(self._tracked[key])
                else:
                    return 0

    def db_stats(self, add_config=False, add_run_code=False, add_run_mode=False,
                 best_prefix: Optional[str] = None, best_key: Optional[str] = None,
                 rename_key: Optional[str] = None) -> Dict[str, Any]:
        stats = {}
        if add_config:
            stats['config'] = self.config
        if add_run_code:
            stats['run_code'] = self.run_code
        if add_run_mode:
            stats['run_mode'] = self.run_mode
        best_index = None
        if best_key is not None:
            value = self.get(best_key)
            if len(value) > 0:
                best_index = np.argmax(value)
        for key, variable_info in self._variables_info.items():
            if variable_info.db_save and not variable_info.single_value_is_array:
                stats_key = key
                if best_prefix is not None and best_prefix in key and best_index is not None:
                    value = self._tracked[key][best_index]
                    if rename_key is not None:
                        stats_key = key.replace(best_prefix, rename_key)
                else:
                    value = self._get_aggregable_stats(key, variable_info)
                if hasattr(value, 'item'):
                    value = value.item()
                stats[stats_key] = value
        return stats

    def redis_stats(self,
                    best_prefix: Optional[str] = None,
                    best_key: Optional[str] = None,
                    rename_key: Optional[str] = None
                    ) -> Dict[str, Any]:
        stats = {}
        best_index = None
        if best_key is not None:
            value = self.get(best_key)
            if len(value) > 0:
                best_index = np.argmax(value)
        for key, variable_info in self._variables_info.items():
            if variable_info.redis_save and not variable_info.single_value_is_array:
                if best_prefix is not None and best_prefix in key and best_index is not None:
                    if rename_key is not None:
                        stats_key = key.replace(best_prefix, rename_key)
                    else:
                        stats_key = key
                    stats[stats_key] = self._tracked[key][best_index]
                else:
                    stats[key] = self._get_aggregable_stats(key, variable_info)
        return stats

    def get(self,
            key: str,
            include_info=False,
            add_cumulative=False,
            only_cumulative=False) -> Optional[Union[Any, Tuple[Any, VariableInfo]]]:
        result = None
        if key in self._variables_info:
            variable_info = self._variables_info[key]
            if variable_info.keep_cumulative:
                result = self._tracked[key + self.history_suffix]
                if add_cumulative:
                    result = {
                        key + self.history_suffix: result,
                        key + self.cumulative_suffix: self._tracked[key + self.cumulative_suffix]
                    }
                elif only_cumulative:
                    result = self._tracked[key + self.cumulative_suffix]
            else:
                result = self._tracked[key]
            if include_info:
                result = (result, variable_info)
        return result

    def get_array_data(self, key, index):
        if key in self._variables_info:
            variable_info = self._variables_info[key]
            if variable_info.single_value_is_array:
                return {
                    f'{key}/total': self.get(f'{key}/total')[index],
                    f'{key}/avg': self.get(f'{key}/avg')[index],
                    f'{key}/std': self.get(f'{key}/std')[index],
                    f'{key}/min': self.get(f'{key}/min')[index],
                    f'{key}/max': self.get(f'{key}/max')[index],
                }

    def __getitem__(self, item):
        return self._tracked[item]

    def __contains__(self, item):
        return item in self._tracked

    @classmethod
    def from_dict(cls, variables_dict: Dict[str, Any]) -> 'Tracker':
        tracker = cls(
            run_code=variables_dict['run_code'],
            run_mode=variables_dict['run_mode'],
            config=SingleRunConfig(root_dir=ROOT_DIR, **variables_dict['config'])
        )
        tracker.is_condensed = variables_dict['is_condensed']
        reserved_keys = ['run_code', 'run_mode', 'config']
        for key, value in variables_dict.items():
            if key not in reserved_keys:
                variable_info = {}
                var_key = key
                initial_value = value
                if key.endswith(tracker.history_suffix):
                    variable_info['keep_cumulative'] = True
                    var_key = key.replace(tracker.history_suffix, '')
                    initial_value = (value, variables_dict[var_key + tracker.cumulative_suffix])
                elif key.endswith(tracker.cumulative_suffix):
                    variable_info['keep_cumulative'] = True
                    var_key = key.replace(tracker.cumulative_suffix, '')
                    initial_value = (variables_dict[var_key + tracker.history_suffix], value)
                elif isinstance(value, (int, float)):
                    variable_info['is_only_cumulative'] = True
                tracker.init_tracking(var_key, initial_value=initial_value, **variable_info)
        return tracker

    @classmethod
    def init_condensed_tracker(cls, run_code: str, config: SingleRunConfig,
                               run_mode: Optional[RunMode] = None,
                               tensorboard: Optional[TensorboardWrapper] = None) -> 'Tracker':
        tracker = cls(run_code, config, run_mode, tensorboard)
        if AgentType.is_policy_gradient(config.environment.agent.type) or config.run.use_on_policy_agent:
            init_policy_gradient_tracker(tracker)
        else:
            init_value_based_tracker(tracker)
        return tracker

    @classmethod
    def init_tracker(cls, nodes: List[Node], run_code: str, config: SingleRunConfig,
                     run_mode: Optional[RunMode] = None,
                     tensorboard: Optional[TensorboardWrapper] = None) -> 'Tracker':
        tracker = Tracker.init_condensed_tracker(run_code, config, run_mode, tensorboard)
        for i, node in enumerate(nodes):
            tracker.init_tracking(f'evaluation/nodes/n_{i}/resource_history', tensorboard=True)
            tracker.init_tracking(f'evaluation/nodes/n_{i}/demand_history', tensorboard=True)
            tracker.track(f'nodes/base_station_mapping/n_{i}', node.base_station_type, is_string=True)
        tracker.is_condensed = False
        return tracker


def init_value_based_tracker(tracker: Tracker):
    if tracker.run_mode == RunMode.Train:
        tracker.init_tracking('training/episode/reward/utility', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        tracker.init_tracking('training/episode/reward/remaining_gap', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        tracker.init_tracking('training/episode/reward/surplus', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        tracker.init_tracking('training/episode/reward/cost', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        tracker.init_tracking('training/episode/problem_solved/count', tensorboard=True, redis_save=True,
                              aggregation_fn=last_item)
        tracker.init_tracking('training/episode/resource_utilization', tensorboard=True, redis_save=True,
                              aggregation_fn=last_item)
        tracker.init_tracking('training/episode/hour_satisfied', tensorboard=True, redis_save=True,
                              aggregation_fn=last_item)
        tracker.init_tracking('training/problem_solved/nodes_satisfied', tensorboard=True, aggregation_fn=np.max)
        tracker.init_tracking('training/episode/length', tensorboard=True)
        tracker.init_tracking('training/episode/total', is_only_cumulative=True, tensorboard=True)

        tracker.init_tracking('training/total_steps', is_only_cumulative=True)
        tracker.init_tracking('training/action_history/add_node')
        tracker.init_tracking('training/action_history/remove_node')
        tracker.init_tracking('training/action_history/resource_class')
        tracker.init_tracking('training/action_history/quantity')
        tracker.init_tracking('training/action_history/add')
        tracker.init_tracking('training/action_history/remove')
        tracker.init_tracking('training/action_history/is_random')
        tracker.init_tracking('hp/epsilon', tensorboard=True, redis_save=True, aggregation_fn=last_item)
        tracker.init_tracking('hp/prioritized_buffer_beta', tensorboard=True, redis_save=True, aggregation_fn=last_item)
        tracker.init_tracking('hp/expert_action_theta', tensorboard=True, redis_save=True, aggregation_fn=last_item)

        # loss
        tracker.init_tracking('training/loss/add_node', tensorboard=True)
        tracker.init_tracking('training/loss/movement', tensorboard=True)
        tracker.init_tracking('training/q_estimate_avg/add_node', tensorboard=True)
        tracker.init_tracking('training/q_estimate_avg/movement', tensorboard=True)
        tracker.init_tracking('training/td_error_avg/add_node', tensorboard=True)
        tracker.init_tracking('training/td_error_avg/movement', tensorboard=True)
    # validation and evaluation
    init_validation_and_evaluation_tracks(tracker)


def init_policy_gradient_tracker(tracker: Tracker):
    if tracker.run_mode == RunMode.Train:
        # training metrics
        tracker.init_tracking('training/steps', is_only_cumulative=True, tensorboard=True, db_save=True, redis_save=True)
        tracker.init_tracking('training/reward/utility/total', keep_cumulative=True, db_save=True, redis_save=True,
                              tensorboard=True)
        tracker.init_tracking('training/reward/remaining_gap/total', keep_cumulative=True, db_save=True, redis_save=True,
                              tensorboard=True)
        tracker.init_tracking('training/reward/surplus/total', keep_cumulative=True, db_save=True, redis_save=True,
                              tensorboard=True)
        tracker.init_tracking('training/reward/cost/total', keep_cumulative=True, db_save=True, redis_save=True,
                              tensorboard=True)
        # tracker.init_tracking('training/reward/pre_gap/total', keep_cumulative=True, db_save=True, redis_save=True,
        #                      tensorboard=True)
        tracker.init_tracking('training/batch/reward/utility', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        tracker.init_tracking('training/batch/reward/remaining_gap', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        tracker.init_tracking('training/batch/reward/surplus', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        tracker.init_tracking('training/batch/reward/cost', single_value_is_array=True, tensorboard=True,
                              redis_save=True)
        # tracker.init_tracking('training/batch/reward/pre_gap', single_value_is_array=True, tensorboard=True,
        #                      redis_save=True)
        tracker.init_tracking('training/batch/problem_solved/count', tensorboard=True, redis_save=True)
        tracker.init_tracking('training/batch/dones', tensorboard=True)
        tracker.init_tracking('training/episode/length',  keep_cumulative=True, tensorboard=True)

        tracker.init_tracking('training/action_history/add_node')
        tracker.init_tracking('training/action_history/remove_node')
        tracker.init_tracking('training/action_history/resource_class')
        tracker.init_tracking('training/action_history/quantity')
        tracker.init_tracking('training/action_history/add')
        tracker.init_tracking('training/action_history/remove')
        tracker.init_tracking('training/done', tensorboard=True)
        tracker.init_tracking('training/episodes', is_only_cumulative=True, tensorboard=True,
                              db_save=True, redis_save=True)
        tracker.init_tracking('training/problem_solved/nodes_satisfied', tensorboard=True)
        tracker.init_tracking('training/problem_solved/count', is_only_cumulative=True, tensorboard=True,
                              db_save=True, redis_save=True)
        # loss metrics
        if tracker.config.environment.agent.type == AgentType.TD_AC \
                or tracker.config.environment.agent.type == AgentType.MC_AC:
            tracker.init_tracking('loss/add_node/actor', tensorboard=True)
            tracker.init_tracking('loss/add_node/critic', tensorboard=True)
            tracker.init_tracking('loss/movement/actor', tensorboard=True)
            tracker.init_tracking('loss/movement/critic', tensorboard=True)
        elif tracker.config.environment.agent.type == AgentType.Reinforce:
            tracker.init_tracking('loss/add_node/policy', tensorboard=True)
            tracker.init_tracking('loss/movement/policy', tensorboard=True)

    init_validation_and_evaluation_tracks(tracker)


def init_validation_and_evaluation_tracks(tracker: 'Tracker'):
    eval_seeds = tracker.config.random_seeds.evaluation
    if tracker.run_mode == RunMode.Train:
        val_seeds = tracker.config.random_seeds.validation
        # validation metrics
        for seed in val_seeds:
            tracker.init_tracking(f'validation-{seed}/reward/utility', single_value_is_array=True)
            tracker.init_tracking(f'validation-{seed}/reward/remaining_gap', single_value_is_array=True)
            tracker.init_tracking(f'validation-{seed}/reward/surplus', single_value_is_array=True)
            tracker.init_tracking(f'validation-{seed}/reward/cost', single_value_is_array=True)
            # tracker.init_tracking(f'validation-{seed}/reward/pre_gap', single_value_is_array=True)
            tracker.init_tracking(f'validation-{seed}/problem_solved/count', aggregation_fn=np.max)
            tracker.init_tracking(f'validation-{seed}/hour_satisfied', aggregation_fn=np.max)
            tracker.init_tracking(f'validation-{seed}/resource_utilization', aggregation_fn=np.mean)
            tracker.init_tracking(f'validation-{seed}/action_history/add_node')
            tracker.init_tracking(f'validation-{seed}/action_history/remove_node')
            tracker.init_tracking(f'validation-{seed}/action_history/resource_class')
            tracker.init_tracking(f'validation-{seed}/action_history/quantity')
            tracker.init_tracking(f'validation-{seed}/action_history/add')
            tracker.init_tracking(f'validation-{seed}/action_history/remove')
            tracker.init_tracking(f'validation-{seed}/episode/length')
        tracker.init_tracking('validation/avg/reward/utility', redis_save=True, db_save=True, tensorboard=True,
                              aggregation_fn=np.max)
        tracker.init_tracking('validation/avg/reward/remaining_gap', redis_save=True, db_save=True, tensorboard=True,
                              aggregation_fn=np.max)
        tracker.init_tracking('validation/avg/reward/surplus', redis_save=True, db_save=True, tensorboard=True,
                              aggregation_fn=np.max)
        tracker.init_tracking('validation/avg/reward/cost', redis_save=True, db_save=True, tensorboard=True,
                              aggregation_fn=np.max)
        tracker.init_tracking('validation/avg/problem_solved/count', tensorboard=True, db_save=True, redis_save=True,
                              aggregation_fn=np.max)
        tracker.init_tracking('validation/avg/hour_satisfied', tensorboard=True, db_save=True, redis_save=True,
                              aggregation_fn=np.max)
        tracker.init_tracking('validation/avg/resource_utilization', tensorboard=True, db_save=True, redis_save=True,
                              aggregation_fn=np.max)
        tracker.init_tracking('validation/avg/episode/length', tensorboard=True, db_save=True, redis_save=True,
                              aggregation_fn=np.max)
        # tracker.init_tracking('validation/avg/reward/pre_gap', redis_save=True, db_save=True, tensorboard=True,
        #                      aggregation_fn=np.max)

    # evaluation metrics
    for seed in eval_seeds:
        tracker.init_tracking(f'evaluation-{seed}/times/choose_action', aggregation_fn=np.mean)
        tracker.init_tracking(f'evaluation-{seed}/reward/utility', single_value_is_array=True)
        tracker.init_tracking(f'evaluation-{seed}/reward/remaining_gap', single_value_is_array=True)
        tracker.init_tracking(f'evaluation-{seed}/reward/surplus', single_value_is_array=True)
        tracker.init_tracking(f'evaluation-{seed}/reward/cost', single_value_is_array=True)
        tracker.init_tracking(f'evaluation-{seed}/problem_solved/count', aggregation_fn=np.max)
        tracker.init_tracking(f'evaluation-{seed}/hour_satisfied', aggregation_fn=np.max)
        tracker.init_tracking(f'evaluation-{seed}/resource_utilization', aggregation_fn=np.mean)

        tracker.init_tracking(f'evaluation-{seed}/reward_history/utility', keep_cumulative=True, tensorboard=True)
        tracker.init_tracking(f'evaluation-{seed}/reward_history/remaining_gap', keep_cumulative=True, tensorboard=True)
        tracker.init_tracking(f'evaluation-{seed}/reward_history/surplus', keep_cumulative=True, tensorboard=True)
        tracker.init_tracking(f'evaluation-{seed}/reward_history/cost', keep_cumulative=True, tensorboard=True)
        tracker.init_tracking(f'evaluation-{seed}/problem_solved_history/count', keep_cumulative=True, tensorboard=True)
        tracker.init_tracking(f'evaluation-{seed}/hour_satisfied_history', keep_cumulative=True, tensorboard=True)

        # tracker.init_tracking(f'evaluation-{seed}/reward/pre_gap', single_value_is_array=True)
        tracker.init_tracking(f'evaluation-{seed}/action_history/add_node')
        tracker.init_tracking(f'evaluation-{seed}/action_history/remove_node')
        tracker.init_tracking(f'evaluation-{seed}/action_history/resource_class')
        tracker.init_tracking(f'evaluation-{seed}/action_history/quantity')
        tracker.init_tracking(f'evaluation-{seed}/action_history/add')
        tracker.init_tracking(f'evaluation-{seed}/action_history/remove')

        tracker.init_tracking(f'evaluation-{seed}/episode/length')

    tracker.init_tracking('evaluation/avg/times/choose_action', tensorboard=True, db_save=True, aggregation_fn=np.mean)
    tracker.init_tracking('evaluation/avg/reward/utility', redis_save=True, db_save=True, tensorboard=True,
                          aggregation_fn=last_item)
    tracker.init_tracking('evaluation/avg/reward/remaining_gap', redis_save=True, db_save=True, tensorboard=True,
                          aggregation_fn=last_item)
    tracker.init_tracking('evaluation/avg/reward/surplus', redis_save=True, db_save=True, tensorboard=True,
                          aggregation_fn=last_item)
    tracker.init_tracking('evaluation/avg/reward/cost', redis_save=True, db_save=True, tensorboard=True,
                          aggregation_fn=last_item)
    tracker.init_tracking('evaluation/avg/problem_solved/count', tensorboard=True, db_save=True, redis_save=True,
                          aggregation_fn=last_item)
    tracker.init_tracking('evaluation/avg/hour_satisfied', tensorboard=True, db_save=True, redis_save=True,
                          aggregation_fn=last_item)
    tracker.init_tracking('evaluation/avg/resource_utilization', tensorboard=True, db_save=True, redis_save=True,
                          aggregation_fn=last_item)
    tracker.init_tracking('evaluation/avg/episode/length', tensorboard=True, db_save=True, redis_save=True,
                          aggregation_fn=last_item)
    # tracker.init_tracking('evaluation/avg/reward/pre_gap', redis_save=True, db_save=True, tensorboard=True,
    #                       aggregation_fn=last_item)
