import json
import os
from copy import deepcopy
from typing import Union, List, Optional

import numpy as np

from prorl.common.config import AbstractConfig, load_config_dict, set_sub_config, LoggerConfig, \
    ConfigValueError, ExportMode
from prorl.common.data_structure import RunMode
from prorl.common.filesystem import get_absolute_path, get_data_base_dir, filter_out_path
from prorl.common.object_handler import SaverMode
from prorl.core.step import Step
from prorl.emulator.config import EmulatorConfig
from prorl.emulator.data_structure import ModelTypes
from prorl.environment.agent import AgentType
from prorl.environment.config import EnvConfig, ResourceConfig


class RandomSeedsConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.training = 10
        self.validation: List[int] = [100]  # [100, 110, 120, 130, 140]
        self.evaluation: List[int] = [1000]  # [1000, 1100, 1200, 1300, 1400]
        super(RandomSeedsConfig, self).__init__('RandomSeedsConfig', **configs_to_override)


class TensorboardConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = False
        self.save_path: str = 'tensorboard'
        self.save_model_graph: bool = False
        super(TensorboardConfig, self).__init__('TensorboardConfig', **configs_to_override)

    def get_save_path(self):
        return os.path.join(get_data_base_dir(), self.save_path)

    def _after_override_configs(self):
        self.save_path = filter_out_path(self.save_path)


class SaverConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.save_agent: bool = True
        self.mode: SaverMode = SaverMode.Disk
        self.base_path = 'results'
        self.default_bucket = 'results'
        self.save_prefix: str = ''
        self.save_name_with_uuid: bool = True
        self.save_name_with_date: bool = True
        self.save_name: str = ''
        self.stats_condensed: bool = True
        self.tensorboard: TensorboardConfig = set_sub_config('tensorboard', TensorboardConfig, **configs_to_override)
        super(SaverConfig, self).__init__('SaverConfig', **configs_to_override)

    def get_base_path(self):
        return os.path.join(get_data_base_dir(), self.base_path)

    def _after_override_configs(self):
        self.base_path = filter_out_path(self.base_path)
        if isinstance(self.mode, str):
            try:
                self.mode: SaverMode = SaverMode(self.mode)
            except Exception:
                raise ConfigValueError('mode', self.mode, module=self.name(),
                                       extra_msg=f'Possible values are: {SaverMode.list()}')


class RunConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.run_mode: RunMode = RunMode.Train
        self.episode_per_iteration: int = 1
        self.stop_step: int = -1
        self.step_per_second: int = 1
        self.step_size: int = 3600
        self.stop_date: Union[Step, None] = Step.from_str('1w')  # Step.from_str('2w 2M')
        self.initial_date: Union[Step, None] = None  # Step.from_str('4w')
        self.bootstrapping_steps: int = 0
        self.rollout_batch_size: int = 3600
        self.training_iterations: int = 1000
        self.use_on_policy_agent: bool = True
        self.debug_frequency: int = 1
        self.info_frequency: int = 4
        self.last_validation_metrics: int = 4
        self.validation_run: ValRunConfig = set_sub_config('validation_run', ValRunConfig, **configs_to_override)
        self.evaluation_episode_length: int = 3600
        self.save_n_models_after_best: int = 0
        self.train_every_hour_only: bool = False
        super(RunConfig, self).__init__('RunConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.run_mode, str):
            try:
                self.run_mode: RunMode = RunMode(self.run_mode)
            except Exception:
                raise ConfigValueError('run_mode', self.run_mode, module=self.name(),
                                       extra_msg=f'Possible values are: {RunMode.list()}')
        if self.stop_date is not None and isinstance(self.stop_date, str):
            self.stop_date = Step.from_str(self.stop_date, step_per_second=self.step_per_second)
        if self.initial_date is not None and isinstance(self.initial_date, str):
            self.initial_date = Step.from_str(self.initial_date, step_per_second=self.step_per_second)
        if self.use_on_policy_agent:
            total_steps = self.training_iterations * self.rollout_batch_size
            if self.run_mode == RunMode.Eval:
                total_steps = self.evaluation_episode_length
            self.stop_date = None
            self.stop_step = total_steps


class ValRunConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.stop_step: int = -1
        self.step_per_second: int = 1
        self.step_size: int = 1
        self.stop_date: Union[Step, None] = Step(hour=1)
        self.initial_date: Union[Step, None] = None
        self.rollout_batch_size: int = 1800
        self.debug_frequency: float = Step(week_day=1).to_second()
        self.info_frequency: float = Step(week_day=1).to_second()
        self.validation_frequency: int = 4
        self.keep_n_validation_best: int = 0
        self.keep_metric: str = 'evaluation/avg/reward/utility'
        self.validation_keep_metric: str = 'validation/avg/reward/utility'
        self.logger_level: int = 20
        super(ValRunConfig, self).__init__('ValRunConfig', **configs_to_override)

    def _after_override_configs(self):
        if self.stop_date is not None and isinstance(self.stop_date, str):
            self.stop_date = Step.from_str(self.stop_date, step_per_second=self.step_per_second)
        if self.initial_date is not None and isinstance(self.initial_date, str):
            self.initial_date = Step.from_str(self.initial_date, step_per_second=self.step_per_second)


class RedisConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.update_frequency: int = 1
        self.host: str = 'localhost'
        self.port: int = 6379
        self.db: int = 0
        super(RedisConfig, self).__init__('RedisConfig', **configs_to_override)


class MultiRunParamConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.key: str = 'environment.agent.double_dqn.learning_rate'
        self.key_short: str = 'lr'
        self.value: Union[str, int, float, list, dict] = 0.1
        self.filename_key_val: str = 'lr=0.1'
        super(MultiRunParamConfig, self).__init__(MultiRunParamConfig, **configs_to_override)


class MultiRunConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.is_multi_run: bool = False
        self.multi_run_code: Optional[str] = None
        self.multi_run_params: List[MultiRunParamConfig] = [
            # MultiRunParamConfig(
            #     key='environment.agent.double_dqn.learning_rate',
            #     value=0.1,
            #     filename_key_val='lr=0.1'
            # ),
            # MultiRunParamConfig(
            #     key='environment.agent.double_dqn.target_net_update_frequency',
            #     value=3600,
            #     filename_key_val='target-update=3600'
            # )
        ]
        super(MultiRunConfig, self).__init__(MultiRunConfig, **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.multi_run_params, list):
            multi_run_params: List[MultiRunParamConfig] = []
            for param in self.multi_run_params:
                if isinstance(param, MultiRunParamConfig):
                    multi_run_params.append(param)
                elif isinstance(param, dict):
                    multi_run_params.append(MultiRunParamConfig(**param))
                else:
                    raise ConfigValueError('multi_run_params', self.multi_run_params, self.name(),
                                           extra_msg='multi_run_params entries must be an object')
            self.multi_run_params: List[MultiRunParamConfig] = multi_run_params


def get_n_resource_units(total: int, n: int) -> int:
    module = total % n
    if module == 0:
        return total
    if module >= (n/2):
        add = n - module
        return total + add
    else:
        return total - module


class SingleRunConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.version = '2.0.0'
        self.run: RunConfig = set_sub_config('run', RunConfig, **configs_to_override)
        self.saver: SaverConfig = set_sub_config('saver', SaverConfig, **configs_to_override)
        self.emulator: EmulatorConfig = set_sub_config('emulator', EmulatorConfig, root_dir, **configs_to_override)
        self.environment: EnvConfig = set_sub_config('environment', EnvConfig, root_dir, **configs_to_override)
        self.logger: LoggerConfig = set_sub_config('logger', LoggerConfig, 'global', **configs_to_override)
        self.random_seeds: RandomSeedsConfig = set_sub_config('random_seeds', RandomSeedsConfig, **configs_to_override)
        self.redis: RedisConfig = set_sub_config('redis', RedisConfig, **configs_to_override)
        self.multi_run: MultiRunConfig = set_sub_config('multi_run', MultiRunConfig, **configs_to_override)
        super(SingleRunConfig, self).__init__(root_dir=root_dir,
                                              config_object_name='SingleRunConfig', **configs_to_override)

    def _after_override_configs(self):
        for res in self.environment.resources:
            self.emulator.model.nodes_demand_model_options.resource_absolute_values[res.name] = res.total_available
            self.emulator.model.auto_configurable_model_options.resource_absolute_values[res.name] = res.total_available
        if self.emulator.model.type == ModelTypes.TimDatasetModel:
            model_options = self.emulator.model.tim_dataset_model_options
            n_nodes = len(self.emulator.model.base_station_names())
            if model_options.load_ran_configurations:
                self.environment.nodes.n_nodes = n_nodes
                path = get_absolute_path(model_options.get_magic_numbers_path())
                with open(path, 'r') as f:
                    magic_numbers = json.load(f)
                load_ratio = model_options.load_ratio
                run_ratio_conf = None
                for ratio_conf in magic_numbers['ratios_capacities']:
                    if ratio_conf['ratio'] == load_ratio:
                        run_ratio_conf = ratio_conf
                # total_resource_units = get_n_resource_units(magic_numbers['total_bs_units'], n_nodes)
                resource_capacity = run_ratio_conf['unit_capacity']
                node_units = run_ratio_conf['nodes_units']
                node_capacity = resource_capacity * node_units
                total_capacity = node_capacity * n_nodes
                # assert total_capacity == run_ratio_conf['total_capacity']
                self.environment.resources = [
                    ResourceConfig(
                        name='res_1',
                        total_available=total_capacity,
                        allocated=node_capacity,
                        total_units=node_units * n_nodes,
                        units_allocated=node_units,
                        bucket_size=1,
                        min_resource_buckets_per_node=0,
                        classes={
                            'resource': {'cost': 1, 'capacity': resource_capacity, 'allocated': node_units},
                        }
                    )
                ]
                # set the actions
                # self.environment.action_space.bucket_move = run_ratio_conf['actions_from_max_difference']
            # set the time step properties and the batch properties
            # self.run.step_size = model_options.step_size
            step_size_multiplier = model_options.step_size // self.run.step_size
            if model_options.step_size > 3600:
                step_size_multiplier = step_size_multiplier / (model_options.step_size / 3600)
            self.run.rollout_batch_size = int(24 * 7 * self.run.episode_per_iteration * step_size_multiplier)
            eval_weeks = model_options.time_step[RunMode.Eval]['end_date'].weeks_difference(
                model_options.time_step[RunMode.Eval]['start_date'])
            eval_n_loads = len(model_options.loads_with_respect_to_capacity[RunMode.Eval])
            if eval_weeks == 0:
                eval_hours = model_options.time_step[RunMode.Eval]['end_date'].hours_difference(
                    model_options.time_step[RunMode.Eval]['start_date'])
                eval_len = int(eval_hours * step_size_multiplier * eval_n_loads)
            else:
                eval_len = int(24 * 7 * eval_weeks * step_size_multiplier * eval_n_loads)
            self.run.evaluation_episode_length = eval_len
            validation_weeks = model_options.time_step[RunMode.Validation]['end_date'].weeks_difference(
                model_options.time_step[RunMode.Validation]['start_date'])
            val_n_loads = len(model_options.loads_with_respect_to_capacity[RunMode.Validation])
            if validation_weeks == 0:
                validation_hours = model_options.time_step[RunMode.Validation]['end_date'].hours_difference(
                    model_options.time_step[RunMode.Validation]['start_date'])
                validation_len = int(validation_hours * step_size_multiplier * val_n_loads)
            else:
                validation_len = int(24 * 7 * validation_weeks * step_size_multiplier * val_n_loads)
            self.run.validation_run.rollout_batch_size = validation_len
            self.run.stop_step = None
            # self.run.initial_date = model_options.time_step[self.run.run_mode]['start_date']
            # self.run.stop_date = model_options.time_step[self.run.run_mode]['end_date']
            if AgentType.is_mc_method(self.environment.agent.type):
                self.run.episode_per_iteration = 1
        # updated for the TestModel
        if self.emulator.model.type == ModelTypes.TestModel:
            self.emulator.model.test_model_options.time_step_size = self.run.step_size
            self.run.rollout_batch_size = int(24 * 7 * self.run.episode_per_iteration)
            self.run.evaluation_episode_length = int(24 * 7 * 1)
            self.run.validation_run.rollout_batch_size = int(24 * 7 * 1)
            self.run.stop_step = self.run.step_size * 24 * 7
            self.emulator.model.test_model_options.n_nodes = self.environment.nodes.n_nodes
            self.emulator.model.base_station_name_mappings = {str(i): i for i in range(self.environment.nodes.n_nodes)}
            self.emulator.model.n_base_stations = self.environment.nodes.n_nodes
        if self.emulator.model.type == ModelTypes.SyntheticModel:
            model_options = self.emulator.model.synthetic_model
            episode_length = model_options.episode_length * len(model_options.distribution_multipliers)
            step_sub_steps = model_options.model_step_size // self.run.step_size
            self.run.rollout_batch_size = episode_length * step_sub_steps
            self.run.evaluation_episode_length = episode_length * step_sub_steps
            self.run.validation_run.rollout_batch_size = episode_length * step_sub_steps
            total_units = 0
            res_capacity = 1
            for res_conf in self.environment.resources:
                for res_class_name, res_class in res_conf.classes.items():
                    res_capacity = res_class['capacity']
                total_units += res_conf.total_units
            max_demand = self.emulator.model.synthetic_model.demand_absolute_value
            max_multiplier = max(self.emulator.model.synthetic_model.distribution_multipliers)
            max_demand *= max_multiplier
            max_std = 0
            for couple in self.emulator.model.synthetic_model.couples_config:
                if couple.std > max_std:
                    max_std = couple.std
            test_values = np.random.normal(max_demand, max_std, 1000000)
            max_demand = np.ceil(test_values.max())
            del test_values
            max_units_demand = int(np.ceil(max_demand / res_capacity).item())
            if max_units_demand > total_units:
                max_units_demand = total_units
            actions = [i+1 for i in range(max_units_demand)]
            # self.environment.action_space.bucket_move = actions
        if AgentType.is_policy_gradient(self.environment.agent.type):
            self.run.use_on_policy_agent = True
        if AgentType.is_value_based(self.environment.agent.type):
            self.run.use_on_policy_agent = False

    def to_validation_run_config(self):
        val_config = SingleRunConfig(root_dir=self.root_dir, **deepcopy(self.export(mode=ExportMode.DICT)))
        val_config.run.run_mode = RunMode.Validation
        val_config.saver.save_batch = -1
        val_config.logger.level = val_config.run.validation_run.logger_level
        val_config.random_seeds.environment = val_config.random_seeds.validation_env
        val_config.saver.tensorboard.enabled = False
        val_config.run.stop_step = val_config.run.validation_run.stop_step
        val_config.run.step_per_second = val_config.run.validation_run.step_per_second
        val_config.run.step_size = val_config.run.validation_run.step_size
        val_config.run.stop_date = val_config.run.validation_run.stop_date
        val_config.run.initial_date = val_config.run.validation_run.initial_date
        val_config.run.debug_frequency = val_config.run.validation_run.debug_frequency
        val_config.run.info_frequency = val_config.run.validation_run.info_frequency
        val_config.run.validation_run.enabled = False
        return val_config


def get_single_run_config(root_dir, config_path: str = None, log_level: int or None = None) -> SingleRunConfig:
    config = SingleRunConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    if log_level is not None:
        config.set_configs(logger={'level': log_level})
    return config
