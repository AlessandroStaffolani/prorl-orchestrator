import json
import os
import sys

from prorl.common.data_structure import RunMode
from prorl.core.step import Step
from prorl.emulator.data_structure import ModelTypes

if sys.version_info >= (3, 8):
    from typing import Dict, List, Optional, Union
else:
    from typing import Dict, List

from prorl.common.config import AbstractConfig, ConfigValueError, load_config_dict, set_sub_config
from prorl.common.filesystem import get_data_base_dir, filter_out_path


class ModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: ModelTypes = ModelTypes.TimDatasetModel
        self.n_base_stations = 4
        self.base_station_name_mappings: Dict[str, str] = {
            'bs_1': 'bs_1',
            'bs_2': 'bs_2',
            'bs_3': 'bs_3',
            'bs_4': 'bs_4',
            'bs_5': 'bs_5',
            'bs_6': 'bs_6',
            'bs_7': 'bs_7',
            'bs_8': 'bs_8',
        }
        self.n_resources = 1
        self.resource_name_mappings: Dict[str, str] = {
            'res_1': 'bandwidth',
            # 'res_2': 'computation'
        }
        self.model_disk: ModelDiskConfig = set_sub_config('model_disk', ModelDiskConfig, **configs_to_override)
        self.tim_dataset_model_options: TimDatasetModelConfig = set_sub_config(
            'tim_dataset_model_options', TimDatasetModelConfig, **configs_to_override)
        self.synthetic_model: SyntheticModelConfig = set_sub_config(
            'synthetic_model', SyntheticModelConfig, **configs_to_override)
        # if self.type == ModelTypes.GaussianModel:
        #     self.model_options: GaussianModelConfig = self.gaussian_model_options
        super(ModelConfig, self).__init__('ModelConfig', **configs_to_override)
        self.export_exclude.append('model_options')

    def base_station_names(self) -> List[str]:
        return list(self.base_station_name_mappings.keys())

    def resource_names(self) -> List[str]:
        return list(self.resource_name_mappings.keys())

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: ModelTypes = ModelTypes(self.type)
            except Exception:
                raise ConfigValueError('loader_mode', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {ModelTypes.list()}')
        if self.type == ModelTypes.TimDatasetModel:
            self.model_disk.use_disk = False
            self.base_station_name_mappings = {str(bs_id): bs_id for bs_id in self.tim_dataset_model_options.bs_ids}
        elif self.type == ModelTypes.TestModel:
            self.model_disk.use_disk = False
            self.base_station_name_mappings = {str(i): i for i in range(self.test_model_options.n_nodes)}
        self.n_base_stations = len(self.base_station_name_mappings)


class ModelDiskConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.use_disk: bool = False
        self.episodes_to_generate: Dict[str, int] = {
            'training': 100,
            'validation': 1,
            'evaluation': 1
        }
        self.seeds: Dict[str, List[int]] = {
            'training': [10],  # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'validation': [10],  # [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            'evaluation': [10]  # [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        self.base_folder: str = 'models'
        super(ModelDiskConfig, self).__init__('ModelDiskConfig', **configs_to_override)

    def get_base_folder(self):
        return os.path.join(get_data_base_dir(), self.base_folder)

    def _after_override_configs(self):
        self.base_folder = filter_out_path(self.base_folder)


class TimDatasetModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.full_data_path: Optional[str] = 'models/tim_dataset/12-nodes/full_data.csv'
        self.chunks_path: Optional[str] = None
        self.time_step: Dict[str, Dict[str, Step]] = {
            RunMode.Train: {
                'start_date': Step.from_str('4w 2W'),
                'end_date': Step.from_str('4w 2M')
            },
            RunMode.Validation: {
                'start_date': Step.from_str('4w'),
                'end_date': Step.from_str('4w 1W')
            },
            RunMode.Eval: {
                'start_date': Step.from_str('4w 1W'),
                'end_date': Step.from_str('4w 2W')
            }
        }
        self.step_size: int = 3600  # 1 hour
        self.chunks_extension: str = 'csv'
        self.separator: str = ','
        self.node_id_column: str = 'aggregated_bs_id'
        self.demand_column: str = 'internet'
        self.idx_column: str = 'idx'
        self.hour_col: str = 'hour'
        self.week_day_col: str = 'weekday'
        self.week_col: str = 'week'
        self.month_col: str = 'month'
        # if it is string is a path to a sub_area-json file
        self.bs_ids: Union[List[int], str] = [
            2841,
            1760,
            2042,
            2338,
            2340,
            2920
        ]
        # self.bs_ids: Union[List[int], str] = [4611, 4518, 4147, 4425, 4714, 4808, 4528, 4430, 3505, 3506, 3501, 3503]
        self.bs_ids: Union[List[int], str] = 'models/tim_dataset/sub_area-top_12_nodes.json'
        self.ran_configurations_path: str = 'models/tim_dataset/run_magic_numbers-12_nodes-new.json'
        self.load_ran_configurations: bool = False
        self.load_ratio: float = 0.8
        self.bs_data_path: str = 'models/tim_dataset/aggregated_bs_data-LTE.csv'
        self.use_index: bool = True
        self.index_data_path = 'models/tim_dataset/indexes/12-nodes-data-index.json'
        self.change_load_frequencies: Dict[RunMode, int] = {
            RunMode.Train: 168,
            RunMode.Validation: 168,
            RunMode.Eval: 168
        }
        self.loads_with_respect_to_capacity: Dict[RunMode, List[float]] = {
            RunMode.Train: [0.8],  # [0.7, 0.8, 0.9, 1, 1.1, 1.2],
            RunMode.Validation: [0.8],  # [0.7, 0.8, 0.9, 1, 1.1, 1.2],
            RunMode.Eval: [0.8],  # [0.7, 0.8, 0.9, 1, 1.1, 1.2],
        }
        super(TimDatasetModelConfig, self).__init__('TimDatasetModelConfig', **configs_to_override)

    def get_full_data_path(self):
        return os.path.join(get_data_base_dir(), self.full_data_path)

    def get_chunks_path(self):
        if self.chunks_path is not None:
            return os.path.join(get_data_base_dir(), self.chunks_path)
        else:
            return None

    def get_bs_ids(self):
        if isinstance(self.bs_ids, str):
            return os.path.join(get_data_base_dir(), self.bs_ids)
        else:
            return self.bs_ids

    def get_magic_numbers_path(self):
        return os.path.join(get_data_base_dir(), self.ran_configurations_path)

    def get_bs_data_path(self):
        return os.path.join(get_data_base_dir(), self.bs_data_path)

    def get_index_data_path(self):
        return os.path.join(get_data_base_dir(), self.index_data_path)

    def _after_override_configs(self):
        self.full_data_path = filter_out_path(self.full_data_path)
        self.chunks_path = filter_out_path(self.chunks_path) if self.chunks_path is not None else None
        self.ran_configurations_path = filter_out_path(self.ran_configurations_path)
        self.bs_data_path = filter_out_path(self.bs_data_path)
        self.index_data_path = filter_out_path(self.index_data_path)
        if isinstance(self.get_bs_ids(), str):
            self.bs_ids = filter_out_path(self.bs_ids)
            full_path = self.get_bs_ids()
            assert os.path.exists(full_path), 'bs_ids path not exists. Provide a list of int or a path to json file'
            with open(full_path, 'r') as f:
                data = json.load(f)
                assert 'aggregated_bs_id' in data, \
                    'The bs_ids json file must have the list of int in the aggregated_bs_id field'
                assert isinstance(data['aggregated_bs_id'], list), \
                    'The bs_ids json file must have a list of int in the aggregated_bs_id field'
                self.bs_ids = data['aggregated_bs_id']
        for mode, conf in self.time_step.items():
            for key, value in conf.items():
                if isinstance(value, str):
                    self.time_step[mode][key] = Step.from_str(value)


class CoupleConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.calm_load: float = 0.5
        self.calm_load_equal: bool = True
        self.stress_load: float = 0.8
        self.stress_every: Union[int, Dict[str, Union[float, List[float]]]] = 12
        self.keep_stress: int = 0
        self.swap_stress: bool = True
        self.std: float = 0.5
        super(CoupleConfig, self).__init__('CoupleConfig', **configs_to_override)


class SyntheticModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.episode_length: int = 168
        self.change_distribution_frequency: int = 168
        self.couples_config: List[CoupleConfig] = [
            CoupleConfig(
                calm_load=0.3,
                stress_load=0.7,
                stress_every={
                    'hour': [9, 18],
                    'week_day': [0, 1, 2, 3, 4],
                },
                keep_stress=0,
                std=1,
                swap_stress=True
            ),
            CoupleConfig(
                calm_load=0.3,
                stress_load=0.85,
                stress_every={
                    'hour': [12, 20],
                    'week_day': 5
                },
                keep_stress=0,
                std=1,
                swap_stress=True
            ),
            CoupleConfig(
                calm_load=0.3,
                stress_load=0.8,
                stress_every={
                    'hour': [10, 28],
                    'week_day': 6
                },
                keep_stress=0,
                std=1,
                swap_stress=True
            ),
        ]
        self.demand_absolute_value: float = 80.0
        self.model_step_size: int = 3600
        # every steps the means are multiplied by increase_multiplier
        self.distribution_multipliers: List[float] = [1]  # [1, 1.5, 2, 0.6]
        super(SyntheticModelConfig, self).__init__('SyntheticModelConfig', **configs_to_override)

    def _after_override_configs(self):
        couples: List[CoupleConfig] = []
        for conf in self.couples_config:
            if isinstance(conf, CoupleConfig):
                couples.append(conf)
            elif isinstance(conf, dict):
                couples.append(CoupleConfig(**conf))
            else:
                raise ConfigValueError('couples_config', self.couples_config, self.name(),
                                       extra_msg='Couple Config entries must be a dict')
        self.couples_config = couples


class EmulatorConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.model: ModelConfig = set_sub_config('model', ModelConfig, **configs_to_override)
        super(EmulatorConfig, self).__init__(config_object_name='EmulatorConfig',
                                             root_dir=root_dir, **configs_to_override)


def get_emulator_config(root_dir, config_path: str = None, log_level: int or None = None) -> EmulatorConfig:
    config = EmulatorConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    if log_level is not None:
        config.set_configs(logger={'level': log_level})
    return config
