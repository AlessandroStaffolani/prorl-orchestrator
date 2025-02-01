import json
import os
from abc import abstractmethod
from glob import glob
from logging import Logger
from typing import List, Optional, Any, Dict, Tuple, Union

from numpy.random import RandomState

from prorl.emulator import logger, EmulatorConfig, emulator_config
from prorl.core.step_data import StepDataEntry, StepData
from prorl.core.step import Step
from prorl.common.data_structure import RunMode
from prorl.environment.node import POOL_NODE


def parse_entry(entry: Dict[str, dict]) -> List[StepDataEntry]:
    entries: List[StepDataEntry] = []
    step = Step(**entry['step'])
    for res_name, res_data in entry['data'].items():
        for bs_data in res_data:
            bs_name = list(bs_data.keys())[0]
            bs_value = list(bs_data.values())[0]
            entries.append(StepDataEntry(res_name, value=bs_value, base_station=bs_name, step=step))
    return entries


def parse_step_data(entry: Dict[str, dict], generator_model) -> StepData:
    entries = parse_entry(entry)
    return StepData(*entries, generator_model=generator_model)


class AbstractModel:

    def __init__(
            self,
            base_station_names: List[str],
            resource_names: List[str],
            model_name: str,
            emulator_configuration: Optional[EmulatorConfig] = None,
            random_state: Optional[RandomState] = None,
            log: Optional[Logger] = None,
            disk_data_folder: Optional[str] = None,
            disable_log: bool = False,
            run_mode: RunMode = RunMode.Train,
            random_seed: int = 42,
            use_pool_node: bool = False,
            **kwargs,
    ):
        self.emulator_config: EmulatorConfig = emulator_configuration if emulator_configuration is not None\
            else emulator_config
        self.logger: Logger = log if log is not None else logger
        self.run_mode: RunMode = run_mode
        self.random: RandomState = random_state if random_state is not None else RandomState()
        self.random_seed = random_seed
        self.base_station_names: List[str] = base_station_names
        self.resource_names: List[str] = resource_names
        self._name: str = model_name
        self.disable_log: bool = disable_log
        self.disk_data_folder = disk_data_folder
        self.use_disk_data = self.emulator_config.model.model_disk.use_disk and self.disk_data_folder is not None
        self.model_metadata: Optional[Dict[str, Any]] = None
        self.episodes_available: int = 0
        self.episodes_files: Dict[int, str] = {}
        self.current_episodes_order: List[int] = []
        self.current_episode: int = -1
        self.current_episode_step: int = 0
        self.current_episode_data: List[StepData] = []
        self.model_step_size: int = 1
        self.use_pool_node: bool = use_pool_node
        self.load_model_metadata()

    @property
    def system_load(self) -> float:
        return 1.0

    @property
    def pool_bs_name(self) -> str:
        return POOL_NODE

    def model_name(self):
        return self._name

    def get_step_data(self, step: Step) -> StepData:
        step_data = self.current_episode_data[self.current_episode_step]
        assert step_data.step == step
        self.current_episode_step += 1
        return step_data

    def load_model_metadata(self):
        if self.use_disk_data:
            assert self.disk_data_folder is not None
            assert os.path.isdir(self.disk_data_folder),\
                f'disk_data_folder does not exists on path: "{self.disk_data_folder}"'
            with open(os.path.join(self.disk_data_folder, 'metadata.json'), 'r') as file:
                self.model_metadata = json.load(file)
            self.episodes_available = self.model_metadata['model_disk']['episodes_to_generate'][self.run_mode.value]
            episodes_files = glob(f'{self.disk_data_folder}/{self.run_mode.value}-{self.random_seed}-*.json')
            assert len(episodes_files) == self.episodes_available
            for e_file in episodes_files:
                last_part = e_file.split('/')[-1]
                filtered = last_part.replace(f'{self.run_mode.value}-{self.random_seed}-', '').replace('.json', '')
                self.episodes_files[int(filtered)] = e_file
            self._shuffle_episodes()

    def _shuffle_episodes(self):
        ids = list(self.episodes_files.keys())
        self.current_episodes_order = self.random.choice(ids, replace=False, size=len(ids))

    def load_current_episode(self):
        if self.use_disk_data:
            if self.current_episode >= len(self.current_episodes_order):
                self._shuffle_episodes()
            episode_id = self.current_episodes_order[self.current_episode % len(self.current_episodes_order)]
            episode_file = self.episodes_files[episode_id]
            with open(episode_file, 'r') as f:
                episode_data = json.load(f)
            self.current_episode_data = list(map(lambda entry: parse_step_data(entry, self.model_name()), episode_data))

    def generate(
            self,
            step: Step,
            resources: List[str] = None,
            base_stations: List[str] = None,
            is_last_step: bool = False,
    ) -> StepData:
        if self.use_disk_data:
            data: StepData = self.get_step_data(step)
            return data
        else:
            list_base_stations = self.base_station_names
            list_resources = self.resource_names
            if base_stations is not None:
                list_base_stations = base_stations
            if resources is not None:
                list_resources = resources
            data = StepData(generator_model=self.model_name())
            for i, resource in enumerate(list_resources):
                for j, base_station in enumerate(list_base_stations):
                    if base_station == self.pool_bs_name:
                        entry = self.pool_step_data_entry(resource, step)
                    else:
                        entry = self._generate_single_base_station(step, resource, base_station)
                    data.add_entry(entry)
            return data

    def pool_step_data_entry(self, resource: str, step: Step) -> StepDataEntry:
        return StepDataEntry(resource=resource, value=0, base_station=self.pool_bs_name, step=step)

    def reset(self, show_log=True, full_reset=True):
        self.current_episode += 1
        self.current_episode_step = 0
        self.load_current_episode()
        if show_log:
            self.logger.debug(f'Model {self._name} has been reset')

    def is_full_reset(self) -> bool:
        return True

    def get_start_step(self, step_size: int) -> Union[None, Step]:
        return None

    def get_stop_step(self, step_size) -> Union[None, int, Step]:
        return None

    @abstractmethod
    def _generate_single_base_station(self, step: Step, resource_name: str, base_station_name: str) -> StepDataEntry:
        pass
