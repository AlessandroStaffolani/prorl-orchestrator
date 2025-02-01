import json
import os
import time
from glob import glob
from logging import Logger
from typing import List, Optional, Tuple, Dict, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.random import RandomState

from prorl import SingleRunConfig
from prorl.common.data_structure import RunMode
from prorl.common.encoders import NumpyEncoder
from prorl.common.filesystem import get_absolute_path, create_directory, load_file, ROOT_DIR, \
    create_directory_from_filepath, get_data_base_dir
from prorl.common.print_utils import print_status
from prorl.core.step import Step
from prorl.core.step_data import StepDataEntry, StepData
from prorl.core.timestep import time_step_factory_get, TimeStep
from prorl.emulator import EmulatorConfig
from prorl.emulator.models.abstract import AbstractModel
from prorl.environment.data_structure import EnvResource
from prorl.run.remote import MongoRunWrapper


def get_step_idx(step: Step) -> int:
    return step.hour + step.week_day * 24


class TimDatasetModel(AbstractModel):

    def __init__(
            self,
            base_station_names: List[str],
            resource_names: List[str],
            em_config: Optional[EmulatorConfig] = None,
            random_state: Optional[RandomState] = None,
            log: Optional[Logger] = None,
            disk_data_folder: Optional[str] = None,
            disable_log: bool = False,
            run_mode: RunMode = RunMode.Train,
            random_seed: int = 42,
            use_pool_node: bool = False,
            **kwargs
    ):
        super(TimDatasetModel, self).__init__(
            base_station_names=base_station_names,
            resource_names=resource_names,
            model_name='TimDatasetModel',
            emulator_configuration=em_config,
            random_state=random_state,
            log=log,
            disk_data_folder=disk_data_folder,
            disable_log=disable_log,
            run_mode=run_mode,
            random_seed=random_seed,
            use_pool_node=use_pool_node,
            **kwargs
        )
        model_config = self.emulator_config.model.tim_dataset_model_options
        self.full_data_path: Optional[str] = model_config.get_full_data_path()
        self.chunks_path: Optional[str] = model_config.get_chunks_path()
        self.index_data_path: Optional[str] = model_config.get_index_data_path()
        self.use_chunks = False
        self.use_index = False
        if not model_config.use_index:
            if self.chunks_path is not None:
                assert os.path.exists(self.chunks_path), f'{self.chunks_path} not exists'
                assert os.path.isdir(self.chunks_path), f'{self.chunks_path} is not a directory'
                self.use_chunks = True
            elif self.full_data_path is not None:
                assert os.path.exists(self.full_data_path), f'{self.full_data_path} not exists'
                if os.path.isdir(self.full_data_path):
                    self.full_data_path = os.path.join(self.full_data_path, f'{self.run_mode.value}-data.csv')
                assert os.path.exists(self.full_data_path), f'{self.full_data_path} not exists'
            else:
                raise AttributeError('one between full_data_path and chunks_path must be defined')
        else:
            self.index_data_path = get_absolute_path(self.index_data_path)
            assert os.path.exists(self.index_data_path), f'{self.index_data_path} not exists'
            self.use_index = True
        self.chunks_extension: str = model_config.chunks_extension
        self.separator: str = model_config.separator
        self.node_id_column: str = model_config.node_id_column
        self.demand_column: str = model_config.demand_column
        self.idx_column: str = model_config.idx_column
        self.hour_col: str = model_config.hour_col
        self.week_day_col: str = model_config.week_day_col
        self.week_col: str = model_config.week_col
        self.month_col: str = model_config.month_col
        self.model_step_size: int = model_config.step_size
        self.load_ratio: float = model_config.load_ratio
        self.change_load_frequency: int = model_config.change_load_frequencies[self.run_mode]
        self.loads_with_respect_to_capacity: List[float] = model_config.loads_with_respect_to_capacity[self.run_mode]
        self.model_time_step = model_config.time_step[self.run_mode]

        self.chunks_files_path: List[str] = self.get_chunks_files_path()

        self.current_chunk_path: Optional[str] = None
        self.current_chunk: Optional[pd.DataFrame] = None
        self.current_chunk_idx_range: Optional[Tuple[int, int]] = None
        self.next_chunk_index: int = 0

        self.full_data_df: Optional[pd.DataFrame] = None
        self.index_data: Optional[Dict[str, dict]] = None
        self.bs_ids_no_pool = [int(bs_id) for bs_id in self.base_station_names if bs_id != self.pool_bs_name]

        self.internal_counter: int = 0
        self.current_load_index: int = 0
        self.last_step: Optional[Step] = None

        # before finishing the init, we load the first chunk
        if self.use_chunks:
            self.load_next_chunk()
        elif self.use_index:
            self.load_index_data()
        else:
            self.load_full_data()

    @property
    def system_load(self) -> float:
        current_index = self.current_load_index % len(self.loads_with_respect_to_capacity)
        return self.loads_with_respect_to_capacity[current_index]

    def load_full_data(self):
        df = pd.read_csv(filepath_or_buffer=self.full_data_path, sep=self.separator)
        self.full_data_df = df[df[self.node_id_column].isin(self.bs_ids_no_pool)]

    def get_chunks_files_path(self) -> List[str]:
        if self.use_chunks:
            abs_path = get_absolute_path(self.chunks_path)
            return sorted(glob(f'{abs_path}/*.{self.chunks_extension}'), key=lambda f: f.split('/')[-1])
        else:
            return []

    def load_index_data(self):
        self.index_data = load_file(self.index_data_path, is_json=True)

    def load_next_chunk(self):
        self.current_chunk_path: str = self.chunks_files_path[self.next_chunk_index % len(self.chunks_files_path)]
        self.current_chunk: pd.DataFrame = pd.read_csv(filepath_or_buffer=self.current_chunk_path, sep=self.separator)
        self.current_chunk = self.current_chunk[self.current_chunk[self.node_id_column].isin(self.bs_ids_no_pool)]
        self.next_chunk_index += 1
        self.current_chunk_idx_range = (
            self.current_chunk[self.idx_column].min(),
            self.current_chunk[self.idx_column].max()
        )

    def _step_to_model_step_size(self, step: Step):
        if self.model_step_size < 3600 * 24:
            return Step(
                second_step=0,
                second=0,
                minute=0,
                hour=step.hour,
                week_day=step.week_day,
                week=step.week,
                month=step.month,
                year=step.year
            )

    def post_generate(self, step_data: StepData, step: Step) -> StepData:
        current_index = self.current_load_index % len(self.loads_with_respect_to_capacity)
        current_multiplier = self.loads_with_respect_to_capacity[current_index] / self.load_ratio
        if current_multiplier != 1.0:
            # if the multiplier is different to 1.0 we need to multiply the demands by the multiplier
            step_data = step_data * current_multiplier
        if step != self.last_step:
            self.internal_counter += 1
            self.last_step = step
        if self.internal_counter % self.change_load_frequency == 0:
            self.current_load_index += 1
            if self.current_load_index >= len(self.loads_with_respect_to_capacity):
                self.current_load_index = 0
        return step_data

    def generate(
            self,
            step: Step,
            resources: List[str] = None,
            base_stations: List[str] = None,
            is_last_step: bool = False,
    ) -> StepData:
        list_base_stations = self.base_station_names
        list_resources = self.resource_names
        if base_stations is not None:
            list_base_stations = base_stations
        if resources is not None:
            list_resources = resources
        if self.use_disk_data:
            data: StepData = self.get_step_data(step)
            return self.post_generate(data, step)
        elif self.use_index:
            model_step = self._step_to_model_step_size(step)
            key = model_step.to_str(full_key=True, no_total_step=True)
            step_data_raw = {}
            if is_last_step and key not in self.index_data:
                for res in list_resources:
                    step_data_raw[res] = []
                    for bs_name in list_base_stations:
                        step_data_raw[res].append({bs_name: 0})
            else:
                raw_data = self.index_data[key]
                # if self.use_pool_node:
                #     for res in list_resources:
                #         step_data_raw[res] = [{self.pool_bs_name: 0}]
                #         for bs_data in raw_data[res]:
                #             step_data_raw[res].append(bs_data)
                # else:
                #     step_data_raw = raw_data
                step_data_raw = raw_data
            data = StepData.from_dict({
                'data': step_data_raw,
                'step': step.to_dict(),
                'generator_model': self.model_name()
            })
            return self.post_generate(data, step)
        else:
            data = StepData(generator_model=self.model_name())
            for i, resource in enumerate(list_resources):
                if self.use_chunks:
                    rows = self.get_rows_from_chunks(step)
                else:
                    rows = self.get_rows_from_full_data(step)
                for j, base_station in enumerate(list_base_stations):
                    if base_station == self.pool_bs_name:
                        entry = self.pool_step_data_entry(resource, step)
                    else:
                        bs_sub_row = rows[rows[self.node_id_column] == int(base_station)]
                        if len(bs_sub_row) == 0:
                            value = 0
                        else:
                            value = bs_sub_row[self.demand_column].item()
                        entry = StepDataEntry(
                            resource=resource,
                            value=value,
                            base_station=base_station,
                            step=step
                        )
                    data.add_entry(entry)
            return self.post_generate(data, step)

    def get_rows_from_chunks(self, step: Step):
        idx = get_step_idx(step)
        if idx < self.current_chunk_idx_range[0] or idx > self.current_chunk_idx_range[1]:
            # idx not in range, firstly change the chunk and then get the data
            self.load_next_chunk()
        assert self.current_chunk_idx_range[0] <= idx <= self.current_chunk_idx_range[1], \
            'Current chunk does not contain the step idx'
        return self.current_chunk[(self.current_chunk[self.idx_column] == idx)]

    def get_rows_from_full_data(self, step: Step):
        hour = step.hour
        week_day = step.week_day
        week = step.week
        month = step.month
        return self.full_data_df[
            (self.full_data_df[self.hour_col] == hour) &
            (self.full_data_df[self.week_day_col] == week_day) &
            (self.full_data_df[self.week_col] == week) &
            (self.full_data_df[self.month_col] == month)
        ]

    def _generate_single_base_station(self, step: Step, resource_name: str, base_station_name: str) -> StepDataEntry:
        pass

    def reset(self, show_log=True, full_reset=True):
        super(TimDatasetModel, self).reset(show_log)
        if self.use_chunks:
            self.next_chunk_index = 0
            self.load_next_chunk()
        if full_reset:
            self.internal_counter = 0
            self.current_load_index = 0

    def is_full_reset(self) -> bool:
        if len(self.loads_with_respect_to_capacity) == 1:
            return True
        else:
            return self.current_load_index == len(self.loads_with_respect_to_capacity)

    def get_start_step(self, step_size: int) -> Union[None, Step]:
        return self.model_time_step['start_date']

    def get_stop_step(self, step_size: int) -> Union[None, int, Step]:
        return self.model_time_step['end_date']


def generate_tim_model_index(
        config: SingleRunConfig,
        save_path: str,
        base_station_names: List[str],
        resource_names: List[str],
        em_config: Optional[EmulatorConfig] = None,
        random_state: Optional[RandomState] = None,
        log: Optional[Logger] = None,
        disk_data_folder: Optional[str] = None,
        disable_log: bool = False,
        run_mode: RunMode = RunMode.Train,
        random_seed: int = 42,
        **kwargs
):
    index = {}
    config.emulator.model.tim_dataset_model_options.use_index = False
    model = TimDatasetModel(base_station_names, resource_names, em_config, random_state, log,
                            disk_data_folder, disable_log, run_mode, random_seed, use_pool_node=False, **kwargs)
    config = config.run
    config.initial_date = Step.from_str('4w 0W')
    config.step_size = 3600
    config.stop_step = None
    config.stop_date = Step.from_str('4w 2M')
    time_step: TimeStep = time_step_factory_get(
        run_code=f'test-model-{model.run_mode}-{str(uuid4())}',
        step_per_second=config.step_per_second,
        step_size=config.step_size,
        stop_step=config.stop_step,
        stop_date=config.stop_date,
        initial_date=config.initial_date,
        logger=log
    )
    data = model.generate(time_step.current_step)
    index[time_step.current_step.to_str(full_key=True, no_total_step=True)] = data.to_dict()['data']
    for step in time_step:
        data = model.generate(step)
        index[step.to_str(full_key=True, no_total_step=True)] = data.to_dict()['data']
        if (step.total_steps // 3600) % 168 == 0:
            print(f'Generated data for step: {step}')
    save_path = get_absolute_path(save_path)
    create_directory(save_path)
    full_path = os.path.join(save_path, f'{len(model.bs_ids_no_pool)}-nodes-data-index.json')
    with open(full_path, 'w') as f:
        json.dump(index, f, cls=NumpyEncoder)


def tim_model_index_on_mongo(
        index_path: str,
        mongo: MongoRunWrapper,
        collection_name
):
    index_path = get_absolute_path(index_path)
    assert os.path.exists(index_path), f'Index path not exists: "{index_path}"'

    with open(index_path, 'r') as f:
        index_data = json.load(f)

    documents = []
    count = 1
    total = len(index_data)
    for step_str, step_data in index_data.items():
        step = Step.from_str(step_str)
        step_idx = get_step_idx(step)
        step_dict = step.to_dict()
        if 'total_steps' in step_dict:
            del step_dict['total_steps']
        document = {
            'step': step_str,
            'hour_idx': step_idx,
        }
        document = {**document, **step_dict, 'data': step_data}
        documents.append(document)
        print_status(count, total, 'Preparing documents')
        count += 1
    print()
    mongo.bulk_save(collection=collection_name, documents=documents)
    print(f'Index saved correctly into: {collection_name}')


def get_dataset_df(
        config: SingleRunConfig,
        model_nodes: List[str],
        data_path: str,
        log: Logger,
        initial_step: Step = Step.from_str('4w 0W'),
        stop_step: Step = Step.from_str('4w 2M'),
        step_size: int = 3600
) -> Tuple[pd.DataFrame, dict]:
    resources_info = []
    for res_conf in config.environment.resources:
        resources_info.append(EnvResource(
            name=res_conf.name,
            bucket_size=res_conf.bucket_size,
            min_buckets=res_conf.min_resource_buckets_per_node,
            total_available=res_conf.total_available,
            allocated=res_conf.allocated,
            classes=res_conf.classes
        ))
    resource_names: List[str] = [res.name for res in resources_info]
    config.emulator.model.tim_dataset_model_options.full_data_path = data_path
    model = TimDatasetModel(
        base_station_names=model_nodes,
        resource_names=resource_names,
        em_config=config.emulator,
        random_state=np.random.RandomState(42),
        run_mode=RunMode.Train,
        log=log
    )
    config = config.run
    config.initial_date = initial_step
    config.step_size = step_size
    config.stop_step = None
    config.stop_date = stop_step
    time_step: TimeStep = time_step_factory_get(
        run_code=f'test-model-{time.time()}',
        step_per_second=config.step_per_second,
        step_size=config.step_size,
        stop_step=config.stop_step,
        stop_date=config.stop_date,
        initial_date=config.initial_date,
        logger=log
    )
    simulated_data = {'idx': [], 'demand': [], 'node': []}
    t = time_step.current_step
    steps = 0
    while not time_step.is_last:
        data = model.generate(t)
        idx = get_step_idx(t)
        total = 0
        for bs_data in data.to_dict()['data']['res_1']:
            bs_name = list(bs_data.keys())[0]
            demand = list(bs_data.values())[0]
            total += demand
            simulated_data['idx'].append(idx)
            simulated_data['demand'].append(demand)
            simulated_data['node'].append(bs_name)
        simulated_data['idx'].append(idx)
        simulated_data['demand'].append(total)
        simulated_data['node'].append('total')
        t = time_step.next()
        steps += 1
        if steps % 60 == 0:
            print(f'Executed step {steps}', end='\r')
    print(f'Steps performed {steps}')
    df = pd.DataFrame(simulated_data)
    return df, simulated_data


def _get_unit_capacity(
        data_frame: pd.DataFrame,
        current_ratio: float,
        total_sub_bs: float,
        df_percentile: int = 1
) -> float:
    only_total = data_frame[data_frame['node'] == 'total']['demand'].sort_values(ascending=False)
    reversed_percentile = 1-df_percentile
    items_to_select = int(np.floor(len(only_total) * reversed_percentile))
    if items_to_select == 0:
        items_to_select = 1
    avg_max_demand = only_total[0: items_to_select].mean()
    total_capacity = avg_max_demand * (1/current_ratio)
    per_sub_bs_capacity = total_capacity // total_sub_bs
    return per_sub_bs_capacity


def _get_n_resource_units(total: int, n: int) -> int:
    module = total % n
    if module == 0:
        return total
    if module >= (n/2):
        add = n - module
        return total + add
    else:
        return total - module


def _get_max_next_hour_difference(
        nodes: List[int],
        df: pd.DataFrame,
) -> float:
    max_difference = None
    max_difference_node = None
    df['node'] = df['node'].astype(str)
    for node in nodes:
        small_df = df[df['node'].isin([str(node)])].reset_index()
        for i in small_df.index:
            if (i+1) < len(small_df.index):
                current_demand = small_df.iloc[i]['demand']
                next_demand = small_df.iloc[i+1]['demand']
                diff = next_demand - current_demand
                if max_difference is None or diff > max_difference:
                    max_difference = diff
                    max_difference_node = str(node)
    return max_difference


def generate_magic_numbers(
        sub_area_files: List[str],
        sub_area_units_capacity_multipliers: Dict[str, int],
        sub_area_base_folder: str,
        config: SingleRunConfig,
        ratios: List[float],
        bs_data_path: str,
        log: Logger,
        initial_step: Step = Step.from_str('4w 0W'),
        stop_step: Step = Step.from_str('4w 2M'),
        percentile: int = 1,
        units_multiplier: int = 1,
        step_size: int = 3600

):
    bs_data_df = pd.read_csv(bs_data_path)
    sub_area_info = {}
    for sub_area_file in sub_area_files:
        full_path = os.path.join(sub_area_base_folder, sub_area_file)
        with open(full_path, 'r') as f:
            sub_area_nodes = json.load(f)['aggregated_bs_id']
        sub_area_info[sub_area_file] = {
            'nodes': sub_area_nodes,
            'n_nodes': len(sub_area_nodes),
            'unit_capacity_multiplier': sub_area_units_capacity_multipliers[sub_area_file]
        }
    for sub_area, sub_a_info in sub_area_info.items():
        nodes = sub_a_info['nodes']
        full_data_path = os.path.join(ROOT_DIR, f'data/models/tim_dataset/{sub_a_info["n_nodes"]}-nodes/full_data.csv')
        index_data_path = os.path.join(ROOT_DIR, f'data/models/tim_dataset/indexes/{sub_a_info["n_nodes"]}-nodes-data-index.json')
        filtered_bs_data_df = bs_data_df[bs_data_df['aggregated_bs_id'].isin(nodes)]
        config.emulator.model.tim_dataset_model_options.full_data_path = full_data_path
        config.emulator.model.tim_dataset_model_options.index_data_path = index_data_path

        print()
        df, simulated_data = get_dataset_df(config, nodes, full_data_path, log, initial_step, stop_step, step_size)

        total_sub_bs = filtered_bs_data_df['n_base_stations'].sum() * units_multiplier
        total_sub_bs = _get_n_resource_units(total_sub_bs, sub_a_info['n_nodes'])
        total_sub_bs = total_sub_bs / sub_a_info['unit_capacity_multiplier']
        total_sub_bs = _get_n_resource_units(total_sub_bs, sub_a_info['n_nodes'])
        nodes_units = total_sub_bs // sub_a_info['n_nodes']
        total_demand = df['demand'].max().item()
        df_internet_max = df[df['node'] != 'total'].groupby(['node'])['demand'].idxmax()
        peaks_df = df.loc[df_internet_max.values]
        max_peak = peaks_df['demand'].max()
        capacities = []
        for ratio in ratios:
            unit_capacity = _get_unit_capacity(df, ratio, total_sub_bs, percentile)
            # unit_capacity = unit_capacity * sub_a_info['unit_capacity_multiplier']
            # total_sub_bs = _get_n_resource_units(total_sub_bs / sub_a_info['unit_capacity_multiplier'],
            #                                      sub_a_info['n_nodes'])
            total_capacity = unit_capacity * total_sub_bs
            # nodes_units = total_sub_bs // sub_a_info['n_nodes']
            node_capacity = nodes_units * unit_capacity
            max_peak_to_base_capacity_diff = max_peak - node_capacity
            missing_units = int(np.ceil(max_peak_to_base_capacity_diff / unit_capacity))
            actions_from_base_capacity = [i+1 for i in range(missing_units)]
            total_demand_units = int(np.ceil(max_peak / unit_capacity))
            actions_from_zero_capacity = [i+1 for i in range(total_demand_units)]
            actions_from_node_capacity = [i+1 for i in range(int(nodes_units.item()))]
            max_next_hour_difference = _get_max_next_hour_difference(nodes, df[df['node'] != 'total'])
            max_next_hour_difference_value = np.ceil(max_next_hour_difference / unit_capacity)
            actions_from_max_difference = [i+1 for i in range(int(max_next_hour_difference_value.item()))]
            capacities.append({
                'ratio': ratio,
                'unit_capacity': unit_capacity,
                'total_capacity': total_capacity,
                'total_demand': total_demand,
                'nodes_units': nodes_units,
                'nodes_capacity': node_capacity,
                'actions_from_base_capacity': actions_from_base_capacity,
                'actions_from_zero_capacity': actions_from_zero_capacity,
                'actions_from_node_capacity': actions_from_node_capacity,
                'actions_from_max_difference': actions_from_max_difference
            })
        magic_config = {
            'aggregated_bs_id': nodes,
            'total_bs_units': total_sub_bs,
            'ratios_capacities': capacities,
            'datasets': {
                'training': 24*7*6,  # 6 weeks,
                'validation': 24*7,  # 1 week
                'evaluation': 24*7,  # 1 week
            }
        }

        filename = f'run_magic_numbers-{len(nodes)}_nodes-new.json'
        with open(os.path.join(ROOT_DIR, f'data/models/tim_dataset/{filename}'), 'w') as file:
            json.dump(magic_config, file, cls=NumpyEncoder, indent=2)
        print(f'Generated magic numbers for sub area {sub_area}')
    print()


def generate_sub_dataset(
        nodes: List[int],
        chunks_folder: str = os.path.join(ROOT_DIR, 'data/models/tim_dataset/chunks-full-LTE'),
        output_folder: str = os.path.join(ROOT_DIR, 'data/models/tim_dataset')
) -> Tuple[str, str]:
    filename = os.path.join(output_folder, f'{len(nodes)}-nodes/full_data.csv')
    files = sorted(glob(f'{chunks_folder}/*.csv'), key=lambda x: x.split('/')[-1])
    create_directory_from_filepath(filename)

    day_count = 4
    week_count = 0
    month_count = 0
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        df = df[df['aggregated_bs_id'].isin(nodes)]
        df['week'] = week_count
        df['month'] = month_count
        if i == 0:
            header = True
        else:
            header = False
        df.to_csv(filename, index=False, header=header, mode='a')
        day_count += 1
        if day_count % 7 == 0:
            day_count = 0
            week_count += 1
            if week_count % 4 == 0:
                week_count = 0
                month_count += 1
        print_status(i + 1, len(files), 'Processing chunks')
    return filename, os.path.join(output_folder, f'{len(nodes)}-nodes')


def split_sub_dataset(
        full_data_path: str,
        output_path: str
):
    full_df = pd.read_csv(full_data_path)
    n_nodes = len(full_df.groupby(['aggregated_bs_id'], as_index=False)['aggregated_bs_id'].count())
    week_rows = 24 * 7 * n_nodes
    train_df = full_df.copy()

    val_df = pd.concat([
        full_df[
            (full_df['weekday'].isin([4, 5, 6])) &
            (full_df['week'] == 0) &
            (full_df['month'] == 0)
            ],
        full_df[
            (full_df['weekday'].isin([0, 1, 2, 3])) &
            (full_df['week'] == 1) &
            (full_df['month'] == 0)
            ]
    ], ignore_index=True)
    train_df.drop(train_df[
                      (train_df['weekday'].isin([4, 5, 6])) &
                      (train_df['week'] == 0) &
                      (train_df['month'] == 0)
                      ].index, inplace=True)
    train_df.drop(train_df[
                      (train_df['weekday'].isin([0, 1, 2, 3])) &
                      (train_df['week'] == 1) &
                      (train_df['month'] == 0)
                      ].index, inplace=True)
    eval_df = pd.concat([
        full_df[
            (full_df['weekday'].isin([4, 5, 6])) &
            (full_df['week'] == 1) &
            (full_df['month'] == 0)
            ],
        full_df[
            (full_df['weekday'].isin([0, 1, 2, 3])) &
            (full_df['week'] == 2) &
            (full_df['month'] == 0)
            ]
    ], ignore_index=True)
    train_df.drop(train_df[
                      (train_df['weekday'].isin([4, 5, 6])) &
                      (train_df['week'] == 1) &
                      (train_df['month'] == 0)
                      ].index, inplace=True)
    train_df.drop(train_df[
                      (train_df['weekday'].isin([0, 1, 2, 3])) &
                      (train_df['week'] == 2) &
                      (train_df['month'] == 0)
                      ].index, inplace=True)
    train_df.drop(train_df[
                      (train_df['weekday'].isin([4, 5, 6])) &
                      (train_df['week'] == 0) &
                      (train_df['month'] == 2)
                      ].index, inplace=True)
    train_df.drop(train_df[
                      (train_df['weekday'].isin([0, 1, 2, 3])) &
                      (train_df['week'] == 1) &
                      (train_df['month'] == 2)
                      ].index, inplace=True)

    print(val_df.info())
    print(eval_df.info())
    print(train_df.info())

    val_df.to_csv(os.path.join(output_path, f'{RunMode.Validation.value}-data.csv'), header=True, index=False)
    eval_df.to_csv(os.path.join(output_path, f'{RunMode.Eval.value}-data.csv'), header=True, index=False)
    train_df.to_csv(os.path.join(output_path, f'{RunMode.Train.value}-data.csv'), header=True, index=False)


def sub_dataset_pipeline(
        config: SingleRunConfig,
        logger: Logger,
        sub_area_file_path: str,
        chunks_folder: str = os.path.join(ROOT_DIR, 'data/models/tim_dataset/chunks-full-LTE'),
        output_folder: str = os.path.join(ROOT_DIR, 'data/models/tim_dataset'),
        ratios: List[float] = [0.6, 0.8, 0.9, 0.95, 1, 1.2],
        bs_data_path: str = os.path.join(ROOT_DIR, 'data/models/tim_dataset/aggregated_bs_data-LTE.csv'),
        initial_step: Step = Step.from_str('4w 0W'),
        stop_step: Step = Step.from_str('4w 2M'),
        percentile: int = 1,
        units_multiplier: int = 1,
        step_size: int = 3600
):
    with open(sub_area_file_path, 'r') as f:
        nodes = json.load(f)['aggregated_bs_id']
    full_data_path, output_path = generate_sub_dataset(nodes, chunks_folder, output_folder)
    split_sub_dataset(full_data_path, output_path)
    config.emulator.model.tim_dataset_model_options.full_data_path = full_data_path.replace(get_data_base_dir(), '')[1:]
    resources_info = []
    for res_conf in config.environment.resources:
        resources_info.append(EnvResource(
            name=res_conf.name,
            bucket_size=res_conf.bucket_size,
            min_buckets=res_conf.min_resource_buckets_per_node,
            total_available=res_conf.total_available,
            allocated=res_conf.allocated,
            classes=res_conf.classes
        ))
    resource_names: List[str] = [res.name for res in resources_info]
    generate_tim_model_index(
        config,
        save_path=os.path.join(output_folder, 'indexes'),
        base_station_names=[str(n) for n in nodes],
        resource_names=resource_names,
        em_config=config.emulator,
        random_state=np.random.RandomState(42),
        run_mode=RunMode.Train,
        log=logger
    )
    sub_area_file = sub_area_file_path.split('/')[-1]
    sub_area_units_capacity_multipliers = {sub_area_file: 1}
    sub_area_base_folder = os.path.join(*sub_area_file_path.split('/')[:-1])
    generate_magic_numbers(
        sub_area_files=[sub_area_file],
        sub_area_units_capacity_multipliers=sub_area_units_capacity_multipliers,
        sub_area_base_folder=sub_area_base_folder,
        config=config,
        ratios=ratios,
        bs_data_path=bs_data_path,
        log=logger,
        initial_step=initial_step,
        stop_step=stop_step,
        percentile=percentile,
        units_multiplier=units_multiplier,
        step_size=step_size
    )
