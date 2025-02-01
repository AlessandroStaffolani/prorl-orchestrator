import json
import os
from glob import glob
from logging import Logger
from typing import List, Dict, Any
from uuid import uuid4

from numpy.random import RandomState

from prorl import SingleRunConfig
from prorl.common.config import ExportMode
from prorl.common.filesystem import create_directory
from prorl.common.print_utils import print_status
from prorl.core.timestep import TimeStep, time_step_factory_get, time_step_factory_reset
from prorl.emulator.models import create_model_from_type
from prorl.emulator.models.abstract import AbstractModel


def get_model_metadata(config: SingleRunConfig, model_name: str) -> Dict[str, Any]:
    metadata = {
        'model_name': model_name,
        'model': config.emulator.model.nodes_demand_model_options.export(mode=ExportMode.DICT),
        'model_disk': config.emulator.model.model_disk.export(mode=ExportMode.DICT),
        'base_station_name_mappings': config.emulator.model.base_station_name_mappings,
        'nodes': config.environment.nodes.export(mode=ExportMode.DICT),
        'episode_lengths': {
            'training': config.environment.reset_options.time_interval,
            'validation': config.run.validation_run.rollout_batch_size,
            'evaluation': config.run.evaluation_episode_length,
        },
        'resource_classes': [
            res.export(mode=ExportMode.DICT) for res in config.environment.resources
        ],
        'step_per_second': config.run.step_per_second,
        'step_size': config.run.step_size,
        'stack_n_states': config.environment.state.stack_n_states
    }

    return metadata


def metadata_are_equals(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a == b


def generate_episodes(random_state: RandomState, model_uuid: str, time_step_config: dict,
                      length: int, n_episodes: int, config: SingleRunConfig, log: Logger) -> List[Dict[str, Any]]:
    n_nodes = config.environment.nodes.get_n_nodes()
    base_stations = config.emulator.model.base_station_names()
    if n_nodes <= len(base_stations):
        base_station_names: List[str] = config.emulator.model.base_station_names()[0: config.environment.nodes.n_nodes]
    else:
        base_station_names: List[str] = []
        for i in range(n_nodes):
            base_station_names.append(base_stations[i % len(base_stations)])
    resource_names: List[str] = [res.name for res in config.environment.resources]
    model: AbstractModel = create_model_from_type(
        model_type=config.emulator.model.type,
        base_station_names=base_station_names,
        resource_names=resource_names,
        em_config=config.emulator,
        random_state=random_state,
        log=log,
        disable_log=True
    )
    stack_n_states = config.environment.state.stack_n_states
    step_size = config.run.step_size
    total_steps = length + (step_size * 2)
    if stack_n_states > 1:
        total_steps += step_size
    total_steps *= stack_n_states
    for i in range(n_episodes):
        episode_data = []
        time_step: TimeStep = time_step_factory_reset(model_uuid, recreate=True, **time_step_config)
        steps = 0
        while steps < total_steps:
            data = model.generate(time_step.current_step)
            episode_data.append(data.to_dict())
            time_step.next()
            steps += 1
        yield episode_data


def generate_disk_data(model_metadata: Dict[str, Any],
                       config: SingleRunConfig, base_folder: str, log: Logger) -> str:
    model_uuid = str(uuid4())
    folder_path = os.path.join(base_folder, model_uuid)
    create_directory(folder_path)
    # save the metadata
    with open(os.path.join(folder_path, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, sort_keys=False, indent=2)
    seeds = config.emulator.model.model_disk.seeds
    episodes_to_generate = config.emulator.model.model_disk.episodes_to_generate
    episodes_length = model_metadata['episode_lengths']
    time_step_config = {
        'stop_step': -1,
        'step_per_second': model_metadata['step_per_second'],
        'step_size': model_metadata['step_size']
    }
    # generate training data
    time_step_factory_get(run_code=model_uuid, **time_step_config)
    for i, seed in enumerate(seeds['training']):
        random_state = RandomState(seed)
        time_step: TimeStep = time_step_factory_reset(model_uuid, recreate=True, **time_step_config)
        episodes = generate_episodes(random_state=random_state, model_uuid=model_uuid,
                                     time_step_config=time_step_config,
                                     length=episodes_length['training'],
                                     n_episodes=episodes_to_generate['training'],
                                     config=config, log=log)
        for j, episode in enumerate(episodes):
            with open(os.path.join(folder_path, f'training-{seed}-{j}.json'), 'w') as f:
                json.dump(episode, f)
            print_status((i+1) * (j+1), len(seeds['training']) * episodes_to_generate['training'],
                         'Generating training data', 40)
    print()
    # generate validation data
    for i, seed in enumerate(seeds['validation']):
        random_state = RandomState(seed)
        time_step: TimeStep = time_step_factory_reset(model_uuid, recreate=True, **time_step_config)
        episodes = generate_episodes(random_state=random_state, model_uuid=model_uuid,
                                     time_step_config=time_step_config,
                                     length=episodes_length['validation'],
                                     n_episodes=episodes_to_generate['validation'],
                                     config=config, log=log)
        for j, episode in enumerate(episodes):
            with open(os.path.join(folder_path, f'validation-{seed}-{j}.json'), 'w') as f:
                json.dump(episode, f)
            print_status((i+1) * (j+1), len(seeds['validation']) * episodes_to_generate['validation'],
                         'Generating validation data', 40)
    print()
    for i, seed in enumerate(seeds['evaluation']):
        random_state = RandomState(seed)
        time_step: TimeStep = time_step_factory_reset(model_uuid, recreate=True, **time_step_config)
        episodes = generate_episodes(random_state=random_state, model_uuid=model_uuid,
                                     time_step_config=time_step_config,
                                     length=episodes_length['evaluation'],
                                     n_episodes=episodes_to_generate['evaluation'],
                                     config=config, log=log)
        for j, episode in enumerate(episodes):
            with open(os.path.join(folder_path, f'evaluation-{seed}-{j}.json'), 'w') as f:
                json.dump(episode, f)
            print_status((i+1) * (j+1), len(seeds['evaluation']) * episodes_to_generate['evaluation'],
                         'Generating evaluation data', 40)
    print()
    return folder_path


def manage_disk_data(config: SingleRunConfig, model_name: str, log: Logger) -> str:
    should_generate = False
    n_nodes = config.environment.nodes.get_n_nodes()
    base_folder = config.emulator.model.model_disk.get_base_folder()
    base_folder = os.path.join(base_folder, model_name, f'{n_nodes}-nodes')
    model_metadata = get_model_metadata(config, model_name)
    data_uuid = None
    if not os.path.exists(base_folder):
        # if the folder is empty we have to generate the data for sure
        should_generate = True
        create_directory(base_folder)
    else:
        meta_files = glob(f'{base_folder}/*/metadata.json')
        if len(meta_files) == 0:
            # no meta, we need to generate data
            should_generate = True
        else:
            equal_file_path = None
            for meta_file_path in meta_files:
                with open(meta_file_path, 'r') as f:
                    metadata = json.load(f)
                if metadata_are_equals(model_metadata, metadata):
                    equal_file_path = meta_file_path
                    break

            if equal_file_path is None:
                # not found compatible meta
                should_generate = True
            else:
                should_generate = False
                data_uuid = equal_file_path.split('/')[-2]

    # if should_generate, we generate the data
    if should_generate:
        data_folder = generate_disk_data(model_metadata, config, base_folder, log)
    else:
        print('Skipping data generation, model already into the disk')
        data_folder = os.path.join(base_folder, data_uuid)
    print(f'Using data folder: "{data_folder}"')
    return data_folder
