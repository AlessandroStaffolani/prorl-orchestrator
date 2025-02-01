from logging import Logger
from typing import Optional, Union

from numpy.random import RandomState

from prorl.emulator import emulator_config, EmulatorConfig
from prorl.emulator.data_structure import ModelTypes
from prorl.common.data_structure import RunMode
from prorl.emulator.models.auto_configurable_model import AutoConfigurableModel
from prorl.emulator.models.nodes_demand_model import NodesDemandModel
from prorl.emulator.models.synthetic_model import SyntheticModel
from prorl.emulator.models.tim_dataset_model import TimDatasetModel
from prorl.emulator.models.test_model import TestModel

MAPPING = {
    ModelTypes.AutoConfigurableModel: AutoConfigurableModel,
    ModelTypes.NodesDemandModel: NodesDemandModel,
    ModelTypes.TimDatasetModel: TimDatasetModel,
    ModelTypes.TestModel: TestModel,
    ModelTypes.SyntheticModel: SyntheticModel,
}


def get_value(key, kwargs, default):
    if key in kwargs:
        return kwargs[key]
    else:
        return default


def create_model_from_type(
        model_type: ModelTypes = None,
        em_config: Optional[EmulatorConfig] = None,
        log: Optional[Logger] = None,
        random_state: Optional[RandomState] = None,
        disk_data_folder: Optional[str] = None,
        disable_log: bool = False,
        run_mode: RunMode = RunMode.Train,
        random_seed: int = 42,
        use_pool_node: bool = False,
        **kwargs
) -> Union[AutoConfigurableModel, NodesDemandModel]:
    config: EmulatorConfig = em_config if em_config is not None else emulator_config
    model_name = config.model.type
    if model_type is not None:
        model_name = model_type
    base_station_names = get_value('base_station_names', kwargs, config.model.base_station_names())
    resource_names = get_value('resource_names', kwargs, config.model.resource_names())

    if model_name in MAPPING:
        model_class = MAPPING[model_name]
        return model_class(
            base_station_names=base_station_names,
            resource_names=resource_names,
            em_config=config,
            random_state=random_state,
            log=log,
            disk_data_folder=disk_data_folder,
            disable_log=disable_log,
            run_mode=run_mode,
            random_seed=random_seed,
            use_pool_node=use_pool_node
        )
    else:
        raise AttributeError(f'ModelTypes {model_name} is not available')
