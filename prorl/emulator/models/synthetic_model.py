from dataclasses import dataclass
from logging import Logger
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
from numpy.random import RandomState

from prorl.common.data_structure import RunMode
from prorl.core.step import Step
from prorl.core.step_data import StepDataEntry, StepData
from prorl.emulator import EmulatorConfig
from prorl.emulator.config import CoupleConfig
from prorl.emulator.models.abstract import AbstractModel


@dataclass
class CoupleStatus:
    current_load: float
    config: CoupleConfig
    bs_1_name: str
    bs_1_index: int
    bs_2_name: str
    bs_2_index: int
    load_multiplier: float = 1.0
    is_stressing: bool = False
    stressing_steps: int = 0
    swapped: bool = False
    last_step_updated: Step = None

    def set_current_load(self, step: Step, step_size: int):
        diff = step.total_steps - self.last_step_updated.total_steps
        if diff != step_size:
            # we are in a step in the middle, we do not change anything
            return
        else:
            # we can change the load, so update last_step_update
            self.last_step_updated = step
        is_stress_step = False
        if self.is_stressing and self.stressing_steps <= self.config.keep_stress:
            is_stress_step = True
        elif isinstance(self.config.stress_every, int):
            stress_frequency = self.config.stress_every * step_size
            if step.total_steps > 0 and step.total_steps % stress_frequency == 0:
                is_stress_step = True
        else:
            dict_step = step.to_dict()
            match = True
            for unit_name, unit_value in self.config.stress_every.items():
                if isinstance(unit_value, list):
                    if dict_step[unit_name] not in unit_value:
                        match = False
                else:
                    if dict_step[unit_name] != unit_value:
                        match = False
            if match:
                is_stress_step = True

        if is_stress_step:
            self.current_load = self.config.stress_load
            self.stressing_steps += 1
            self.is_stressing = True
            if self.stressing_steps == 1:
                self.swapped = not self.swapped
        else:
            self.current_load = self.config.calm_load
            self.stressing_steps = 0
            self.is_stressing = False

    def get_high_node_demand_profile(self, absolute_demand: float) -> float:
        demand = absolute_demand * self.current_load
        return demand * self.load_multiplier

    def get_low_node_demand_profile(self, absolute_demand: float) -> float:
        low_node_load = 1 - self.current_load
        if not self.is_stressing and self.config.calm_load_equal:
            low_node_load = self.current_load
        demand = absolute_demand * low_node_load
        return demand * self.load_multiplier

    def is_swapped(self) -> bool:
        if self.config.swap_stress and self.swapped and self.is_stressing:
            return True
        else:
            return False

    def get_bs_1_data_entry(
            self,
            demand_profile: float,
            resource: str,
            step: Step,
            random_state: RandomState
    ) -> StepDataEntry:
        return StepDataEntry(
            resource=resource,
            value=max(0, random_state.normal(demand_profile, self.config.std)),
            base_station=self.bs_1_name,
            step=step,
        )

    def get_bs_2_data_entry(
            self,
            demand_profile: float,
            resource: str,
            step: Step,
            random_state: RandomState
    ) -> StepDataEntry:
        return StepDataEntry(
            resource=resource,
            value=max(0, random_state.normal(demand_profile, self.config.std)),
            base_station=self.bs_2_name,
            step=step,
        )


class SyntheticModel(AbstractModel):

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
        super(SyntheticModel, self).__init__(
            base_station_names=base_station_names,
            resource_names=resource_names,
            model_name='SyntheticModel',
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
        model_config = self.emulator_config.model.synthetic_model
        self.episode_length: int = model_config.episode_length
        self.change_distribution_frequency: int = model_config.change_distribution_frequency
        self.couples_config: List[CoupleConfig] = model_config.couples_config
        # every steps the means are multiplied by increase_multiplier
        self.distribution_multipliers: List[float] = model_config.distribution_multipliers
        self.model_step_size: int = model_config.model_step_size
        self.demand_absolute_value: float = model_config.demand_absolute_value

        self.couples_status: List[CoupleStatus] = []

        self.distribution_index = 0
        self.overall_steps = 0

        self._init_bs_status()

    def _init_bs_status(self):
        self.couples_status: List[CoupleStatus] = []
        counter = 0
        if self.use_pool_node:
            base_station_names = [bs_name for bs_name in self.base_station_names if bs_name != self.pool_bs_name]
        else:
            base_station_names = self.base_station_names
        for i in range(0, len(base_station_names), 2):
            config = self.couples_config[counter % len(self.couples_config)]
            status = CoupleStatus(
                current_load=config.calm_load,
                config=config,
                bs_1_name=base_station_names[i],
                bs_1_index=i,
                bs_2_name=base_station_names[i+1],
                bs_2_index=i+1,
                load_multiplier=self.distribution_multipliers[self.distribution_index],
                last_step_updated=Step.from_str('')
            )
            self.couples_status.append(status)
            counter += 1

    def generate_couple_data(self, step: Step, couple_index: int, resource: str) -> Tuple[StepDataEntry, StepDataEntry]:
        couple_status: CoupleStatus = self.couples_status[couple_index]
        couple_status.set_current_load(step, step_size=self.model_step_size)
        if couple_status.is_swapped():
            bs_1_demand = couple_status.get_low_node_demand_profile(self.demand_absolute_value)
            bs_2_demand = couple_status.get_high_node_demand_profile(self.demand_absolute_value)
        else:
            bs_1_demand = couple_status.get_high_node_demand_profile(self.demand_absolute_value)
            bs_2_demand = couple_status.get_low_node_demand_profile(self.demand_absolute_value)
        bs_1_data = couple_status.get_bs_1_data_entry(bs_1_demand, resource, step, self.random)
        bs_2_data = couple_status.get_bs_2_data_entry(bs_2_demand, resource, step, self.random)
        return bs_1_data, bs_2_data

    def _counter_increase(self, step: Step) -> int:
        increase = 0
        if step.total_steps % self.model_step_size == 0:
            increase = 1
        return increase

    def _generate_single_base_station(self, step: Step, resource_name: str, base_station_name: str) -> StepDataEntry:
        pass

    def _generate_step_data(self, step: Step, list_resources: List[str], list_base_stations: List[str]) -> StepData:
        data = StepData(generator_model=self.model_name())
        for i, resource in enumerate(list_resources):
            if self.use_pool_node:
                data.add_entry(self.pool_step_data_entry(resource, step))
            for j, couple_state in enumerate(self.couples_status):
                bs_1_data, bs_2_data = self.generate_couple_data(step, j, resource)
                data.add_entry(bs_1_data)
                data.add_entry(bs_2_data)

        if self.overall_steps > 0 and self.overall_steps % self.change_distribution_frequency == 0:
            self.distribution_index += 1
            current_multiplier = self.distribution_multipliers[
                self.distribution_index % len(self.distribution_multipliers)]
            for couple in self.couples_status:
                couple.load_multiplier = current_multiplier

        self.overall_steps += self._counter_increase(step)
        return data

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

        return self._generate_step_data(step, list_resources, list_base_stations)

    def reset(self, show_log=True, full_reset=True):
        super(SyntheticModel, self).reset(show_log)
        self.distribution_index = 0
        self.overall_steps = 0
        self._init_bs_status()

    def get_stop_step(self, step_size) -> Union[None, int, Step]:
        return self.episode_length * self.model_step_size * len(self.distribution_multipliers)
