import os
from logging import Logger
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import torch

from prorl import SingleRunConfig, ROOT_DIR
from prorl.common.data_structure import RunMode
from prorl.common.object_handler import get_save_folder_from_config, ObjectHandler, MinioObjectHandler
from prorl.common.stats_tracker import Tracker
from prorl.common.tensorboard_wrapper import TensorboardWrapper
from prorl.core.state import State
from prorl.core.step import Step
from prorl.core.step_data import StepData
from prorl.environment.action_space import Action
from prorl.environment.agent.abstract import AgentAbstract
from prorl.environment.node import Node


class RunStatsManager:

    def __init__(
            self,
            run_code: str,
            config: SingleRunConfig,
            nodes: List[Node],
            agent_name: str,
            total_steps: int,
            step_size: int,
            object_handler: Union[ObjectHandler, MinioObjectHandler],
            logger: Logger,
            val_run_code: Optional[str] = None,
            save_folder: Optional[str] = None
    ):
        self.config: SingleRunConfig = config
        self.logger: Logger = logger
        self.run_code = run_code
        self.run_mode: RunMode = self.config.run.run_mode
        self.val_run_code: Optional[str] = val_run_code
        self.nodes: List[Node] = nodes
        self.stats_tracker: Optional[Tracker] = None
        self.object_handler: Union[ObjectHandler, MinioObjectHandler] = object_handler
        self.agent_name: str = agent_name
        self.total_steps: int = total_steps
        self.step_size: int = step_size
        self.cumulative_problem_solved: int = 0
        self.save_folder: Optional[str] = save_folder
        self.tensorboard: Optional[TensorboardWrapper] = None
        self.average_reward: Dict[str, List[float]] = {
            'add_60': [],
            'move_60': [],
        }
        self._init()

    def _init(self):
        if self.val_run_code is None:
            self._init_saver()

    def _init_saver_single_run(self):
        final_folder = get_save_folder_from_config(
            save_name=self.config.saver.save_name,
            save_prefix=self.config.saver.save_prefix,
            save_name_with_date=self.config.saver.save_name_with_date,
            save_name_with_uuid=self.config.saver.save_name_with_uuid,
            uuid=self.run_code
        )
        model_type_value = self.config.emulator.model.type.value
        self.save_folder: str = os.path.join(model_type_value,
                                             self.run_mode.value,
                                             self.agent_name, final_folder)

    def _init_saver_multi_run(self):
        multi_run_code = self.config.multi_run.multi_run_code
        if self.run_mode == RunMode.Train or self.run_mode == RunMode.Validation:
            final_folder = '_'.join([param.filename_key_val for param in self.config.multi_run.multi_run_params])
        elif self.run_mode == RunMode.Eval:
            final_folder = f'seed={self.config.random_seeds.evaluation[0]}_'
            final_folder += '_'.join([
                param.filename_key_val for param in self.config.multi_run.multi_run_params
                if 'seed' not in param.filename_key_val
            ])
            if multi_run_code.endswith('-evaluation'):
                multi_run_code = multi_run_code.replace('-evaluation', '')
        else:
            final_folder = ''
        model_type_value = self.config.emulator.model.type.value
        self.save_folder: str = os.path.join(
            multi_run_code,
            model_type_value,
            self.run_mode.value,
            self.agent_name,
            final_folder
        )

    def _init_saver(self):
        if self.config.multi_run.is_multi_run:
            self._init_saver_multi_run()
        else:
            self._init_saver_single_run()
        if self.config.saver.tensorboard.enabled:
            tensorboard_path = os.path.join(
                ROOT_DIR,
                self.config.saver.tensorboard.get_save_path(),
                self.save_folder
            )
            self.tensorboard: TensorboardWrapper = TensorboardWrapper(log_dir=tensorboard_path, logger=self.logger)
        if self.config.saver.stats_condensed:
            self.stats_tracker: Tracker = Tracker.init_condensed_tracker(
                run_code=self.run_code, config=self.config, run_mode=self.run_mode, tensorboard=self.tensorboard
            )
        else:
            self.stats_tracker: Tracker = Tracker.init_tracker(
                nodes=self.nodes,
                run_code=self.run_code, config=self.config, run_mode=self.run_mode, tensorboard=self.tensorboard
            )

    def step_stats(
            self,
            t: Step,
            action: Action,
            reward: Tuple[float, float],
            nodes_satisfied: int,
            resource_name: str,
            current_demand: StepData,
            choose_action_time: float = None,
            learn_time: float = None,
            penalty: Optional[Union[int, float]] = None
    ):
        nodes_resources: List[float] = [n.get_current_allocated(resource_name) for n in self.nodes]
        current_demand: Dict[str, Dict[str, List[Dict[str, float]]]] = current_demand.to_dict()
        nodes_demand: List[float] = []
        for node_data in current_demand['data'][resource_name]:
            nodes_demand.append(list(node_data.values())[0])

        step = t.total_steps // self.config.run.step_size

        self.stats_tracker.track('reward/add_node', reward[0], step)
        self.stats_tracker.track('reward/movement', reward[1], step)
        self.average_reward['add_60'].append(reward[0])
        self.average_reward['move_60'].append(reward[1])
        if step % 60 == 0:
            self.stats_tracker.track('reward/add_node/60-steps-avg',
                                     float(np.mean(self.average_reward['add_60'])), step)
            self.stats_tracker.track('reward/movement/60-steps-avg',
                                     float(np.mean(self.average_reward['move_60'])), step)
            self.average_reward['add_60'] = []
            self.average_reward['move_60'] = []
        if penalty is not None:
            self.stats_tracker.track('reward/penalties', penalty, step)
        self.stats_tracker.track('action_history/add_node', action.add_node, step)
        self.stats_tracker.track('action_history/remove_node', action.remove_node, step)
        self.stats_tracker.track('action_history/resource_class', action.resource_class, step)
        self.stats_tracker.track('action_history/quantity', action.quantity, step)
        self.stats_tracker.track('problem_solved/nodes_satisfied', nodes_satisfied, step)
        if nodes_satisfied == len(nodes_resources):
            self.stats_tracker.track('problem_solved/count', 1, step)
            self.cumulative_problem_solved += 1
        else:
            self.stats_tracker.track('problem_solved/count', 0, step)
        if choose_action_time is not None:
            self.stats_tracker.track('times/choose_action', choose_action_time, step)
        if learn_time is not None:
            self.stats_tracker.track('times/learn_step', learn_time, step)
        if not self.config.saver.stats_condensed and self.run_mode == RunMode.Train:
            for i, node in enumerate(self.nodes):
                self.stats_tracker.track(f'nodes/n_{i}/resource_history', nodes_resources[i], step)
                self.stats_tracker.track(f'nodes/n_{i}/demand_history', nodes_demand[i], step)
        if t.to_second() % self.config.run.info_frequency == 0:
            self.log_run_status(t)
        elif t.to_second() % self.config.run.debug_frequency == 0:
            self.log_run_status(t)

    def save_stats(self, agent: Optional[AgentAbstract] = None):
        if self.val_run_code is None:
            if self.config.saver.enabled:
                if agent is not None:
                    self.logger.debug('Starting stats and agent saving procedure')
                data_to_save = self.stats_tracker.disk_stats()
                filename = 'full_data.json'
                message = f'Saved all the run stats into {os.path.join(self.save_folder, filename)}'
                self.object_handler.save(
                    obj=data_to_save,
                    path=self.save_folder,
                    filename=filename,
                    max_retries=3,
                    wait_timeout=2,
                    use_progress=True
                )
                self.logger.info(message)
            if self.config.saver.save_agent and agent is not None:
                self.save_agent(agent)

    def save_agent(self, agent: AgentAbstract, filename: Optional[str] = None):
        if self.config.saver.enabled and agent.save_agent_state:
            agent_state = agent.get_agent_state()
            agent_filename = 'agent_state.pth' if filename is None else filename
            self.object_handler.save_agent_model(
                agent_model=agent_state,
                filename=agent_filename,
                path=self.save_folder,
                max_retries=3,
                wait_timeout=2,
                use_progress=True
            )

    def add_agent_net_graph(self, agent: AgentAbstract, state: State):
        if self.tensorboard is not None and self.config.saver.tensorboard.save_model_graph \
                and self.val_run_code is None:
            agent_model = agent.get_model()
            if agent_model is not None:
                test_state = [
                    state.tolist(),
                    state.tolist(),
                    state.tolist(),
                    state.tolist(),
                ]
                self.tensorboard.add_graph(agent_model, input_to_model=torch.tensor(test_state, dtype=torch.float))

    def save_validation_run_stats(self):
        if self.val_run_code is not None:
            save_index = self.val_run_code.split('_')[-1]
            if self.config.saver.enabled:
                filename = 'validation_run_data.json'
                self.object_handler.save(
                    obj=self.stats_tracker.disk_stats(),
                    path=os.path.join(self.save_folder, f'validation_run-{save_index}'),
                    filename=filename
                )

    def save_validation_run_agent(self, agent: AgentAbstract, index=None):
        if self.val_run_code is not None or index is not None:
            if index is not None:
                save_index = index
            else:
                save_index = self.val_run_code.split('_')[-1]
            if self.config.saver.enabled:
                agent_state = agent.get_agent_state()
                filename = 'validation_run_agent_state.pth'
                self.object_handler.save_agent_model(
                    agent_model=agent_state,
                    path=os.path.join(self.save_folder, f'validation_run-{save_index}'),
                    filename=filename,
                )

    def log_run_status(self, t: Step):
        add_node_reward = self.stats_tracker.get('reward/add_node', only_cumulative=True)
        movement_reward = self.stats_tracker.get('reward/movement', only_cumulative=True)
        total_cost = self.stats_tracker.get('movement_cost', only_cumulative=True)
        problems_solved_count = self.stats_tracker.get('problem_solved/count')

        total_steps = self.total_steps // self.config.run.step_size
        steps_completed = t.total_steps // self.config.run.step_size
        current_step = str(t) if t.total_steps > 0 else '0'

        message = f'Steps completed {steps_completed}/{total_steps} | problem solved count: {problems_solved_count} ' \
                  f'| add node total reward: {add_node_reward} | movement total reward: {movement_reward} ' \
                  f'| total cost: {total_cost} | current step: {current_step} | run mode: {self.run_mode.value}'

        self.logger.info(message)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
