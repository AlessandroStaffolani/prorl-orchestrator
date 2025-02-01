import itertools
import time
from datetime import timedelta
from logging import Logger
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Optional, Dict, Tuple, List, Generator

import numpy as np
from numpy.random import RandomState

from prorl import SingleRunConfig
from prorl.common.data_structure import RunMode
from prorl.common.stats_tracker import Tracker
from prorl.core.state import State
from prorl.core.step_data import StepData
from prorl.environment.action_space import ActionSpaceWrapper, ActionType
from prorl.environment.agent import AgentType
from prorl.environment.agent.baseline.random import BaselineAgent
from prorl.environment.node import Node
from prorl.environment.node_groups import NodeGroups
from prorl.environment.rollouts import sample_trajectory
from prorl.environment.state_builder import StateType
from prorl.environment.wrapper import EnvWrapper
from prorl.run.config import RunConfig

SolutionType = Tuple[Tuple[int, int], ...]


def beautify_seconds(seconds: float) -> str:
    return f"{str(timedelta(seconds=seconds))} (h:mm:ss.ms)"


class OracleAgent(BaselineAgent):

    def __init__(self,
                 action_space_wrapper: ActionSpaceWrapper,
                 random_state: np.random.RandomState,
                 action_spaces: Dict[ActionType, int],
                 state_spaces: Dict[StateType, int],
                 log: Logger,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(OracleAgent, self).__init__(action_space_wrapper,
                                          random_state=random_state,
                                          name=AgentType.Oracle,
                                          action_spaces=action_spaces,
                                          state_spaces=state_spaces,
                                          log=log,
                                          config=config,
                                          **kwargs)
        self.best_solution: Optional[SolutionType] = None
        self.best_score = None
        self.n_possible_solutions = 0
        self.log_intervals = 1
        self.solution_time_sum = 0
        self.n_processes = 1
        self.lock = Lock()
        self.use_pool = False
        self.run_code = None

        self.current_index = 0

    def get_current_add_action(self):
        return self.best_solution[self.current_index % len(self.best_solution)][0]

    def get_current_remove_action(self):
        return self.best_solution[self.current_index % len(self.best_solution)][1]

    def set_stats_tracker(self, tracker: Tracker):
        super().set_stats_tracker(tracker)
        self.run_code = self.stats_tracker.run_code

    def get_single_step_possible_actions(self, n_nodes) -> List[Tuple[int, int]]:
        all_actions = list(
            itertools.product(
                [i for i in range(n_nodes + 1)],
                [i for i in range(n_nodes + 1)],
            )
        )
        available_actions = []
        for add, remove in all_actions:
            if add != remove:
                available_actions.append((add, remove))
            else:
                if add == n_nodes and remove == n_nodes:
                    available_actions.append((add, remove))
        return available_actions

    def _get_all_possible_solutions(self, n_nodes: int, time_horizon: int) -> Generator[SolutionType, None, None]:
        actions_per_step: List[List[Tuple[int, int]]] = [
            self.get_single_step_possible_actions(n_nodes)
            for _ in range(time_horizon)
        ]
        return itertools.product(*actions_per_step)

    def _init_solution_env(self, index, seed, batch_size):
        mode = RunMode.Eval
        random_state = RandomState(seed)
        run_config = RunConfig(
            stop_date=None,
            stop_step=batch_size,
            bootstrapping_steps=0,
            rollout_batch_size=batch_size
        )
        model_options = self.config.emulator.model.tim_dataset_model_options
        run_config.step_size = self.config.run.step_size
        run_config.stop_step = None
        run_config.initial_date = model_options.time_step[mode]['start_date']
        run_config.stop_date = model_options.time_step[mode]['end_date']
        disk_data_path = None
        return EnvWrapper(
            base_stations_mappings=self.config.emulator.model.base_station_name_mappings,
            run_code=f'{self.run_code}_solution-{index}',
            auto_init=True,
            nodes_type_distribution=self.config.environment.nodes.nodes_type_distribution,
            state_features=self.config.environment.state.features,
            run_config=run_config,
            env_configuration=self.config.environment,
            emulator_configuration=self.config.emulator,
            random_state=random_state,
            log=self.logger,
            run_mode=mode,
            disk_data_path=disk_data_path,
            random_seed=seed,
            no_init_reset=True
        )

    def _init_sub_oracle_agent(self, solution: SolutionType):
        return OracleSubAgent(
            solution=solution,
            action_space_wrapper=self.action_space_wrapper,
            random_state=self.random,
            action_spaces=self.action_spaces,
            state_spaces=self.state_spaces,
            log=self.logger,
            config=self.config,
        )

    def simulate_one_solution(
            self,
            index: int,
            solution: SolutionType,
            resource: Optional[str] = None
    ):
        start = time.time()
        batch_size = len(solution)
        seed = self.config.random_seeds.evaluation[0]
        tracker = Tracker.init_condensed_tracker(
            run_code=f'{self.run_code}_solution-{index}', config=self.config, run_mode=RunMode.Eval, tensorboard=None
        )
        solution_env = self._init_solution_env(index, seed, batch_size)
        initial_state = solution_env.reset(show_log=False)
        solution_agent = self._init_sub_oracle_agent(solution)

        solution_agent.set_stats_tracker(tracker)
        solution_env.set_stats_tracker(tracker)

        _, _, _, _, _, _ = sample_trajectory(
            env=solution_env,
            initial_state=initial_state,
            agent=solution_agent,
            batch_size=batch_size,
            resource_name=resource,
            stats_tracker=tracker,
            config=self.config,
            iteration=0,
            mode=RunMode.Eval,
            seed=seed
        )

        score = tracker.get(f'{RunMode.Eval.value}-{seed}/reward/utility/total')[0]
        final_solution = tuple(solution_agent.final_solution)
        if self.use_pool:
            with self.lock:
                self.update_best_score(score, final_solution)
        else:
            self.update_best_score(score, final_solution)
        sol_time = time.time() - start
        self.solution_time_sum += sol_time
        self.log_status(index)

    def update_best_score(self, score: float, final_solution: SolutionType):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_solution = final_solution

    def get_number_of_possible_solutions(self, n: int, t: int) -> int:
        return ((n+1) * (n+1) - n)**t

    def set_log_intervals(self):
        if self.n_possible_solutions < 1000:
            n_logs = 10
        elif self.n_possible_solutions < 10000:
            n_logs = 20
        elif self.n_possible_solutions < 100000:
            n_logs = 50
        elif self.n_possible_solutions < 1000000:
            n_logs = 100
        else:
            n_logs = 1000
        self.n_processes = int(max(1., min(n_logs * 0.2, 8)))
        self.log_intervals = self.n_possible_solutions // n_logs

    def log_status(self, index):
        if index % self.log_intervals == 0:
            solution_time_avg = self.solution_time_sum / (index + 1)
            remaining_time = solution_time_avg * (self.n_possible_solutions - (index + 1))
            if self.use_pool:
                remaining_time /= self.n_processes
            self.logger.info(f'Oracle solutions processed {index + 1}/{self.n_possible_solutions} | '
                             f'Best score so far: {self.best_score} | '
                             f'Average solution time: {beautify_seconds(solution_time_avg)} | '
                             f'Estimated remaining time: {beautify_seconds(remaining_time)}')

    def compute_exact_solution(self, resource: Optional[str] = None):
        start = time.time()
        n_nodes = self.config.environment.nodes.n_nodes
        start_date = self.config.emulator.model.tim_dataset_model_options.time_step[RunMode.Eval]['start_date']
        end_date = self.config.emulator.model.tim_dataset_model_options.time_step[RunMode.Eval]['end_date']
        step_size = self.config.emulator.model.tim_dataset_model_options.step_size
        time_horizon = (end_date.total_steps - start_date.total_steps) // step_size
        self.n_possible_solutions = self.get_number_of_possible_solutions(n_nodes, time_horizon)
        self.logger.info('Oracle is trying to find optimal solution for the instance of the problem')
        self.logger.info(f'Problem parameters: number of node: {n_nodes} | time horizon: {time_horizon} | '
                         f'Possible solutions: {self.n_possible_solutions:_}')
        self.set_log_intervals()
        all_solutions = self._get_all_possible_solutions(n_nodes, time_horizon)
        if self.use_pool:
            self.logger.info(f'Oracle will use {self.n_processes} threads to find optimal solution')
            pool = ThreadPool(processes=self.n_processes)
        else:
            pool = None
        index = 0
        for solution in all_solutions:
            if self.use_pool:
                pool.apply_async(
                    func=self.simulate_one_solution,
                    args=(index, solution, resource)
                )
            else:
                self.simulate_one_solution(index, solution, resource)
            index += 1

        if self.use_pool:
            pool.close()
            pool.join()

        total_time = time.time() - start

        self.logger.info(f'Oracle found best solution after {beautify_seconds(total_time)},'
                         f' with score (total utility): {self.best_score}')

    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        if self.best_solution is None:
            self.compute_exact_solution(resource)
        action = self.get_current_add_action()
        if not self.action_space_wrapper.add_action_space.is_action_available(action):
            action = self.action_space_wrapper.add_action_space.get_available_actions()[0].item()
        return action

    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        action = self.get_current_remove_action()
        if not self.action_space_wrapper.remove_action_space.is_action_available(action):
            action = self.action_space_wrapper.remove_action_space.get_available_actions()[0].item()

        self.current_index += 1
        return action

    def _choose_node_remove(self, state: State,
                            epsilon: Optional[float] = None, random: Optional[float] = None,
                            resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None, **kwargs) -> int:
        raise NotImplementedError("Oracle cannot execute _choose_node_remove method")

    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        raise NotImplementedError("Oracle cannot execute _choose_node_add method")

    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        raise NotImplementedError("Oracle cannot execute _choose_combined_sub_action method")

    def _choose_quantity(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None, **kwargs) -> int:
        raise NotImplementedError("Oracle cannot execute _choose_quantity method")


class OracleSubAgent(BaselineAgent):

    def __init__(self,
                 solution: SolutionType,
                 action_space_wrapper: ActionSpaceWrapper,
                 random_state: np.random.RandomState,
                 action_spaces: Dict[ActionType, int],
                 state_spaces: Dict[StateType, int],
                 log: Logger,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(OracleSubAgent, self).__init__(action_space_wrapper,
                                             random_state=random_state,
                                             name=AgentType.Oracle,
                                             action_spaces=action_spaces,
                                             state_spaces=state_spaces,
                                             log=log,
                                             config=config,
                                             **kwargs)
        self.solution: SolutionType = solution
        self.current_index = 0
        self.final_solution = []
        self.selected_add_action = None

    def get_current_add_action(self):
        return self.solution[self.current_index % len(self.solution)][0]

    def get_current_remove_action(self):
        return self.solution[self.current_index % len(self.solution)][1]

    def _choose_add(self, state: State,
                    epsilon: Optional[float] = None, random: Optional[float] = None,
                    resource: Optional[str] = None,
                    nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                    demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                    ) -> int:
        action = self.get_current_add_action()
        if not self.action_space_wrapper.add_action_space.is_action_available(action):
            action = self.action_space_wrapper.add_action_space.get_available_actions()[0].item()
        self.selected_add_action = action
        return action

    def _choose_remove(self, state: State,
                       epsilon: Optional[float] = None, random: Optional[float] = None,
                       resource: Optional[str] = None,
                       nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                       demand: Optional[StepData] = None, pool_node: Optional[Node] = None,
                       ) -> int:
        action = self.get_current_remove_action()
        if not self.action_space_wrapper.remove_action_space.is_action_available(action):
            action = self.action_space_wrapper.remove_action_space.get_available_actions()[0].item()

        if len(self.final_solution) < len(self.solution):
            self.final_solution.append((self.selected_add_action, action))
        self.current_index += 1
        return action

    def _choose_node_remove(self, state: State,
                            epsilon: Optional[float] = None, random: Optional[float] = None,
                            resource: Optional[str] = None,
                            nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                            demand: Optional[StepData] = None, **kwargs) -> int:
        raise NotImplementedError("SubOracle cannot execute _choose_node_remove method")

    def _choose_node_add(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None
                         ) -> int:
        raise NotImplementedError("SubOracle cannot execute _choose_node_add method")

    def _choose_combined_sub_action(self, state: State, add_node: int,
                                    nodes: List[Node],
                                    epsilon: Optional[float] = None, random: Optional[float] = None,
                                    resource: Optional[str] = None,
                                    demand: Optional[StepData] = None) -> Tuple[int, int, int]:
        raise NotImplementedError("SubOracle cannot execute _choose_combined_sub_action method")

    def _choose_quantity(self, state: State,
                         epsilon: Optional[float] = None, random: Optional[float] = None,
                         resource: Optional[str] = None,
                         nodes: Optional[List[Node]] = None, node_groups: Optional[NodeGroups] = None,
                         demand: Optional[StepData] = None, **kwargs) -> int:
        raise NotImplementedError("SubOracle cannot execute _choose_quantity method")
