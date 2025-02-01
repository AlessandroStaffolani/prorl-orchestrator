import os
import os
import shutil
import socket
from copy import deepcopy
from logging import Logger
from typing import Optional, List, Dict, Union, Any, Tuple
from uuid import uuid4

import torch
from numpy.random import RandomState

from prorl import SingleRunConfig, get_logger, ROOT_DIR, single_run_config, logger
from prorl.common.data_structure import RunMode
from prorl.common.filesystem import create_directory
from prorl.common.logger import BaseLogger
from prorl.common.object_handler import create_object_handler, SaverMode
from prorl.common.object_handler.base_handler import ObjectHandler
from prorl.common.object_handler.minio_handler import MinioObjectHandler
from prorl.common.print_utils import print_status
from prorl.common.stats_tracker import Tracker
from prorl.core.state import State
from prorl.core.step import Step
from prorl.emulator.data_structure import ModelTypes
from prorl.emulator.generate_disk_data import manage_disk_data
from prorl.environment.action_space import Action
from prorl.environment.agent import AgentType
from prorl.environment.agent.abstract import AgentAbstract, AgentLoss, AgentQEstimate
from prorl.environment.agent.factory import create_agent
from prorl.environment.config import ModelLoadConfig
from prorl.environment.rollouts import sample_trajectory, Rollout, track_average_values, \
    track_off_policy_training_iteration
from prorl.environment.wrapper import EnvWrapper
from prorl.run import RunStatus
from prorl.run.config import RunConfig
from prorl.run.remote import MongoRunWrapper, PersistenceInfo, RedisRunWrapper
from prorl.run.remote.redis_constants import REDIS_VALIDATION_RUNS_QUEUE
from prorl.run.run_stats_manager import RunStatsManager


def get_run_logger(run_code: str, run_config: SingleRunConfig) -> BaseLogger:
    model_name = run_config.emulator.model.type.value
    agent_name = run_config.environment.agent.type.value
    log_folder = os.path.join('load_balancer', model_name, agent_name, run_code)
    create_directory(os.path.join(ROOT_DIR, 'logs', log_folder))
    run_config.environment.logger.update_file_handler_folder(log_folder)
    run_config.environment.logger.name = run_code
    return get_logger(run_config.environment.logger)


def clean_log_folder(run_code: str, run_config: SingleRunConfig):
    model_name = run_config.emulator.model.type.value
    agent_name = run_config.environment.agent.type.value
    log_folder = os.path.join(ROOT_DIR, 'logs', 'load_balancer', model_name, agent_name, run_code)
    warning_path = os.path.join(log_folder, 'warning.log')
    if os.path.getsize(warning_path) == 0:
        # no warning or error we can delete the log
        shutil.rmtree(log_folder, ignore_errors=True)
        try:
            shutil.rmtree(log_folder)
            os.rmdir(log_folder)
        except Exception:
            pass


def save_best_val_runs(
        run_code: str,
        mongo: MongoRunWrapper,
        config: SingleRunConfig,
):
    best_runs: List[Dict[str, Any]] = mongo.get_run_best_validation_runs(
        run_code=run_code,
        limit=0,
        metric=config.run.validation_run.keep_metric
    )
    for i, val_run in enumerate(best_runs):
        mongo.update_scheduled_validation_run(
            run_code=run_code,
            val_code=val_run['validation_run_code'],
            status=RunStatus.COMPLETED,
            best_index=i
        )


def complete_training_run(
        run_code: str,
        mongo: MongoRunWrapper,
        config: SingleRunConfig,
        redis: Optional[RedisRunWrapper] = None,
):
    is_executed = mongo.is_training_run_in_status(status=RunStatus.EXECUTED,
                                                  run_code=run_code, check_validation_runs=True)
    if is_executed:
        save_best_val_runs(run_code, mongo, config)
        mongo.update_scheduled_run(run_code=run_code, status=RunStatus.COMPLETED)
        is_completed = True
    else:
        is_completed = mongo.is_training_run_in_status(status=RunStatus.COMPLETED,
                                                       run_code=run_code, check_validation_runs=True)
    if is_completed:
        if config.multi_run.is_multi_run:
            is_saved = mongo.is_run_saved_in_multi_run_table(
                run_code=run_code,
                multi_run_code=config.multi_run.multi_run_code
            )
            if not is_saved:
                mongo.save_multi_run_single_run(
                    multi_run_code=config.multi_run.multi_run_code,
                    run_code=run_code,
                    run_params=config.multi_run.multi_run_params
                )
        if redis is not None:
            redis.delete_single_run_stats(run_code=run_code)


def load_agent_model(
        model_load: ModelLoadConfig,
        config_saver_mode: SaverMode,
        handler: Union[ObjectHandler, MinioObjectHandler],
        log: Logger,
        device: torch.device,
) -> Tuple[Dict[str, Any], str]:
    object_handler = handler
    if model_load.mode != config_saver_mode:
        minio_endpoint = None
        object_handler = create_object_handler(
            logger=log,
            enabled=True,
            mode=model_load.mode,
            base_path=model_load.base_path,
            default_bucket=model_load.base_path,
            minio_endpoint=minio_endpoint
        )
    if not object_handler.exists(object_handler.get_path(model_load.path)):
        raise AttributeError(f'Agent model to load path not exists. Path "{model_load.path}"')
    agent_state = object_handler.load_agent_model(file_path=model_load.path, map_location=device,
                                                  base_path=model_load.base_path, bucket=model_load.base_path)
    return agent_state, model_load.path


class BaseRunner:

    def __init__(
            self,
            run_code: str,
            config: Optional[SingleRunConfig] = None,
            log: Optional[Logger] = None
    ):
        self.initialized = False
        self.run_code: str = run_code
        self.config: SingleRunConfig = config if config is not None else single_run_config
        self.logger: Logger = log if log is not None else logger
        self.run_mode: RunMode = self.config.run.run_mode
        self.resource_name: str = self.config.environment.resources[0].name
        self.total_steps = self.config.run.training_iterations
        if AgentType.is_value_based(self.config.environment.agent.type):
            bootstrapping_steps = self.config.environment.agent.double_dqn.bootstrap_steps
            batch_size = self.config.environment.agent.double_dqn.batch_size
            additional_steps = bootstrapping_steps if bootstrapping_steps > batch_size else batch_size
            self.total_steps += additional_steps
            step_sub_steps = 1
            if self.config.emulator.model.type == ModelTypes.TimDatasetModel:
                step_sub_steps = self.config.emulator.model.tim_dataset_model_options.step_size // self.config.run.step_size
            self.config.environment.agent.double_dqn.bootstrap_steps *= step_sub_steps
            self.total_steps *= step_sub_steps
        if self.run_mode == RunMode.Eval:
            self.total_steps = self.config.run.evaluation_episode_length
        self.n_iterations: int = self.config.run.training_iterations
        self.env: Optional[EnvWrapper] = None
        self.agent: Optional[AgentAbstract] = None
        self.disk_data_path: Optional[str] = None
        # random_states
        self.run_random: Optional[RandomState] = None
        # stats tracker
        self.stats_tracker: Tracker = Tracker.init_condensed_tracker(
            run_code=self.run_code, config=self.config, run_mode=self.run_mode, tensorboard=None
        )
        self.validation_counter = 0
        self.best_validation_performance: Optional[float] = None
        self.best_validation_performance_iteration: int = 0
        self.best_agent_model = None

    def _init(self):
        if not self.initialized:
            self._init_seeds()
            self._init_env()
            self._init_agent()
            self._set_env_info_on_agent(self.agent)
            self.initialized = True

    def _init_seeds(self):
        if self.run_mode == RunMode.Eval:
            run_seed = self.config.random_seeds.evaluation[0]
        else:
            run_seed = self.config.random_seeds.training
        self.run_random: RandomState = RandomState(run_seed)
        self.logger.info(
            f'Initialized environment and agent with seed: {run_seed}')

    def _return_new_env(self, run_config, random_state, run_code, run_mode: RunMode,
                        random_seed: int, no_init_reset: bool) -> EnvWrapper:
        env = EnvWrapper(
            base_stations_mappings=self.config.emulator.model.base_station_name_mappings,
            run_code=run_code,
            auto_init=True,
            nodes_type_distribution=self.config.environment.nodes.nodes_type_distribution,
            state_features=self.config.environment.state.features,
            run_config=run_config,
            env_configuration=self.config.environment,
            emulator_configuration=self.config.emulator,
            random_state=random_state,
            log=self.logger,
            run_mode=run_mode,
            disk_data_path=self.disk_data_path,
            random_seed=random_seed,
            no_init_reset=no_init_reset
        )
        return env

    def _init_val_eval_env(self, random_seed: int, batch_size, mode: RunMode, no_init_reset=True):
        random_state = RandomState(random_seed)
        run_config = RunConfig(
            stop_date=None,
            stop_step=batch_size,
            bootstrapping_steps=0,
            rollout_batch_size=batch_size
        )
        if self.config.emulator.model.type == ModelTypes.TimDatasetModel:
            model_options = self.config.emulator.model.tim_dataset_model_options
            run_config.step_size = self.config.run.step_size
            run_config.stop_step = None
            run_config.initial_date = model_options.time_step[mode]['start_date']
            run_config.stop_date = model_options.time_step[mode]['end_date']
        elif self.config.emulator.model.type == ModelTypes.SyntheticModel:
            # model_options = self.config.emulator.model.synthetic_model
            run_config.step_size = self.config.run.step_size
        return self._return_new_env(run_config=run_config, random_state=random_state,
                                    run_code=f'{self.run_code}_{mode.value}',
                                    run_mode=mode,
                                    random_seed=random_seed, no_init_reset=no_init_reset)

    def _init_env(self):
        if self.config.emulator.model.model_disk.use_disk:
            self.disk_data_path = manage_disk_data(config=self.config,
                                                   model_name=self.config.emulator.model.type.value,
                                                   log=self.logger)
        if self.run_mode == RunMode.Eval:
            run_seed = self.config.random_seeds.evaluation[0]
        else:
            run_seed = self.config.random_seeds.training
        self.env: EnvWrapper = self._return_new_env(run_config=self.config.run, random_state=self.run_random,
                                                    run_code=self.run_code, run_mode=self.config.run.run_mode,
                                                    random_seed=run_seed, no_init_reset=False)

    def _return_new_agent(self, env: EnvWrapper, init_seed=True, run_mode=RunMode.Train):
        return create_agent(
            agent_type=self.config.environment.agent.type,
            action_space_wrapper=env.action_space_wrapper,
            random_state=self.run_random,
            action_spaces=env.action_spaces,
            state_spaces=env.state_spaces,
            config=self.config,
            log=self.logger,
            mode=run_mode,
            node_groups=env.node_groups,
            init_seed=init_seed,
            run_mode=run_mode,
            **self.config.environment.agent.global_parameters
        )

    def _init_agent(self):
        self.agent: AgentAbstract = self._return_new_agent(self.env)
        bootstrapping_steps = self.agent.bootstrap_steps
        message = f'Initialized agent {self.config.environment.agent.type} using device {self.agent.device}'
        if bootstrapping_steps > 0:
            message += f' and bootstrapping of {bootstrapping_steps} steps'
        self.logger.info(message)
        self._eventually_load_agent_model()
        self._after_init_agent()
        self.agent.set_mode(self.run_mode)

    def _eventually_load_agent_model(self):
        model_load_wrapper = self.config.environment.agent.model_load
        if model_load_wrapper.load_model:
            object_handler: Union[ObjectHandler, MinioObjectHandler] = create_object_handler(
                logger=self.logger,
                enabled=self.config.saver.enabled,
                mode=self.config.saver.mode,
                base_path=self.config.saver.get_base_path(),
                default_bucket=self.config.saver.default_bucket,
                minio_endpoint=None
            )
            model_config = model_load_wrapper.model_load_options
            agent_state, model_path = load_agent_model(
                model_load=model_config,
                config_saver_mode=self.config.saver.mode,
                handler=object_handler,
                log=self.logger,
                device=self.agent.device,
            )
            self.agent.load_agent_state(agent_state)
            self.logger.info(f'Loaded agent state from path: {model_path}')

    def _after_init_agent(self):
        pass

    def _after_env_step(
            self,
            t: Step,
            action: Action,
            state: Dict[str, State],
            reward: float,
            agent_loss: Optional[AgentLoss] = None,
            agent_Q_estimate: Optional[AgentQEstimate] = None,
            step_info: Optional[Dict[str, Union[int, float]]] = None,
            action_info: Optional = None,
    ):
        pass

    def run_off_policy(self):
        if self.run_mode == RunMode.Train:
            self.train_off_policy()
        elif self.run_mode == RunMode.Eval:
            self.evaluate_only()

    def train_off_policy(self):
        step_sub_steps = 1
        if self.config.emulator.model.type == ModelTypes.TimDatasetModel:
            step_sub_steps = self.config.emulator.model.tim_dataset_model_options.step_size // self.config.run.step_size
        self.logger.info(f'Starting training for {self.total_steps} time steps with {step_sub_steps} steps per hour')
        # self.do_validations(0, self.total_steps - self.agent.bootstrap_steps)
        self._pre_run_start(n_iterations=self.total_steps)
        previous_state = None
        previous_action = None
        step_info = {}
        reward = None
        episode_stats = {}
        n_episodes = 0

        done = False
        state = self.env.reset(show_log=False, add_plus_one=False)
        for current_i in range(self.total_steps):
            if done:
                state = self.env.reset(show_log=False, add_plus_one=False)
                n_episodes += 1
                self._end_off_policy_episode(current_i, n_episodes)
            action, state, action_info = self.agent.choose(state, nodes=self.env.nodes, resource=self.resource_name,
                                                           node_groups=self.env.node_groups,
                                                           demand=self.env.current_demand, pool_node=self.env.pool_node)
            # The full next state must contain the information about the next action selected (remove and add nodes)
            # So we need to wait to push into the experience replay (agent.learn do the experience replay push),
            # but we need to do it before we perform the new step, so before we get the new demand and the new reward
            if previous_state is not None:
                self.agent.push_experience(state_wrapper=previous_state, actions=previous_action,
                                           next_state_wrapper=state, reward=reward, done=done)

                track_off_policy_training_iteration(
                    iteration=current_i - 1, episode_stats=episode_stats, tracker=self.stats_tracker,
                    actions=previous_action, reward=reward, done=done, step_info=step_info
                )
                if self.config.run.train_every_hour_only:
                    if current_i % step_sub_steps == 0:
                        # we do one learn step per hour, independently to the number of actions taken
                        _, _ = self.agent.learn()
                else:
                    _, _ = self.agent.learn()
                if done:
                    self._log_off_policy_last_training_episode(current_i)

            next_state, reward, step_info = self.env.step(action, resource=self.resource_name)
            done = step_info['done']
            previous_state = state
            previous_action = action
            state = next_state
        self._end_off_policy_episode(self.total_steps, n_episodes)
        self._after_run()

    def _pre_run_start(self, n_iterations: int):
        pass

    def _set_env_info_on_agent(self, agent: AgentAbstract):
        agent.set_max_factors(
            max_cost=self.env.reward_class.max_cost,
            max_remaining_gap=self.env.reward_class.remaining_gap_max_factor
        )

    def run_on_policy(self):
        if self.run_mode == RunMode.Train:
            self.train_on_policy()
        elif self.run_mode == RunMode.Eval:
            self.evaluate_only()

    def train_on_policy(self):
        assert AgentType.is_baseline(self.agent.name), \
            'on policy agent only available for run_on_policy'
        # until the environment has reached the stop step we collect rollouts and we train
        n_iterations = self.config.run.training_iterations
        self._pre_run_start(n_iterations)
        # since we are dealing with a non-episodic task, we can not reset the environment at each iteration,
        # therefore we use initial state for keeping track of the state from which start the next rollout.
        # The first time this state is the initial, after that it will be the last next state received
        initial_state = self.env.reset()
        self.agent.reset()
        batch_size = self.config.run.rollout_batch_size
        previous_state, previous_action, state, reward, previous_done = None, None, None, None, None
        iteration = 0
        executed_steps = 0
        while iteration < n_iterations:
            # collect a rollout
            rollout, previous_state, previous_action, state, reward, previous_done = sample_trajectory(
                env=self.env,
                initial_state=initial_state,
                agent=self.agent,
                batch_size=batch_size,
                resource_name=self.resource_name,
                stats_tracker=self.stats_tracker,
                config=self.config,
                iteration=iteration,
                mode=RunMode.Train,
                previous_state=previous_state,
                previous_action=previous_action,
                reward=reward,
                previous_done=previous_done
            )
            initial_state = state
            executed_steps += len(rollout) * self.config.environment.state.stack_n_states

            iteration += 1

            # check if we need to start an evaluation and log the stats
            self._end_on_policy_iteration(iteration, n_iterations, executed_steps, rollout)
        self._after_run()

    def evaluate_only(self):
        self._pre_run_start(1)
        if not AgentType.is_baseline(self.config.environment.agent.type):
            object_handler: Union[ObjectHandler, MinioObjectHandler] = create_object_handler(
                logger=self.logger,
                enabled=self.config.saver.enabled,
                mode=self.config.saver.mode,
                base_path=self.config.saver.get_base_path(),
                default_bucket=self.config.saver.default_bucket,
                minio_endpoint=None
            )
            agent_state, model_path = load_agent_model(
                model_load=self.config.environment.agent.model_load.model_load_options,
                config_saver_mode=self.config.saver.mode,
                handler=object_handler,
                log=self.logger,
                device=self.agent.device
            )
            self.best_agent_model = agent_state
        self.do_evaluations(full_reset=True, use_best_model=True)
        self._after_eval_run()

    def do_evaluations(self, full_reset=True, use_best_model=False):
        seeds = self.config.random_seeds.evaluation
        for i, seed in enumerate(seeds):
            batch_size = self.config.run.evaluation_episode_length
            eval_env = self._init_val_eval_env(random_seed=seed, batch_size=batch_size, mode=RunMode.Eval,
                                               no_init_reset=not full_reset)
            eval_env.set_stats_tracker(self.stats_tracker)
            # if getattr(eval_env.reward_class, 'normalization_range', None) is not None:
            #     eval_env.reward_class.normalization_range = (0, 1)
            if full_reset:
                initial_state = eval_env.reset(show_log=False)
            else:
                initial_state = eval_env.reset(reset_time_step=True, reset_reward=True, reset_nodes=True,
                                               reset_generator=True, reset_node_groups=True, show_log=False)
            if use_best_model and not AgentType.is_baseline(self.agent.name):
                # create a new agent and load the best model
                if self.config.run.validation_run.enabled:
                    eval_agent = self._return_new_agent(eval_env, init_seed=False, run_mode=RunMode.Eval)
                    eval_agent.set_stats_tracker(self.stats_tracker)
                    eval_agent.load_agent_state(self.best_agent_model)
                else:
                    eval_agent = self.agent
            else:
                eval_agent = self.agent
            eval_agent.reset()
            eval_agent.set_mode(RunMode.Eval)
            self._set_env_info_on_agent(eval_agent)
            rollout, _, _, _, _, _ = sample_trajectory(
                env=eval_env,
                initial_state=initial_state,
                agent=eval_agent,
                batch_size=batch_size,
                resource_name=self.resource_name,
                stats_tracker=self.stats_tracker,
                config=self.config,
                iteration=0,
                mode=RunMode.Eval,
                seed=seed
            )
            print_status(i + 1, len(seeds), 'Evaluations performed')
        print()
        # average the eval rollouts metrics
        track_average_values(tracker=self.stats_tracker, mode=RunMode.Eval, iteration=0, seeds=seeds)

    def _after_eval_run(self):
        prefix = RunMode.Eval.value
        steps_completed = self.config.run.evaluation_episode_length
        utility_reward = self.stats_tracker.get(f'{prefix}/avg/reward/utility')[-1]
        remaining_gap_total = self.stats_tracker.get(f'{prefix}/avg/reward/remaining_gap')[-1]
        if len(self.stats_tracker.get(f'{prefix}/avg/reward/surplus')) > 0:
            surplus_total = self.stats_tracker.get(f'{prefix}/avg/reward/surplus')[-1]
        else:
            surplus_total = 0
        cost_total = self.stats_tracker.get(f'{prefix}/avg/reward/cost')[-1]
        problems_solved_count = self.stats_tracker.get(f'{prefix}/avg/problem_solved/count')[-1]
        hours_solved_count = self.stats_tracker.get(f'{prefix}/avg/hour_satisfied')[-1]
        resource_utilization = self.stats_tracker.get(f'{prefix}/avg/resource_utilization')[-1]
        episodes_avg_len = self.stats_tracker.get(f'{prefix}/avg/episode/length')[-1]

        message = f'Evaluation completed | average episode length: {episodes_avg_len} ' \
                  f'| hours solved count: {hours_solved_count} | problem solved count: {problems_solved_count} ' \
                  f'| utility total reward: {utility_reward} ' \
                  f'| remaining gap total: {remaining_gap_total} | surplus total: {surplus_total} ' \
                  f'| cost total: {cost_total} | avg resource utilization: {resource_utilization}'

        self.logger.info(message)

    def do_validations(self, n_iterations: int, agent_state=None):
        if self.config.run.validation_run.enabled:
            seeds = self.config.random_seeds.validation
            val_agent = None
            for i, val_seed in enumerate(seeds):
                batch_size = self.config.run.validation_run.rollout_batch_size
                val_env = self._init_val_eval_env(random_seed=val_seed, batch_size=batch_size, mode=RunMode.Validation)
                val_env.set_stats_tracker(self.stats_tracker)
                # if getattr(val_env.reward_class, 'normalization_range', None) is not None:
                #     val_env.reward_class.normalization_range = (0, 1)
                initial_state = val_env.reset(reset_time_step=True, reset_reward=True, reset_nodes=True,
                                              reset_generator=True, reset_node_groups=True, show_log=False)
                val_agent = self._return_new_agent(val_env, init_seed=False, run_mode=RunMode.Validation)
                val_agent.set_stats_tracker(self.stats_tracker)
                self._set_env_info_on_agent(val_agent)
                val_agent.set_mode(RunMode.Validation)
                if agent_state is None:
                    val_agent.load_agent_state(deepcopy(self.agent.get_agent_state()))
                else:
                    val_agent.load_agent_state(agent_state)
                val_agent.reset()
                rollout, _, _, _, _, _ = sample_trajectory(
                    env=val_env,
                    initial_state=initial_state,
                    agent=val_agent,
                    batch_size=batch_size,
                    resource_name=self.resource_name,
                    stats_tracker=self.stats_tracker,
                    config=self.config,
                    iteration=self.validation_counter,
                    mode=RunMode.Validation,
                    seed=val_seed
                )
                print_status(i + 1, len(seeds), f'Performing validations iteration {self.validation_counter}')
            #  save the average values in the stats tracker
            track_average_values(tracker=self.stats_tracker, mode=RunMode.Validation,
                                 iteration=self.validation_counter, seeds=seeds)
            val_performance = self.stats_tracker.get(
                self.config.run.validation_run.validation_keep_metric)[self.validation_counter]
            self._log_validation_run(n_iterations, level=20)
            self._after_validation_run(val_agent, val_performance)
            self.validation_counter += 1

    def _after_validation_run(self, val_agent: AgentAbstract, val_performance: float):
        if self.best_validation_performance is None or val_performance > self.best_validation_performance:
            iteration_divider = self.config.run.validation_run.validation_frequency
            self.best_validation_performance = val_performance
            self.best_validation_performance_iteration = self.validation_counter
            self.best_agent_model = deepcopy(val_agent.get_agent_state())
            n_seeds = len(self.config.random_seeds.validation)
            self.logger.info(f'Validation iteration {self.validation_counter} '
                             f'obtained the current best validation performance with a value of {val_performance}'
                             f' over {n_seeds} seeds')
            return True
        else:
            return False

    def _after_run(self):
        self.logger.info(
            f'Best validation run performance obtained at iteration {self.best_validation_performance_iteration}'
            f' with performance {self.best_validation_performance}')
        self.logger.info('Starting model evaluation')
        self.do_evaluations(full_reset=False, use_best_model=True)
        self._after_eval_run()
        keep_metric = self.config.run.validation_run.keep_metric
        run_performance = self.stats_tracker.get(keep_metric)[-1]
        return run_performance

    def _end_off_policy_episode(self, iteration: int, n_episodes: int):
        val_frequency = self.config.run.validation_run.validation_frequency
        bootstrap_steps = self.agent.bootstrap_steps
        # no_random_iterations = iteration - bootstrap_steps
        no_random_iterations = n_episodes
        if no_random_iterations % val_frequency == 0 and no_random_iterations >= 0 and not self.agent.is_bootstrapping:
            self.do_validations(self.total_steps - bootstrap_steps)

    def _end_on_policy_iteration(self, iteration: int, n_iterations: int, executed_steps: int, rollout: Rollout):
        episode_length = len(rollout)
        if iteration % self.config.run.validation_run.validation_frequency == 0:
            self.do_validations(n_iterations)
            self._log_policy_training_last_batch(iteration, executed_steps, n_iterations, episode_length, level=20)
        elif iteration % self.config.run.debug_frequency == 0:
            # log the stats
            self._log_policy_training_last_batch(iteration, executed_steps, n_iterations, episode_length, level=10)

    def _log_validation_run(self, n_iterations: int, level=20):
        if self.config.run.validation_run.enabled:
            prefix = RunMode.Validation.value
            iteration_divider = self.config.run.validation_run.validation_frequency
            # entire run performances
            utility_reward = self.stats_tracker.get(f'{prefix}/avg/reward/utility', only_cumulative=True)[-1]
            remaining_gap_reward = self.stats_tracker.get(f'{prefix}/avg/reward/remaining_gap', only_cumulative=True)[
                -1]
            if len(self.stats_tracker.get(f'{prefix}/avg/reward/surplus')) > 0:
                surplus_reward = self.stats_tracker.get(f'{prefix}/avg/reward/surplus')[-1]
            else:
                surplus_reward = 0
            cost_reward = self.stats_tracker.get(f'{prefix}/avg/reward/cost', only_cumulative=True)[-1]
            pre_gap_reward = 0  # self.stats_tracker.get(f'{prefix}/avg/reward/pre_gap', only_cumulative=True)[-1]
            problems_solved_count = self.stats_tracker.get(f'{prefix}/avg/problem_solved/count')[-1]
            hours_solved_count = self.stats_tracker.get(f'{prefix}/avg/hour_satisfied')[-1]
            resource_utilization = self.stats_tracker.get(f'{prefix}/avg/resource_utilization')[-1]
            episodes_avg_len = self.stats_tracker.get(f'{prefix}/avg/episode/length')[-1]
            n_seeds = len(self.config.random_seeds.validation)
            n_steps = self.config.run.validation_run.rollout_batch_size

            message = f'Validation iteration {self.validation_counter} ' \
                      f'| Average values over {n_seeds} seeds | episode length: {episodes_avg_len}' \
                      f'| hours solved count: {hours_solved_count} | problem solved count: {problems_solved_count} ' \
                      f'| utility total reward: {utility_reward} ' \
                      f'| remaining gap total: {remaining_gap_reward} | surplus total: {surplus_reward} ' \
                      f'| cost total: {cost_reward} | avg resource utilization: {resource_utilization}'
            self.logger.log(level, message)

    def _log_policy_training_last_batch(self, iteration: int, executed_steps: int,
                                        n_iterations: int, episode_length: int, level=10):
        prefix = RunMode.Train.value
        # last batch performances
        utility_reward = self.stats_tracker.get(f'{prefix}/batch/reward/utility/total')[-1]
        utility_reward_avg = self.stats_tracker.get(f'{prefix}/batch/reward/utility/avg')[-1]
        utility_reward_std = self.stats_tracker.get(f'{prefix}/batch/reward/utility/std')[-1]
        remaining_gap_avg = self.stats_tracker.get(f'{prefix}/batch/reward/remaining_gap/avg')[-1]
        remaining_gap_std = self.stats_tracker.get(f'{prefix}/batch/reward/remaining_gap/std')[-1]
        cost_avg = self.stats_tracker.get(f'{prefix}/batch/reward/cost/avg')[-1]
        cost_std = self.stats_tracker.get(f'{prefix}/batch/reward/cost/std')[-1]
        utility_reward_min = self.stats_tracker.get(f'{prefix}/batch/reward/utility/min')[-1]
        utility_reward_max = self.stats_tracker.get(f'{prefix}/batch/reward/utility/max')[-1]
        problems_solved_count = self.stats_tracker.get(f'{prefix}/batch/problem_solved/count')[-1]
        pre_gap_avg = 0  # self.stats_tracker.get(f'{prefix}/batch/reward/pre_gap/avg')[-1]
        pre_gap_std = 0  # self.stats_tracker.get(f'{prefix}/batch/reward/pre_gap/std')[-1]
        n_dones = self.stats_tracker.get(f'{prefix}/batch/dones')[-1]
        n_episodes = self.stats_tracker.get(f'{prefix}/episodes')

        message = f'Last training batch {iteration}/{n_iterations} | episodes completed {n_episodes} ' \
                  f'| last episode length {episode_length}' \
                  f'| steps executed since start {executed_steps} ' \
                  f'| problem solved count: {problems_solved_count} | utility total reward: {utility_reward} | ' \
                  f'utility reward avg: {utility_reward_avg} | utility reward std: {utility_reward_std} | ' \
                  f'utility reward min: {utility_reward_min} | utility reward max: {utility_reward_max} | ' \
                  f'remaining gap avg: {remaining_gap_avg} | remaining gap std: {remaining_gap_std} | ' \
                  f'cost avg: {cost_avg} | cost std: {cost_std} ' \
                  f'| pre gap avg: {pre_gap_avg} | pre gap std: {pre_gap_std} | episodes terminated: {n_dones}'

        self.logger.log(level, message)

    def _log_off_policy_last_training_episode(self, iteration: int, level=10):
        prefix = RunMode.Train.value
        # last batch performances
        utility_reward = self.stats_tracker.get(f'{prefix}/episode/reward/utility/total')[-1]
        utility_reward_avg = self.stats_tracker.get(f'{prefix}/episode/reward/utility/avg')[-1]
        utility_reward_std = self.stats_tracker.get(f'{prefix}/episode/reward/utility/std')[-1]
        remaining_gap_avg = self.stats_tracker.get(f'{prefix}/episode/reward/remaining_gap/avg')[-1]
        remaining_gap_std = self.stats_tracker.get(f'{prefix}/episode/reward/remaining_gap/std')[-1]
        if len(self.stats_tracker.get(f'{prefix}/episode/reward/surplus/avg')) > 0:
            surplus_avg = self.stats_tracker.get(f'{prefix}/episode/reward/surplus/avg')[-1]
            surplus_std = self.stats_tracker.get(f'{prefix}/episode/reward/surplus/std')[-1]
        else:
            surplus_avg = 0
            surplus_std = 0
        cost_avg = self.stats_tracker.get(f'{prefix}/episode/reward/cost/avg')[-1]
        cost_std = self.stats_tracker.get(f'{prefix}/episode/reward/cost/std')[-1]
        utility_reward_min = self.stats_tracker.get(f'{prefix}/episode/reward/utility/min')[-1]
        utility_reward_max = self.stats_tracker.get(f'{prefix}/episode/reward/utility/max')[-1]
        problems_solved_count = self.stats_tracker.get(f'{prefix}/episode/problem_solved/count')[-1]
        hours_solved_count = self.stats_tracker.get(f'{prefix}/episode/hour_satisfied')[-1]
        n_episodes = self.stats_tracker.get(f'{prefix}/episode/total')
        episode_length = self.stats_tracker.get(f'{prefix}/episode/length')[-1]
        steps_since_start = self.stats_tracker.get(f'{prefix}/total_steps')
        current_epsilon = self.agent.get_action_epsilon()
        current_prior_beta = self.agent.get_prioritized_beta_parameter()
        current_expert_theta = self.agent.get_expert_train_theta()

        message = f'Last training iteration {iteration}/{self.total_steps} | episodes completed: {n_episodes} ' \
                  f'| steps since start: {steps_since_start} | last episode length: {episode_length}' \
                  f'| hours solved count: {hours_solved_count} | problem solved count: {problems_solved_count} ' \
                  f'| utility total reward: {utility_reward} | ' \
                  f'utility reward avg: {utility_reward_avg} | utility reward std: {utility_reward_std} | ' \
                  f'utility reward min: {utility_reward_min} | utility reward max: {utility_reward_max} | ' \
                  f'remaining gap avg: {remaining_gap_avg} | remaining gap std: {remaining_gap_std} | ' \
                  f'surplus avg: {surplus_avg} | surplus std: {surplus_std} | ' \
                  f'cost avg: {cost_avg} | cost std: {cost_std} | current epsilon: {current_epsilon}'

        if current_prior_beta is not None:
            message += f' | current prioritized buffer beta: {current_prior_beta}'

        if current_expert_theta is not None:
            message += f' | current expert training theta: {current_expert_theta}'

        self.logger.log(level, message)

    def run(self):
        self._init()
        if self.agent.collect_rollouts or (self.config.run.use_on_policy_agent and self.agent.is_baseline):
            # Collect rollouts and feed them to a policy gradient algorithm
            self.run_on_policy()
        else:
            # Off-policy algorithm, thus interact and learn directly in the same loop
            self.run_off_policy()

    def close_runner(self):
        pass


class Runner(BaseRunner):

    def __init__(self, run_code: str,
                 validation_run_queue: Optional[str] = None
                 ):
        self.run_code: str = run_code
        mongo_host = None
        mongo_port = None
        self.mongo: MongoRunWrapper = MongoRunWrapper(host=mongo_host, port=mongo_port)
        run_info = self.mongo.get_by_run_code(run_code=self.run_code)
        if run_info is None:
            raise AttributeError(f'No run with code {run_code} exists')
        self.run_status = RunStatus(run_info['status'])
        if self.run_status == RunStatus.RUNNING or self.run_status == RunStatus.COMPLETED \
                or self.run_status == RunStatus.ERROR:
            raise AttributeError(
                f'The run with code {run_code} can not be started because the status is {self.run_status}')
        self.config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **run_info['config'])
        self.logger: BaseLogger = get_run_logger(self.run_code, self.config)

        super(Runner, self).__init__(run_code=self.run_code, config=self.config, log=self.logger)

        self.redis: Optional[RedisRunWrapper] = None
        minio_endpoint = None
        self.object_handler: Union[ObjectHandler, MinioObjectHandler] = create_object_handler(
            logger=self.logger,
            enabled=self.config.saver.enabled,
            mode=self.config.saver.mode,
            base_path=self.config.saver.get_base_path(),
            default_bucket=self.config.saver.default_bucket,
            minio_endpoint=minio_endpoint
        )
        self.run_stats_manager: Optional[RunStatsManager] = None
        self.result_path: Optional[PersistenceInfo] = None
        self.iteration_checkpoint: List[int] = []
        self.last_batches_results: List[int] = []
        self.best_batch_results: Optional[float] = None
        self.validation_run_index = 0
        self.validation_run_queue_name: str = REDIS_VALIDATION_RUNS_QUEUE
        if validation_run_queue is not None:
            self.validation_run_queue_name = validation_run_queue
        self.last_validation_start_step: Optional[Step] = None
        self._init()

    def _init(self):
        try:
            if not self.initialized:
                self._init_seeds()
                self._init_env()
                self._init_agent()
                self._set_env_info_on_agent(self.agent)
                self._init_run_stats_manager()
                self._init_redis()
                self.initialized = True
        except Exception as e:
            self.logger.exception(e)
            raise e

    def _init_run_stats_manager(self):
        self.run_stats_manager: RunStatsManager = RunStatsManager(
            run_code=self.run_code,
            config=self.config,
            logger=self.logger,
            nodes=self.env.nodes,
            agent_name=self.agent.name,
            total_steps=self.env.time_step.stop_step,
            step_size=self.env.time_step.step_size,
            object_handler=self.object_handler
        )
        self.env.set_stats_tracker(self.run_stats_manager.stats_tracker)
        self.agent.set_stats_tracker(self.run_stats_manager.stats_tracker)
        self.stats_tracker = self.run_stats_manager.stats_tracker
        host = socket.gethostbyname(socket.gethostname())
        if self.config.saver.mode == SaverMode.Minio:
            host = self.object_handler.endpoint
        self.result_path: PersistenceInfo = PersistenceInfo(
            path=self.run_stats_manager.save_folder,
            save_mode=self.config.saver.mode,
            host=host
        )

    def _init_redis(self):
        if self.config.redis.enabled:
            redis_host = os.getenv('REDIS_HOST')
            redis_port = int(os.getenv('REDIS_PORT'))
            self.redis: RedisRunWrapper = RedisRunWrapper(
                run_code=self.run_code,
                run_total_steps=self.total_steps,
                logger=self.logger,
                config=self.config,
                host=redis_host,
                port=redis_port
            )

    def _schedule_validation_run(self, force_schedule=False):
        is_enabled = self.config.run.validation_run.enabled
        current_step = self.env.time_step.current_step
        if is_enabled and self.agent.requires_validation and self.agent.mode == RunMode.Train:
            if force_schedule:
                if current_step != self.last_validation_start_step:
                    self._send_validation_run_schedule()
            elif current_step.to_second() % self.config.run.validation_run.validation_frequency == 0:
                self._send_validation_run_schedule()

    def _send_validation_run_schedule(self):
        current_step = self.env.time_step.current_step
        val_run_code = f'{self.run_code}_{self.validation_run_index}'
        val_config: SingleRunConfig = self.config.to_validation_run_config()
        agent_state_binaries = self.agent.serialize_agent_state()
        self.mongo.add_scheduled_validation_run(
            run_code=self.run_code,
            val_code=val_run_code,
            val_config=val_config,
            agent_state=agent_state_binaries,
            training_current_step=current_step.total_steps,
            save_folder_path=self.result_path
        )
        if self.redis is not None:
            self.redis.add_scheduled_validation_run(val_run_code, queue_name=self.validation_run_queue_name)
        self.validation_run_index += 1
        self.last_validation_start_step = current_step

    def _update_redis_stats(self, iteration=None, n_iterations=0):
        if self.redis is not None:
            check_step = self.env.time_step.current_step.to_second()
            run_stats = self.stats_tracker.redis_stats()
            if iteration is not None:
                check_step = iteration
                run_stats['iteration'] = iteration
            if check_step % self.config.redis.update_frequency == 0:
                if isinstance(check_step, Step):
                    check_step = check_step.total_steps
                self.redis.add_stats(
                    agent_name=self.agent.name,
                    current_step=check_step,
                    status=RunStatus.RUNNING,
                    run_stats=run_stats
                )

    def _pre_run_start(self, n_iterations: int):
        self.mongo.update_scheduled_run(self.run_code, status=RunStatus.RUNNING)
        if self.redis is not None:
            self.redis.add_stats(status=RunStatus.RUNNING,
                                 agent_name=self.agent.name,
                                 current_step=self.env.time_step.current_step, run_stats={})

    def _end_off_policy_episode(self, iteration: int, n_episodes: int):
        super(Runner, self)._end_off_policy_episode(iteration, n_episodes)
        self._update_redis_stats(iteration, self.total_steps)

    def _end_on_policy_iteration(self, iteration: int, n_iterations: int, total_steps: int, rollout: Rollout):
        super(Runner, self)._end_on_policy_iteration(iteration, n_iterations, total_steps, rollout)
        self._update_redis_stats(iteration, n_iterations)
        # self._checkpoint(iteration)

    def _after_validation_run(self, val_agent: AgentAbstract, val_performance: float):
        is_new_best = super(Runner, self)._after_validation_run(val_agent, val_performance)
        if is_new_best:
            # override the new best agent
            self.run_stats_manager.save_agent(val_agent, 'best_validation_agent_state.pth')
        else:
            n_next_val = self.config.run.save_n_models_after_best
            if self.validation_counter <= self.best_validation_performance_iteration + n_next_val:
                index = self.validation_counter - self.best_validation_performance_iteration
                # self.run_stats_manager.save_agent(val_agent, f'after_best_{index}_agent_state.pth')
                self.run_stats_manager.save_agent(val_agent, f'after_best_{index}_agent_state.pth')

    def _checkpoint(self, iteration: int):
        frequency = self.config.run.last_validation_metrics
        keep_metric = f'train/{frequency}_batches/reward/utility/total'
        metric = self.stats_tracker.get(keep_metric)
        if iteration % frequency == 0 and len(metric) > 0:
            metric_value = metric[-1]
            if self.best_batch_results is None or metric_value > self.best_batch_results:
                # save the current model
                self.iteration_checkpoint.append(iteration)
                self.run_stats_manager.save_validation_run_agent(
                    agent=self.agent,
                    index=iteration
                )
                self.logger.info(f'At iteration {iteration} new best average {keep_metric} in the latest {frequency} '
                                 f'batch iterations, average: {metric_value}')
                self.best_batch_results = metric_value

    def _after_run(self):
        run_performance = super(Runner, self)._after_run()
        self._save_validation_data()
        self._save_evaluation_data()
        self.save_run_data(run_performance, self.agent)

    def _save_validation_data(self):
        prefix = RunMode.Validation.value
        is_enabled = self.config.run.validation_run.enabled
        if is_enabled and self.agent.requires_validation and self.agent.mode == RunMode.Train:
            utility_reward = self.stats_tracker.get(f'{prefix}/avg/reward/utility')
            for iteration in range(len(utility_reward)):
                validation_data = {
                    f'{prefix}/avg/reward/utility': self.stats_tracker.get(f'{prefix}/avg/reward/utility')[iteration],
                    f'{prefix}/avg/reward/remaining_gap': self.stats_tracker.get(f'{prefix}/avg/reward/remaining_gap')[
                        iteration],
                    f'{prefix}/avg/reward/cost': self.stats_tracker.get(f'{prefix}/avg/reward/cost')[iteration],
                    f'{prefix}/avg/problem_solved/count': self.stats_tracker.get(f'{prefix}/avg/problem_solved/count')[
                        iteration],
                }
                if len(self.stats_tracker.get(f'{prefix}/avg/reward/surplus')) > 0:
                    validation_data[f'{prefix}/avg/reward/surplus'] = \
                        self.stats_tracker.get(f'{prefix}/avg/reward/surplus')[iteration]
                for key, value in validation_data.items():
                    if hasattr(value, 'item'):
                        value = value.item()
                    validation_data[key] = value

                self.mongo.add_validation_run(
                    run_code=self.run_code,
                    iteration=iteration + 1,
                    status=RunStatus.COMPLETED,
                    validation_data=validation_data,
                    validation_steps=self.config.run.validation_run.rollout_batch_size
                )

    def _save_evaluation_data(self):
        prefix = RunMode.Eval.value
        if self.run_mode == RunMode.Train:
            utility_reward = self.stats_tracker.get(f'{prefix}/avg/reward/utility')
            for iteration in range(len(utility_reward)):
                evaluation_data = {
                    f'{prefix}/avg/reward/utility': self.stats_tracker.get(f'{prefix}/avg/reward/utility')[iteration],
                    f'{prefix}/avg/reward/remaining_gap': self.stats_tracker.get(f'{prefix}/avg/reward/remaining_gap')[
                        iteration],
                    f'{prefix}/avg/reward/cost': self.stats_tracker.get(f'{prefix}/avg/reward/cost')[iteration],
                    f'{prefix}/avg/problem_solved/count': self.stats_tracker.get(f'{prefix}/avg/problem_solved/count')[
                        iteration],
                    f'{prefix}/avg/times/choose_action': self.stats_tracker.get(f'{prefix}/avg/times/choose_action')[
                        iteration],
                }
                if len(self.stats_tracker.get(f'{prefix}/avg/reward/surplus')) > 0:
                    evaluation_data[f'{prefix}/avg/reward/surplus'] = \
                        self.stats_tracker.get(f'{prefix}/avg/reward/surplus')[iteration]
                for key, value in evaluation_data.items():
                    if hasattr(value, 'item'):
                        value = value.item()
                    evaluation_data[key] = value

                self.mongo.add_evaluation_run(
                    run_code=self.run_code,
                    iteration=iteration + 1,
                    status=RunStatus.COMPLETED,
                    evaluation_data=evaluation_data,
                    steps=self.config.run.evaluation_episode_length
                )

    def _after_eval_run(self):
        super(Runner, self)._after_eval_run()
        if self.run_mode == RunMode.Eval:
            keep_metric = self.config.run.validation_run.keep_metric
            run_performance = self.stats_tracker.get(keep_metric)[-1]
            self.save_run_data(run_performance=run_performance, agent=None)

    def save_run_data(self, run_performance: Optional[float], agent: Optional[AgentAbstract]):
        # save stats and agent
        self.run_stats_manager.save_stats(agent=agent)
        # set as completed the run on redis
        if self.redis is not None:
            redis_stats = self.stats_tracker.redis_stats()
            if self.run_mode == RunMode.Train:
                redis_stats = self.stats_tracker.redis_stats()
            self.redis.add_stats(
                agent_name=self.agent.name,
                current_step=self.env.time_step.current_step,
                status=RunStatus.COMPLETED,
                run_stats=redis_stats
            )
        # set as completed the run on mongo
        db_stats = self.stats_tracker.db_stats()
        data_folder = None
        if self.config.emulator.model.model_disk.use_disk:
            data_folder = self.env.disk_data_path
            data_folder_parts = data_folder.split('/')[-3:]
            data_folder = '/'.join(data_folder_parts)
        self.mongo.update_scheduled_run(
            run_code=self.run_code,
            status=RunStatus.COMPLETED,
            result_path=self.result_path,
            total_steps=self.env.time_step.current_step.total_steps,
            run_stats=db_stats,
            run_performance=run_performance,
            data_folder=data_folder,
            best_validation_performance=self.best_validation_performance,
            best_validation_performance_iteration=self.best_validation_performance_iteration,
        )
        complete_training_run(
            run_code=self.run_code,
            mongo=self.mongo,
            config=self.config,
            redis=self.redis
        )
        # close the opened connections
        self.close_runner()

    def close_runner(self):
        self.run_stats_manager.close()
        if self.redis is not None:
            self.redis.close()
        self.mongo.close()
        clean_log_folder(run_code=self.run_code, run_config=self.config)
        self.logger.close()
        del self.logger


class ValRunRunner(BaseRunner):

    def __init__(self, val_run_code: str):
        self.val_run_code: str = val_run_code
        mongo_host = None
        mongo_port = None
        self.mongo: MongoRunWrapper = MongoRunWrapper(host=mongo_host, port=mongo_port)
        run_info = self.mongo.get_validation_run_by_code(val_code=self.val_run_code)
        if run_info is None:
            raise AttributeError(f'No validation run with code {val_run_code} exists')
        self.run_status = RunStatus(run_info['status'])
        if self.run_status == RunStatus.RUNNING or self.run_status == RunStatus.COMPLETED \
                or self.run_status == RunStatus.ERROR:
            raise AttributeError(
                f'The validation run with code {val_run_code} can not be started because the status is {self.run_status}')
        self.config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **run_info['config'])
        self.logger: Logger = get_run_logger(self.val_run_code, self.config)

        super(ValRunRunner, self).__init__(run_code=self.val_run_code, config=self.config, log=self.logger)

        self.original_run_code: str = run_info['run_code']
        self.agent_state = run_info['agent_state']
        self.save_path_info: PersistenceInfo = PersistenceInfo(**run_info['save_folder_path'])
        self.redis: Optional[RedisRunWrapper] = None
        minio_endpoint = None
        self.object_handler: Union[ObjectHandler, MinioObjectHandler] = create_object_handler(
            logger=self.logger,
            enabled=self.config.saver.enabled,
            mode=self.config.saver.mode,
            base_path=self.config.saver.get_base_path(),
            default_bucket=self.config.saver.default_bucket,
            minio_endpoint=minio_endpoint
        )
        self.run_stats_manager: Optional[RunStatsManager] = None
        self._init()

    def _init(self):
        try:
            if not self.initialized:
                self._init_seeds()
                self._init_env()
                self._init_agent()
                self._init_run_stats_manager()
                self._init_redis()
                self.initialized = True
        except Exception as e:
            self.logger.exception(e)
            raise e

    def _after_init_agent(self):
        self.agent.load_serialized_agent_state(self.agent_state)
        self.logger.info(f'Loaded agent state')
        self.agent_state = None

    def _init_run_stats_manager(self):
        self.run_stats_manager: RunStatsManager = RunStatsManager(
            run_code=self.original_run_code,
            val_run_code=self.val_run_code,
            config=self.config,
            logger=self.logger,
            nodes=self.env.nodes,
            agent_name=self.agent.name,
            total_steps=self.env.time_step.stop_step,
            step_size=self.env.time_step.step_size,
            object_handler=self.object_handler,
            save_folder=self.save_path_info['path']
        )
        self.run_stats_manager.stats_tracker = Tracker.init_condensed_tracker(
            run_code=self.run_code, config=self.config, run_mode=self.run_mode, tensorboard=None
        )
        self.env.set_stats_tracker(self.run_stats_manager.stats_tracker)
        self.agent.set_stats_tracker(self.run_stats_manager.stats_tracker)
        self.stats_tracker = self.run_stats_manager.stats_tracker

    def _init_redis(self):
        if self.config.redis.enabled:
            redis_host = os.getenv('REDIS_HOST')
            redis_port = int(os.getenv('REDIS_PORT'))
            self.redis: RedisRunWrapper = RedisRunWrapper(
                run_code=self.original_run_code,
                val_run_code=self.val_run_code,
                run_total_steps=self.total_steps,
                logger=self.logger,
                config=self.config,
                host=redis_host,
                port=redis_port
            )

    def _update_redis_stats(self):
        if self.redis is not None:
            if self.env.time_step.current_step.to_second() % self.config.redis.update_frequency == 0:
                self.redis.add_stats(
                    agent_name=self.agent.name,
                    current_step=self.env.time_step.current_step,
                    status=RunStatus.RUNNING,
                    run_stats=self.run_stats_manager.stats_tracker.redis_stats(),
                )

    def run_off_policy(self):
        try:
            self.logger.info(f'Starting execution for {self.env.time_step.stop_step} time steps')
            self.mongo.update_scheduled_validation_run(
                run_code=self.original_run_code,
                val_code=self.val_run_code, status=RunStatus.RUNNING)
            if self.redis is not None:
                self.redis.add_stats(status=RunStatus.RUNNING, run_stats={},
                                     agent_name=self.agent.name,
                                     current_step=self.env.time_step.current_step)
            state = self.env.reset()
            previous_state = None
            previous_action = None
            reward = None
            while not self.env.is_last_step():
                t: Step = self.env.current_time_step
                action, state, _ = self.agent.choose(
                    state, t, nodes=self.env.nodes, resource=self.resource_name,
                    node_groups=self.env.node_groups, is_bootstrapping_phase=False,
                    demand=self.env.current_demand
                )
                if previous_state is not None and previous_action is not None and reward is not None:
                    self.agent.learn(state_wrapper=previous_state, action=previous_action,
                                     next_state_wrapper=state, reward=reward, t=t, is_bootstrapping_phase=False)
                next_state, reward, step_info = self.env.step(action, resource=self.resource_name)
                self.run_stats_manager.step_stats(
                    t=t,
                    action=action,
                    reward=reward,
                    nodes_satisfied=step_info['nodes_satisfied'],
                    resource_name=self.resource_name,
                    current_demand=self.env.current_demand,
                    penalty=step_info['penalty'],
                )
                self._update_redis_stats()
                previous_state = state
                previous_action = action
                state = next_state
            self.run_stats_manager.log_run_status(t=self.env.current_time_step)
            self.run_stats_manager.save_validation_run_stats()
            self.run_stats_manager.save_validation_run_agent(self.agent)
            if self.redis is not None:
                self.redis.add_stats(
                    current_step=self.env.time_step.current_step,
                    agent_name=self.agent.name,
                    status=RunStatus.EXECUTED,
                    run_stats=self.run_stats_manager.stats_tracker.redis_stats()
                )

            self.mongo.update_scheduled_validation_run(
                run_code=self.original_run_code,
                val_code=self.val_run_code,
                status=RunStatus.EXECUTED,
                total_steps=self.env.time_step.current_step.total_steps,
                run_stats=self.run_stats_manager.stats_tracker.db_stats(),
                delete_agent_state=True)
            complete_training_run(
                run_code=self.original_run_code,
                mongo=self.mongo,
                config=self.config,
                redis=self.redis
            )
            self.close_runner()
        except Exception as e:
            self.logger.exception(e)
            raise e

    def close_runner(self):
        self.run_stats_manager.close()
        if self.redis is not None:
            self.redis.close()
        self.mongo.close()
        clean_log_folder(run_code=self.val_run_code, run_config=self.config)
        del self.logger


class TestRunner(BaseRunner):

    def __init__(
            self,
            run_code: Optional[str] = None,
            config: Optional[SingleRunConfig] = None,
            log: Optional[Logger] = None
    ):
        super(TestRunner, self).__init__(
            run_code=run_code if run_code is not None else str(uuid4()),
            config=config,
            log=log
        )
        self.problem_solved_count: int = 0
        self.state_count: Dict[State, int] = {}
        self.reward_info_history: List[Dict[str, float]] = []
        self.action_cost_history: List[float] = []

    def _after_init_agent(self):
        self.env.set_stats_tracker(self.stats_tracker)
        self.agent.set_stats_tracker(self.stats_tracker)
