import itertools
import os
from copy import deepcopy
from datetime import datetime
from typing import List, Union, Dict, Optional, Tuple, Any
from uuid import uuid4

from prorl.common.config import AbstractConfig, ConfigValueError, set_sub_config, load_config_dict, ExportMode
from prorl.common.dict_utils import set_config_sub_value
from prorl.common.enum_utils import ExtendedEnum
from prorl.common.filesystem import ROOT_DIR
from prorl.core.step import Step
from prorl.common.data_structure import RunMode
from prorl.environment.agent import AgentType
from prorl.run.config import SingleRunConfig, MultiRunParamConfig

AGENT_WITH_VALIDATION = [
    AgentType.DoubleDQN,
]

AGENT_ON_POLICY = [

]

AGENT_ALIAS = {

}


class HyperParamType(str, ExtendedEnum):
    Agent = 'agent'
    Environment = 'env'
    Run = 'run'
    Root = 'root'


class HyperParamValueMode(str, ExtendedEnum):
    Array = 'array'
    MultiArray = 'multi-array'


class RandomSeedsConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.run: List[int] = [10, 11, 12, 13, 14]
        super(RandomSeedsConfig, self).__init__('RandomSeedsConfig', **configs_to_override)


class AgentHyperParamConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: HyperParamType = HyperParamType.Root
        self.key: Optional[str] = 'param.key'
        self.values_mode: HyperParamValueMode = HyperParamValueMode.Array
        self.values: List[Union[str, int, float, Dict[str, Union[str, int, float]]]] = []
        super(AgentHyperParamConfig, self).__init__('AgentHyperParamConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: HyperParamType = HyperParamType(self.type)
            except Exception:
                raise ConfigValueError('type', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {HyperParamType.list()}')
        if isinstance(self.values_mode, str):
            try:
                self.values_mode: HyperParamValueMode = HyperParamValueMode(self.values_mode)
            except Exception:
                raise ConfigValueError('values_mode', self.values_mode, module=self.name(),
                                       extra_msg=f'Possible values are: {HyperParamValueMode.list()}')


class MultiRunConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.base_run_config: SingleRunConfig = set_sub_config('base_run_config', SingleRunConfig,
                                                               root_dir,
                                                               **configs_to_override)
        self.random_seeds: RandomSeedsConfig = set_sub_config('random_seeds', RandomSeedsConfig, **configs_to_override)
        self.hyperparameters: Dict[AgentType, List[AgentHyperParamConfig]] = {
            AgentType.DoubleDQN: [
                AgentHyperParamConfig(
                    type=HyperParamType.Agent,
                    key='double_dqn.learning_rate',
                    values_mode=HyperParamValueMode.Array,
                    values=[0.1, 0.01, 0.001, 0.0001]
                ),
                AgentHyperParamConfig(
                    type=HyperParamType.Agent,
                    key='double_dqn.batch_size',
                    values_mode=HyperParamValueMode.Array,
                    values=[32, 64, 128, 256]
                ),
            ],
        }
        self.multi_run_name: str = '*auto*'
        self.skip_name_date: bool = False
        super(MultiRunConfig, self).__init__('MultiRunConfig', **configs_to_override)

    def _after_override_configs(self):
        if self.multi_run_name == '*auto*' or self.multi_run_name is None:
            self.multi_run_name = str(uuid4())
        if isinstance(self.hyperparameters, dict):
            hyperparameters: Dict[AgentType, List[AgentHyperParamConfig]] = {}
            for agent_name, conf_list in self.hyperparameters.items():
                agent_type = agent_name if isinstance(agent_name, AgentType) else AgentType(agent_name)
                if agent_type not in hyperparameters:
                    hyperparameters[agent_type] = []
                if isinstance(conf_list, list):
                    for conf in conf_list:
                        if isinstance(conf, AgentHyperParamConfig):
                            hyperparameters[agent_type].append(conf)
                        elif isinstance(conf, dict):
                            hyperparameters[agent_type].append(AgentHyperParamConfig(**conf))
                        else:
                            raise ConfigValueError('hyperparameters', self.hyperparameters, self.name(),
                                                   extra_msg='Hyperaparameters entries must be an object')
                else:
                    raise ConfigValueError('hyperparameters', self.hyperparameters, self.name(),
                                           extra_msg='Hyperaparameters must be'
                                                     ' "Dict[AgentType, List[AgentHyperParamConfig]]"')
            self.hyperparameters: Dict[AgentType, List[AgentHyperParamConfig]] = hyperparameters

    def generate_runs_config(self) -> List[SingleRunConfig]:
        runs: List[SingleRunConfig] = []
        scheduled_at = f'scheduled_at={datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
        for run_seed in self.random_seeds.run:
            for agent, agent_hp_config in self.hyperparameters.items():
                agent_combinations: List[
                    Tuple[Tuple[str, Any, HyperParamType], ...]] = generate_combinations(agent_hp_config)
                for param_combination in agent_combinations:
                    # create a new single_run_config
                    base_config = deepcopy(self.base_run_config.export(mode=ExportMode.DICT))
                    run_config = SingleRunConfig(root_dir=self.root_dir, **base_config)
                    run_config.random_seeds.training = run_seed
                    run_config.multi_run.is_multi_run = True
                    if self.skip_name_date is False:
                        run_config.multi_run.multi_run_code = f'{self.multi_run_name}-{scheduled_at}'
                    else:
                        run_config.multi_run.multi_run_code = self.multi_run_name
                    run_config.multi_run.multi_run_params.append(MultiRunParamConfig(
                        key=build_filename_key('seed'),
                        key_short=build_filename_key('seed'),
                        value=run_seed,
                        filename_key_val=f'seed={run_seed}'
                    ))
                    run_config.environment.agent.type = agent
                    for param in param_combination:
                        update_single_run_config_param(run_config, param)
                    runs.append(run_config)
        return runs

    @classmethod
    def generate_evaluation_config(cls,
                                   best_runs: List[dict],
                                   env_random_seed,
                                   agents: List[AgentType],
                                   batch_size: int,
                                   stop_step: Optional[Step] = None,
                                   step_size: Optional[int] = None,
                                   use_ssh_tunnel: bool = False,
                                   skip_name_date: bool = False,
                                   use_on_policy_agent: bool = True,
                                   seed_multiplier: int = 100,
                                   agent_filename: Optional[str] = 'agent_state.pth'
                                   ):
        if len(best_runs):
            base_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **best_runs[0]['config'])
            base_config.run.run_mode = RunMode.Eval
            if stop_step is not None:
                base_config.run.stop_date = stop_step
            if step_size is not None:
                base_config.run.step_size = step_size
            base_config.run.rollout_batch_size = batch_size
            base_config.run.use_on_policy_agent = use_on_policy_agent
            base_config.multi_run.multi_run_params = []
            hyperparameters: Dict[str, List[dict]] = {}
            for agent_type in agents:
                agent_values: List[Dict[str, Union[int, float, str]]] = []
                for run in best_runs:
                    run_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **run['config'])
                    if run_config.environment.agent.type == agent_type:
                        agent_type_str = agent_type.value.replace('-', '_')
                        if agent_type_str in AGENT_ALIAS:
                            agent_type_str = AGENT_ALIAS[agent_type_str]
                        base_config.environment.agent[agent_type_str] = run_config.environment.agent[agent_type_str]
                        base_config.saver.stats_condensed = False
                        base_config.saver.save_agent = False
                        run_seed = run_config.random_seeds.training
                        if agent_type in AGENT_WITH_VALIDATION:
                            model_path_base = run['save_folder_path']['path']
                            validation_index = run['validation_run_code'].replace(f'{run["run_code"]}_', '')
                            validation_name = f'validation_run_agent_state.pth'
                            model_path_full = os.path.join(model_path_base, f'validation_run-{validation_index}',
                                                           validation_name)
                            save_mode = run['save_folder_path']['save_mode']
                        # elif agent_type in AGENT_ON_POLICY:
                        #     model_path_base = run['result_path']['path']
                        #     validation_name = f'validation_run_agent_state.pth'
                        #     model_path_full = os.path.join(model_path_base, f'validation_run-{run["best_iteration"]}',
                        #                                    validation_name)
                        #     save_mode = run['result_path']['save_mode']
                        else:
                            model_path_base = run['result_path']['path']
                            model_path_full = os.path.join(model_path_base, agent_filename)
                            save_mode = run['result_path']['save_mode']
                        if save_mode == 'minio':
                            base_path = run_config.saver.default_bucket
                        else:
                            base_path = run_config.saver.get_base_path()
                        agent_values.append({
                            'environment.agent.model_load.load_model': True,
                            'environment.agent.model_load.use_single_model': True,
                            'environment.agent.model_load.add_node_model.path': model_path_full,
                            'environment.agent.model_load.add_node_model.mode': save_mode,
                            'environment.agent.model_load.add_node_model.base_path': base_path,
                            'environment.agent.model_load.add_node_model.use_ssh_tunnel': use_ssh_tunnel,
                            'random_seeds.evaluation': [run_seed * seed_multiplier],
                        })
                hyperparameters[agent_type.value] = [
                    AgentHyperParamConfig(
                        type=HyperParamType.Root,
                        values_mode=HyperParamValueMode.MultiArray,
                        values=agent_values
                    ).export(mode=ExportMode.DICT)
                ]
            multi_run_config = {
                'base_run_config': base_config.export(mode=ExportMode.DICT),
                'random_seeds': {'run': [0]},
                'hyperparameters': hyperparameters,
                'multi_run_name': f'{base_config.multi_run.multi_run_code}-evaluation',
                'skip_name_date': skip_name_date,
            }
            return cls(root_dir=ROOT_DIR, **multi_run_config)
        else:
            raise AttributeError('Best runs array is empty')

    @classmethod
    def generate_evaluation_config_sub_agents(cls,
                                              add_node_best_runs: List[dict],
                                              movement_best_runs: List[dict],
                                              env_random_seed,
                                              agents: List[AgentType],
                                              stop_step: Optional[Step] = None,
                                              step_size: Optional[int] = None,
                                              use_ssh_tunnel: bool = False,
                                              skip_name_date: bool = False,
                                              ):
        if len(add_node_best_runs) != len(movement_best_runs):
            raise AttributeError('add_node_best_runs length is different to the length of movement_best_runs')
        n_runs = len(add_node_best_runs)
        if n_runs > 0:
            base_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **movement_best_runs[0]['config'])
            add_node_base_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR,
                                                                    **add_node_best_runs[0]['config'])
            base_config.environment.state.base_features = add_node_base_config.environment.state.base_features
            base_config.environment.agent.type = AgentType.DoubleDQN
            base_config.environment.agent.double_dqn.q_net_addition_parameters['add_node_units'] = \
                add_node_base_config.environment.agent.double_dqn.q_net_addition_parameters['add_node_units']
            base_config.run.run_mode = RunMode.Eval
            if stop_step is not None:
                base_config.run.stop_date = stop_step
            if step_size is not None:
                base_config.run.step_size = step_size
            base_config.multi_run.multi_run_params = []
            hyperparameters: Dict[str, List[dict]] = {}
            for agent_type in agents:
                agent_values: List[Dict[str, Union[int, float, str]]] = []
                for i in range(n_runs):
                    add_node_run = add_node_best_runs[i]
                    add_node_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **add_node_run['config'])
                    movement_run = movement_best_runs[i]
                    movement_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **movement_run['config'])
                    agent_type_str = agent_type.value.replace('-', '_')
                    if agent_type_str in AGENT_ALIAS:
                        agent_type_str = AGENT_ALIAS[agent_type_str]
                    base_config.environment.agent[agent_type_str] = movement_config.environment.agent[agent_type_str]
                    base_config.environment.state.base_features = \
                        add_node_config.environment.state.base_features
                    base_config.environment.agent.type = AgentType.DoubleDQN
                    base_config.environment.agent.double_dqn.q_net_addition_parameters['add_node_units'] = \
                        add_node_config.environment.agent.double_dqn.q_net_addition_parameters['add_node_units']
                    base_config.saver.stats_condensed = False
                    base_config.saver.save_agent = False
                    run_seed = movement_config.random_seeds.training
                    tmp = {
                        'random_seeds.training': run_seed,
                        'environment.agent.model_load.load_model': True,
                    }
                    add_agent_model_params(
                        container=tmp, agent_type=agent_type, run=add_node_run, run_config=add_node_config,
                        use_ssh_tunnel=use_ssh_tunnel, sub_agent_name='add_node_model'
                    )
                    add_agent_model_params(
                        container=tmp, agent_type=agent_type, run=movement_run, run_config=movement_config,
                        use_ssh_tunnel=use_ssh_tunnel, sub_agent_name='movement_model'
                    )

                    agent_values.append(tmp)
                hyperparameters[agent_type.value] = [
                    AgentHyperParamConfig(
                        type=HyperParamType.Root,
                        values_mode=HyperParamValueMode.MultiArray,
                        values=agent_values
                    ).export(mode=ExportMode.DICT)
                ]
            multi_run_config = {
                'base_run_config': base_config.export(mode=ExportMode.DICT),
                'random_seeds': {'environment': env_random_seed, 'agent': [0]},
                'hyperparameters': hyperparameters,
                'multi_run_name': f'{base_config.multi_run.multi_run_code}-evaluation',
                'skip_name_date': skip_name_date,
            }
            return cls(root_dir=ROOT_DIR, **multi_run_config)
        else:
            raise AttributeError('Best runs array is empty')


def add_agent_model_params(
        container: Dict[str, Any],
        agent_type: AgentType,
        run: Dict[str, Any],
        run_config: SingleRunConfig,
        use_ssh_tunnel: bool,
        sub_agent_name: str
):
    if agent_type in AGENT_WITH_VALIDATION:
        model_path_base = run['save_folder_path']['path']
        validation_index = run['validation_run_code'].replace(f'{run["run_code"]}_', '')
        validation_name = f'validation_run_agent_state.pth'
        model_path_full = os.path.join(model_path_base, f'validation_run-{validation_index}',
                                       validation_name)
        save_mode = run['save_folder_path']['save_mode']
        if save_mode == 'minio':
            base_path = run_config.saver.default_bucket
        else:
            base_path = run_config.saver.get_base_path()
    else:
        model_path_base = run['result_path']['path']
        model_path_full = os.path.join(model_path_base, 'agent_state.pth')
        save_mode = run['result_path']['save_mode']
        if save_mode == 'minio':
            base_path = run_config.saver.default_bucket
        else:
            base_path = run_config.saver.get_base_path()

    container[f'environment.agent.model_load.{sub_agent_name}.path'] = model_path_full
    container[f'environment.agent.model_load.{sub_agent_name}.mode'] = save_mode
    container[f'environment.agent.model_load.{sub_agent_name}.base_path'] = base_path
    container[f'environment.agent.model_load.{sub_agent_name}.use_ssh_tunnel'] = use_ssh_tunnel


def get_multi_run_config(root_dir, config_path: str = None) -> MultiRunConfig:
    config = MultiRunConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    return config


def generate_combinations(
        parameters: List[AgentHyperParamConfig]) -> List[Tuple[Tuple[str, Any, HyperParamType], ...]]:
    possibilities: List[List[Tuple[str, Any, HyperParamType]]] = []
    # I need to use itertools.combinatinos to generate all the possible combinations
    for param_config in parameters:
        if param_config.values_mode == HyperParamValueMode.Array:
            param_possibilities: List[Tuple[str, Any, HyperParamType]] = []
            for value in param_config.values:
                param_possibilities.append(
                    (param_config.key, value, param_config.type)
                )
            possibilities.append(param_possibilities)
        elif param_config.values_mode == HyperParamValueMode.MultiArray:
            param_possibilities: List[Tuple[Tuple[str, Any, HyperParamType], ...]] = []
            for dict_values in param_config.values:
                sub_param: List[Tuple[str, Any, HyperParamType]] = []
                for key, value in dict_values.items():
                    sub_param.append(
                        (key, value, param_config.type)
                    )
                param_possibilities.append(tuple(sub_param))
            return param_possibilities
        else:
            raise ValueError('Parameter values_mode is invalid')

    return list(itertools.product(*possibilities))


SHORT_NAME_MAPPING = {
    'learning_rate': 'lr',
    'target_net_update_frequency': 'target-update',
}


def build_filename_key(key):
    key_parts = key.split('.')
    filename_key = key_parts[-1]
    if key == 'random_seeds.run':
        return 'run-seed'
    if key == 'environment.agent.model_load.path':
        return 'model-path'
    if filename_key in SHORT_NAME_MAPPING:
        filename_key = SHORT_NAME_MAPPING[filename_key]
    else:
        filename_key = filename_key.replace('_', '-')
    return filename_key


def build_filename_key_val(key: str, value: Any) -> str:
    filename_key = build_filename_key(key)
    if isinstance(value, list):
        filename_value = ','.join([str(v) for v in value])
    else:
        filename_value = str(value)
    return f'{filename_key}={filename_value}'


HYPERPARAMETER_TYPE_MAPPING = {
    HyperParamType.Root: '',
    HyperParamType.Environment: 'environment',
    HyperParamType.Agent: 'environment.agent',
    HyperParamType.Run: 'run',
}


def update_single_run_config_param(config: SingleRunConfig, param: Tuple[str, Any, HyperParamType]):
    key, value, param_type = param
    key_start = HYPERPARAMETER_TYPE_MAPPING[param_type]
    if len(key_start) > 0:
        full_key = f'{key_start}.{key}'
    else:
        full_key = key
    set_config_sub_value(config, key=full_key, value=value)
    if 'base_features' in full_key and isinstance(value, list):
        value = f'{len(value)}-features'
    multi_run_param: MultiRunParamConfig = MultiRunParamConfig(
        key=build_filename_key(full_key),
        key_short=build_filename_key(full_key),
        value=value,
        filename_key_val=build_filename_key_val(full_key, value)
    )
    if multi_run_param.key == 'agent-seed':
        first_param = config.multi_run.multi_run_params[0]
        if first_param.key == build_filename_key('seed'):
            del config.multi_run.multi_run_params[0]
    config.multi_run.multi_run_params.append(multi_run_param)
