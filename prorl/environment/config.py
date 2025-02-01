import sys

from prorl.environment.scalarization_function import ScalarizationFunction

if sys.version_info >= (3, 8):
    from typing import Union, List, Dict, Any, Optional
else:
    from typing import Union, List, Dict, Any, Optional

from prorl.common.config import AbstractConfig, ConfigValueError, load_config_dict, LoggerConfig, set_sub_config
from prorl.common.enum_utils import ExtendedEnum
from prorl.common.object_handler import SaverMode
from prorl.environment.agent import AgentType
from prorl.environment.agent.parameter_schedulers import EpsilonType
from prorl.environment.agent.experience_replay import ReplayBufferType
from prorl.environment.data_structure import ResourceClass, ResourceDistribution, NodeTypeDistribution, \
    EnvResource
from prorl.environment.reward import RewardFunctionType
from prorl.environment.state_builder import StateFeatureName


class NetworkType(str, ExtendedEnum):
    FullyConnected = 'fully-connected'
    Dueling = 'dueling'


class NodesConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.n_nodes: int = 6
        self.nodes_type_distribution: NodeTypeDistribution = NodeTypeDistribution.Equally
        self.node_resource_distribution: ResourceDistribution = ResourceDistribution.Pool
        self.resource_distribution_parameters: Dict[str, Union[int, float, bool, str]] = {
            'initial_node_units': 3
        }
        self.use_pool_node: bool = True
        super(NodesConfig, self).__init__('NodesConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.nodes_type_distribution, str):
            try:
                self.nodes_type_distribution: NodeTypeDistribution = NodeTypeDistribution(self.nodes_type_distribution)
            except Exception:
                raise ConfigValueError('nodes_type_distribution', self.nodes_type_distribution, module=self.name(),
                                       extra_msg=f'Possible values are: {NodeTypeDistribution.list()}')

    def get_n_nodes(self, no_pool: bool = False) -> int:
        return self.n_nodes


class ResourceConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.name: str = ''
        self.total_available: int = 0
        self.allocated: int = 0
        self.units_allocated: int = 0
        self.total_units: int = 0
        self.min_resource_buckets_per_node: int = 0
        self.bucket_size: int = 0
        self.classes: Dict[str, ResourceClass] = {}
        super(ResourceConfig, self).__init__('ResourcesConfig', **configs_to_override)

    def _after_override_configs(self):
        if len(self.classes) > 0:
            allocated = 0
            units = 0
            for name, res_class in self.classes.items():
                allocated += res_class['allocated'] * res_class['capacity']
                units += res_class['allocated']
            self.allocated = allocated
            self.units_allocated = units


class ProvidersConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.n_providers: int = 1
        super(ProvidersConfig, self).__init__('ProvidersConfig', **configs_to_override)


class SubAgentSetupConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.full_action_split: bool = True
        self.same_state_features: bool = True
        self.same_reward: bool = True
        self.add_full_space: bool = True
        super(SubAgentSetupConfig, self).__init__('SubAgentSetupConfig', **configs_to_override)


class StateConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.features: List[StateFeatureName] = [
            StateFeatureName.PoolCapacity,
            StateFeatureName.NodeCapacity,
            StateFeatureName.NodeDemand,
            StateFeatureName.NodeDelta,
            StateFeatureName.TimeEncoded,
            StateFeatureName.CurrentLives,
            StateFeatureName.NodeAdd,
        ]
        self.base_features: List[StateFeatureName] = [
            StateFeatureName.PoolCapacity,
            StateFeatureName.NodeCapacity,
            StateFeatureName.NodeDemand,
            StateFeatureName.NodeDelta,
            StateFeatureName.TimeEncoded,
            StateFeatureName.CurrentLives,
            StateFeatureName.NodeAdd,
        ]
        self.normalized: bool = True
        self.floor_deltas: bool = True
        self.additional_properties: Dict[str, Any] = {
            'units_to_skip': ['second_step', 'second', 'minute', 'week', 'month', 'year', 'total_steps'],
            'resource_units_normalization': 'total',  # possible values: ["by_class", "total"],
            'delta_with_units': True
        }
        self.node_groups_with_resource_classes: bool = True
        self.stack_n_states: int = 1
        super(StateConfig, self).__init__('StateConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.features, list) and isinstance(self.features[0], str):
            self.features: List[StateFeatureName] = [StateFeatureName(feature) for feature in self.features]
        else:
            ConfigValueError('features', self.features, module=self.name(),
                             extra_msg=f'Possible values are: {StateFeatureName.list()}')
        if isinstance(self.base_features, list) and isinstance(self.base_features[0], str):
            self.base_features: List[StateFeatureName] = [
                StateFeatureName(feature) for feature in self.base_features]
        else:
            ConfigValueError('add_node_features', self.base_features, module=self.name(),
                             extra_msg=f'Possible values are: {StateFeatureName.list()}')


class ActionSpaceConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.bucket_move: List[int] = [1]
        super(ActionSpaceConfig, self).__init__('ActionSpaceConfig', **configs_to_override)


class RewardConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: RewardFunctionType = RewardFunctionType.GapSurplusCost
        self.multi_reward: bool = True
        self.invalid_action_penalty: int = -100
        self.disable_cost: bool = False
        self.null_reward: float = 0
        self.null_between_interval: Optional[int] = 3600
        self.parameters: Dict[str, Any] = {
            'alpha': 1,
            'scalarized': True,
            'scalarization_function': ScalarizationFunction.Linear,
            'normalize_objectives': True,
            'training_normalization_range': (0, 10),
            'val_eval_normalization_range': (0, 1),
            'clipped_remaining_gap': False,
            'zero_remaining_gap_reward': 0,
            'weights': [0.7, 0.3, 0],
            'gap_with_units': True,
            'hour_satisfied_bonus': 0,
            'satisfied_or_nothing': False,
            'delta_target': 0,
            'training_success_reward': 10,
            'val_eval_success_reward': 1,
        }
        super(RewardConfig, self).__init__('RewardConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: RewardFunctionType = RewardFunctionType(self.type)
            except Exception:
                raise ConfigValueError('type', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {RewardFunctionType.list()}')
        if 'scalarization_function' in self.parameters and isinstance(self.parameters['scalarization_function'], str):
            try:
                self.parameters['scalarization_function'] = ScalarizationFunction(
                    self.parameters['scalarization_function'])
            except Exception:
                raise ConfigValueError('parameters.scalarization_function', self.parameters['scalarization_function'],
                                       module=self.name(),
                                       extra_msg=f'Possible values are: {ScalarizationFunction.list()}')


class BudgetFeatureConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.budget: float = 1000
        self.penalty: float = self.budget
        super(BudgetFeatureConfig, self).__init__('BudgetFeatureConfig', **configs_to_override)


class ReplayBufferConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.type: ReplayBufferType = ReplayBufferType.UniformBuffer
        self.capacity: int = 0
        self.alpha: float = 0.6
        self.beta: float = 0.4
        self.beta_annealing_steps: int = 1000
        self.prioritized_epsilon = 1e-6
        super(ReplayBufferConfig, self).__init__('ReplayBufferConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: ReplayBufferType = ReplayBufferType(self.type)
            except Exception:
                raise ConfigValueError('type', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {ReplayBufferType.list()}')


class DoubleDQNConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.learning_rate: float = 0.0001
        self.gamma: float = 0.99
        self.batch_size: int = 128
        self.bootstrap_steps: int = 500
        self.target_net_update_frequency: int = 168
        self.q_net_type: NetworkType = NetworkType.Dueling
        self.q_net_addition_parameters: Dict[str, Any] = {
            'hidden_units': [64, 128, 128, 256],
            'batch_normalization': True,
        }
        self.use_stochastic_policy: bool = False
        self.use_full_stochastic_policy: bool = False
        self.updates_per_step: int = 1
        self.epsilon_type: EpsilonType = EpsilonType.LinearDecay
        self.epsilon_parameters: Dict[str, Union[int, float]] = {
            'start': 1,
            'end': 0.001,
            'total': 500,
            'decay': 1800,
            'alternate_values': [(100, 150), (150, 250)]
        }
        self.train_with_expert: bool = True
        self.theta_parameters: Dict[str, Union[int, float]] = {
            'start': 1,
            'end': 0,
            'total': 0,
        }
        super(DoubleDQNConfig, self).__init__('DoubleDQNConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.epsilon_type, str):
            try:
                self.epsilon_type: EpsilonType = EpsilonType(self.epsilon_type)
            except Exception:
                raise ConfigValueError('epsilon_type', self.epsilon_type, module=self.name(),
                                       extra_msg=f'Possible values are: {EpsilonType.list()}')
        if isinstance(self.q_net_type, str):
            try:
                self.q_net_type: NetworkType = NetworkType(self.q_net_type)
            except Exception:
                raise ConfigValueError('q_net_type', self.q_net_type, module=self.name(),
                                       extra_msg=f'Possible values are: {NetworkType.list()}')


class SamplingOptimalConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.batch_size: int = 15
        super(SamplingOptimalConfig, self).__init__('SamplingOptimalConfig', **configs_to_override)


class ModelLoadConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.path: Optional[str] = None
        self.mode: Optional[SaverMode] = SaverMode.Disk
        self.base_path: str = ''
        self.use_ssh_tunnel: bool = False
        super(ModelLoadConfig, self).__init__('ModelLoadConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.mode, str):
            try:
                self.mode: SaverMode = SaverMode(self.mode)
            except Exception:
                raise ConfigValueError('mode', self.mode, module=self.name(),
                                       extra_msg=f'Possible values are: {SaverMode.list()}')


class ModelLoadWrapperConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.load_model: bool = False
        self.model_load_options: ModelLoadConfig = set_sub_config('model_load_options',
                                                                  ModelLoadConfig, **configs_to_override)
        super(ModelLoadWrapperConfig, self).__init__('ModelLoadWrapperConfig', **configs_to_override)


class AgentConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: AgentType = AgentType.DoubleDQN
        self.replay_buffer: ReplayBufferConfig = set_sub_config('replay_buffer',
                                                                ReplayBufferConfig, **configs_to_override)
        self.global_parameters: Dict[str, object] = {}
        self.double_dqn: DoubleDQNConfig = set_sub_config('double_dqn', DoubleDQNConfig, **configs_to_override)
        self.sampling_optimal: SamplingOptimalConfig = set_sub_config('sampling_optimal', SamplingOptimalConfig,
                                                                      **configs_to_override)
        self.model_load: ModelLoadWrapperConfig = set_sub_config('model_load', ModelLoadWrapperConfig,
                                                                 **configs_to_override)
        self.combine_last_sub_actions: bool = True
        super(AgentConfig, self).__init__('AgentConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: AgentType = AgentType(self.type)
            except Exception:
                raise ConfigValueError('type', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {AgentType.list()}')


class NodeGroupsConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = False
        super(NodeGroupsConfig, self).__init__('NodeGroupsConfig', **configs_to_override)


class EpisodeConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.lives: int = 300
        self.loose_life_threshold: int = 0  # remove a life if more than loose_life_threshold nodes are unsatisfied
        super(EpisodeConfig, self).__init__('EpisodeConfig', **configs_to_override)


class EnvConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.node_groups: NodeGroupsConfig = set_sub_config('node_groups', NodeGroupsConfig, **configs_to_override)
        self.nodes: NodesConfig = set_sub_config('nodes', NodesConfig, **configs_to_override)
        self.resources: List[ResourceConfig] = [
            ResourceConfig(name='res_1',
                           total_available=100,
                           allocated=50,
                           units_allocated=5,
                           total_units=10,
                           bucket_size=1,
                           min_resource_buckets_per_node=0,
                           classes={
                               'resource': {'cost': 1, 'capacity': 10, 'allocated': 5},
                           }),
            # ResourceConfig(name='res_2', total_available=200, resource_per_node=20,
            #                bucket_size=2, min_resource_buckets_per_node=3)
        ]
        self.providers: ProvidersConfig = set_sub_config('providers', ProvidersConfig, **configs_to_override)
        self.sub_agents_setup: SubAgentSetupConfig = set_sub_config('sub_agents_setup', SubAgentSetupConfig,
                                                                    **configs_to_override)
        self.agent: AgentConfig = set_sub_config('agent', AgentConfig, **configs_to_override)
        self.state: StateConfig = set_sub_config('state', StateConfig, **configs_to_override)
        self.action_space: ActionSpaceConfig = set_sub_config('action_space', ActionSpaceConfig, **configs_to_override)
        self.reward: RewardConfig = set_sub_config('reward', RewardConfig, **configs_to_override)
        self.budget_feature: BudgetFeatureConfig = set_sub_config('budget_feature', BudgetFeatureConfig,
                                                                  **configs_to_override)
        self.episode: EpisodeConfig = set_sub_config('episode', EpisodeConfig, **configs_to_override)
        self.logger: LoggerConfig = set_sub_config('logger', LoggerConfig, 'environment', **configs_to_override)
        super(EnvConfig, self).__init__(config_object_name='EnvConfig', root_dir=root_dir, **configs_to_override)

    def get_env_resources(self) -> List[EnvResource]:
        resources = []
        for res_conf in self.resources:
            resources.append(EnvResource(
                name=res_conf.name,
                bucket_size=res_conf.bucket_size,
                min_buckets=res_conf.min_resource_buckets_per_node,
                total_available=res_conf.total_available,
                allocated=res_conf.allocated,
                units_allocated=res_conf.units_allocated,
                total_units=res_conf.total_units,
                classes=res_conf.classes
            ))
        return resources

    def _after_override_configs(self):
        if isinstance(self.resources, list):
            resources: List[ResourceConfig] = []
            for conf in self.resources:
                if isinstance(conf, ResourceConfig):
                    resources.append(conf)
                elif isinstance(conf, dict):
                    resources.append(ResourceConfig(**conf))
                else:
                    raise ConfigValueError('resources', self.resources, self.name(),
                                           extra_msg='Resource entries must be an object')
            self.resources: List[ResourceConfig] = resources
        # for resource_config in self.resources:
        #     if len(resource_config.classes) > 0:
        #         resource_config.total_available = resource_config.allocated * self.nodes.get_n_nodes()
        #         resource_config.total_units = resource_config.units_allocated * self.nodes.get_n_nodes()


def get_env_config(root_dir, config_path: str = None, log_level: Union[int, None] = None) -> EnvConfig:
    config = EnvConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    if log_level is not None:
        config.set_configs(logger={'level': log_level})
    return config
