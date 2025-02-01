from prorl.common.enum_utils import ExtendedEnum


class AgentType(str, ExtendedEnum):
    Random = 'random'
    DoubleDQN = 'prorl'
    DoubleDQNFullSpace = 'prorl-no-split'
    Heuristic = 'heuristic'
    Greedy = 'greedy'
    Oracle = 'oracle'

    @staticmethod
    def is_value_based(agent_type: 'AgentType') -> bool:
        v_based_types = [AgentType.DoubleDQN, AgentType.DoubleDQNFullSpace]
        return agent_type in v_based_types

    @staticmethod
    def is_baseline(agent_type: 'AgentType') -> bool:
        baselines = [AgentType.Random, AgentType.Heuristic, AgentType.Greedy, AgentType.Oracle]
        return agent_type in baselines

