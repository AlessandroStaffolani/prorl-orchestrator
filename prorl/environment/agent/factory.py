from logging import Logger
from typing import Union, Optional, Dict

from numpy.random import RandomState

from prorl import SingleRunConfig
from prorl.common.data_structure import RunMode
from prorl.environment.action_space import ActionSpaceWrapper, ActionType
from prorl.environment.agent import AgentType
from prorl.environment.agent.baseline import RandomAgent, GreedyOptimalAgent, \
    ExhaustiveSearchAgent, OracleAgent
from prorl.environment.agent.dqn import DoubleDQNAgent, DoubleDQNFullSpaceAgent
from prorl.environment.node_groups import NodeGroups
from prorl.environment.state_builder import StateType
from prorl.environment.wrapper import EnvWrapper

AGENTS_MAPPING = {
    AgentType.Random: RandomAgent,
    AgentType.DoubleDQN: DoubleDQNAgent,
    AgentType.DoubleDQNFullSpace: DoubleDQNFullSpaceAgent,
    AgentType.Heuristic: GreedyOptimalAgent,
    AgentType.Greedy: ExhaustiveSearchAgent,
    AgentType.Oracle: OracleAgent,
}


def create_agent(
        agent_type: AgentType,
        action_space_wrapper: ActionSpaceWrapper,
        random_state: RandomState,
        action_spaces: Dict[ActionType, int],
        state_spaces: Dict[StateType, int],
        log: Logger,
        config: Optional[SingleRunConfig] = None,
        mode: RunMode = RunMode.Train,
        node_groups: Optional[NodeGroups] = None,
        env_wrapper: Optional[EnvWrapper] = None,
        **parameters) -> Union[RandomAgent]:
    if agent_type in AGENTS_MAPPING:
        return AGENTS_MAPPING[agent_type](
            action_space_wrapper=action_space_wrapper,
            random_state=random_state,
            action_spaces=action_spaces,
            state_spaces=state_spaces,
            log=log,
            config=config,
            mode=mode,
            node_groups=node_groups,
            env_wrapper=env_wrapper,
            **parameters
        )
    else:
        raise AttributeError(f'AgentType "{agent_type}" not available')
