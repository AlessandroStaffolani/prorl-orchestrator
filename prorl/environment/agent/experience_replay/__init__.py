from prorl.environment.agent.experience_replay.replay_buffer import ExperienceEntry, ReplayBuffer,\
    ReplayBufferType, PrioritizedReplayBuffer
import numpy as np


MAPPING = {
    ReplayBufferType.UniformBuffer: ReplayBuffer,
    ReplayBufferType.PrioritizedBuffer: PrioritizedReplayBuffer,
}


def replay_buffer_factory(
        buffer_type: ReplayBufferType,
        capacity: int,
        random_state: np.random.RandomState,
        alpha: float,
) -> ReplayBuffer:
    if buffer_type in MAPPING:
        buffer = MAPPING[buffer_type]
        return buffer(
            capacity=capacity,
            random_state=random_state,
            alpha=alpha,
            )
    else:
        raise AttributeError('ReplayBufferType has no replay buffer associated')
