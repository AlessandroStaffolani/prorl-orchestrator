from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch
from torch import Tensor, tensor

from prorl.common.enum_utils import ExtendedEnum
from prorl.core.state import State
from prorl.environment.action_space import CombinedActionSpaceWrapper, ActionType
from prorl.environment.agent.experience_replay.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBufferType(str, ExtendedEnum):
    UniformBuffer = 'uniform-buffer'
    PrioritizedBuffer = 'prioritized-buffer'


@dataclass
class ExperienceEntry:
    state: Union[Tensor, List[Tensor]]
    action: Union[Tensor, List[Tensor]]
    reward: Union[Tensor, List[Tensor]]
    next_state: Union[Tensor, List[Tensor]]
    done: Union[Tensor, List[Tensor]]
    weights: Optional[Tensor] = None
    indexes: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'weights': self.weights,
            'indexes': self.indexes,
        }

    def __dict__(self):
        return self.to_dict()

    def __iter__(self):
        for value in self.__dict__().values():
            yield value


class ReplayBuffer(object):
    """
    ReplayBuffer Class, used by the agents for storing and retrieving samples from the previous sequences of
    iteration with the environment
    """

    def __init__(
            self,
            capacity: int,
            random_state: np.random.RandomState = None,
            **kwargs
    ):
        self.capacity: int = capacity
        self.memory: deque[ExperienceEntry] = deque(maxlen=self.capacity)
        if random_state is not None:
            self.random: np.random.RandomState = random_state
        else:
            self.random: np.random.RandomState = np.random.RandomState()

    def _push_pre_processing(self,
                             state: State,
                             action: int,
                             reward: float,
                             next_state: State,
                             done: bool,
                             device: torch.device) -> ExperienceEntry:
        return ExperienceEntry(
            state=state.to_tensor(device),
            action=tensor([action], dtype=torch.long, device=device),
            reward=tensor([reward], dtype=torch.float, device=device),
            next_state=next_state.to_tensor(device),
            done=tensor([done], dtype=torch.float, device=device)
        )

    def push(
            self,
            state: State,
            action: int,
            reward: float,
            next_state: State,
            done: bool,
            device: torch.device = torch.device('cpu')
    ):
        """
        Add a new entry to the experience replay buffer
        """
        new_entry: ExperienceEntry = self._push_pre_processing(state, action, reward, next_state, done, device)
        self.memory.append(new_entry)

    def sample(self, batch_size: int, device: torch.device, **kwargs) -> ExperienceEntry:
        """
        Random choice of `batch_size` elements from the experience replay buffer
        Parameters
        """
        indexes = self.random.choice(len(self), batch_size)
        samples = [self.memory[i] for i in indexes]
        return self._sample_post_processing(samples)

    def _sample_post_processing(
            self,
            samples: List[ExperienceEntry],
            indexes: Optional[List[int]] = None,
            weights: Optional[Tensor] = None
    ) -> ExperienceEntry:
        batch = ExperienceEntry(*zip(*samples))
        state = torch.stack(batch.state)
        action = torch.stack(batch.action)
        reward = torch.stack(batch.reward)
        next_state = torch.stack(batch.next_state)
        done = torch.stack(batch.done)
        return ExperienceEntry(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            weights=weights,
            indexes=indexes,
        )

    def update_priorities(self, indexes: List[int], priorities: np.ndarray):
        pass

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return f'<ReplayBuffer capacity={self.capacity} size={len(self)} >'


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            capacity: int,
            random_state: np.random.RandomState = None,
            alpha: float = 0.6,
            **kwargs
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            capacity=capacity,
            random_state=random_state,
        )
        self.memory: List[ExperienceEntry] = []
        self.max_priority: float = 1.0
        self.alpha: float = alpha
        self._next_idx: int = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.sum_tree: SumSegmentTree = SumSegmentTree(tree_capacity)
        self.min_tree: MinSegmentTree = MinSegmentTree(tree_capacity)

    def push(
            self,
            state: State,
            action: int,
            reward: float,
            next_state: State,
            done: bool,
            device: torch.device = torch.device('cpu')
    ):
        new_entry: ExperienceEntry = self._push_pre_processing(state, action, reward, next_state, done, device)
        # store the entry
        if self._next_idx >= len(self.memory):
            self.memory.append(new_entry)
        else:
            self.memory[self._next_idx] = new_entry
        # add the priority to the trees
        self.sum_tree[self._next_idx] = self.max_priority ** self.alpha
        self.min_tree[self._next_idx] = self.max_priority ** self.alpha
        self._next_idx = (self._next_idx + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device, beta: float = 0.4) -> ExperienceEntry:
        assert len(self) >= batch_size
        assert beta > 0

        indexes = self._sample_proportional(batch_size)
        samples = [self.memory[i] for i in indexes]
        weights = torch.tensor([[self._calculate_weight(i, beta)] for i in indexes], dtype=torch.float, device=device)

        return self._sample_post_processing(samples, indexes=indexes, weights=weights)

    def update_priorities(self, indexes: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indexes) == len(priorities)

        for idx, priority in zip(indexes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = self.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float) -> float:
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

    def __str__(self):
        return f'<PrioritizedReplayBuffer capacity={self.capacity} size={len(self)} >'


class RolloutBuffer:

    def __init__(
            self,
            capacity: int,
            state_dim: int,
            action_space_wrapper: Optional[CombinedActionSpaceWrapper] = None,
            n_rewards: int = 1,
            **kwargs
    ):
        self.capacity = int(capacity)
        self.action_space_wrapper: CombinedActionSpaceWrapper = action_space_wrapper
        self.n_rewards: int = n_rewards
        if self.capacity != -1:
            self.states: np.ndarray = np.zeros((self.capacity, state_dim), dtype=np.float64)
            self.actions: np.ndarray = np.zeros((self.capacity, 4), dtype=np.float64)
            self.next_states: np.ndarray = np.zeros((self.capacity, state_dim), dtype=np.float64)
            self.rewards: np.ndarray = np.zeros((self.capacity, self.n_rewards), dtype=np.float64)
            self.done: np.ndarray = np.zeros(self.capacity, dtype=np.float64)
        else:
            self.states: Optional[np.ndarray] = None
            self.actions: Optional[np.ndarray] = None
            self.next_states: Optional[np.ndarray] = None
            self.rewards: Optional[np.ndarray] = None
            self.done: Optional[np.ndarray] = None

    def push(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            next_state: np.ndarray,
            done: np.ndarray,
            **kwargs
    ):
        if self.capacity != -1:
            self.states[-self.capacity:] = state
            self.actions[-self.capacity:] = action
            self.next_states[-self.capacity:] = next_state
            self.rewards[-self.capacity:] = reward
            self.done[-self.capacity:] = done
        else:
            self.states = state
            self.actions = action
            self.next_states = next_state
            self.rewards = reward
            self.done = done

    def sample(
            self,
            batch_size: int,
            device: torch.device,
            action_type: ActionType,
            **kwargs
    ) -> ExperienceEntry:
        if batch_size != -1:
            # sample the latest batch size rollout transitions
            actions = self.actions[-batch_size:]
            actions_list = []
            for a in actions:
                if action_type == ActionType.Combined:
                    remove_node = int(a[-3])
                    res_class = int(a[-2])
                    quantity = int(a[-1])
                    combined_action = self.action_space_wrapper.combined_inverted_mapping[(remove_node, res_class, quantity)]
                    actions_list.append(combined_action)
                elif action_type == ActionType.Add:
                    actions_list.append(a[0])
            return ExperienceEntry(
                state=torch.from_numpy(self.states[-batch_size:]).float().to(device),
                action=torch.tensor(actions_list, dtype=torch.float, device=device),
                reward=torch.from_numpy(self.rewards[-batch_size:]).float().to(device),
                next_state=torch.from_numpy(self.next_states[-batch_size:]).float().to(device),
                done=torch.from_numpy(self.done[-batch_size:]).bool().to(device)
            )
        else:
            # sample the latest batch size rollout transitions
            actions = self.actions
            actions_list = []
            for a in actions:
                if action_type == ActionType.Combined:
                    remove_node = int(a[-3])
                    res_class = int(a[-2])
                    quantity = int(a[-1])
                    combined_action = self.action_space_wrapper.combined_inverted_mapping[(remove_node, res_class, quantity)]
                    actions_list.append(combined_action)
                elif action_type == ActionType.Add:
                    actions_list.append(a[0])
            return ExperienceEntry(
                state=torch.from_numpy(self.states).float().to(device),
                action=torch.tensor(actions_list, dtype=torch.float, device=device),
                reward=torch.from_numpy(self.rewards).float().to(device),
                next_state=torch.from_numpy(self.next_states).float().to(device),
                done=torch.from_numpy(self.done).bool().to(device)
            )
