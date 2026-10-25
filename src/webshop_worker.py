"""Shared disjoint worker contract; only the native-class loading transport differs."""
import numpy as np
from task_split import manifest, partition_goals, assert_allowed, task_identity
from hint_webshop.environment import native_parts
NativeWebshopWorker = native_parts()[0]


class DisjointWebshopWorker(NativeWebshopWorker):
    def __init__(self, seed, env_kwargs, is_train):
        self.is_train = bool(is_train)
        self.worker_seed = int(seed)
        super().__init__(seed, dict(env_kwargs))
        server = self.env.unwrapped.server
        server.goals = partition_goals(server.goals, self.is_train)
        server.weights = [goal['weight'] for goal in server.goals]
        server.cum_weights = [0] + np.cumsum(server.weights).tolist()
        server.user_sessions.clear()
        self.env.reset(session=0)
        self._current_id()

    def _current_id(self):
        env = self.env.unwrapped
        return assert_allowed(env.server.user_sessions[env.session]['goal'], self.is_train)

    def reset(self, idx):
        if not isinstance(idx, (int,np.integer)) or not 0 <= idx < len(self.env.unwrapped.server.goals):
            raise RuntimeError('Task index outside this split')
        observation, info = super().reset(int(idx))
        info.update(task_id=self._current_id(), task_split='train' if self.is_train else 'validation')
        return observation, info

    def step(self, action):
        current = self._current_id()
        observation, reward, done, info = super().step(action)
        self._current_id()
        info.update(task_id=current, task_split='train' if self.is_train else 'validation')
        return observation, reward, done, info

    def get_split_audit(self):
        goals = self.env.unwrapped.server.goals
        ids = {task_identity(goal) for goal in goals}
        validation = set(manifest()['validation_task_ids'])
        overlap = len(ids & validation) if self.is_train else len(ids - validation)
        return {'is_train':self.is_train, 'worker_seed':self.worker_seed, 'goal_entries':len(goals),
                'unique_task_families':len(ids), 'opposite_split_overlap':overlap}
