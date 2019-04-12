import tensorflow as tf
import numpy as np
import gym

from stable_baselines.common import BaseRLModel, SetVerbosity
from .utils import HERGoalEnvWrapper


class HER(BaseRLModel):
    """
    Hindsight Experience replay.
    """
    def __init__(self, policy, env, model_class, sampling_strategy, get_achieved_goal,
                 verbose=0, _init_setup_model=True):
        # super().__init__(policy=policy, env=env, verbose=verbose, policy_base=None, requires_vec_env=False)

        self.model_class = model_class
        self.env = env
        assert isinstance(self.env, gym.GoalEnv), "HER only supports gym.GoalEnv"
        self.wrapped_env = HERGoalEnvWrapper(env)

        self.model = self.model_class(policy, self.wrapped_env)



    def _get_pretrain_placeholders(self):
        raise NotImplementedError()


    def setup_model(self):
        pass

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="HER",
              reset_num_timesteps=True):
        with SetVerbosity(self.verbose):
            self._setup_learn(seed)

        return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    def action_probability(self, observation, state=None, mask=None, actions=None):
        pass

    def save(self, save_path):
        pass

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        pass
