import functools

import gym

from stable_baselines.common import BaseRLModel
from .replay_buffer import HindsightExperienceReplayWrapper, KEY_TO_GOAL_STRATEGY
from .utils import HERGoalEnvWrapper


class HER(BaseRLModel):
    """
    Hindsight Experience replay.
    """

    def __init__(self, policy, env, model_class, n_sampled_goal=4,
                 goal_selection_strategy='future', *args, **kwargs):
        # super().__init__(policy=policy, env=env, verbose=verbose, policy_base=None, requires_vec_env=False)

        self.model_class = model_class
        self.env = env
        assert isinstance(self.env, gym.GoalEnv), "HER only supports gym.GoalEnv"
        self.wrapped_env = HERGoalEnvWrapper(env)
        if isinstance(goal_selection_strategy, str):
            assert goal_selection_strategy in KEY_TO_GOAL_STRATEGY.keys()
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]
        self.replay_wrapper = functools.partial(HindsightExperienceReplayWrapper, n_sampled_goal=n_sampled_goal,
                                                goal_selection_strategy=goal_selection_strategy,
                                                wrapped_env=self.wrapped_env)
        self.model = self.model_class(policy, self.wrapped_env, *args, **kwargs)

    def _get_pretrain_placeholders(self):
        raise NotImplementedError()

    def setup_model(self):
        pass

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="HER",
              reset_num_timesteps=True):
        return self.model.learn(total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="HER",
                                reset_num_timesteps=True, replay_wrapper=self.replay_wrapper)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    def action_probability(self, observation, state=None, mask=None, actions=None):
        pass

    def save(self, save_path):
        pass

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        pass
