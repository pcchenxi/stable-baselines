import functools

import gym

from stable_baselines.common import BaseRLModel
from stable_baselines.common import OffPolicyRLModel
from stable_baselines.common.base_class import _UnvecWrapper
from .replay_buffer import HindsightExperienceReplayWrapper, KEY_TO_GOAL_STRATEGY
from .utils import HERGoalEnvWrapper


class HER(BaseRLModel):
    """
    Hindsight Experience Replay (HER) https://arxiv.org/abs/1707.01495

    :param policy: (BasePolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param model_class: (OffPolicyRLModel) The off policy RL model to apply Hindsight Experience Replay
        currently supported: DQN, DDPG, SAC
    :param n_sampled_goal: (int)
    :param goal_selection_strategy: (GoalSelectionStrategy or str)
    """

    def __init__(self, policy, env, model_class, n_sampled_goal=4,
                 goal_selection_strategy='future', *args, **kwargs):

        super().__init__(policy=policy, env=env, verbose=kwargs.get('verbose', 0),
                         policy_base=None, requires_vec_env=False)

        self.model_class = model_class
        self.replay_wrapper = None

        # Convert string to GoalSelectionStrategy object
        if isinstance(goal_selection_strategy, str):
            assert goal_selection_strategy in KEY_TO_GOAL_STRATEGY.keys(), "Unknown goal selection strategy"
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy]

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy

        if self.env is not None:
            self._create_replay_wrapper(self.env)

        assert issubclass(model_class, OffPolicyRLModel),\
            "Error: HER only works with Off policy model (such as DDPG, SAC and DQN)."

        self.model = self.model_class(policy, self.env, *args, **kwargs)
        self.model._save_to_file = self._save_to_file


    def _create_replay_wrapper(self, env):
        # if isinstance(env, VecEnv):
        #     assert isinstance(env, _UnvecWrapper)

        # TODO: check if the env is not already wrapped
        if not isinstance(env, HERGoalEnvWrapper):
            env = HERGoalEnvWrapper(env)

        self.env = env
        # TODO: check for TimeLimit wrapper too
        # TODO: support VecEnv
        # assert isinstance(self.env, gym.GoalEnv), "HER only supports gym.GoalEnv"

        self.replay_wrapper = functools.partial(HindsightExperienceReplayWrapper,
                                                n_sampled_goal=self.n_sampled_goal,
                                                goal_selection_strategy=self.goal_selection_strategy,
                                                wrapped_env=self.env)

    def set_env(self, env):
        # Unwrap VecEnv if needed
        # TODO: save/load correct observation_space
        # which is different between HER and the wrapped env
        # super().set_env(env)
        self._create_replay_wrapper(env)
        self.model.set_env(self.env)

    def get_env(self):
        return self.env

    def __getattr__(self, attr):
        """
        Wrap the RL model.

        :param attr: (str)
        :return: (Any)
        """
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.model, attr)

    def __set_attr__(self, attr, value):
        if attr in self.__dict__:
            setattr(self, attr, value)
        else:
            set_attr(self.model, attr, value)

    def _get_pretrain_placeholders(self):
        return self.model._get_pretrain_placeholders()

    def setup_model(self):
        pass

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="HER",
              reset_num_timesteps=True):
        return self.model.learn(total_timesteps, callback=callback, seed=seed, log_interval=log_interval,
                                tb_log_name=tb_log_name, reset_num_timesteps=reset_num_timesteps,
                                replay_wrapper=self.replay_wrapper)

    def _check_obs(self, observation):
        if isinstance(observation, dict):
            if self.env is not None:
                if len(observation['observation'].shape) > 1:
                    observation = _UnvecWrapper.unvec_obs(observation)
                    return [self.env.convert_dict_to_obs(observation)]
                return self.env.convert_dict_to_obs(observation)
            else:
                raise ValueError("You must either pass an env to HER or wrap your env using HERGoalEnvWrapper")
        return observation

    def predict(self, observation, state=None, mask=None, deterministic=True):
        return self.model.predict(self._check_obs(observation), state, mask, deterministic)

    def action_probability(self, observation, state=None, mask=None, actions=None):
        return self.model.action_probability(self._check_obs(observation), state, mask, actions)

    def _save_to_file(self, save_path, data=None, params=None):
        # HACK to save the replay wrapper
        # or better to save only the replay strategy and its params?
        # it will not work with VecEnv
        data['n_sampled_goal'] = self.n_sampled_goal
        data['goal_selection_strategy'] = self.goal_selection_strategy
        data['model_class'] = self.model_class
        super()._save_to_file(save_path, data, params)

    def save(self, save_path):
        # Is there something more to save? (the replay wrapper?)
        self.model.save(save_path)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, _ = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=env, model_class=data['model_class'],
                    n_sampled_goal=data['n_sampled_goal'],
                    goal_selection_strategy=data['goal_selection_strategy'],
                    _init_setup_model=False)
        # model.__dict__.update(data)
        # model.__dict__.update(kwargs)
        model.model = data['model_class'].load(load_path, model.get_env(), **kwargs)
        model.model._save_to_file = model._save_to_file
        return model
