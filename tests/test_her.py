import os

import pytest

from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.her import GoalSelectionStrategy
from stable_baselines.her.replay_buffer import KEY_TO_GOAL_STRATEGY
from stable_baselines.common.bit_flipping_env import BitFlippingEnv
from stable_baselines.common.vec_env import DummyVecEnv

N_BITS = 10


@pytest.mark.parametrize('goal_selection_strategy', list(GoalSelectionStrategy))
@pytest.mark.parametrize('model_class', [DQN, SAC, DDPG])
def test_her(model_class, goal_selection_strategy):
    env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC], max_steps=N_BITS)

    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=0)
    model.learn(1000)


@pytest.mark.parametrize('goal_selection_strategy', [list(KEY_TO_GOAL_STRATEGY.keys())[0]])
@pytest.mark.parametrize('model_class', [DQN, SAC, DDPG])
def test_model_manipulation(model_class, goal_selection_strategy):
    env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC], max_steps=N_BITS)

    model = HER('MlpPolicy', env, model_class, n_sampled_goal=3, goal_selection_strategy=goal_selection_strategy,
                verbose=0)
    model.learn(1000)

    model.save('./test_her')

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    del model

    model = HER.load('./test_her')
    model.set_env(env)
    model.learn(1000)

    env = model.get_env()
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    assert model.n_sampled_goal == 3

    del model

    model = HER.load('./test_her', env=env)
    model.learn(1000)

    assert model.n_sampled_goal == 3

    if os.path.isfile('./test_her.pkl'):
        os.remove('./test_her.pkl')


@pytest.mark.parametrize('goal_selection_strategy', [list(KEY_TO_GOAL_STRATEGY.keys())[0]])
@pytest.mark.parametrize('model_class', [DQN, SAC, DDPG])
def test_her_vec_env(model_class, goal_selection_strategy):
    env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC], max_steps=N_BITS)
    # Test vec env
    env = DummyVecEnv([lambda: env])

    model = HER('MlpPolicy', env, model_class, n_sampled_goal=3, goal_selection_strategy=goal_selection_strategy,
                verbose=0)
    model.learn(1000)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

#     model.save('./test_her')
#     del model
#
#     # TODO: wrap into an _UnvecWrapper if needed
#     # model = HER.load('./test_her')
#     # model.set_env(env)
#     # model.learn(1000)
#     #
#     # assert model.n_sampled_goal == 3
#     #
#     # del model
#
#     env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC], max_steps=N_BITS)
#     model = HER.load('./test_her', env=env)
#     model.learn(1000)
#
#     assert model.n_sampled_goal == 3
#
#     if os.path.isfile('./test_her.pkl'):
#         os.remove('./test_her.pkl')
