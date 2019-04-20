import pytest

from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.her import GoalSelectionStrategy
from stable_baselines.common.bit_flipping_env import BitFlippingEnv

N_BITS = 10

@pytest.mark.parametrize('goal_selection_strategy', list(GoalSelectionStrategy))
@pytest.mark.parametrize('model_class', [DQN, SAC, DDPG])
def test_her(model_class, goal_selection_strategy):
    env = BitFlippingEnv(N_BITS, continuous=model_class in [DDPG, SAC], max_steps=N_BITS)

    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=0)
    model.learn(1000)
