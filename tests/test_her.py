import pytest

from stable_baselines import HER, DQN, SAC, DDPG
from stable_baselines.her import GoalSelectionStrategy
from stable_baselines.common.bit_flipping_env import BitFlippingEnv

N_BITS = 10


@pytest.mark.parametrize('goal_selection_strategy', list(GoalSelectionStrategy))
def test_dqn_her(goal_selection_strategy):
    env = BitFlippingEnv(N_BITS, continuous=False, max_steps=N_BITS)
    model = HER('MlpPolicy', env, DQN, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                prioritized_replay=False, verbose=1)
    model.learn(5000)
