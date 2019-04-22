import numpy as np
from gym import GoalEnv, spaces


class BitFlippingEnv(GoalEnv):
    """
    Simple bit flipping env, useful to test HER.
    The goal is to flip all the bits to get a vector of ones.
    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.

    :param n_bits: (int) Number of bits to flip
    :param continuous: (bool) Wether to use the continuous version or not
    :param max_steps: (int) Max number of steps, by defaults, equal to n_bits
    """
    def __init__(self, n_bits=10, continuous=False, max_steps=None):
        super(BitFlippingEnv, self).__init__()
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        self.observation_space = spaces.Dict({
            'observation': spaces.MultiBinary(n_bits),
            'achieved_goal': spaces.MultiBinary(n_bits),
            'desired_goal': spaces.MultiBinary(n_bits)
        })
        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)
        self.continuous = continuous
        self.state = None
        self.desired_goal = np.ones((n_bits,))
        if max_steps is None:
            max_steps = n_bits
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def reset(self):
        self.current_step = 0
        self.state = self.observation_space.spaces['observation'].sample()
        return {
            'observation': self.state.copy(),
            'achieved_goal': self.state.copy(),
            'desired_goal': self.desired_goal.copy()
        }

    def step(self, action):
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = {
            'observation': self.state.copy(),
            'achieved_goal': self.state.copy(),
            'desired_goal': self.desired_goal.copy()
        }
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], None)
        done = (obs['achieved_goal'] == obs['desired_goal']).all()
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {'is_success': done}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, _info):
        # Deceptive reward: it is positive only when the goal is achieved
        return 0 if (achieved_goal == desired_goal).all() else -1

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.state.copy()
        print(self.state)

    def close(self):
        pass
