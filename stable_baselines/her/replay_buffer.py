import copy
from enum import Enum

import numpy as np


class GoalSelectionStrategy(Enum):
    FUTURE = 0
    FINAL = 1
    EPISODE = 2
    RANDOM = 3


KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}


class HindsightExperienceReplayWrapper(object):
    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env):
        """
        Inspired by https://github.com/NervanaSystems/coach/.

        :param n_sampled_goal: The number of artificial transitions to generate for each actual transition
        :param goal_selection_strategy: The method that will be used for generating the goals for the
                                                hindsight transitions. Should be one of GoalSelectionStrategy
        :param wrapped_env:
        """
        super(HindsightExperienceReplayWrapper, self).__init__()
        self.n_sampled_goal = n_sampled_goal
        assert isinstance(goal_selection_strategy, GoalSelectionStrategy)
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        self.current_episode = []
        self.replay_buffer = replay_buffer

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        self.current_episode.append((obs_t, action, reward, obs_tp1, done))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            self.current_episode = []

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)

    def __len__(self):
        return len(self.replay_buffer)

    def _sample_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling method.

        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_idx: the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # The achieved goal at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random achieved goal during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random achieved goal from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer._storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goal according to the sampling strategy.

        :param episode_transitions: () a list of all the transitions in the current episode
        :param transition_idx: the transition to start sampling from
        :return: a goal corresponding to the sampled obs
        """
        return [
            self._sample_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        last_episode_transitions = copy.deepcopy(self.current_episode)

        # for each transition in the last episode, create a set of hindsight transitions
        for transition_idx, transition in enumerate(last_episode_transitions):

            obs_t, action, reward, obs_tp1, done = transition
            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)
            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(last_episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_goals(last_episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                obs, action, reward, next_obs, done = copy.deepcopy(transition)

                # Convert concatenated obs to dict
                # so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # update the reward and terminal signal according to the goal
                reward = self.env.compute_reward(goal, next_obs_dict['achieved_goal'], None)
                # Can we ensure that done = reward == 0
                done = False

                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)
