from enum import Enum
import copy

import numpy as np

from stable_baselines.deepq.replay_buffer import ReplayBuffer


class GoalSelectionStrategy(Enum):
    FUTURE = 0
    FINAL = 1
    EPISODE = 2
    RANDOM = 3


class HindsightExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, size, n_sampled_goal, goal_selection_strategy, env):
        """

        Inspired by https://github.com/NervanaSystems/coach/.

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        :param n_sampled_goal: The number of artificial transitions to generate for each actual transition
        :param goal_selection_strategy: The method that will be used for generating the goals for the
                                                hindsight transitions. Should be one of GoalSelectionStrategy
        :param env:
        """
        super(HER, self).__init__(size)
        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = env
        self.current_episode = []
        self.get_achieved_goal = None

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        self.current_episode.append((obs_t, action, reward, obs_tp1, done))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            self.current_episode = []

    # def _encode_sample(self, idxes):
    #     obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
    #     for i in idxes:
    #         data = self._storage[i]
    #         obs_t, action, reward, obs_tp1, done = data
    #         obses_t.append(np.array(obs_t, copy=False))
    #         actions.append(np.array(action, copy=False))
    #         rewards.append(reward)
    #         obses_tp1.append(np.array(obs_tp1, copy=False))
    #         dones.append(done)
    #     return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
    #

    def _sample_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling method.

        :param episode_transitions: a list of all the transitions in the current episode
        :param transition_idx: the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_obs = np.random.choice(episode_transitions[transition_idx + 1:])
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # The achieved goal at the end of the episode
            selected_obs = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random achieved goal during the episode
            selected_obs = np.random.choice(episode_transitions)
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random achieved goal from the entire replay buffer
            # selected_obs = np.random.choice(self._storage)
            raise NotImplementedError()
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.get_achieved_goal(selected_obs)

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
            super().add(obs_t, action, reward, obs_tp1, done)
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
                reward = self.env.compute_reward(goal, next_obs_dict['achieved_goal'])
                # Can we ensure that done = reward == 0
                done = False

                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add to the replay buffer
                super().add(obs, action, reward, next_obs, done)
