import gym
import absl.logging as log
import numpy as np
from tqdm import tqdm
from os.path import exists
from numba import jit

from config import GYM_ENVIRONMENT, LEARNING_RATE, DISCOUNT, EPISODES, SHOW_EVERY, DISCRETE_SIZE, EPSILON, \
    START_EPSILON_DECAYING, END_EPSILON_DECAYING, EPSILON_DECAY_VALUE

log.set_verbosity(log.INFO)


def compute_discrete_state_diff(state: np.ndarray, env_low: np.ndarray) -> np.ndarray:
    return state[0] - env_low


@jit(nopython=True)
def get_discrete_state_scaled(diff: np.ndarray, discrete_os_win_size: np.ndarray) -> np.ndarray:
    return (diff / discrete_os_win_size).astype(np.int_)


class SimpleQLearning:
    def __init__(self) -> None:
        log.info("Initialize Simple QLearning")
        self.env = gym.make(GYM_ENVIRONMENT)

        discrete_os_size = ([DISCRETE_SIZE] * len(self.env.observation_space.high))
        self.discrete_os_win_size = ((self.env.observation_space.high - self.env.observation_space.low) / discrete_os_size) - 1

        self.ep_rewards = []
        self.aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

        self.save_path = r'simpleQLearningModel'
        if exists(self.save_path + '.npy'):
            self.best_game = np.load(self.save_path + '.npy', allow_pickle=True).item()
            self.q_table = self.best_game['qtable']
            self.epsilon = self.best_game['epsilon']
        else:
            self.best_game = None
            self.q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [self.env.action_space.n]))
            self.epsilon = EPSILON

        self.random_action_probabilities = np.random.random(size=EPISODES)

    def train(self) -> None:
        for episode in tqdm(range(EPISODES)):
            episode_reward = 0
            diff = compute_discrete_state_diff(self.env.reset(), self.env.observation_space.low)
            discrete_state = get_discrete_state_scaled(diff, self.discrete_os_win_size)
            done = False

            if episode % SHOW_EVERY == 0:
                render = True
            else:
                render = False

            while not done:
                if self.random_action_probabilities[episode] > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.q_table[discrete_state])
                else:
                    # Get random action
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, done, _, () = self.env.step(action)

                episode_reward += reward

                new_diff = compute_discrete_state_diff(new_state, self.env.observation_space.low)
                new_discrete_state = get_discrete_state_scaled(new_diff, self.discrete_os_win_size)

                if render:
                    self.env.render()

                # If simulation did not end yet after last step - update Q table
                if not done:
                    # Maximum possible Q value in next step (for new state)
                    max_future_q = np.max(self.q_table[new_discrete_state])

                    # Current Q value (for current state and performed action)
                    current_q = self.q_table[discrete_state + (action,)]

                    # And here's our equation for a new Q value for current state and action
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                    # Update Q table with new Q value
                    self.q_table[discrete_state + (action,)] = new_q

                # Simulation ended (for any reason) - if goal position is achieved - update Q value with reward directly
                elif new_state[0] >= self.env.goal_position:
                    self.q_table[discrete_state + (action,)] = 0

                discrete_state = new_discrete_state

            # Decaying is being done every episode if episode number is within decaying range
            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                self.epsilon -= EPSILON_DECAY_VALUE

            self.ep_rewards.append(episode_reward)

            if self.best_game is None:
                self.best_game = {'qtable': self.q_table, 'reward': episode_reward, 'episode_number': episode,
                                  'epsilon': self.epsilon}
            else:
                if episode_reward > self.best_game['reward']:
                    self.best_game['qtable'] = self.q_table
                    self.best_game['reward'] = episode_reward
                    self.best_game['episode_number'] = episode
                    self.best_game['epsilon'] = self.epsilon

            if not episode % SHOW_EVERY:
                average_reward = sum(self.ep_rewards[-SHOW_EVERY:]) / SHOW_EVERY
                self.aggr_ep_rewards['ep'].append(episode)
                self.aggr_ep_rewards['avg'].append(average_reward)
                self.aggr_ep_rewards['max'].append(max(self.ep_rewards[-SHOW_EVERY:]))
                self.aggr_ep_rewards['min'].append(min(self.ep_rewards[-SHOW_EVERY:]))
                log.info(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: '
                         f'{self.epsilon:>1.2f}')

        np.save(self.save_path, self.best_game)

    def play_best_game(self):
        log.info('Rendering best founded game...')
        log.info(f"Best model -> epsilon: {self.best_game['epsilon']}, reward: {self.best_game['reward']}")
        played_reward = 0
        diff = compute_discrete_state_diff(self.env.reset(), self.env.observation_space.low)
        discrete_state = get_discrete_state_scaled(diff, self.discrete_os_win_size)
        done = False

        while not done:
            action = np.argmax(self.q_table[discrete_state])

            new_state, reward, done, _, () = self.env.step(action)

            played_reward += reward

            if SHOW_EVERY == 0:
                self.env.render()

            new_diff = compute_discrete_state_diff(new_state, self.env.observation_space.low)
            new_discrete_state = get_discrete_state_scaled(new_diff, self.discrete_os_win_size)

            discrete_state = new_discrete_state
        log.info(f"Played reward: {played_reward}")

    def close_environment(self) -> None:
        self.env.close()
