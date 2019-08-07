import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
from collections import deque
import random
import absl.logging as log
from tqdm import tqdm
import gym
from os.path import exists
from config import GYM_ENVIROMENT, EPSILON, MODEL_NAME, DISCOUNT, DEEP_EPISODES, AGGREGATE_STATS_EVERY, MIN_EPSILON, \
    EPSILON_DECAY_VALUE, UPDATE_TARGET_EVERY, REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE


log.set_verbosity(log.INFO)


class DeepQLearning:
    def __init__(self):
        log.info("Initialize Deep QLearning")
        self.env = gym.make(GYM_ENVIROMENT)
        self.model = self.create_model()

        if exists(f'{MODEL_NAME}.model'):
            self.model = load_model(f'{MODEL_NAME}.model')

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0

        self.ep_rewards = []

    def create_model(self):
        model = Sequential()

        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))

        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train_on_batch(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        mini_batch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def play_games(self, n_games: int = 5):
        log.info(f"Start playing {n_games}...")
        for episode in range(1, n_games+1):
            episode_reward = 0
            step = 1

            current_state = self.env.reset()

            done = False
            while not done:
                action = np.argmax(self.get_qs(current_state))

                new_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                self.env.render()

                self.update_replay_memory((current_state, action, reward, new_state, done))
                self.train_on_batch(done)

                current_state = new_state
                step += 1
            log.info(f"Episode {episode} reward: {episode_reward}")

    def train(self):
        epsilon = EPSILON
        for episode in tqdm(range(1, DEEP_EPISODES + 1), ascii=True, unit='episodes'):
            episode_reward = 0
            step = 1

            current_state = self.env.reset()

            done = False
            while not done:
                if np.random.random() > epsilon:
                    action = np.argmax(self.get_qs(current_state))
                else:
                    action = np.random.randint(0, self.env.action_space.n)

                new_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                if not episode % AGGREGATE_STATS_EVERY:
                    self.env.render()

                self.update_replay_memory((current_state, action, reward, new_state, done))
                self.train_on_batch(done)

                current_state = new_state
                step += 1

            if episode > 1 and episode_reward > max(self.ep_rewards):
                self.model.save(f'{MODEL_NAME}.model')

            self.ep_rewards.append(episode_reward)

            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                min_reward = min(self.ep_rewards)
                max_reward = max(self.ep_rewards)
                avg_reward = sum(self.ep_rewards)/len(self.ep_rewards)

                log.info(f"Statistics:\nMin reward: {min_reward}\nMax reward: {max_reward}\nMean reward: {avg_reward}")

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY_VALUE
                epsilon = max(MIN_EPSILON, epsilon)

    def close_environment(self) -> None:
        self.env.close()
