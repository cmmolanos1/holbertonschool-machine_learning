import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Input
from keras import Model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
# from rl.processors import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

class AtariProcessor(Processor):
    """Class atari processor"""
    def process_observation(self, observation):
        """Recizing and convert to grayscale"""
        # checks if (height, width, channel)
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """We could perform this processing step in `process_observation`.
        In this case, however, we would need to store a `float32` array
        instead, which is 4x more memory intensive than an `uint8` array. This
        matters if we store 1M observations."""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Rewards"""
        return np.clip(reward, -1., 1.)

def create_q_model(num_actions, window):
    """Build model, using the same model that was described by Mnih et al.
    (2015)."""
    inputs = Input(shape=(window, 84, 84))
    inputs_sort = Permute((2, 3, 1))(inputs)
    layer1 = Conv2D(32, 8, strides=4, activation="relu",
                           data_format="channels_last")(inputs_sort)
    layer2 = Conv2D(64, 4, strides=2, activation="relu",
                           data_format="channels_last")(layer1)
    layer3 = Conv2D(64, 3, strides=1, activation="relu",
                           data_format="channels_last")(layer2)

    layer4 = Flatten()(layer3)

    layer5 = Dense(512, activation="relu")(layer4)
    action = Dense(num_actions, activation="linear")(layer5)

    return Model(inputs=inputs, outputs=action)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    window = 4
    model = create_q_model(num_actions, window)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=window)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy,
                   memory=memory, processor=processor,
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.fit(env,
            nb_steps=17500,
            log_interval=10000,
            visualize=False,
            verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
