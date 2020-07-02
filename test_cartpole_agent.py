from agent import Agent
import gym
import tensorflow as tf
import os

env = gym.make("CartPole-v0")
env.seed(1)

n_actions = env.action_space.n
file_name = 'model/cartpole_model.h5'

os.makedirs('model', exist_ok = True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=n_actions)
])

if os.path.isfile(file_name):
    model.load_weights(file_name)

cartpole_agent = Agent(model, env)
cartpole_agent.train_loop(model_file = file_name)