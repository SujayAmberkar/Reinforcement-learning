import gymnasium as gym
from stable_baselines3 import A2C


env = gym.make("CartPole-v1", render_mode="human")

model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=250000)
# model.save("./MountainCar")

# # del model # remove to demonstrate saving and loading

model = A2C.load("./a2c_cartpole")

observation, info = env.reset(seed=42)

# print(env.action_space.sample)

for _ in range(10000):
   # action = env.action_space.sample()
   action,_states = model.predict(observation)  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   # print(env.step(action)[1])
   if terminated or truncated:
      observation, info = env.reset()

env.close()