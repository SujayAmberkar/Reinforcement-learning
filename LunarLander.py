import gymnasium as gym
from stable_baselines3 import A2C


env = gym.make("LunarLander-v2",render_mode="human")

model = A2C("MlpPolicy",env,verbose=1)

# model.learn(total_timesteps=250000)
# model.save('./LanderModel')

model = A2C.load('LanderModel')

observation, info = env.reset(seed=42)

for _ in range(5000):
    action,_state = model.predict(observation)
    observation,reward, terminated,truncated, info = env.step(action)
    
    if terminated or truncated:
        env.reset()

env.close()