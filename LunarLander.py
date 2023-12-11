import gymnasium as gym
from stable_baselines3 import A2C,DQN,PPO


env = gym.make("CartPole-v1",render_mode="rgb_array")

model = A2C("MlpPolicy",env,verbose=1,tensorboard_log='./DQNlogs_tensor')

model.learn(total_timesteps=100000)
model.save('./CartA2C')

# model = A2C.load('CartDQN')

observation, info = env.reset(seed=42)

for _ in range(5000):
    action,_state = model.predict(observation)
    observation,reward, terminated,truncated, info = env.step(action)
    
    if terminated or truncated:
        env.reset()

env.close()