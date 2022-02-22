import numpy as np
import gym
import random 
import time

# Create environment
env = gym.make("FrozenLake-v1")

# Create Q Table
# Row representes the states, cols represents the actions (left, right, up and down)
# q_table = np.zeros((env.action_space.n, env.observation_space.n))
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table= np.zeros((state_space_size, action_space_size))
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration = 1
min_exploration = 0.01
exploration_decay = 0.001

rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    for step in range(max_steps_per_episode):
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])

        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :])) 
        state = new_state
        episode_reward += reward

        if done == True:
            break

    exploration_rate = min_exploration + (max_exploration - min_exploration) * np.exp(-exploration_decay*episode)
    rewards.append(episode_reward)


    # print("***Average rewards per thousand episodes")
    if episode > 0 and episode % 1000 == 0:
        print(episode, ": ", sum(rewards[-1000:])/1000)

# for episode in range(3):
#     # initialize new episode params
#     state = env.reset()
#     done = False
#     time.sleep(1)

#     for step in range(max_steps_per_episode):        
#         # Show current state of environment on screen
#         # Choose action with highest Q-value for current state       
#         # Take new action
#         env.render()
#         time.sleep(0.3)
#         action = np.argmax(q_table[state,:])        
#         new_state, reward, done, info = env.step(action)

#         if done:
#             env.render()
#             if reward == 1:
#                 # Agent reached the goal and won episode
#                 print("***Goal Reached!***")
#                 time.sleep(3)
#                 break
#             else:
#                 # Agent stepped in a hole and lost episode     
#                 print("***Agent fell into the hole***")
#                 time.sleep(3)
#                 break

#         # Set new state
#         state = new_state

env.close()