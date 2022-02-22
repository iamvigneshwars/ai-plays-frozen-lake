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
max_steps_per_episode = 1000

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration = 1
min_exploration = 0.01
exploration_decay = 0.001

# Total rewards
rewards = []
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    for step in range(max_steps_per_episode):
        exploration_threshold = random.uniform(0, 1)
        # If the exploration rate is greater than the threshold,
        # exploit the available information about the environment.
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])

        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        # Calculate the new q value for current state, action pair.
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :])) 
        state = new_state
        episode_reward += reward

        if done == True:
            break

    # Decay exploration rate
    exploration_rate = min_exploration + (max_exploration - min_exploration) * np.exp(-exploration_decay*episode)
    rewards.append(episode_reward)

    if episode > 0 and episode % 1000 == 0:
        print("***Average rewared after ",episode, " episodes : ", sum(rewards[-1000:])/1000, "***")

# for episode in range(3):
#     # Reset the environment to play the game.
#     state = env.reset()
#     done = False

#     for step in range(max_steps_per_episode):        
#         env.render()
#         time.sleep(0.3)
#         # Select the best action for the current state.
#         action = np.argmax(q_table[state,:])        
#         new_state, reward, done, info = env.step(action)

#         if done:
#             env.render()
#             if reward == 1:
#                 # If the agent reached the goal
#                 print("Episode :",episode)
#                 print("***Goal Reached!***")
#                 time.sleep(2)
#                 break
#             else:
#                 # If the agent stepping onto the hole.
#                 print("Episode :",episode)
#                 print("***Agent fell into the hole***")
#                 time.sleep(2)
#                 break

#         # Set new state
#         state = new_state

env.close()