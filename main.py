import numpy as np
import matplotlib.pyplot as plt

from IPython.display import HTML
from IPython import display as ipythondisplay
from gym.wrappers import Monitor

from pyvirtualdisplay import Display

from agents import Deep_Agent
from preprocessing import define_env
from visuals import video
from visuals import plot


def learn():
    """Train the Mario agent"""

    env = define_env()
    algo = "duelddqn"
    agent = Deep_Agent(state_dim = (4, 84, 84), action_dim = env.action_space.n, algo = algo)
    get_xp_steps = 30000 #Size of replay buffer
    state = env.reset()

    for i in range(get_xp_steps):
        if(i % 10000 == 0):
            print(i)
        action = np.random.randint(env.action_space.n)

        if i%200==0:
            agent.explo *= agent.explo_decay  #Reduce exploration rate at every episode

        if(agent.explo < agent.explo_min):
            agent.explo = agent.explo_min

        next_state, reward, done, info = env.step(action)
        agent.get_xp(xp=(state, next_state, action, reward, done))

        if done or info['flag_get']:
            state = env.reset()
            
    print("Begin training...")

    episodes = 10000
    q_learning_rewards = []
    flag=[]

    for e in range(episodes):

        if (e % 20== 0):
            print(e)

        if (e % 100 == 0):
            plot(q_learning_rewards)
            print(agent.explo)

        state = env.reset()
        actions=[]

        ep_reward = 0.0
        ep_length = 0

        while True:
            action = agent.training_act(state, action_dim=env.action_space.n)
            actions.append(action)
            next_state, reward, done, info = env.step(action)
            agent.get_xp(xp=(state, next_state, action, reward, done))

            if(agent.counter % 5000 == 0):  #Sync target network network with online
                agent.sync_target_q()

            if(agent.counter % 3 == 0):  #Update weights of online network
                agent.learn()
                
            ep_reward += reward
            ep_length += 1
            state = next_state

            if done or info['flag_get']:
                q_learning_rewards.append(ep_reward)
                break
        
        agent.explo *= agent.explo_decay

        if(agent.explo < agent.explo_min):
            agent.explo = agent.explo_min

        if info['flag_get']:
            flag.append(1)

        else:
            flag.append(0)
            
        if ep_reward>max_reward:
            max_reward=ep_reward

    agent.sync_target_q()

    video()  # Create video for agent

    plot()  # Create moving average plot


if __name__ == "__learn__":
    learn()
