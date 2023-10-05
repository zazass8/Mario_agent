import numpy as np
import matplotlib.pyplot as plt

from IPython.display import HTML
from IPython import display as ipythondisplay
import glob
import io
import base64
from gym.wrappers import Monitor
from pyvirtualdisplay import Display

from preprocessing import define_env


def plot(q_learning_rewards):
  """Plot the moving averages"""

  avg_episodes = 100
  q_learning_rewards2 = []
  q_learning_rewards = np.array(q_learning_rewards)
  for i in range(q_learning_rewards.shape[0] - avg_episodes):
      q_learning_rewards2.append(np.mean(q_learning_rewards[i:(i+avg_episodes)]))
  plt.figure(figsize = (12, 6))
  plt.plot(range(len(q_learning_rewards2)), q_learning_rewards2)
  plt.xlabel("Episodes of Training")
  plt.ylabel("Average Undiscounted Return")
  plt.grid()
  plt.show()
  q_learning_rewards = q_learning_rewards.tolist()
  return


def show_video():
  display = Display(visible=0, size=(1400, 900))
  display.start()
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")


def video():
  """Create video for the trained agent"""
  
  global env
  global agent
  env = Monitor(env, './video', force=True, mode = 'evaluation')
  state=env.reset()
  for _ in range(5000):
    env.render()
    action = agent.training_act(state, action_dim=env.action_space.n)
    next_state, reward, done, info = env.step(action)
    if done:
      break
  env.close()
  show_video()
  env=define_env()
  return 