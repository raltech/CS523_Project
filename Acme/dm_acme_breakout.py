import acme

import IPython

from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import sonnet as snt

from absl import app
from absl import flags
import acme
from acme.agents.tf import dqn
from acme.tf import networks

import gym

# Imports required for visualization
import pyvirtualdisplay
import imageio
import base64

# Set up a virtual display for rendering.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# helper functions
import functools

from acme import wrappers
import dm_env
import gym

def make_environment(level: str, evaluation: bool = False) -> dm_env.Environment:
  env = gym.make(level, full_action_space=True)

  max_episode_len = 108_000 if evaluation else 50_000

  return wrappers.wrap_all(env, [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=True,
      ),
      wrappers.SinglePrecisionWrapper,
  ])

# Create a simple helper function to render a frame from the current state of
# the environment.
def render(env):
    return env.environment.render(mode='rgb_array')

def display_video(frames, filename='breakout.mp4'):
  """Save and display video."""

  # Write video
  with imageio.get_writer(filename, fps=60) as video:
    for frame in frames:
      video.append_data(frame)

  # Read video and display the video
  video = open(filename, 'rb').read()
  b64_video = base64.b64encode(video)
  video_tag = ('<video  width="320" height="240" controls alt="test" '
               'src="data:video/mp4;base64,{0}">').format(b64_video.decode())

  return IPython.display.HTML(video_tag)

environment = make_environment('BreakoutNoFrameskip-v4')
environment_spec = acme.make_environment_spec(environment)
network = networks.DQNAtariNetwork(environment_spec.actions.num_values)

# Create a logger for the agent and environment loop.
# agent_logger = loggers.TerminalLogger(label='agent', time_delta=0)
agent_logger = loggers.TerminalLogger(label='agent')
# env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=0)
env_loop_logger = loggers.TerminalLogger(label='env_loop')

agent = dqn.DQN(environment_spec, network, logger=agent_logger)

# Create an loop connecting this agent to the environment created above.
env_loop = acme.EnvironmentLoop(
    environment, agent, logger=env_loop_logger)

# Run a `num_episodes` training episodes.
# Rerun this cell until the agent has learned the given task.
env_loop.run(num_episodes=10)

# timestep = environment.reset()
# frames = [render(environment)]

# while not timestep.last():
#   # Simple environment loop.
#   action = agent.select_action(timestep.observation)
#   timestep = environment.step(action)

#   # Render the scene and add it to the frame stack.
#   frames.append(render(environment))

# # Save and display a video of the behaviour.
# display_video(np.array(frames))