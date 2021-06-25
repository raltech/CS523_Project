# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for different experiment flavours."""

import functools

from acme import wrappers
import dm_env
import gym

def make_environment(evaluation: bool = False,
                     level: str = 'PongNoFrameskip-v4') -> dm_env.Environment:
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
environment_library = 'gym'
if environment_library == 'dm_control':
  def render(env):
    return env.physics.render(camera_id=0)
elif environment_library == 'gym':
  def render(env):
    return env.environment.render(mode='rgb_array')
else:
  raise ValueError("choose among ['dm_control', 'gym'].")

def display_video(frames, filename='temp.mp4'):
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