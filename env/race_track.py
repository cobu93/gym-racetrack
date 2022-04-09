import gym
from .rtspaces.action import RaceTrackMovementsSpace, RaceTrackAction
import random
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image
import time

from .track import test_track

class RaceTrackEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {
    "render.modes": [
      "human"
    ]
  }

  OUT_CODE = 1
  TRACK_CODE = 2
  OIL_CODE = 3
  GOAL_CODE = 4
  CAR_CODE = 5

  VALID_CELL_VALUES = [
    OUT_CODE,
    TRACK_CODE,
    OIL_CODE,
    GOAL_CODE
    ]  

  def __init__(
    self, 
    max_speed: int = 4,
    min_speed: int = 1,
    accepted_incrementals: List[int] = [-1, 0, 1],
    track: List[List[int]] = test_track,
    keep_car_trace: bool = False,
    render_delay: int = 1,
    color_map: Dict[int, Tuple] = {
      OUT_CODE: (148, 203, 124),
      TRACK_CODE: (185, 185, 185),
      OIL_CODE: (70, 70, 70),
      GOAL_CODE: (225, 155, 51),
      CAR_CODE: (51, 155, 225)
    }):

    super(RaceTrackEnv, self).__init__()

    self.keep_car_trace = keep_car_trace
    self.render_delay = render_delay
    self.color_map = color_map
        
    ########################################
    #
    # Track validation
    #
    ########################################

    if len(track) <= 0:
        raise ValueError("The track must have at least one row")

    self.box_height = len(track)
    self.box_width = len(track[0])

    for row_idx, row in enumerate(track):
        if len(row) != self.box_width:
            raise ValueError(f"All rows in the track must have the same dimension. Row {row_idx} has {len(row)} and must have {self.box_width}")

        for column_idx, cell in enumerate(row):
            if cell not in RaceTrackEnv.VALID_CELL_VALUES:
                raise ValueError(f"Cell in position ({row_idx}, {column_idx}) hasn't a valid value: {cell}")
    
    self.track = np.array(track)

    ########################################
    #
    # Action space
    #
    ########################################

    self.action_space = RaceTrackMovementsSpace(
      max_speed=max_speed,
      min_speed=min_speed,
      accepted_incrementals=accepted_incrementals
    )

    ########################################
    #
    # Control variables
    #
    ########################################
    self.current_speed = min_speed
    self.current_state = None
    self.last_action = None
    self.current_observation = None
    self.ignore_next_action = False
    
    self.observation_image = np.zeros((*self.track.shape, 3), dtype=np.uint8)

    for row_idx, row in enumerate(self.track):
      for column_idx, value in enumerate(row):
        self.observation_image[row_idx, column_idx] = self.color_map[value]

    self.reset()

    
  def step(
    self, 
    action: RaceTrackAction
  ):
    info = {}

    info["keep_car_trace"] = self.keep_car_trace

    if not self.keep_car_trace:
      value = self.track[self.current_state]
      self.current_observation[self.current_state] = value
      self.observation_image[self.current_state[0], self.current_state[1]] = self.color_map[value]

    info["ignored_action"] = self.ignore_next_action

    if self.ignore_next_action:
      action = self.last_action
      action.action = 0
      self.ignore_next_action = False

    
    info["speed_increment"] = action.action

    current_speed = self.current_speed + action.action
    self.current_speed = min(self.action_space.max_speed, max(self.action_space.min_speed, current_speed))

    action.speed = self.current_speed

    if not self.action_space.contains(action):
      raise ValueError("Given action is not valid")
    
    info["current_speed"] = self.current_speed

    next_position = (
      self.current_state[0] + action.vertical_moves,
      self.current_state[1] + action.horizontal_moves
    )

    info["from_position"] = self.current_state
    info["to_position"] = next_position
    
    info["horizontal_moves"] = action.horizontal_moves
    info["vertical_moves"] = action.vertical_moves

    if action.horizontal_moves == 0:
      crossed_states = self.track[
        min(self.current_state[0], next_position[0]):max(self.current_state[0], next_position[0]) + 1,
        self.current_state[1]:self.current_state[1] + 1
      ].flatten()

    elif action.vertical_moves == 0:
      crossed_states = self.track[
        self.current_state[0]:self.current_state[0] + 1,
        min(self.current_state[1], next_position[1]):max(self.current_state[1], next_position[1]) + 1
      ].flatten()
    else:
      crossed_states = self.track[
        min(self.current_state[0], next_position[0]):max(self.current_state[0], next_position[0]) + 1,
        min(self.current_state[1], next_position[1]):max(self.current_state[1], next_position[1]) + 1
      ].flatten()

    info["crossed_states"] =  crossed_states
    
    self.current_state = next_position

    # If win
    if RaceTrackEnv.GOAL_CODE in crossed_states:
      info["status"] = "Goal crossed"
      return self.current_state, 0, True, info
    
    # If out of track
    if next_position[1] >= self.track.shape[1] \
      or next_position[1] < 0 \
      or next_position[0] >= self.track.shape[0] \
      or next_position[0] < 0 \
      or RaceTrackEnv.OUT_CODE in crossed_states:
      
      if RaceTrackEnv.OUT_CODE in crossed_states \
        and next_position[1] < self.track.shape[1] \
        and next_position[1] >= 0 \
        and next_position[0] < self.track.shape[0] \
        and next_position[0] >= 0:
        self.current_observation[next_position] = RaceTrackEnv.CAR_CODE

      info["status"] = "Out of track"
      return self.current_state, -1, True, info

    # If oil
    # If out of track
    if RaceTrackEnv.OIL_CODE in crossed_states:
      info["status"] = "Oil crossed"
      self.current_observation[next_position] = RaceTrackEnv.CAR_CODE
      self.last_action = action
      self.ignore_next_action = True
      return self.current_state, -1, False, info

    info["status"] = "Regular movement"
    self.current_observation[next_position] = RaceTrackEnv.CAR_CODE
    
    return self.current_state, -1, False, info
    

  def reset(self):
    self.current_speed = self.action_space.min_speed

    self.current_state = (
      self.box_height - 1,
      random.choice(np.where(self.track[-1] == RaceTrackEnv.TRACK_CODE)[0])      
    )

    self.last_action = None
    self.current_observation = np.copy(self.track)
    self.current_observation[self.current_state] = RaceTrackEnv.CAR_CODE
    self.ignore_next_action = False

    self.observation_image = np.zeros((*self.track.shape, 3), dtype=np.uint8)

    for row_idx, row in enumerate(self.track):
      for column_idx, value in enumerate(row):
        self.observation_image[row_idx, column_idx] = self.color_map[value]
    
  def render(self, mode="human", close=False):
    car_indices = np.where(self.current_observation == RaceTrackEnv.CAR_CODE)
    self.observation_image[car_indices] = self.color_map[RaceTrackEnv.CAR_CODE]
    image = Image.fromarray(self.observation_image, "RGB")
    time.sleep(self.render_delay)
    return image

