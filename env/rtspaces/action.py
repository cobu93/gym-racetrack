import gym
import random
from typing import List

class RaceTrackAction():

    def __init__(
        self, 
        action: int,
        horizontal_moves: int, 
        vertical_moves: int,
        speed: int = -1,
    ):
        self.speed = speed
        self.action = action
        self.horizontal_moves= horizontal_moves
        self.vertical_moves = vertical_moves

    def __repr__(self):
        return f"speed:{self.speed} - action:{self.action} - horizontal_moves:{self.horizontal_moves} - vertical_moves:{self.vertical_moves}"


class RaceTrackMovementsSpace(gym.Space):
  
    def __init__(
        self,
        *args,
        max_speed: int = 4,
        min_speed: int = 1,
        accepted_incrementals: List[int] = [-1, 0, 1],
        **kwargs
    ):  
        super(RaceTrackMovementsSpace, self).__init__(*args, **kwargs)
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.accepted_incrementals = accepted_incrementals
    
    def sample(self, current_speed) -> RaceTrackAction:
        """Randomly sample an element of this space. Can be
        uniform or non-uniform sampling based on boundedness of space."""
        speed_action = random.choice(self.accepted_incrementals)
        current_speed += speed_action
        current_speed = min(self.max_speed, max(self.min_speed, current_speed))

        vertical_moves = random.randint(0, current_speed)
        horizontal_moves = current_speed - vertical_moves
        horizontal_moves *= random.choice([-1, 1])

        return RaceTrackAction( 
            action=speed_action,
            horizontal_moves=horizontal_moves, 
            vertical_moves=-vertical_moves,
            speed=current_speed
        )
       
    def contains(self, x: RaceTrackAction) -> bool:
        if x.action not in self.accepted_incrementals:
            return False

        if x.speed > self.max_speed or x.speed < self.min_speed:
            return False

        if x.vertical_moves > 0:
            return False

        if abs(x.vertical_moves) + abs(x.horizontal_moves) != x.speed:
            return False
        
        return True

        


