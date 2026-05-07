from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
from entities import Skeleton

class ThrowerDetector(ABC):
    @abstractmethod
    def detect(self, obj_frames):
        pass

def filter_obj_frame(func, obj_frame):
    return list(filter(func, obj_frame))

@dataclass
class BallProximityThrowerDetector(ThrowerDetector):
    ball_filter: List[str] = field(default_factory=lambda: ["ball"])

    def detect(self, obj_frames):
        thrower_ids = {}
        relevant_obj_frames = []
        for obj_frame in obj_frames:
            for obj in obj_frame:
                if any(word in obj.name for word in self.ball_filter):
                    relevant_obj_frames.append(obj_frame)
                    break

        for obj_frame in relevant_obj_frames:
            balls = filter_obj_frame(
                lambda x: any(word in x.name for word in self.ball_filter), obj_frame
            )
            players = filter_obj_frame(
                lambda x: not any(word in x.name for word in self.ball_filter),
                obj_frame,
            )

            for ball in balls:
                smallest_dist = float("inf")
                closest_player = None
                for player in players:
                    dist = player.distance_to(ball)
                    if dist < smallest_dist:
                        smallest_dist = dist
                        closest_player = player

                if closest_player is not None:
                    thrower_ids[closest_player] = None

        return list(thrower_ids.keys())

@dataclass
class ActionThrowerDetector(ThrowerDetector):
    action_filter: List[str] = field(default_factory=lambda: ["jump-shot"])

    def detect(self, obj_frames):
        throwers = {}

        for obj_frame in obj_frames:
            for obj in obj_frame:
                if self.action_filter and obj.action:
                    if any(word in obj.action for word in self.action_filter):
                        throwers[obj] = None

        return list(throwers.keys())

@dataclass
class CombinedThrowerDetector(ThrowerDetector):
    ball_filter: List[str] = field(default_factory=lambda: ["ball"])
    action_filter: List[str] = field(default_factory=lambda: ["jump-shot"])

    def __post_init__(self):
        self.ball_detector = BallProximityThrowerDetector(self.ball_filter)
        self.action_detector = ActionThrowerDetector(self.action_filter)

    def detect(self, obj_frames):
        all_balls = set()

        for obj_frame in obj_frames:
            for obj in obj_frame:
                if any(word in obj.name for word in self.ball_filter) and obj.id != -1:
                    all_balls.add(obj.id)

        shooters = self.action_detector.detect(obj_frames)
        players_with_ball = self.ball_detector.detect(obj_frames)

        num_balls = len(all_balls)
        valid_shooters = [s for s in shooters if s.id != -1]
        num_shooters = len(valid_shooters)

        if num_balls == 0 and num_shooters > 0:
            return [shooters[0]]
        elif num_balls == 1:
            overlap = [p for p in players_with_ball if p in shooters]
            return overlap if overlap else players_with_ball
        elif num_balls > 1 and num_shooters == 1:
            return shooters
        elif num_balls > 1 and num_shooters > 1:
            overlap = [p for p in players_with_ball if p in shooters]
            return (
                overlap
                if overlap
                else (players_with_ball if players_with_ball else shooters)
            )

        return []

class ReleaseDetector(ABC):
    @abstractmethod
    def detect(self, obj_frames, fps: int) -> int:
        pass

@dataclass
class ActionReleaseDetector(ReleaseDetector):
    def detect(self, obj_frames, fps: int) -> int:
        required_frames = fps * 0.2
        consecutive = 0
        for i, obj_frame in enumerate(obj_frames):
            has_jump_shot = any(
                getattr(obj, "action", "") == "jump-shot" for obj in obj_frame
            )
            if has_jump_shot:
                consecutive += 1
                if consecutive >= required_frames:
                    return i
            else:
                consecutive = 0
        return -1

@dataclass
class SkeletonReleaseDetector(ReleaseDetector):
    def detect(self, obj_frames, fps: int) -> int:
        for i, obj_frame in enumerate(obj_frames):
            for obj in obj_frame:
                if isinstance(obj, Skeleton):
                    la = (
                        obj.left_knee_angle,
                        obj.left_shoulder_angle,
                        obj.left_elbow_angle,
                    )
                    ra = (
                        obj.right_knee_angle,
                        obj.right_shoulder_angle,
                        obj.right_elbow_angle,
                    )

                    left_shot = (
                        all(x is not None for x in la)
                        and la[0] > 170
                        and la[1] > 130
                        and la[2] > 150
                    )
                    right_shot = (
                        all(x is not None for x in ra)
                        and ra[0] > 170
                        and ra[1] > 130
                        and ra[2] > 150
                    )

                    if left_shot or right_shot:
                        return i
        return -1
