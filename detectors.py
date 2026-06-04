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
class BiggestPersonThrowerDetector(ThrowerDetector):
    person_filter: List[str] = field(
        default_factory=lambda: ["person", "player", "human"]
    )

    def detect(self, obj_frames):
        largest_area = 0
        biggest_obj = None

        for obj_frame in obj_frames:
            for obj in obj_frame:
                if (
                    any(word in obj.name for word in self.person_filter)
                    and getattr(obj, "id", -1) != -1
                ):
                    area = (obj.rect.x2 - obj.rect.x1) * (obj.rect.y2 - obj.rect.y1)
                    if area > largest_area:
                        largest_area = area
                        biggest_obj = obj

        if biggest_obj:
            return [biggest_obj]
        return []


class ReleaseDetector(ABC):
    @abstractmethod
    def detect(self, obj_frames, fps: int) -> int:
        pass


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
