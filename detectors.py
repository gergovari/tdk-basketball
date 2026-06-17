from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
from entities import Skeleton


class ThrowerDetector(ABC):
    @abstractmethod
    def detect(self, obj_frames, video_size=None):
        pass


def filter_obj_frame(func, obj_frame):
    return list(filter(func, obj_frame))


@dataclass
class BiggestPersonThrowerDetector(ThrowerDetector):
    person_filter: List[str] = field(
        default_factory=lambda: ["person", "player", "human"]
    )

    def detect(self, obj_frames, video_size=None):
        # We will build tracks based on Euclidean distance of centers
        tracks = [] # Each track: {'frames': {frame_idx: obj}, 'last_frame': idx, 'last_center': (cx, cy)}
        
        if video_size is not None:
            max_dist = video_size[0] * 0.15
        else:
            max_dist = 200 # Maximum allowed pixel distance between frames to match tracks
        
        for i, obj_frame in enumerate(obj_frames):
            persons = [obj for obj in obj_frame if any(word in obj.name for word in self.person_filter)]
            
            available_tracks = [t for t in tracks if i not in t['frames']]
            pairs = []
            
            for p_idx, person in enumerate(persons):
                cx = (person.rect.x1 + person.rect.x2) / 2
                cy = (person.rect.y1 + person.rect.y2) / 2
                
                for t_idx, track in enumerate(available_tracks):
                    tcx, tcy = track['last_center']
                    dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                    if dist < max_dist:
                        pairs.append((dist, p_idx, t_idx))
                        
            pairs.sort(key=lambda x: x[0])
            used_persons = set()
            used_tracks = set()
            
            for dist, p_idx, t_idx in pairs:
                if p_idx in used_persons or t_idx in used_tracks:
                    continue
                    
                track = available_tracks[t_idx]
                person = persons[p_idx]
                cx = (person.rect.x1 + person.rect.x2) / 2
                cy = (person.rect.y1 + person.rect.y2) / 2
                
                track['frames'][i] = person
                track['last_frame'] = i
                track['last_center'] = (cx, cy)
                
                used_persons.add(p_idx)
                used_tracks.add(t_idx)
                
            for p_idx, person in enumerate(persons):
                if p_idx not in used_persons:
                    cx = (person.rect.x1 + person.rect.x2) / 2
                    cy = (person.rect.y1 + person.rect.y2) / 2
                    tracks.append({
                        'frames': {i: person},
                        'last_frame': i,
                        'last_center': (cx, cy)
                    })
                    
        if not tracks:
            return []
            
        valid_tracks = []
        for track in tracks:
            t_centers = []
            for obj in track['frames'].values():
                t_centers.append(((obj.rect.x1 + obj.rect.x2) / 2, (obj.rect.y1 + obj.rect.y2) / 2))
                
            t_centers.sort(key=lambda p: p[0])
            median_x = t_centers[len(t_centers)//2][0]
            t_centers.sort(key=lambda p: p[1])
            median_y = t_centers[len(t_centers)//2][1]
            
            track['median_x'] = median_x
            track['median_y'] = median_y
            
            if video_size is not None:
                # Reject tracks that have a median center on the outer 15% edges of the frame
                margin = video_size[0] * 0.15
                if median_x < margin or median_x > video_size[0] - margin:
                    continue
                    
                # Score based on length AND proximity to the horizontal center
                center_x = video_size[0] / 2
                dist_to_center = abs(median_x - center_x)
                weight = 1.0 - (dist_to_center / center_x)
                track['score'] = len(track['frames']) * weight
            else:
                track['score'] = len(track['frames'])
                
            valid_tracks.append(track)
            
        if not valid_tracks:
            return []
            
        best_track = max(valid_tracks, key=lambda t: t['score'])
        best_median_x = best_track['median_x']
        best_median_y = best_track['median_y']
        
        for track in tracks:
            if track is best_track:
                continue
                
            near_frames = 0
            for obj in track['frames'].values():
                cx = (obj.rect.x1 + obj.rect.x2) / 2
                cy = (obj.rect.y1 + obj.rect.y2) / 2
                dist = ((best_median_x - cx)**2 + (best_median_y - cy)**2)**0.5
                if dist < max_dist:
                    near_frames += 1
                    
            first_obj = next(iter(track['frames'].values()))
            first_cx = (first_obj.rect.x1 + first_obj.rect.x2) / 2
            first_cy = (first_obj.rect.y1 + first_obj.rect.y2) / 2
            dist_start = ((best_median_x - first_cx)**2 + (best_median_y - first_cy)**2)**0.5
            
            last_obj = list(track['frames'].values())[-1]
            last_cx = (last_obj.rect.x1 + last_obj.rect.x2) / 2
            last_cy = (last_obj.rect.y1 + last_obj.rect.y2) / 2
            dist_end = ((best_median_x - last_cx)**2 + (best_median_y - last_cy)**2)**0.5
                    
            if (near_frames / len(track['frames']) > 0.3) or (dist_start < max_dist) or (dist_end < max_dist):
                best_track['frames'].update(track['frames'])

        unified_id = 99999
        
        # Override the id of the thrower object in every frame it appears
        for frame_idx, obj in best_track['frames'].items():
            obj.id = unified_id
            
        dummy = next(iter(best_track['frames'].values()))
        return [dummy]


class ReleaseDetector(ABC):
    @abstractmethod
    def detect(self, obj_frames, fps: int, start_idx: int = 0) -> int:
        pass


@dataclass
class SkeletonReleaseDetector(ReleaseDetector):
    def detect(self, obj_frames, fps: int, start_idx: int = 0) -> int:
        max_angle = -1
        max_frame = -1
        
        # We need the elbow to at least extend to 120 to consider it a real throw
        min_required_extension = 120 
        
        for i in range(start_idx, len(obj_frames)):
            obj_frame = obj_frames[i]
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
                    
                    # Ensure shoulders are somewhat raised (>90) to rule out just standing up
                    l_val = la[2] if (la[2] is not None and la[1] is not None and la[1] > 90) else -1
                    r_val = ra[2] if (ra[2] is not None and ra[1] is not None and ra[1] > 90) else -1
                    
                    angle = max(l_val, r_val)
                    
                    if angle > max_angle:
                        max_angle = angle
                        max_frame = i
                    elif angle != -1 and angle < max_angle - 15 and max_angle > min_required_extension:
                        # Angle has dropped significantly from the peak, meaning the stroke is over!
                        return max_frame
                        
        if max_angle > min_required_extension:
            return max_frame
            
        return -1


@dataclass
class SkeletonPrepareDetector(ReleaseDetector):
    def detect(self, obj_frames, fps: int, start_idx: int = 0) -> int:
        for i in range(start_idx, len(obj_frames)):
            obj_frame = obj_frames[i]
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

                    # Prepare phase: limbs bent. E.g., elbows < 90, knees < 160.
                    left_prep = (
                        all(x is not None for x in la)
                        and la[0] < 160
                        and la[1] < 120
                        and la[2] < 90
                    )
                    right_prep = (
                        all(x is not None for x in ra)
                        and ra[0] < 160
                        and ra[1] < 120
                        and ra[2] < 90
                    )

                    if left_prep or right_prep:
                        return i
        return -1

    def detect_backward(self, obj_frames, fps: int, start_idx: int, end_idx: int) -> int:
        min_angle = 180
        min_frame = -1
        
        # Search backwards from the release frame (end_idx) to start_idx
        for i in range(end_idx, start_idx - 1, -1):
            obj_frame = obj_frames[i]
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

                    left_prep = (
                        all(x is not None for x in la)
                        and la[0] < 160
                        and la[1] < 120
                        and la[2] < 90
                    )
                    right_prep = (
                        all(x is not None for x in ra)
                        and ra[0] < 160
                        and ra[1] < 120
                        and ra[2] < 90
                    )

                    if left_prep or right_prep:
                        angle = la[2] if left_prep else ra[2]
                        if angle < min_angle:
                            min_angle = angle
                            min_frame = i
                    elif min_frame != -1:
                        # We were in a prepare phase (going backward) and just exited it.
                        # This means we found the full peak flexion for this shot.
                        return min_frame

        return min_frame
