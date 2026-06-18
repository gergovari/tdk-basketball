from dataclasses import dataclass
from typing import List


@dataclass
class InputParams:
    video_id: str
    data_path: str

    @property
    def input_video_path(self):
        return f"{self.data_path}/input/{self.video_id}.mp4"

    @property
    def output_video_path(self):
        return f"{self.data_path}/output/{self.video_id}.mp4"

    @property
    def output_data_path(self):
        return f"{self.data_path}/data/{self.video_id}.csv"


@dataclass
class YOLOParams:
    model_path: str
    name_filter: List[str]
