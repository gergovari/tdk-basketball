from dataclasses import dataclass
from typing import Tuple

@dataclass
class ScaledInt:
    Value: int
    ref_res: Tuple[int, int]
    res: Tuple[int, int]

    @property
    def value(self) -> int:
        scale_factor = self.res[0] / self.ref_res[0]
        return int(self.Value * scale_factor)

    @value.setter
    def value(self, new_val: int):
        scale_factor = self.res[0] / self.ref_res[0]
        self.Value = int(new_val / scale_factor)
