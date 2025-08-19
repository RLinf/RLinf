
from mani_skill.utils.registration import register_env

from rlinf.environment.tasks.put_on_in_scene_multi import (
    PutOnPlateInScene25MainV3,
)


@register_env("PutOnPlateInScene25VisionImage-v1", max_episode_steps=80, asset_download_ids=["bridge_v2_real2sim"])
class PutOnPlateInScene25VisionImage(PutOnPlateInScene25MainV3):
    @property
    def basic_obj_infos(self):
        if self.obj_set == "train":
            lo = 16
            lo_offset = 0
        elif self.obj_set == "test":
            lo = 5
            lo_offset = 16
        elif self.obj_set == "all":
            lo = 21
            lo_offset = 0
        else:
            raise ValueError(f"Unknown obj_set: {self.obj_set}")

        lc = 16
        lc_offset = 0
        lp = len(self.plate_names)
        lp_offset = 0
        l1 = len(self.xyz_configs)
        l2 = len(self.quat_configs)
        return lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2
