from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
import numpy as np

from liberopro.liberopro.envs.arenas.style import get_texture_filename


class LivingRoomTableArena(Arena):
    """Empty workspace."""

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.41),
        xml="arenas/empty_arena.xml",
        floor_style="light-gray",
        wall_style="light-gray-plaster",
    ):
        super().__init__(xml_path_completion(xml))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = (
            self.bottom_pos
            + np.array([0, 0, -self.table_half_size[2]])
            + self.table_offset
        )

        self.living_room_table_body = self.worldbody.find(
            "./body[@name='living_room_table']"
        )

        texplane = self.asset.find("./texture[@name='texplane']")
        plane_file = texplane.get("file")
        plane_file = "/".join(
            plane_file.split("/")[:-1]
            + [get_texture_filename(type="floor", style=floor_style)]
        )
        texplane.set("file", plane_file)

        texwall = self.asset.find("./texture[@name='tex-wall']")
        wall_file = texwall.get("file")
        wall_file = "/".join(
            wall_file.split("/")[:-1]
            + [get_texture_filename(type="wall", style=wall_style)]
        )
        texwall.set("file", wall_file)
