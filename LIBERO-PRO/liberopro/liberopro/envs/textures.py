import os
from liberopro.liberopro import get_libero_path

# This is the mapping from texture name to texture file name. Currently this has some duplication with style.py. We will fix this in the future.

TEXTURE_MAPPING = {
    "Porcelain_Floor_1": "grigia_caldera_porcelain_floor.png",
    "Gray_Plaster_1": "gray_plaster.png",
    "Marble_Floor_1": "marble_floor.png",
    "Gray_Floor_1": "gray_floor.png",
    "Blue_Wall_1": "dark_blue_wall.png",
    "White_Marble_Floor_1": "white_marble_floor.png",
    "Gray_Plaster_2": "new_light_gray_plaster.png",
    "Wood_Floor_1": "seamless_wood_planks_floor.png",
    "Rustic_Floor_1": "rustic_floor.png",
    "Gray_Plaster_3": "light-gray-plaster.png",
    "Light_Floor_1": "light_floor.png",
    "Blue_Wall_2": "canvas_sky_blue.png",
    "Dark_Floor_1": "dark_floor_texture.png",
    "Beige_Plaster_1": "meeka-beige-plaster.png",
    "Gray_Plaster_4": "dark_gray_plaster.png",
    "Stucco_Wall_1": "stucco_wall.png",
    "Light_Gray_Floor_1": "light-gray-floor-tile.png",
    "Wood_Table_1": "martin_novak_wood_table.png",
    "Sky_Image_1": "capriccio_sky.png",
    "Gray_Ceramic_Tile_1": "gray_ceramic_tile.png",
    "Yellow_Linen_Wall_Texture_1": "yellow_linen_wall_texture.png",
    "Gray_Floor_2": "dapper_gray_floor.png",
    "Ceramic_1": "ceramic.png",
    "Cream_Plaster_1": "cream-plaster.png",
    "Gray_Plaster_5": "smooth_light_gray_plaster.png",
    "White_Wall_1": "white_wall.png",
    "Blue_Wall_3": "light_blue_wall.png",
    "Porcelain_Floor_2": "tile_grigia_caldera_porcelain_floor.png",
    "Brown_Ceramic_Tile_1": "brown_ceramic_tile.png",
    "Gray_Wall_1": "gray_wall.png",
    "Kona_Gotham_1": "kona_gotham.png",
    "Gray_Plaster_6": "light_gray_plaster.png",
    "Green_Plaster_Wall_1": "dark_green_plaster_wall.png",
    "Gray_Plaster_7": "light_grey_plaster.png",
    "Wood_Table_2": "table_light_wood.png",
}


def get_texture_file_list(type=None, texture_path="../"):
    texture_mapping_dict = {}
    path = os.path.join(texture_path, "textures")
    for (key, value) in sorted(TEXTURE_MAPPING.items()):
        if type.lower() == "table":
            # Only those with "table" in the name or no other element name will be included
            if "table" in key.lower() or (
                "wall" not in key.lower() and "floor" not in key.lower()
            ):
                texture_mapping_dict[key] = value
        elif type.lower() == "wall":
            if "wall" in key.lower() or (
                "table" not in key.lower() and "floor" not in key.lower()
            ):
                texture_mapping_dict[key] = value
        elif type.lower() == "floor":
            if "floor" in key.lower() or (
                "wall" not in key.lower() and "table" not in key.lower()
            ):
                texture_mapping_dict[key] = value

    texture_list = []
    for key, value in texture_mapping_dict.items():
        texture_list.append((key, os.path.join(path, value)))
    return texture_list
