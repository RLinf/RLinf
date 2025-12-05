FLOOR_STYLE = {
    "dark": "dark_floor_texture.png",
    "rustic": "rustic_floor.png",
    "light-gray": "light-gray-floor-tile.png",
    "white-marble": "white_marble_floor.png",
    "wood-plank": "seamless_wood_planks_floor.png",
    "brown-ceramic": "brown_ceramic_tile.png",
    "gray-ceramic": "gray_ceramic_tile.png",
    "tile_grigia_caldera": "tile_grigia_caldera_porcelain_floor.png",
}

WALL_STYLE = {
    "light-blue": "light_blue_wall.png",
    "dark-blue": "dark_blue_wall.png",
    "dark-gray-plaster": "dark_gray_plaster.png",
    "gray-plaster": "gray_plaster.png",
    "dark-green": "dark_green_plaster_wall.png",
    "light-gray-plaster": "light-gray-plaster.png",
    "ceramic": "ceramic.png",
    "white": "white_wall.png",
    "yellow-linen": "yellow_linen_wall_texture.png",
}


STYLE_MAPPING = {"floor": FLOOR_STYLE, "wall": WALL_STYLE}


def get_texture_filename(type, style):
    assert type in STYLE_MAPPING.keys()
    assert style in STYLE_MAPPING[type].keys()
    return STYLE_MAPPING[type][style]
