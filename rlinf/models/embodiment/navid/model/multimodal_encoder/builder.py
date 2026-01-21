import os
from .eva_vit import EVAVisionTowerLavis

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', "./model_zoo/OpenAI/clip-vit-large-patch14"))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    if not is_absolute_path_exists:
        raise ValueError(f'Not find vision tower: {vision_tower}')
    
    if "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EVAVisionTowerLavis(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
