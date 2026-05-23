from PIL import Image, ImageChops, ImageEnhance
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import TRAIN_IMAGES_DIR

# Load the two images
folder = TRAIN_IMAGES_DIR

for i in range(0,4000):
    left_img_path = folder / f"{i}_left.jpg"
    right_img_path = folder / f"{i}_right.jpg"
    if i%50==0:
        print(f"{i} finished!")
    
    
    try:
        left_img = Image.open(left_img_path)
        right_img = Image.open(right_img_path)
    except:
        continue
    
    # enhancer = ImageEnhance.Brightness
    # left_img = enhancer(left_img).enhance(1.1)
    # right_img = enhancer(left_img).enhance(1.1)

    # Check if sizes are the same
    same_size = left_img.size == right_img.size

    # Function to crop black border
    def crop_black_border(image):
        bg = Image.new(image.mode, image.size, (0, 0, 0))
        diff = ImageChops.difference(image, bg)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image  # return original if bbox is None

    # Process images
    if not same_size:
        print(i,left_img.size,right_img.size)
        left_img = crop_black_border(left_img)
        right_img = crop_black_border(right_img)
        right_img=right_img.resize(left_img.size)

    # Merge images side by side
    total_width = left_img.width + right_img.width
    max_height = max(left_img.height, right_img.height)

    merged_img = Image.new("RGB", (total_width, max_height))
    merged_img.paste(left_img, (0, 0))
    merged_img.paste(right_img, (left_img.width, 0))

    # Save merged image
    merged_path = folder / "merged" / f"{i}_merge.jpg"
    # merged_path="test.jpg"
    merged_img.save(merged_path)
