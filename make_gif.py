import argparse
import fnmatch
import math
import numpy as np
import os
import re

from natsort import natsorted, ns

import matplotlib
from PIL import Image, ImageDraw, ImageFont

types = ["fixed", "tilted", "fixed_ema", "tilted_ema", "random"]

number_re = re.compile(r'\d+')


def list_img_for_type(type, path):
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and not f.startswith('.') and fnmatch.fnmatch(f, '*{}.png'.format(type)):
            yield f


def write_step_on_image(img, step, text_height):
    img_w, img_h = img.size
    new_img = Image.new('RGBA', (img_w, img_h + text_height), (255, 255, 255, 255))
    new_img_w, new_img_h = new_img.size
    offset = ((new_img_w - img_w) // 2, (new_img_h - img_h))
    new_img.paste(img, offset)
    draw = ImageDraw.Draw(new_img)
    draw.text((0, 0), str("Step: {}".format(step)), fill=(0, 0, 0))
    return new_img

def make_gif_for_type(type, path, output_path):
    files = [f for f in list_img_for_type(type, path)]
    files = natsorted(files, alg=ns.IGNORECASE)
    images = []
    for filename in files:
        image = Image.open(os.path.join(path, filename), 'r')
        image = write_step_on_image(image, number_re.findall(filename)[0], text_height=20)
        images.append(image)
    images[0].save(os.path.join(output_path, type + '.gif'), save_all=True, append_images=images[1:], duration=100, loop=0)


def main():
    print("Hello There\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='gif')

    opt = parser.parse_args()
    print("Params:")
    print(opt)
    print("")

    abs_input_dir = os.path.abspath(opt.input_dir)
    if not os.path.exists(abs_input_dir):
        print("Input directory {} does not exist".format(abs_input_dir))
        exit(1)
    
    # Get all files in the input directory
    files = [f for f in os.listdir(abs_input_dir) if os.path.isfile(os.path.join(abs_input_dir, f)) and fnmatch.fnmatch(f, '*.png')]

    # Check if there are any files
    if len(files) == 0:
        print("No images found in {}".format(abs_input_dir))
        exit(1)

    # Create Output Directory
    abs_output_dir = os.path.abspath(opt.output_dir)
    if not os.path.exists(abs_output_dir):
        print("Creating output directory {}".format(abs_output_dir))
        os.makedirs(abs_output_dir,exist_ok=True)
    else:
        print("{} already exists".format(abs_output_dir))
    
    # Make gifs for each type
    for type in types:
        print("Making gif for type {}".format(type))
        make_gif_for_type(type, abs_input_dir, abs_output_dir)
    
    # Done
    print("Done")


if __name__ == '__main__':
    main()