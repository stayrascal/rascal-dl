import argparse
import os
import pickle

import numpy as np
from PIL import Image

from util import randomdata, text2img, mark_image


def generate_sample():
    image = Image.open(randomdata.random_clean_image())
    opacity = randomdata.random_opacity()
    text = randomdata.random_text()
    font, font_size, font_color = randomdata.random_font(), randomdata.random_font_size(), randomdata.random_font_color()
    mark = text2img(text, font, font_size, font_color)
    # mark = mark.rotate(randomdata.cut_gauss(0, 7, -20, 20), expand=1)
    position = randomdata.random_pos(image.size, mark.size)
    # new_size = (min(image.size[0] - position[0], mark.size[0]), min(image.size[1] - position[1], mark.size[1]))
    # mark.resize(new_size)
    image = mark_image(image, mark, position, opacity)
    # image = image.crop((position[0], position[1], position[0] + new_size[0], position[1] + new_size[1]))
    return image, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', help='run a sample instead of generation a data file', action='store_true')
    parser.add_argument('--save_pickle', type=str, default='', help='direct')
    parser.add_argument('--save_image', type=str, default='', help='parent directory to put generated jpeg file if set')
    parser.add_argument('count', type=int, help='how many samples to generate')
    args = parser.parse_args()

    final_size = (200, 32)

    if args.sample:
        image, text = generate_sample()
        print('text: {}'.format(text))
        image = image.convert('RGB')
        image = image.resize(final_size, Image.BILINEAR)
        image = np.asarray(image, dtype=np.uint8)
        image = Image.fromarray(image)
        image.show()
    else:
        images, stats = [], []
        for i in range(args.count):
            image, stat = generate_sample()
            image = image.convert('RGB')
            image = image.resize(final_size, Image.BILINEAR)
            if args.save_image:
                os.makedirs(os.path.join(args.save_image), exist_ok=True)
                image.save(os.path.join(args.save_image, '{}.png'.format(i)))
            images.append(np.asarray(image, dtype=np.uint8))
            stats.append(stat)
            if i > 0 and i % 100 == 0:
                print('generated images: {}'.format(i))
        if args.save_pickle:
            os.makedirs(args.save_pickle, exist_ok=True)
            images = np.array(images)
            output_file = os.path.join(args.save_pickle, 'rcnn.pickle')
            pickle.dump({'images': images, 'stats': stats}, open(output_file, 'wb'))


if __name__ == '__main__':
    main()
