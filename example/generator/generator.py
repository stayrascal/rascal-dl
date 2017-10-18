#!/usr/bin/env python3
# -*- coding: utf-8 -*
import argparse
import os
import pickle

import numpy as np
from PIL import Image

from util import randomdata, text2img, mark_image


def generate_image(mode):
    image = Image.open(randomdata.random_clean_image())
    stat = {'text': []}

    if mode == 'clean':
        return image, stat

    def random_text(image):
        opacity = randomdata.random_opacity()
        text = randomdata.random_text()
        font, font_size, font_color = randomdata.random_font(), randomdata.random_font_size(), randomdata.random_font_color()
        print('text', text)
        mark = text2img(text, font, font_size, font_color)
        position = randomdata.random_pos(image.size, mark.size)
        image = mark_image(image, mark, position, opacity)
        stat = {'text': text, 'font': font, 'font_size': font_size, 'font_color': font_color}
        stat.update({'image_size': image.size, 'mark_size': mark.size, 'position': position, 'opacity': opacity})
        return image, stat

    for i in range(int(randomdata.cut_gauss(1, 0.5, 1, 5))):
        image, single_stat = random_text(image)
        stat['text'].append(single_stat)

    return image, stat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', help='run a sample instead of generation a data file', action='store_true')
    parser.add_argument('--save_pickle', type=str, default='', help='direct')
    parser.add_argument('--save_image', type=str, default='', help='parent directory to put generated jpeg file if set')
    parser.add_argument('mode', type=str, default='generator', help='one of [watermark, clean]')
    parser.add_argument('count', type=int, help='how many samples to generate')

    args = parser.parse_args()
    if args.sample:
        image, stat = generate_image(args.mode)
        print('text: {}'.format(stat['text']))
        image = image.convert('RGB')
        image = image.resize((220, 20), Image.BILINEAR)
        image = np.asanyarray(image, dtype=np.uint8)
        image = Image.fromarray(image)
        image.show()
    else:
        images, stats = [], []
        for i in range(args.count):
            image, stat = generate_image(args.mode)
            image = image.convert('RGB')
            image = image.resize((220, 20), Image.BILINEAR)
            if args.save_image:
                os.makedirs(os.path.join(args.save_image, args.mode), exist_ok=True)
                image.save(os.path.join(args.save_image, '{}/{}.png'.format(args.mode, i)))
            images.append(np.asarray(image, dtype=np.uint8))
            stats.append(stat)
            if i > 0 and i % 100 == 0:
                print('generated images: {}'.format(i))
        if args.save_pickle:
            os.makedirs(args.save_pickle, exist_ok=True)
            images = np.array(images)
            output_file = os.path.join(args.save_pickle, '{}.pickle'.format(args.mode))
            pickle.dump({'images': images, 'stats': stats}, open(output_file, 'wb'))


if __name__ == '__main__':
    main()
