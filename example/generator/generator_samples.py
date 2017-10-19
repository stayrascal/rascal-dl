import argparse
import os
import pickle

import numpy as np
from PIL import Image


def to_data(image_dirs, stat_templates):
    images, stats = [], []
    for i, image_dir in enumerate(image_dirs):
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        image_files = [f for f in image_files if f[-4:] in ['.jpg', '.png']]
        for image_file in image_files:
            image = Image.open(image_file)
            image = image.convert('RGB')
            image = image.resize((220, 20), Image.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
            stats.append(stat_templates[i])
    return np.array(images), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', type=str, default='', help='directory where you put the samples in')
    parser.add_argument('--save_pickle', type=str, default='', help='directory to put generated pickle file if set')
    args = parser.parse_args()

    os.makedirs(args.save_pickle, exist_ok=True)
    images, stats = to_data([os.path.join(args.samples_dir, 'clean')]), [{'text': []}]
    output_file = os.path.join(args.save_pickle, 'clean.pickle')
    pickle.dump({'images': images, 'stats': stats}, open(output_file, 'wb'))


if __name__ == '__main__':
    main()
