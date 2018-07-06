import os
import tensorflow as tf
import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--output_path', type=str, default='/tmp/output', help='Output path')
parser.add_argument('--mode_vision', type=str, default='1.0')


def main(_):
    export_path_base = FLAGS.output_path
    model_vision = FLAGS.model_vision

    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(model_vision)))

    img_input = Input


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
