from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import shutil
import sys

import tensorflow as tf

sys.path.append("../..")

from models.utils.arg_parsers import parsers
from models.utils.logs import hooks_helper
from models.utils.misc import model_helpers

_CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                'marital_status', 'occupation', 'relationship', 'race', 'gender',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket'
                ]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0],
                        [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def build_model_columns():
    # feature_column为模型输入提供了一个规范格式，表明如何去表示和转换这些数据。其本身不会提供这些数据，需要通过一个输入函数来提供数据。

    # 读取连续的变量特征
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    # 编码稀疏列:将分类值转化为向量值
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                      'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                      '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                           'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
                      'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing: output_id = Hash(input_feature_string) % bucket_size
    # 当不知道所有可能值时，可以使用hash函数为特征值分类序列
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)

    # 桶化准换一个连续列位分类列
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns
    base_columns = [education, marital_status, relationship, workclass, occupation, age_buckets]

    # 线性模型只是给单独的特征分配独立的权重，无法学习特征在特定组合的重要性，可以创建feature1*feature2特征来关联两个源特征的值
    crossed_columns = [tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column([age_buckets, 'education', 'occupation'],
                                                        hash_bucket_size=1000)]

    wide_columns = base_columns + crossed_columns

    deep_columns = [age,
                    education_num,
                    capital_gain,
                    capital_loss,
                    hours_per_week,
                    tf.feature_column.indicator_column(workclass),
                    tf.feature_column.indicator_column(education),
                    tf.feature_column.indicator_column(marital_status),
                    tf.feature_column.indicator_column(relationship),
                    tf.feature_column.embedding_column(occupation, dimension=8)
                    ]

    return wide_columns, deep_columns, crossed_columns


def save_model(model, serving_model_dir):
    _, deep_columns, crossed_columns = build_model_columns()
    # make_parse_example_spec returns a dict mapping feature keys
    # from feature_columns to FixedLenFeature or VarLenFeature values.
    feature_spec = tf.feature_column.make_parse_example_spec(crossed_columns + deep_columns)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(serving_model_dir, export_input_fn)


def build_estimator(model_dir, model_type):
    wide_columns, deep_columns, _ = build_model_columns()
    hidden_units = [100, 75, 50, 25]

    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config
        )
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config
        )
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            n_classes=2,
            weight_column=None,
            linear_feature_columns=wide_columns,
            linear_optimizer='Ftrl',
            dnn_feature_columns=deep_columns,  # default activate function is relu
            dnn_optimizer='Adagrad',
            dnn_dropout=None,
            dnn_hidden_units=hidden_units,
            config=run_config,

        )


def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('%s not found. Please make sure you have run data_download.py '
                                        'and set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def main(argv):
    parser = WideDeepArgParser()
    flags = parser.parse_args(args=argv[1:])

    shutil.rmtree(flags.model_dir, ignore_errors=True)
    model = build_estimator(flags.model_dir, flags.model_type)

    train_file = os.path.join(flags.data_dir, 'adult.data')
    test_file = os.path.join(flags.data_dir, 'adult.test')
    print(model)

    def train_input_fn():
        return input_fn(train_file, flags.epochs_between_evals, True, flags.batch_size)

    def eval_input_fn():
        return input_fn(test_file, 1, False, flags.batch_size)

    loss_prefix = LOSS_PREFIX.get(flags.model_type, '')
    train_hooks = hooks_helper.get_train_hooks(flags.hooks, batch_size=flags.batch_size,
                                               tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                                                               'loss': loss_prefix + 'head/weighted_loss/Sum'})

    for n in range(flags.train_epochs // flags.epochs_between_evals):
        model.train(input_fn=train_input_fn, hooks=train_hooks)
        results = model.evaluate(input_fn=eval_input_fn)

        print('Results at epoch', (n + 1) * flags.epochs_between_evals)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

        if model_helpers.past_stop_threshold(flags.stop_threshold, results['accuracy']):
            break
    save_model(model, flags.serving_model_dir)


class WideDeepArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser(multi_gpu=False, num_gpu=False)])
        self.add_argument(
            '--model_type', '-mt', type=str, default='wide_deep', choices=['wide', 'deep', 'wide_deep'],
            help='[defalut %(default)s] Valid mode types: wide, deep, wide_deep.',
            metavar='<MT>'
        )
        self.set_defaults(
            data_dir='/tmp/census_data',
            model_dir='/tmp/census_mode',
            train_epochs=40,
            epochs_between_evals=2,
            batch_size=40
        )
        self.add_argument('--serving_model_dir', type=str, default='/tmp/serving/model',
                          help='Default serving mode directory')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
