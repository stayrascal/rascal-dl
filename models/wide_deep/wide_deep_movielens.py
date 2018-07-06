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

_CSV_COLUMNS = ['user', 'item', 'ratings', 'age', 'gender', 'occupation', 'zipcode',
                'action', 'adventure', 'animation', 'child', 'comedy', 'crime',
                'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical',
                'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western',
                'delay_days', 'watch_year', 'watch_month', 'watch_day', 'watch_wd',
                'watch_season', 'relase_year', 'release_month', 'release_day',
                'release_wd', 'watch_span', 'age_span']

_CSV_COLUMN_DEFAULTS = [[0], [0], [3], [30], [''], [''], [''],
                        ['0'], ['0'], ['0'], ['0'], ['0'], ['0'],
                        ['0'], ['0'], ['0'], ['0'], ['0'], ['0'],
                        ['0'], ['0'], ['0'], ['0'], ['0'], ['0'],
                        [0], [1997], [9], [17], [4],
                        [1], [1994], [1], [1], [5], ['TenYear'], [30]
                        ]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def build_model_columns():
    # feature_column为模型输入提供了一个规范格式，表明如何去表示和转换这些数据。其本身不会提供这些数据，需要通过一个输入函数来提供数据。

    # 读取连续的变量特征
    user = tf.feature_column.numeric_column('user')
    item = tf.feature_column.numeric_column('item')
    age = tf.feature_column.numeric_column('age')

    # action = tf.feature_column.categorical_column_with_vocabulary_list('action', [0, 1])
    # adventure = tf.feature_column.categorical_column_with_vocabulary_list('adventure', [0, 1])
    # animation = tf.feature_column.categorical_column_with_vocabulary_list('animation', [0, 1])
    # child = tf.feature_column.categorical_column_with_vocabulary_list('child', [0, 1])
    # comedy = tf.feature_column.categorical_column_with_vocabulary_list('comedy', [0, 1])
    # crime = tf.feature_column.categorical_column_with_vocabulary_list('crime', [0, 1])
    # documentary = tf.feature_column.categorical_column_with_vocabulary_list('documentary', [0, 1])
    # drama = tf.feature_column.categorical_column_with_vocabulary_list('drama', [0, 1])
    # fantasy = tf.feature_column.categorical_column_with_vocabulary_list('fantasy', [0, 1])
    # film_noir = tf.feature_column.categorical_column_with_vocabulary_list('film_noir', [0, 1])
    # horror = tf.feature_column.categorical_column_with_vocabulary_list('horror', [0, 1])
    # musical = tf.feature_column.categorical_column_with_vocabulary_list('musical', [0, 1])
    # mystery = tf.feature_column.categorical_column_with_vocabulary_list('mystery', [0, 1])
    # romance = tf.feature_column.categorical_column_with_vocabulary_list('romance', [0, 1])
    # thriller = tf.feature_column.categorical_column_with_vocabulary_list('thriller', [0, 1])
    # war = tf.feature_column.categorical_column_with_vocabulary_list('war', [0, 1])
    # western = tf.feature_column.categorical_column_with_vocabulary_list('western', [0, 1])

    action = tf.feature_column.categorical_column_with_vocabulary_list('action', ['0', '1'])
    adventure = tf.feature_column.categorical_column_with_vocabulary_list('adventure', ['0', '1'])
    animation = tf.feature_column.categorical_column_with_vocabulary_list('animation', ['0', '1'])
    child = tf.feature_column.categorical_column_with_vocabulary_list('child', ['0', '1'])
    comedy = tf.feature_column.categorical_column_with_vocabulary_list('comedy', ['0', '1'])
    crime = tf.feature_column.categorical_column_with_vocabulary_list('crime', ['0', '1'])
    documentary = tf.feature_column.categorical_column_with_vocabulary_list('documentary', ['0', '1'])
    drama = tf.feature_column.categorical_column_with_vocabulary_list('drama', ['0', '1'])
    fantasy = tf.feature_column.categorical_column_with_vocabulary_list('fantasy', ['0', '1'])
    film_noir = tf.feature_column.categorical_column_with_vocabulary_list('film_noir', ['0', '1'])
    horror = tf.feature_column.categorical_column_with_vocabulary_list('horror', ['0', '1'])
    musical = tf.feature_column.categorical_column_with_vocabulary_list('musical', ['0', '1'])
    mystery = tf.feature_column.categorical_column_with_vocabulary_list('mystery', ['0', '1'])
    romance = tf.feature_column.categorical_column_with_vocabulary_list('romance', ['0', '1'])
    thriller = tf.feature_column.categorical_column_with_vocabulary_list('thriller', ['0', '1'])
    war = tf.feature_column.categorical_column_with_vocabulary_list('war', ['0', '1'])
    western = tf.feature_column.categorical_column_with_vocabulary_list('western', ['0', '1'])

    delay_days = tf.feature_column.numeric_column('delay_days')
    watch_year = tf.feature_column.numeric_column('watch_year')
    watch_month = tf.feature_column.numeric_column('watch_month')
    watch_day = tf.feature_column.numeric_column('watch_day')
    watch_wd = tf.feature_column.numeric_column('watch_wd')
    watch_season = tf.feature_column.numeric_column('watch_season')
    relase_year = tf.feature_column.numeric_column('relase_year')
    release_day = tf.feature_column.numeric_column('release_day')
    release_wd = tf.feature_column.numeric_column('release_wd')
    age_span = tf.feature_column.numeric_column('age_span')

    # 编码稀疏列:将分类值转化为向量值
    gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', ['F', 'M'])
    occupation = tf.feature_column.categorical_column_with_vocabulary_list(
        'occupation',
        ['administrator', 'artist', 'doctor', 'educator', 'engineer', 'entertainment', 'executive', 'healthcare',
         'homemaker', 'lawyer', 'librarian', 'marketing', 'none', 'other', 'programmer', 'retired', 'salesman',
         'scientist', 'student', 'technician', 'writer'])
    watch_span = tf.feature_column.categorical_column_with_vocabulary_list(
        'watch_span', ['TenYear', 'Other', 'OneYear', 'OneMonth', 'OneDay', 'OneWeek'])

    # To show an example of hashing: output_id = Hash(input_feature_string) % bucket_size
    # 当不知道所有可能值时，可以使用hash函数为特征值分类序列
    zipcode = tf.feature_column.categorical_column_with_hash_bucket('zipcode', hash_bucket_size=1000)

    # 桶化准换一个连续列位分类列
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns
    base_columns = [gender, occupation, watch_span, zipcode, age_buckets, user, item,
                    action, adventure, animation,
                    child, comedy, crime, documentary, drama, fantasy, film_noir, horror, musical, mystery,
                    romance, thriller, war, western,
                    watch_year, watch_month, watch_day, watch_wd, watch_season,
                    relase_year, release_day, release_wd, age_span]

    # 线性模型只是给单独的特征分配独立的权重，无法学习特征在特定组合的重要性，可以创建feature1*feature2特征来关联两个源特征的值
    crossed_columns = [tf.feature_column.crossed_column(['gender', 'occupation'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'war'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'thriller'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'romance'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'mystery'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'musical'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'horror'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'film_noir'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'fantasy'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'drama'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'documentary'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'crime'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'comedy'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'child'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'animation'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'adventure'], hash_bucket_size=1000),
                       tf.feature_column.crossed_column(['western', 'action'], hash_bucket_size=1000)
                       ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [age, delay_days, user, item, watch_year, watch_month, watch_day, watch_wd, watch_season,
                    relase_year, release_day, release_wd, age_span,
                    tf.feature_column.indicator_column(action),
                    tf.feature_column.indicator_column(adventure),
                    tf.feature_column.indicator_column(animation),
                    tf.feature_column.indicator_column(child),
                    tf.feature_column.indicator_column(comedy),
                    tf.feature_column.indicator_column(crime),
                    tf.feature_column.indicator_column(documentary),
                    tf.feature_column.indicator_column(drama),
                    tf.feature_column.indicator_column(fantasy),
                    tf.feature_column.indicator_column(film_noir),
                    tf.feature_column.indicator_column(horror),
                    tf.feature_column.indicator_column(musical),
                    tf.feature_column.indicator_column(mystery),
                    tf.feature_column.indicator_column(romance),
                    tf.feature_column.indicator_column(thriller),
                    tf.feature_column.indicator_column(war),
                    tf.feature_column.indicator_column(western),
                    tf.feature_column.indicator_column(gender),
                    tf.feature_column.embedding_column(occupation, dimension=8),
                    tf.feature_column.embedding_column(watch_span, dimension=8),
                    tf.feature_column.embedding_column(zipcode, dimension=8)
                    ]

    return wide_columns, deep_columns, crossed_columns


# if __name__ == '__main__':
#     import six
#
#     _, deep_columns, crossed_columns = build_model_columns()
#     feature_columns = deep_columns + crossed_columns
#     result = {}
#     for column in feature_columns:
#         config = column._parse_example_spec  # pylint: disable=protected-access
#         print(config)


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
        return tf.estimator.LinearRegressor(
            model_dir=model_dir,
            feature_columns=wide_columns,
            config=run_config
        )
    elif model_type == 'deep':
        return tf.estimator.DNNRegressor(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units,
            config=run_config
        )
    else:
        return tf.estimator.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            weight_column=None,
            linear_feature_columns=wide_columns,
            linear_optimizer='Ftrl',
            dnn_feature_columns=deep_columns,  # default activate function is relu
            dnn_optimizer='Adagrad',
            dnn_dropout=None,
            dnn_hidden_units=hidden_units,
            config=run_config,
        )
        # return tf.estimator.DNNLinearCombinedClassifier(
        #     model_dir=model_dir,
        #     weight_column=None,
        #     n_classes=5,
        #     linear_feature_columns=wide_columns,
        #     linear_optimizer='Ftrl',
        #     dnn_feature_columns=deep_columns,  # default activate function is relu
        #     dnn_optimizer='Adagrad',
        #     dnn_dropout=None,
        #     dnn_hidden_units=hidden_units,
        #     config=run_config,
        # )


def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('%s not found. Please make sure you have run data_download.py '
                                        'and set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('ratings')
        # return features, labels - 1
        return features, labels

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

    train_file = os.path.join(flags.data_dir, 'train.data')
    test_file = os.path.join(flags.data_dir, 'eval.data')
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

        # if model_helpers.past_stop_threshold(flags.stop_threshold, results['accuracy']):
        if model_helpers.past_stop_threshold(flags.stop_threshold, results['loss']):
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
            data_dir='./movielens',
            model_dir='./movielens/model',
            train_epochs=40,
            epochs_between_evals=2,
            batch_size=40
        )
        self.add_argument('--serving_model_dir', type=str, default='./movielens/serving_model',
                          help='Default serving mode directory')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main(argv=sys.argv)
