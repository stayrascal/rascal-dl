from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from models.tf_serving.apis import predict_pb2
from models.tf_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'ai03:9000', 'Server host:port.')
tf.app.flags.DEFINE_string('image', './cat.jpeg', 'path to image in JPEG format.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    with open(FLAGS.image, 'rb') as f:
        data = f.read()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=[1]))
        result = stub.Predict(request, 10.0)
        print(result)


if __name__ == '__main__':
    tf.app.run()
