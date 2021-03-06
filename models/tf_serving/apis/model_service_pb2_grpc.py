# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/apis/model_service.proto
# To regenerate run
# python -m grpc.tools.protoc --python_out=. --grpc_python_out=. -I. tensorflow_serving/apis/model_service.proto

import grpc
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities

import models.tf_serving.apis.get_model_status_pb2 as tensorflow__serving_dot_apis_dot_get__model__status__pb2


class ModelServiceStub(object):
    """ModelService provides access to information about model versions
    that have been handled by the model server.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
          channel: A grpc.Channel.
        """
        self.GetModelStatus = channel.unary_unary(
            '/tensorflow.serving.ModelService/GetModelStatus',
            request_serializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusRequest.SerializeToString,
            response_deserializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusResponse.FromString,
        )


class ModelServiceServicer(object):
    """ModelService provides access to information about model versions
    that have been handled by the model server.
    """

    def GetModelStatus(self, request, context):
        """Gets status of model. If the ModelSpec in the request does not specify
        version, information about all versions of the model will be returned. If
        the ModelSpec in the request does specify a version, the status of only
        that version will be returned.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'GetModelStatus': grpc.unary_unary_rpc_method_handler(
            servicer.GetModelStatus,
            request_deserializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusRequest.FromString,
            response_serializer=tensorflow__serving_dot_apis_dot_get__model__status__pb2.GetModelStatusResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'tensorflow.serving.ModelService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
