import io
import os
from google.cloud import vision
from google.cloud.vision import types


def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    return client.text_detection(image=image)


def detect_text_uri(uri):
    client = vision.ImageAnnotatorClient()
    image = types.Image()
    image.source.image_uri = uri
    return client.text_detection(image=image)


def get_text_local():
    file_name = os.path.join(os.path.dirname(__file__), 'images/NF_050.jpg')
    print(file_name)
    response = detect_text(file_name)

    # uri = 'https://00e9e64bacc77a6b6d9d822158af53859cb0e1e5f8a59e5026-apidata.googleusercontent.com/download/storage/v1/b/stayrascal/o/images%2Finvoice%2FIMAG0001.JPG-0.jpg?qk=AD5uMEvb37hkDsguCpPry81NQN8_KvZTPqDOQiB2lDzW61AV4_Yxp9qq3mLozjw31IrgH_LL6oaHAQKK3hwPz-8pkCg3LKGEam0Fpit992LWVK0tYWKDWCEg01qP8KtN4ue6cpKQY-4c2GiZzZmvBJxE07zSglRXi8gwf01fnNdtz66Parw0ESNztr5qKBGypuYyGTGtzp5BNL6dgQhi3XAVg7yJVEMd3RsYtenrfrpijHWD0aoDMo7xOvme8vy5CAWHYEfpux9yIriz8rvfii3cfVK9dpBV58CZm5NK993z2fDI0aX7-HARAYv0V3TgVBiKh-z8Ws7uOvtekHJUz1ML0s2q2t-h8LsKRs98ho5V2L2mkuoXk3DYMimZKbjDvZGSwwWH5JbLuC_aIEuTuA-DO3i5BypejqMudsOugDVsivcmlHxDcaIXypSeJ1GTgL1sWPw1QwN5wUV8z7334fjeoFllB_IrsZuPNf0v-bpVUyPxKcCVKiN6EjkFH9dxA_khRIWBLra_HrKYVkqdw6bp3-F0Z4XFUHWni3pi9UL7kQEPCN1YAvwoaEpx46_DIyDlNDWuufrP_iVuKQmXCLk1WE0o4PbvFRj9nFHulaI4kLERWmjSudkedK_Nb4tFZ6XlHqiDhNtHhmR8OqfpvnQACn5yys4L7QqQ_Kw3sXk-ZfihzKJklHK5uAlPnCvuVNg0gLOaL4fk7_sRuO9NJynjlrr74VPDedkICEEpuijhq2B7Y6BAhSuJwkSPsmUdawQVyooUuYNv'
    # response = detect_text_uri(uri)

    texts = response.text_annotations
    print('Texts: ')
    for text in texts:
        print('\n"{}"'.format(text.description))
        vertices = (['({},{})'.format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        print('bounds: {}'.format(','.join(vertices)))


if __name__ == '__main__':
    get_text_local()
