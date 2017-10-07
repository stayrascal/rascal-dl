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
    # uri = 'https://00e9e64bacc77a6b6d9d822158af53859cb0e1e5f8a59e5026-apidata.googleusercontent.com/download/storage/v1/b/stayrascal/o/images%2Finvoice%2FIMAG0001.JPG-0.jpg?qk=AD5uMEvb37hkDsguCpPry81NQN8_KvZTPqDOQiB2lDzW61AV4_Yxp9qq3mLozjw31IrgH_LL6oaHAQKK3hwPz-8pkCg3LKGEam0Fpit992LWVK0tYWKDWCEg01qP8KtN4ue6cpKQY-4c2GiZzZmvBJxE07zSglRXi8gwf01fnNdtz66Parw0ESNztr5qKBGypuYyGTGtzp5BNL6dgQhi3XAVg7yJVEMd3RsYtenrfrpijHWD0aoDMo7xOvme8vy5CAWHYEfpux9yIriz8rvfii3cfVK9dpBV58CZm5NK993z2fDI0aX7-HARAYv0V3TgVBiKh-z8Ws7uOvtekHJUz1ML0s2q2t-h8LsKRs98ho5V2L2mkuoXk3DYMimZKbjDvZGSwwWH5JbLuC_aIEuTuA-DO3i5BypejqMudsOugDVsivcmlHxDcaIXypSeJ1GTgL1sWPw1QwN5wUV8z7334fjeoFllB_IrsZuPNf0v-bpVUyPxKcCVKiN6EjkFH9dxA_khRIWBLra_HrKYVkqdw6bp3-F0Z4XFUHWni3pi9UL7kQEPCN1YAvwoaEpx46_DIyDlNDWuufrP_iVuKQmXCLk1WE0o4PbvFRj9nFHulaI4kLERWmjSudkedK_Nb4tFZ6XlHqiDhNtHhmR8OqfpvnQACn5yys4L7QqQ_Kw3sXk-ZfihzKJklHK5uAlPnCvuVNg0gLOaL4fk7_sRuO9NJynjlrr74VPDedkICEEpuijhq2B7Y6BAhSuJwkSPsmUdawQVyooUuYNv'
    # response = detect_text_uri(uri)

    images = os.listdir(os.path.join(os.path.dirname(__file__), 'images'))

    for image in images:
        image_file = os.path.join('images', image)
        print(image)
        response = detect_text(image_file)
        texts = response.text_annotations
        print('Texts: ')
        for text in texts:
            print('\n"{}"'.format(text.description))
            vertices = (['({},{})'.format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
            print('bounds: {}'.format(','.join(vertices)))


if __name__ == '__main__':
    get_text_local()

# IMAG0001.jpg       => 79042895113 . 10013017 21:26:53      [135170042895223 - 20/01/2017 21:26:53]
# IMAG0001.JPG-0.jpg => 79042$95uJ· 2001/2917 21:26:53
# IMAG0001.JPG-0.png => 79042$9su3 2001/201721.26:53

# IMAG0004.jpg       => None
# IMAG0004.JPG-7.jpg => None
# IMAG0004.JPG-7.png => None


# IMAG0005.jpg       => 135160634218030 1/10/2016 20:47:20   [135160634218030 11/10/2016 20:47:20]
# IMAG0005.JPG-1.jpg => 135160634218030 11/10/2016 20:47:20
# IMAG0005.JPG-1.png => 135160634218030 11/10/2016 20:47:20

# IMAG0012.jpg       => 13516045021421922072016 I-46:23      [13516045021439 22/07/2016 17:46:23]
# IMAG0012.JPG-0.jpg => 35160430214239 22 07 2016 1746:23
# IMAG0012.JPG-0.png => 1351 6045021 4210 22 0-2016 46:23

# IMAG0015.jpg       => 3170347554141 02/06/2017 14:34:09    [135170347554141 02/06/2017 14:34:09]
# IMAG0015.JPG-2.jpg => 35170347554141 02/06/2017 14:34:09
# IMAG0015.JPG-2.png => 3170347554141 02/06/201714:34:09

# IMAG0020.jpg       => 1351 70235049204 15042017 1L         [135170235049204 15/04/2017 11:01:13]
# IMAG0020.JPG-3.jpg => 1351 70235049204 15042017 1L
# IMAG0020.JPG-3.png => 1351 70235049204 15042017 1L

# IMAG0024.jpg       => 333170068422725 29/05/2017 07 08:10  [333170068422725 29/05/2017 07:08:10]
# IMAG0024.JPG-1.jpg => 333170068422725 29/05/2017 0708:10
# IMAG0024.JPG-1.png => 333170068422725 29/05/2017 0708:10

# IMAG0029.jpg       => :99 2606L 22104                      [135160612216490 - 01/10/2016 21:39:24]
# IMAG0029.JPG-2.jpg => 6113934 1606L 22104
# IMAG0029.JPG-2.jpg => 6113934 1606L

# IMAG0036.jpg       => 135170512118604 11/08/2017 13:17:21  [135170512118604 11/08/2017 13:17:21] correct totally
# IMAG0036.JPG-0.jpg => 35170512118804 11/08/2017 13:17:21
# IMAG0036.JPG-0.png => 35170512118804 11/08/2017 13:17:21


# IMG_20170913_133052945[1].jpg-6.jpg => 70185927988 24103/2017 20:04:24 [135170186927988 24/03/2017 20:04:24]
# IMG_20170913_133052945[1].jpg-6.png => 70185927988 24103/2017 20:04:24

# img_40.jpg        => None
# img_40.jpg-3.jpg  => 129:70055035761 22/05/2017 14:13 [129170055035761 22/05/2017 14:13:00]
# img_40.jpg-3.png  => None

# img_45.jpg        => 135170254600793 25042017 211         [135170254600793 25/04/2017 21:11:56]
# img_45.jpg-11.jpg => 135 1702S00%) 2MM 2017 21:11:56
# img_45.jpg-11.png => 135 1702S00%) 2MM 2017 21:11:56


# img_87.jpg       => 35170348565377 02/06/2017 2036       [13517034*565377 02/06/2017 20:36:27]
# img_87.jpg-3.jpg => 35170348565377 02/06/2017 20 36 27
# img_87.jpg-3.png => 35170348565377 02/06/2017 20 36 27

# NF_050.jpg       => 131172632416156 01/08/2017 08:28:39  [131172632416156 - 01/08/2017 08:28:39]
# NF_050.jpg-0.jpg => 131172632416156 - 01/08/2017 08:28:39
# NF_050.jpg-0.png => 131172632416156 01/08/2017 08:28:39


# NF_85.jpg            => 1351 60818781493 24/12/2016 12:18:12 [135160811781493 24/12/2016 12:18:12]
# NF_85.jpg-1.jpg      => 1351 60818781493 24/12/2016 12:18:12
# NF_85.jpg-1.png      => 1351 60818781493 24/12/2016 12:18:12

# sc0064.jpg           => 35370224342438 20/04/2017 8 2 4      [135170224342438 10/04/2017 10:21:43]
# sc0064.png-0.jpg     => 35570224342438 10/04/2017 28 2:4
# sc0064.png-0.png     => 135170224342438 10/04/2017 28 2 4

# sc0074(1).jpg        => 141160204598162 23/122016 09:01:4    [141160204598162 23/12/2016 09:01:49]
# sc0074 (1).png-4.jpg => 141160204598162 23/122016 09:01:4
# sc0074 (1).png-4.png => 141160204598162 23/122016 09:01:4

# sc0074(2).jpg        => 141160204598162 23/122016 09:01:4    [141160204598162 23/12/2016 09:01:49]
# sc0074 (2).png-4.jpg => 141160204598162 23/122016 09:01:4
# sc0074 (2).png-4.png => 141160204598162 23/122016 09:01:4

# sc0074.jpg           => 141160204598162 23/122016 09:01:4    [141160204598162 23/12/2016 09:01:49]
# sc0074.png-4.jpg     => 141160204598162 23/122016 09:01:4
# sc0074.png-4.png     => 141160204598162 23/122016 09:01:4
