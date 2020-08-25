# *******-------------------->>>>>>>>>>>DOCUMENTATION <<<<<<<<<<<<-----------*************
# https://sentinelhub-py.readthedocs.io/en/latest/


from sentinelhub import SHConfig

INSTANCE_ID = '03d65be2-ebd2-4c6a-8f84-1a0a7c45f929'


if INSTANCE_ID:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
else:
    config = None

import datetime
import numpy as np

import matplotlib.pyplot as plt
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox
from sentinelhub import CustomUrlParam, Geometry, FisRequest
from shapely.geometry import Polygon
# import sentinelhub
# print(sentinelhub.__version__)


def plot_image(image, factor=1):
    """
    Utility function for plotting RGB images.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image * factor, 1))
    else:
        plt.imshow(image)
    plt.show()
# ?service = WMS
# &request = GetMap
# &layers = SWIR
# &styles =
# &format = image / jpeg
# &transparent = false
# &version = 1.1.1
# &maxcc = 10
# &width = 256
# &height = 256
# &srs = EPSG: 3857
# &bbox = 6887893.492833803, 5009377.085697314, 7200979.560689886, 5322463.153553395


class WMSRequest:
    # ************** True color (PNG) on a specific date ************** #
    bbox = [6887893.492833803, 5009377.085697314, 7200979.560689886, 5322463.153553395]
    # bbox = [46.16, -16.15, 46.51, -15.58]
    _bbox = BBox(bbox=bbox, crs=CRS.POP_WEB)
    geometry = Geometry(
        Polygon([
                (6887893.0, 5422463.0),
                (7200979.0, 5009377.0),
                (7210979, 5422463.0),
                (7200979.0, 5209377.0),
                (6887893.0, 5422463.0)
                ]),
        CRS.POP_WEB)

    def request1(self):
        wms_request = FisRequest(
            layer='TRUE_COLOR',
            geometry_list=[self._bbox, self.geometry],
            # bbox=self._bbox,
            custom_url_params={
                CustomUrlParam.SHOWLOGO: False
            },
            width=512,
            height=512,
            config=config
        )
        img = wms_request.get_data()
        # print(len(img))
        # print(type(img))
        # print(img)
        plot_image(img[-1])

    # ************** True color of the latest acquisition ************** #
    def request2(self):
        wms_true_color_request = WmsRequest(
            layer='TRUE-COLOR',
            bbox=betsiboka_bbox,
            time='latest',
            width=512,
            height=856,
            config=config
        )
        wms_true_color_img = wms_true_color_request.get_data()
        plot_image(wms_true_color_img[-1])
        print('The latest Sentinel-2 image of this area was taken on {}.'.format(
            wms_true_color_request.get_dates()[-1])
        )

    # ************** True color of the multiple acquisitions in certain time window ************** #
    def request3(self):
        wms_true_color_request = WmsRequest(
            layer='TRUE-COLOR',
            bbox=betsiboka_bbox,
            time=('2017-12-01', '2017-12-31'),
            width=512,
            height=856,
            config=config
        )
        wms_true_color_img = wms_true_color_request.get_data()
        print('There are %d Sentinel-2 images available for December 2017.' % len(wms_true_color_img))
        plot_image(wms_true_color_img[2])

        print('These %d images were taken on the following dates:' % len(wms_true_color_img))
        for index, date in enumerate(wms_true_color_request.get_dates()):
            print(' - image %d was taken on %s' % (index, date))

    # ************** True color of the multiple acquisitions in certain time window with cloud coverage less than 30% ************** #
    def request4(self):
        wms_true_color_request = WmsRequest(
            layer='TRUE-COLOR',
            bbox=betsiboka_bbox,
            time=('2017-12-01', '2017-12-31'),
            width=512, height=856,
            maxcc=0.3,
            config=config
        )
        wms_true_color_img = wms_true_color_request.get_data()
        print('There are %d Sentinel-2 images available for December 2017 with cloud coverage less '
              'than %1.0f%%.' % (len(wms_true_color_img), wms_true_color_request.maxcc * 100.0))
        plot_image(wms_true_color_img[-1])
        print('These %d images were taken on the following dates:' % len(wms_true_color_img))

        for index, date in enumerate(wms_true_color_request.get_dates()):
            print(' - image %d was taken on %s' % (index, date))

    # ************** All Sentinel-2’s raw band values ************** #
    def request5(self):
        wms_bands_request = WmsRequest(
            layer='BANDS-S2-L1C',
            bbox=betsiboka_bbox,
            time='2017-12-15',
            width=512,
            height=856,
            image_format=MimeType.TIFF_d32f,
            config=config
        )

        wms_bands_img = wms_bands_request.get_data()
        print("Shape:", wms_bands_img[-1][:, :, 12].shape)
        plot_image(wms_bands_img[-1][:, :, 12])
        plot_image(wms_bands_img[-1][:, :, [3, 2, 1]], 2.5)

    # ************** All Sentinel-2’s raw band values ************** #
    def request6(self):
        wms_bands_request = WmsRequest(
            data_folder='test_dir',
            layer='BANDS-S2-L1C',
            bbox=betsiboka_bbox,
            time='2017-12-15',
            width=512,
            height=856,
            image_format=MimeType.TIFF_d32f,
            config=config
        )
        wms_bands_img = wms_bands_request.get_data(save_data=True)
        import os

        for folder, _, filenames in os.walk(wms_bands_request.data_folder):
            for filename in filenames:
                print(os.path.join(folder, filename))

        wms_bands_request_from_disk = WmsRequest(
            data_folder='test_dir',
            layer='BANDS-S2-L1C',
            bbox=betsiboka_bbox,
            time='2017-12-15',
            width=512,
            height=856,
            image_format=MimeType.TIFF_d32f,
            config=config
        )
        wms_bands_img_from_disk = wms_bands_request_from_disk.get_data()
        if np.array_equal(wms_bands_img[-1], wms_bands_img_from_disk[-1]):
            print('Arrays are equal.')
        else:
            print('Arrays are different.')
        wms_bands_img_redownload = wms_bands_request_from_disk.get_data(redownload=True)

    # ************** Save downloaded data directly to disk ************** #
    def request7(self):
        wms_true_color_request = WmsRequest(
            data_folder='test_dir_tiff',
            layer='TRUE-COLOR',
            bbox=betsiboka_bbox,
            time=('2017-12-01', '2017-12-31'),
            width=512,
            height=856,
            image_format=MimeType.TIFF,
            config=config
        )
        wms_true_color_request.save_data()
        os.listdir(wms_true_color_request.data_folder)

    # ************** Merging two or more download requests into one ************** #
    def request8(self):
        print("asdasd")
        betsiboka_bbox_large = BBox([45.88, -16.12, 47.29, -15.45], crs=CRS.WGS84)

        wms_true_color_request = WmsRequest(
            layer='AGRICULTURE',
            bbox=betsiboka_bbox_large,
            time='2015-12-01',
            width=960,
            image_format=MimeType.PNG,
            config=config
        )

        wms_true_color_img = wms_true_color_request.get_data()
        plot_image(wms_true_color_img[0])
        plot_image(wms_true_color_img[1])

        wms_true_color_request_with_deltat = WmsRequest(
            layer='AGRICULTURE',
            bbox=betsiboka_bbox_large,
            time='2015-12-01',
            width=960,
            image_format=MimeType.PNG,
            time_difference=datetime.timedelta(hours=2),
            config=config
        )

        wms_true_color_img = wms_true_color_request_with_deltat.get_data()
        print('These %d images were taken on the following dates:' % len(wms_true_color_img))
        for index, date in enumerate(wms_true_color_request_with_deltat.get_dates()):
            print(' - image %d was taken on %s' % (index, date))
        plot_image(wms_true_color_img[-1])


obj = WMSRequest()
obj.request1()
