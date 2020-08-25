from sentinelhub import SHConfig
INSTANCE_ID = '03d65be2-ebd2-4c6a-8f84-1a0a7c45f929'

if INSTANCE_ID:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
else:
    config = None

import numpy as np

import matplotlib.pyplot as plt


def plot_image(image, factor=1):
    """
    Utility function for plotting RGB images.
    """
    fig = plt.subplots(nrows=1, ncols=1, figsize=(5, 7))

    if np.issubdtype(image.dtype, np.floating):
        plt.imshow(np.minimum(image * factor, 1))
    else:
        plt.imshow(image)
    plt.show()


from sentinelhub import WmsRequest, WcsRequest
from sentinelhub import MimeType, CRS, BBox, CustomUrlParam
from sentinelhub import BBoxCollection, BBoxSplitter, get_area_dates
from sentinelhub import Geometry, OsmSplitter
from sentinelhub.data_request import GeopediaRequest
from sentinelhub.constants import ServiceType


def utils_args():
    coor = [6887893.492833803, 5009377.085697314,
            7200979.560689886, 5322463.153553395]

    # coor = [7200979.560689886, 5009377.085697314,
    #         7514065.628545968, 5322463.153553395]

    coor = [7712190.405861145, 5053404.813989572,
            7714636.390766275, 5055850.798894699]

    coor = [6887893.492833803, 7200979.560689886,
            5009377.085697314, 5322463.153553395]

    coor = [7200979.560689886, 7514065.628545968,
            5009377.085697314, 5322463.153553395]
    bbox = BBox(bbox=coor, crs=CRS.POP_WEB)
    bbox = bbox.transform(crs=CRS.WGS84)
    coor = [45.0, 43.07, 64.69, 55.78]
    # print(coor)
    # coor = [64.4784, 39.6 / 648, 64.8162, 39.9222]
    # coor = [46.2014, -15.9906, 46.6051, -15.5961]
    # print(bbox)
    return bbox


# utils_args()


def get_polygon():
    polies = "POLYGON ((61.87500000000001 40.97989806962016,61.87500000000001 43.06888777416963, 64.68750000000001 43.06888777416963, 64.68750000000001 40.97989806962016, 61.87500000000001 40.97989806962016))"

    return polies


def main1():
    bbox = utils_args()

    _geo = GeopediaRequest(
        layer='TRUE_COLOR',
        service_type=ServiceType.WMS,
        bbox=BBox(bbox=bbox, crs=CRS.WGS84),
        theme=None,
        image_format=MimeType.JPG,
        config=config
    )
    img = _geo.get_data()

    plt.imshow(img[0])
    plt.show()


# main1()


def main():
    _bbox = utils_args()
    wms_request = WmsRequest(
        layer='TRUE_COLOR',
        # bbox=BBox(
        #     bbox=[7215306.379154978, 4854643.0279353345,
        #           7177702.65516501, 4817350.823068645],
        #     crs=CRS.POP_WEB),
        bbox=_bbox,
        width=512,
        height=512,
        maxcc=0.3,
        custom_url_params={
            CustomUrlParam.SHOWLOGO: False,
        },
        image_format=MimeType.JPG,
        config=config)
    # print(wms_request.get_url_list())
    wms_data = wms_request.get_data()
    # print(len(wms_data))
    # print(wms_data[0].shape)
    # plot_image(wms_data[0])
    # print(len(wms_data))
    plt.imshow(wms_data[0])
    plt.show()


main()
