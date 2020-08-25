from sentinelhub import SHConfig
import math

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
from sentinelhub import OsmSplitter


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


from sentinelhub import CustomUrlParam
from sentinelhub import DataSource


def world(lon, lat):
    if lat > 89.5:
        lat = 89.5

    if lat < -89.5:
        lat = -89.5

    rLat = math.radians(lat)
    rLong = math.radians(lon)

    print(rLat)
    print(rLong)

    a = 6378137.0
    b = 6356752.3142
    f = (a - b) / a
    e = math.sqrt(2 * f - f**2)

    x = a * rLong
    y = a * math.log(math.tan(math.pi / 4 + rLat / 2) * ((1 - e * math.sin(rLat)) / (1 + e * math.sin(rLat)))**(e / 2))
    return x, y
# [46.16, -16.15, 46.51, -15.58]


# x, y = world(46.51, -15.58)
# print(x, y)


class WCSRequest:
    # ------>>>>>>>>>>> Custom URL Parametres <<<<<<<<<<<-------- #
    @staticmethod
    def get_custom_url_params():
        for custom in CustomUrlParam:
            print(custom)

    # ------>>>>>>>>>>> Data Sources <<<<<<<<<<<-------- #
    @staticmethod
    def get_available_srcs():
        for sent in DataSource.get_available_sources():
            print(sent)

    _coords_wgs84 = [61.875000, 40.979898, 61.9, 41.162114]

    # _coords_wgs84 = [46.16, -16.15, 46.51, -15.58] or [5138507.695018, -1822100.150029, 5177469.516795, -1756135.024601]
    # _coords_wgs84 = [6887893.492833803, 5009377.085697314, 7200979.560689886, 5322463.153553395]
    _coords_wgs84 = [5138507.695018, -1822100.150029, 5177469.516795, -1756135.024601]
    # osm_splitter = OsmSplitter(shape_list=((6887893.492833803, 5009377.085697314, 6887893.492833803, 5322463.153553395, 7200979.560689886, 5322463.153553395, 7200979.560689886, 5009377.085697314, 6887893.492833803, 5009377.085697314)), crs=CRS.POP_WEB, zoom_level=5)
    _bbox = BBox(bbox=_coords_wgs84, crs=CRS.POP_WEB)
    osm_splitter = OsmSplitter(shape_list=[_bbox.geometry], crs=CRS.POP_WEB, zoom_level=50)
    # print(_bbox.geometry)
    __bbox = osm_splitter.get_world_bbox()
    # _bbox = BBox(bbox=_coords_wgs84, crs=CRS.WGS84)

    # ************** True color with specified resolution ************** #
    def request1(self):
        wcs_true_color_request = WcsRequest(
            layer='TRUE_COLOR',
            bbox=self._bbox,
            # time='2017-12-15',
            resx='10m',
            resy='10m',
            custom_url_params=self.customUrlParam,
            config=config
        )

        wcs_true_color_img = wcs_true_color_request.get_data()
        print('''Single element in the list is of type = {} and has shape {}
            '''.format(type(wcs_true_color_img[-1]),
                       wcs_true_color_img[-1].shape))
        plot_image(wcs_true_color_img[-1])

    # ************** Using Custom URL Parameters ************** #
    def request2(self):
        custom_wms_request = WmsRequest(
            layer='SWIR',
            bbox=self.__bbox,
            # time='2019-11-05',
            width=512,
            height=512,
            maxcc=100,
            custom_url_params={
                # CustomUrlParam.ATMFILTER: 'ATMCOR',
                CustomUrlParam.TRANSPARENT: True,
                CustomUrlParam.SHOWLOGO: False
            },
            image_format=MimeType.PNG,
            config=config
        )
        wms_data = custom_wms_request.get_data()
        print(len(wms_data))
        plot_image(wms_data[0])

    # ************** Evalscript ************** #
    def request3(self):
        eval_script = '''
        var bRatio = (B01 - 0.175) / (0.39 - 0.175);
        var NGDR = (B01 - B02) / (B01 + B02);

        function clip(a) {
            return a>0 ? (a<1 ? a : 1) : 0;
        }

        if (bRatio > 1) {
            var v = 0.5*(bRatio - 1);
            return [0.5*clip(B04), 0.5*clip(B03), 0.5*clip(B02) + v];
        }

        if (bRatio > 0 && NGDR > 0) {
            var v = 5 * Math.sqrt(bRatio * NGDR);
            return [0.5 * clip(B04) + v, 0.5 * clip(B03), 0.5 * clip(B02)];
        }

        return [2*B04, 2*B03, 2*B02];
        '''

        evalscript_wms_request = WmsRequest(
            layer='TRUE_COLOR',
            bbox=self._bbox,
            time='2017-12-20',
            width=512,
            custom_url_params={
                CustomUrlParam.EVALSCRIPT: eval_script,
                CustomUrlParam.SHOWLOGO: False
            },
            config=config
        )

        evalscript_wms_data = evalscript_wms_request.get_data()
        plot_image(evalscript_wms_data[0])

    # ************** Evalscript URL ************** #
    def request4(self):
        _url = 'https://raw.githubusercontent.com/sentinel-hub/custom-scripts/master/sentinel-2/ndmi_special/script.js'

        evalscripturl_wms_request = WmsRequest(
            layer='TRUE_COLOR',  # Layer parameter can be any existing layer
            bbox=self._bbox,
            time='2017-12-20',
            width=512,
            custom_url_params={
                CustomUrlParam.EVALSCRIPTURL: _url,
                CustomUrlParam.SHOWLOGO: False
            },
            config=config
        )

        evalscripturl_wms_data = evalscripturl_wms_request.get_data()
        plot_image(evalscripturl_wms_data[0])

    # ************** Sentinel-2 L2A ************** #
    def request5(self):
        volcano_bbox = BBox(bbox=[(-2217485.0, 9228907.0), (-2150692.0, 9284045.0)], crs=CRS.POP_WEB)

        l2a_request = WmsRequest(
            data_source=DataSource.SENTINEL2_L2A,
            layer='TRUE_COLOR',
            bbox=volcano_bbox,
            time='2017-08-30',
            width=512,
            custom_url_params={CustomUrlParam.SHOWLOGO: False},
            config=config
        )

        l2a_data = l2a_request.get_data()
        plot_image(l2a_data[0])

    # ************** DEM ************** #
    def request6(self):
        dem_request = WmsRequest(
            data_source=DataSource.DEM,
            layer='DEM',
            bbox=self._bbox,
            width=512,
            image_format=MimeType.TIFF_d32f,
            custom_url_params={CustomUrlParam.SHOWLOGO: False},
            config=config
        )

        dem_image = dem_request.get_data()[0]

        plot_image(dem_image, 1 / np.amax(dem_image))

    # ************** Landsat8 ************** #
    def request7(self):
        l8_request = WmsRequest(
            data_source=DataSource.LANDSAT8,
            layer='TRUE_COLOR',
            bbox=self._bbox,
            time='2017-08-20',
            width=512,
            config=config
        )

        l8_data = l8_request.get_data()

        plot_image(l8_data[-1])

    # ************** Sentinel-1 ************** #
    def request8(self):
        s1_request = WmsRequest(
            data_source=DataSource.SENTINEL1_IW,
            layer='TRUE_COLOR',
            bbox=self._bbox,
            time='2017-08-20',
            width=512,
            config=config
        )

        s1_data = s1_request.get_data()

        plot_image(s1_data[-1])

    # ************** Sentinel-1, ascending orbit direction ************** #
    def request9(self):
        s1_asc_request = WmsRequest(
            # data_source=DataSource.SENTINEL1_IW_ASC,
            layer='TRUE_COLOR',
            bbox=self._bbox,
            time=('2017-10-03', '2017-10-05'),
            width=512,
            config=config
        )

        s1_asc_data = s1_asc_request.get_data()
        plot_image(s1_asc_data[-1])


boj = WCSRequest()
boj.request2()
