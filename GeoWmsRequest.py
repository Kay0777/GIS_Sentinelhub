from sentinelhub import GeopediaWmsRequest, WmsRequest
from sentinelhub import BBox, ServiceType, CRS
from sentinelhub import CustomUrlParam, MimeType
from PIL import Image
from io import BytesIO
import math


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

# def world(lon, lat):
#     if lat > 89.5:
#         lat = 89.5

#     if lat < -89.5:
#         lat = -89.5

#     rLat = math.radians(lat)
#     rLong = math.radians(lon)

#     a = 6378137.0
#     b = 6356752.3142
#     f = (a - b) / a
#     e = math.sqrt(2 * f - f**2)
#     x = a * rLong
#     y = a * math.log(math.tan(math.pi / 4 + rLat / 2) * ((1 - e * math.sin(rLat)) / (1 + e * math.sin(rLat)))**(e / 2))
#     return {'x': x, 'y': y}


# res = world(37.617778, 55.751667)
# print(res['x'], res['y'])


from sentinelhub import SHConfig

INSTANCE_ID = '03d65be2-ebd2-4c6a-8f84-1a0a7c45f929'


if INSTANCE_ID:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
else:
    config = None


def change(lon, lat):
    alpha = 20037508.34 / 180
    x = lon / alpha
    y = ((360 / math.pi) * math.atan(math.exp((lat / alpha) * math.pi / 180))) - 90

    return x, y


# x, y = change(7200979.560689886, 5322463.153553395)
# print(x, y)


def check1():
    bbox = [61.875000008613085, 40.979898074349165, 64.6875000090046, 43.06888777903171]
    bbox = [5138507.695017507, -1810221.6137981834, 5177469.516795154, -1744665.2835185674]
    # bbox = [61.5230, 40.58633, 64.4115, 43.50]
    geo_wms = WmsRequest(
        # request=None,
        layer='TRUE_COLOR',
        # styles=None,
        image_format=MimeType.JPG,
        custom_url_params={
            CustomUrlParam.TRANSPARENT: False,
            CustomUrlParam.SHOWLOGO: False
        },
        width=512,
        height=512,
        # maxcc=0.5,
        # bbox=BBox(bbox=[5009377.085697314, 6887893.492833803, 5322463.153553395, 7200979.560689886], crs=CRS.POP_WEB),
        # bbox=BBox(bbox=bbox, crs=CRS.WGS84),
        bbox=BBox(bbox=[5009377.085697314, 6887893.492833803, 5322463.153553395, 7200979.560689886], crs='EPSG:3857'),
        config=config
    )
    return geo_wms.get_data(decode_data=False)


def check2():
    geo_wms = GeopediaWmsRequest(
        # request=None,
        layer='SWIR',
        # styles=None,
        image_format=MimeType.PNG,
        custom_url_params={
            CustomUrlParam.TRANSPARENT: False
        },
        theme=None,
        width=512,
        height=512,
        bbox=BBox(bbox=[6887893.492833803, 5009377.085697314, 7200979.560689886, 5322463.153553395], crs=CRS.POP_WEB),
        config=config
    )

    return geo_wms.get_data(decode_data=False)


img_byte = BytesIO(check1()[-1])
img = Image.open(img_byte)
img.save('img.JPG')
