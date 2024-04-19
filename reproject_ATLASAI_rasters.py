"""
Rasters from Atlas AI that contain data for the case study are in a weird projection.
This script reprojects the rasters to EPSG:4326 and saves them in a new folder.
"""
import  glob
import code
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from raster2xyz.raster2xyz import Raster2xyz
if __name__=="__main__":
    target_crs = {'init': 'EPSG:4326'}
    data_files = sorted(glob.glob('Atlas_AI_data/Spending/*.tiff'))
    for file_name in data_files:
        current_raster = rasterio.open(file_name)
        code.interact(local=locals())
        transform, width, height = calculate_default_transform(current_raster.crs, target_crs, current_raster.width, current_raster.height, *current_raster.bounds)
        kwargs = current_raster.meta.copy()
        kwargs.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
        save_file = 'ATLAS_AI_FINAL/spending/{}'.format(file_name.split('/').pop())
        dstRst = rasterio.open(save_file, 'w', **kwargs)
        for i in range(1, current_raster.count + 1):
            reproject(source=rasterio.band(current_raster, i), destination=rasterio.band(dstRst, i), src_crs=current_raster.crs, dst_crs=target_crs, resampling=Resampling.nearest)
        dstRst.close()
        print(f"saved: {save_file}")
        del dstRst

    data_files = sorted(glob.glob("ATLAS_AI_FINAL/spending/*.tiff"))
    for file_name in data_files:
        rtxyz = Raster2xyz()
        save_name_csv = 'ATLAS_AI_FINAL/spending_csv/{}'.format(file_name.split('/').pop().replace('tiff', 'csv'))
        rtxyz.translate(file_name, save_name_csv)
    # code.interact(local=locals())
