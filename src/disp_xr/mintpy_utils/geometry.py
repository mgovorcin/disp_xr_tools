import warnings
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from pyproj import Transformer

import asf_search as asf
import dem_stitcher, tile_mate
from opera_utils.geometry import stitch_geometry_layers
from mintpy.utils import writefile, utils as ut

def _reproject_raster(output_name,  
                      src_data, atr,
                      src_transform, dst_transform,
                      src_crs, dst_crs,
                      dtype='float32',
                      resampling_mode = Resampling.bilinear):
    
    with rasterio.open(output_name, 'w', 
                        height=atr['rows'],
                        width=atr['cols'],
                        count=1,
                        dtype=dtype,
                        crs=dst_crs,
                        transform=dst_transform) as dst: 
        
        reproject(source=src_data,
            destination=rasterio.band(dst, 1),
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_mode)

# I have this function as duplicate, fix it later      
def open_image(file):
    with rasterio.open(file) as dataset:
        # Read the data into an array (e.g., the first band)
        data = dataset.read(1)
        
        # Get some metadata
        width = dataset.width
        height = dataset.height
        crs = dataset.crs
        bounds = dataset.bounds
        gt = dataset.transform
    return data, {'width':width, 'height':height,
                  'crs':crs, 'bounds':bounds, 'gt':gt}

def prepare(metadata:dict,
                 burst_ids: list,
                 work_dir:str,
                 num_threads:int=5):
    print('Generating MINTPY geometry cube') 
    # Make static directory
    static_dir = Path(work_dir) / 'static_lyr'
    static_dir.mkdir(exist_ok=True)

    temp_dir = Path(work_dir) / 'temp_geometry'
    temp_dir.mkdir(exist_ok=True)

    # Download static layers
    results = asf.search(
        operaBurstID=list(burst_ids),
        processingLevel='CSLC-STATIC',
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        results.download(path=static_dir, processes=num_threads)

    list_static_files = [Path(f'{static_dir}/{results[ii].properties["fileName"]}')
                        for ii in range(len(results)) ] 

    print('number of static layer files to download: ', len(results))

    # generating los_east.tif and los_north.tif from downloaded static layers
    _ = stitch_geometry_layers(list_static_files,
                               output_dir=static_dir)

    los_east, los_east_atr = open_image(static_dir / 'los_east.tif')
    los_north, _ = open_image(static_dir / 'los_north.tif')

    # Get incidence and azimuth 
    az_angle = -1 * np.rad2deg(np.arctan2(los_east, los_north)) % 360
    up = np.sqrt(1 - los_east**2 - los_north**2)
    incidence_angle = np.rad2deg(np.arccos(up))
    
    print('Reproject incidence angle')
    atr = dict(rows=metadata['LENGTH'], cols=metadata['WIDTH'],
               bounds=los_east_atr['bounds'])
    
    # Reproject incidence & azimuth angle
    print('Save incidence angle: ', {Path(temp_dir) / 'incidence_angle.tif'})
    _reproject_raster(Path(temp_dir) / 'incidence_angle.tif',
                      incidence_angle,
                      atr, los_east_atr['gt'], metadata['GT'], 
                      los_east_atr['crs'], metadata['crs'],
                      'float32',
                      Resampling.bilinear)
    
    print('Save azimuth angle: ', {Path(temp_dir) / 'azimuth_angle.tif'})
    _reproject_raster(Path(temp_dir) / 'azimuth_angle.tif',
                      az_angle,
                      atr, los_east_atr['gt'], metadata['GT'], 
                      los_east_atr['crs'], metadata['crs'],
                      'float32',
                      Resampling.bilinear)

    # Init projection converter
    transformer = Transformer.from_crs(f'EPSG:{los_east_atr["crs"].to_epsg()}',
                                       "EPSG:4326",
                                       always_xy=True)

    # Find bounds
    snwe = np.zeros((4))
    snwe[2], snwe[1] = transformer.transform(atr['bounds'].left-1e3,
                                             atr['bounds'].top+1e3)
    snwe[3], snwe[0] = transformer.transform(atr['bounds'].right+1e3,
                                             atr['bounds'].bottom-1e3)

    bounds = [snwe[2], snwe[0], snwe[3], snwe[1]]

    # DEM
    print('Download DEM')
    dat_arr, dat_prof = dem_stitcher.stitch_dem(bounds,
                                                dem_name='glo_30',
                                                dst_ellipsoidal_height=True,
                                                dst_area_or_point='Point')
    
    print('Save DEM: ', {Path(temp_dir) / 'dem.tif'})
    _reproject_raster(Path(temp_dir) / 'dem.tif',
                      dat_arr,
                      atr, dat_prof['transform'], metadata['GT'], 
                      dat_prof['crs'], metadata['crs'],
                      'float32',
                      Resampling.bilinear) 

    # WATER MASK
    print('Download Water Mask')
    X, p = tile_mate.get_raster_from_tiles(bounds,
                                           tile_shortname='esa_world_cover_2021')

    # Make byte mask
    X[X == 80] = 0
    X[X != 0] = 1
    X = X.astype('byte')

    # Save
    print('Save water mask: ', {Path(temp_dir) / 'water_mask.tif'})
    _reproject_raster(Path(temp_dir) / 'water_mask.tif',
                      X,
                      atr, p['transform'], metadata['GT'], 
                      p['crs'], metadata['crs'],
                      'uint8',
                      Resampling.nearest)
    
    # Make Mintpy cube
    print('\nCreating MINTPY cube: ', Path(work_dir) / 'geometry.h5')
    inc, atr = open_image(Path(temp_dir) / 'incidence_angle.tif')
    azi, atr = open_image(Path(temp_dir) / 'azimuth_angle.tif')
    mask = np.ma.masked_equal(inc, 0).mask
    dem, atr = open_image(Path(temp_dir) / 'dem.tif')
    dem = np.ma.masked_array(dem, mask=mask).filled(0)
    water, atr = open_image(Path(temp_dir) / 'water_mask.tif')
    water = np.ma.masked_array(water, mask=mask).filled(0)


    # Save mintpy geometry
    meta = {key: value for key, value in metadata.items()}
    meta["FILE_TYPE"] = "geometry"

    dsDict = {
        "incidenceAngle": inc,
        "azimuthAngle": azi,
        "height": dem,
        "waterMask" : water,
    }

    writefile.write(dsDict, Path(work_dir) / 'geometry.h5', metadata=meta)  