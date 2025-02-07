#!/usr/bin/env python3
# Load modules
import re
import yaml
import argparse
import warnings

import h5py
import rasterio
import random
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from concurrent.futures import ThreadPoolExecutor, as_completed

import asf_search as asf
import dem_stitcher, tile_mate
from opera_utils.geometry import stitch_geometry_layers
from mintpy.utils import writefile, utils as ut

# UTILS
def _get_filename_info(products_path: Path) -> pd.DataFrame:
    """
    Get filename information from OPERA DISP products.

    Args:
        products_path (Path): The path to the OPERA DISP products.

    Returns:
        pd.DataFrame: A DataFrame containing the filename information.

    """
    # Get all OPERA DISP products in the specified path
    disp_products = list(Path(products_path).rglob('*.nc'))
    print(f'Found OPERA DISP: {len(disp_products)} products')

    # Read the metadata from the filenames
    disp_df = pd.DataFrame([product.stem.split('_') for product in disp_products],
                            columns=['project', 'level', 'product', 'mode', 'frame_id',
                                     'polarizarion', 'start_date', 'end_date', 'version',
                                     'production_date'])
    
    # Add path, first and second date
    disp_df['path'] = disp_products
    disp_df['date12'] = [f'{d.start_date.split("T")[0]}_{d.end_date.split("T")[0]}'
                        for i, d in  disp_df.iterrows()]
    disp_df['date1'] = [f'{d.start_date.split("T")[0]}' for i, d in  disp_df.iterrows()]
    disp_df['date2'] = [f'{d.end_date.split("T")[0]}' for i, d in  disp_df.iterrows()]

    # Dates string to datetime
    format = '%Y%m%dT%H%M%SZ'
    disp_df['start_date'] = pd.to_datetime(disp_df['start_date'], format=format)
    disp_df['end_date'] = pd.to_datetime(disp_df['end_date'], format=format)
    print(' Starting date:', disp_df.start_date.min())
    print(' Ending date:', disp_df.end_date.max())

    # Group by reference dates
    groupped_df = disp_df.groupby(['date1', 'date2']).apply(lambda x: x, include_groups=False)
    reference_dates = groupped_df.index.get_level_values(0).unique()
    print(' Number of reference dates:', len(reference_dates))

    return disp_df

def _find_duplicates(input_df):
    """
    Find and remove duplicates in the input DataFrame based on the 'date12' column.

    Args:
        input_df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        tuple: A tuple containing two DataFrames. The first DataFrame is the input DataFrame
        with the duplicates removed, and the second DataFrame contains the removed duplicate rows.

    """
    input_df2 = input_df.copy()
    list_duplicates = input_df2.date12.value_counts() 
    duplicates = list_duplicates[list_duplicates.values > 1].index

    duplicate_list = []
    for date in duplicates:
        selected_df = input_df2[input_df2.date12 == date]
        latest_production_date = selected_df.production_date.max()
        for ix, key in (selected_df.production_date != latest_production_date).items():
            if key == True:
                input_df2.drop(ix, inplace=True)
            else:
                duplicate_list.append(input_df.iloc[ix])
    
    if len(duplicate_list) > 0:
        duplicate_list = pd.concat(duplicate_list, axis=1).T

    return input_df2, duplicate_list

def _open_image(file):
    with rasterio.open(file) as dataset:
        # Read the data into an array (e.g., the first band)
        data = dataset.read(1)
        
        # Get some metadata
        width = dataset.width
        height = dataset.height
        crs = dataset.crs
        bounds = dataset.bounds
        gt = dataset.transform
    return data, {'width':width, 'height':height, 'crs':crs, 'bounds':bounds, 'gt':gt}

def _write_geotiff(output_file, data, bounds, epsg=4326):
    """
    Write a GeoTIFF file with the given output file path, data, bounds, and EPSG code.

    Args:
        output_file (str): The output file path for the GeoTIFF file.
        data (ndarray): The data array to be written to the GeoTIFF file.
        bounds (tuple): The bounds of the data in the form (min_x, min_y, max_x, max_y).
        epsg (int, optional): The EPSG code for the coordinate reference system (CRS). Defaults to 4326.

    Returns:
        None
    """
    min_x, min_y, max_x, max_y = bounds
    transform = rasterio.transform.from_bounds(min_x, min_y,
                                               max_x, max_y,
                                               data.shape[1], data.shape[0])

    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,  # Number of bands
        'dtype': data.dtype,
        'crs': f'EPSG:{epsg}',  # Replace with your desired CRS
        'transform': transform  # Adjust as needed
    }

    # Write the array to a raster file
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(data, 1)  # Write data to the first band

def get_metadata(disp_nc : str | Path, reference_date :str = None) -> dict:
    """
    Get metadata for MINTPY from a DISP NetCDF file.

    Args:
        disp_nc (str or Path): The path to the DISP NetCDF file.
        reference_date (str, optional): The reference date. Defaults to None.

    Returns:
        dict: A dictionary containing the metadata.

    """
    def _get_track(static_lyr_path: str):
        return re.search(r'T\d{3}', static_lyr_path.split('/')[-1]).group(0)
    def _get_burst_id(static_lyr_path: str):
        return re.search(r'\d{6}-IW\d', static_lyr_path.split('/')[-1]).group(0)

    import datetime
    import h5py
    from pyproj import CRS

    # Get high-level metadata from DISP
    ds = h5py.File(disp_nc, 'r')
    length, width = ds['displacement'][:].shape 

    # Get general metadata
    metadata = {}
    for key, value in ds.attrs.items():
        metadata[key] = value

    for key, value in ds['identification'].items():
        value = value[()]
        if isinstance(value, (bytes, bytearray)):
            value = value.decode('utf-8')
        metadata[key] = value

    for key, value in ds['metadata'].items():
        # Skip unnecessary keys
        if key not in ['reference_orbit',
                    'secondary_orbit',
                    'processing_information']:
            metadata[key] = value[()]

    metadata['x'] = ds['x'][:]
    metadata['y'] = ds['y'][:]
    metadata['length'] = length
    metadata['width'] = width
    ds.close()
    del ds

    # Get info from pge_runconfig
    runconfig = metadata.pop('pge_runconfig', None)
    runconfig = yaml.safe_load(runconfig.decode())
    static_lyrs = runconfig['dynamic_ancillary_file_group']['static_layers_files']
    metadata['TRACK'] = np.unique(list(map(_get_track, static_lyrs)))[0]
    burst_ids = np.unique(list(map(_get_burst_id, static_lyrs))) 
    metadata['DISP_BURSTS_ID'] =  ' '.join(burst_ids).encode('utf-8')

     # Get crs and transform
    with rasterio.open(f'NETCDF:"{str(disp_nc)}":/displacement', 'r') as rd:
        metadata['crs'] = rd.crs
        gt = rd.transform.to_gdal()
        td = rd.transform 
        rows, cols = rd.shape

    ## Prepare it in mintpy atr format
    metadata["LENGTH"] = rows
    metadata["WIDTH"] = cols

    metadata["X_FIRST"] = gt[0]
    metadata["Y_FIRST"] = gt[3]
    metadata["X_STEP"] = gt[1]
    metadata["Y_STEP"] = gt[5]
    metadata['GT'] = td
    metadata["X_UNIT"] = metadata["Y_UNIT"] = "meters"
    metadata['WAVELENGTH'] = metadata['radar_wavelength']
    metadata['REF_DATE'] = reference_date

    proj = CRS.from_wkt(metadata['crs'].wkt)
    metadata['UTM_ZONE'] = proj.name.split(' ')[-1]
    metadata['EPSG'] = proj.to_authority()[-1]
    metadata['ALOOKS'] = metadata['RLOOkS'] = 1
    metadata['EARTH_RADIUS'] = 6371000.0  # Hardcoded
    metadata["FILE_TYPE"] = "timeseries"
    metadata["UNIT"] = "m"
    metadata["AZIMUTH_PIXEL_SIZE"] = 14.1 # where this comes from

    #Get  mid UTC
    t = pd.to_datetime([metadata['reference_zero_doppler_start_time'],
                        metadata['reference_zero_doppler_end_time']])
    t_mid = t[0] + t.diff()[1] /2
    total_seconds = t_mid.hour*3600 + t_mid.minute*60 + t_mid.second + t_mid.microsecond/1e6
    metadata["CENTER_LINE_UTC"] = total_seconds

    # Clean up of metadata dicts
    for key in ['reference_datetime', 'secondary_datetime']:
        del metadata[key]

    return metadata

### QUALITY LAYERS
def _get_mean_quality_layers(stack_xr: xr.Dataset, output_dir: str,
                             n_workers: int = 10, n_threads: int = 4,
                             memory_limit: str = '4GB'):
    from dask.distributed import Client
    def get_ccomp_counts(array, conn_label=0):
        return np.sum(array == conn_label, axis=0)

    # Initialize the Dask client
    client = Client(n_workers=n_workers,
                    threads_per_worker=n_threads,
                    memory_limit=memory_limit)
    print(f'Dask client: {client.dashboard_link}')

    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Get the mean quality layers
    layers =['temporal_coherence',
             'phase_similarity',
             'estimated_phase_quality']
    for lyr in layers:
        print(f'Get mean {lyr}')
        avg_data = stack_xr[lyr].mean(dim='time')
        avg_data = avg_data.values
        output = out_dir / f'mean_{lyr}.tif'
        print(' ', output)
        _write_geotiff(output,
                       avg_data, stack_xr.rio.bounds(),
                       stack_xr.rio.crs.to_epsg())
        
    # Get the mean connected component labels
    print('Get mean connected component labels')
    ccomp = stack_xr.apply(get_ccomp_counts)
    zero_count = ccomp.connected_component_labels.values
    zero_count = (zero_count / np.int64(stack_xr.time.size)) * 100
    output = out_dir / f'pct_conncomp0.tif'
    print(' ', output)
    _write_geotiff(output,
                   zero_count, stack_xr.rio.bounds(),
                   stack_xr.rio.crs.to_epsg())

    print('Get mean recommended mask')
    mask_count = ccomp.recommended_mask.values
    mask_count = (mask_count  / np.int64(stack_xr.time.size)) * 100
    output = out_dir / f'pct_mask.tif'
    print(' ', output)
    _write_geotiff(output,
                    mask_count, stack_xr.rio.bounds(),
                    stack_xr.rio.crs.to_epsg())
    client.close()

def _find_reference_point(mean_quality_path: str):
    # Load percentage of recommended mask
    pct_mask, atr = _open_image(Path(mean_quality_path) / 'pct_mask.tif')

    # Load mean of temporal coherence
    mean_tcoh, _ = _open_image(Path(mean_quality_path) / 'mean_temporal_coherence.tif')

    # Get mask, where the percentage of recommended mask not 0
    mask = pct_mask != 0
    # Get mask, where the mean temporal coherence is greater than 90th percentile
    threshold = np.nanpercentile(np.ma.masked_array(mean_tcoh, mask=mask).filled(np.nan), 90)
    mask = np.ma.masked_array(mean_tcoh, mask=mask) > threshold
    mask = mask.filled(0)

    # Get the random reference point
    nrow, ncol = np.shape(mask)
    y = random.choice(list(range(nrow)))
    x = random.choice(list(range(ncol)))

    while mask[y, x] == 0:
        y = random.choice(list(range(nrow)))
        x = random.choice(list(range(ncol)))
    print(f'random select pixel\ny/x: {(y, x)}')
    return y, x


# EXTRACTING AND WRITING TO MINTPY H5 CONTAINER FUNCTIONS
def _get_chunks_indices(xr_array: xr.Dataset) -> list:
    """
    Get the indices for chunked slices of an xarray Dataset.

    Parameters:
        xr_array (xr.Dataset): The input xarray Dataset.

    Returns:
        list: A list of slice objects representing the chunked slices.

    """
    chunks = xr_array.chunks
    it, iy, ix = chunks['time'], chunks['y'], chunks['x']

    idx = [sum(ix[:i]) for i in range(len(ix) + 1)]
    idy = [sum(iy[:i]) for i in range(len(iy) + 1)]

    slices = []

    for i in range(len(idy) - 1):  # Y-axis slices for idy
        for j in range(len(idx) - 1):  # X-axis slices for idx
            # Create a slice using the ranges of idt, idy, and idx
            # skip first date
            slice_ = np.s_[:, idy[i]:idy[i + 1], idx[j]:idx[j + 1]]
            slices.append(slice_)
    return slices

def process_and_write(i, f, stack, ref_data, mask=True, drop_nan=True):
    """
    Process and write the data to the specified file.

    Args:
        i (tuple): The index of the data to be processed and written.
        f (h5py.File): The file object to write the data to.
        stack (Stack): The stack object containing the displacement data.
        ref_data (ndarray): The reference data to be subtracted from the displacement data.
        mask (bool, optional): Whether to apply a mask to the data. Defaults to True.
        drop_nan (bool, optional): Whether to convert NaN values to 0 and apply a water mask. Defaults to True.
    """
    # Create a mask based on the unwrapper mask and apply it to the data
    data = stack.displacement[i].values
    # Change slice to skip writing to the first data
    write_ix = (slice(1, None, None), i[1], i[2])
    #write_ix = (slice(None, None, None), i[1], i[2])

    # Use mask
    if mask:
        mask = stack.recommended_mask[i].values == 0
    else:
        mask = stack.water_mask[i].values == 0 

    data = np.ma.masked_array(data, mask=mask).filled(np.nan)

    # Re-reference: subtract reference data
    if ref_data is not None:
        if data.ndim == 3:
            ref_tile = np.tile(ref_data.reshape(-1, 1, 1),
                            (1, data.shape[-2], data.shape[-1]))
        else:
            print('Ref_2d')
            ref_tile = np.tile(ref_data.reshape(-1, 1),
                                (data.shape[-2], data.shape[-1]))
        data -= ref_tile

    # Convert NaNs to 0 and apply water mask
    if drop_nan:
        data = np.nan_to_num(data)

    # Write the re-referenced data to the "timeseries" dataset
    f["timeseries"][write_ix] = data

    # Add connected component labels to the "connectComponent" dataset
    f["connectComponent"][write_ix] = stack.connected_component_labels[i].values

def write_data_parallel(output, stack, ref_data, chunks_ix,
                        mask=True, water_mask=False, drop_nan=True, num_threads=4):
    """
    Write data to an HDF5 file in append mode using parallel execution.

    Parameters:
    output (str): The path to the HDF5 file.
    stack (numpy.ndarray): The stack of data to be written.
    ref_data (numpy.ndarray): The reference data.
    chunks_ix (list): The list of chunk indices.
    mask (bool, optional): Whether to apply a mask to the data. Defaults to True.
    drop_nan (bool, optional): Whether to drop NaN values from the data. Defaults to True.
    num_threads (int, optional): The number of threads to use for parallel execution. Defaults to 4.
    """
    print("Writing data to HDF5 file in append mode...")

    # Initialize the HDF5 file and the progress bar
    with h5py.File(output, "a") as f:
        process_and_write2 = lambda x: process_and_write(x, f, stack,
                                                         ref_data,
                                                         mask=mask,
                                                         drop_nan=drop_nan) 
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = list(tqdm(executor.map(process_and_write2, chunks_ix), total=len(chunks_ix)))

    print("Data writing complete.")

def write_mintpy_container_parallel(stack_xr, metadata, ref_lalo, 
                                    output='timeseries.h5',  mask=True,
                                    drop_nan=True, num_threads=4):
    """
    Write MintPy container in parallel.

    Args:
        stack_xr (xr.Dataset): The stack of data to be written.
        metadata (dict): The metadata for MintPy layout.
        ref_lalo (list): The reference location latlon for the stack.
        output (str, optional): The path to the output HDF5 file. Defaults to 'timeseries.h5'.
        mask (bool, optional): Whether to apply a mask to the data. Defaults to True.
        drop_nan (bool, optional): Whether to drop NaN values from the data. Defaults to True.
        num_threads (int, optional): The number of threads to use for parallel execution. Defaults to 4.
    """
    # map chunks indices
    chunks_ix = _get_chunks_indices(stack_xr)
    print(f'number of chunks: {len(chunks_ix)}')
    stack_xr['displacement'].attrs['units'] = 'm'

    # Get metadata for MintPy layout
    date_list = list(stack_xr['time'].dt.strftime('%Y%m%d').data)
    date_list = [metadata['REF_DATE']] + date_list
    dates = np.array(date_list, dtype=np.string_)
    num_date = len(date_list)
    rows, cols = stack_xr.displacement.shape[1:] 

    #pbase
    pbase = np.zeros(num_date, dtype=np.float32)

    # define dataset structure
    dates = np.array(date_list, dtype=np.string_)

    ds_name_dict = {
        "date": [dates.dtype, (num_date,), dates],
        "bperp": [np.float32, (num_date,), pbase],
        "timeseries": [np.float32, (num_date, rows, cols), None],
        "connectComponent" : [np.int16, (num_date, rows, cols)],
    }

    # Change spatial reference
    if ref_lalo is not None:
        coord = ut.coordinate(metadata)
        ref_lat, ref_lon = ut.latlon2utm(metadata, *ref_lalo)
        yx = coord.lalo2yx(ref_lat, ref_lon)
        print(f'Re-referencing to x: {yx[1]}, y:{yx[0]}')
        metadata['REF_LAT'] = ref_lat
        metadata['REF_LON'] = ref_lon
        metadata['REF_Y'] = yx[0]
        metadata['REF_X'] = yx[1]

        # Get reference data
        ref_data = stack_xr.displacement[:, yx[0], yx[1]].values
    else:
        ref_data = None

    # initiate HDF5 file
    writefile.layout_hdf5(output, ds_name_dict, metadata=metadata)

    # Write
    write_data_parallel(output, stack_xr, ref_data, chunks_ix,
                        mask=mask, drop_nan=drop_nan, num_threads=num_threads)

# GEOMETRY
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

def _get_geometry(metadata:dict,
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

    los_east, los_east_atr = _open_image(static_dir / 'los_east.tif')
    los_north, _ = _open_image(static_dir / 'los_north.tif')

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
    inc, atr = _open_image(Path(temp_dir) / 'incidence_angle.tif')
    azi, atr = _open_image(Path(temp_dir) / 'azimuth_angle.tif')
    mask = np.ma.masked_equal(inc, 0).mask
    dem, atr = _open_image(Path(temp_dir) / 'dem.tif')
    dem = np.ma.masked_array(dem, mask=mask).filled(0)
    water, atr = _open_image(Path(temp_dir) / 'water_mask.tif')
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

def create_timeseries_h5(products_path:str, output_dir: str, ref_lalo:list=None,
         mask:bool = True, drop_nan:bool=True, num_threads:int=4):
    # Get filename information from the DISP products
    disp_df = _get_filename_info(products_path)

    # Get duplicates
    disp_df, duplicates = _find_duplicates(disp_df)
    print(f'Skip {len(duplicates)} duplicates ')

    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get reference dates
    groupped_df = disp_df.groupby(['date1', 'date2']).apply(lambda x: x,
                                                            include_groups=False)
    reference_dates = groupped_df.index.get_level_values(0).unique()
    print(' Ref dates:', list(reference_dates)) 

    # Get metadata from the DISP NetCDF file
    meta = get_metadata(disp_df.path.iloc[0], 
                        disp_df.start_date.min().strftime('%Y%m%d'))

    # Load stacks 
    stacks = []
    for ix,date in enumerate(reference_dates):
        # Load first stack
        stack_files = groupped_df.loc[date].sort_index().path.to_list()
        stacks.append(xr.open_mfdataset(stack_files, 
                                chunks={'time':-1, 'x':512, 'y':512}))
        if ix>0:
            # use last date displacement to stack displacements
            stacks[ix]['displacement'] += stacks[ix-1].displacement.isel(time=-1)

    # Combine stacks
    stack = xr.concat(stacks, dim='time')

    # Get mean quality layers
    stack.rio.write_crs(meta['crs'].to_epsg(), inplace=True)
    _get_mean_quality_layers(stack, output_dir / 'mean_quality_layers',
                             n_workers= 10, n_threads = 10, memory_limit= '4GB')

    # Get reference point
    if ref_lalo is None:
        y,x = _find_reference_point(output_dir / 'mean_quality_layers')
        ref_utm = ut.coordinate(meta).yx2lalo(y,x)
        ref_lalo = list(ut.utm2latlon(meta, *reversed(ref_utm)))
        print(f'random select pixel\nlat/lon: {(ref_lalo[0], ref_lalo[1])}')

    # Get geometry
    burst_ids = (meta['DISP_BURSTS_ID'].decode()).split(' ')
    burst_ids =[meta['TRACK'] + '_' + burst for burst in burst_ids]
    burst_ids =[re.sub(r'-', '_', burst) for burst in burst_ids]
    _get_geometry(meta, burst_ids, output_dir)

    # Get this size of one chunk
    chunk_mb = stack.displacement[_get_chunks_indices(stack)[0]].nbytes / (1024 ** 2)
    print(f'Chunk size: {chunk_mb} MB')
    print(f'Size of parallel run: {chunk_mb * num_threads} MB')


    # Call the write_mintpy_container_parallel function
    write_mintpy_container_parallel(stack, meta, ref_lalo=ref_lalo,
                                    output=output_dir /'timeseries.h5',
                                    mask=mask, drop_nan=drop_nan, num_threads=num_threads)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process and write OPERA DISP to MintPy container.')
    parser.add_argument('disp_nc', type=str, help='Path to the DISP NetCDF files')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory.')
    parser.add_argument('--ref_lalo', type=list, default=None, help='Spatial reference [latitute, longitude].')
    parser.add_argument('--mask', action='store_true', help='Apply mask to the data.')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads for parallel execution.')
    args = parser.parse_args()
    create_timeseries_h5(args.disp_nc, args.output_dir, args.ref_lalo, args.mask, True, args.num_threads)