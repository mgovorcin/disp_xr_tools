import re
import random
import yaml
import rasterio
import xarray as xr
import numpy as np

import pandas as pd
from pathlib import Path

# I have this function as duplicate, fix it later      
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

import numpy as np
import random

def find_reference_point(pct_mask, mean_tcoh, percentile=90):
    """
    Selects a reference point where the percentage mask is nonzero and 
    the mean temporal coherence is above the 90th percentile.

    Parameters:
        pct_mask (np.ndarray): Percentage mask (nonzero values indicate valid regions).
        mean_tcoh (np.ndarray): Mean temporal coherence.

    Returns:
        (int, int): y, x coordinates of the selected reference point.
    """
    # Create a mask where pct_mask is nonzero
    valid_mask = pct_mask == 100
    
    # Compute 90th percentile threshold over valid pixels
    masked_tcoh = np.ma.masked_array(mean_tcoh, mask=valid_mask)
    threshold = np.nanpercentile(masked_tcoh, percentile)
    
    # Create a mask for high-coherence pixels
    high_coherence_mask = (masked_tcoh >= threshold) 
    
    # Get indices of valid pixels
    valid_indices = np.argwhere(high_coherence_mask.filled(0))
    
    if valid_indices.size == 0:
        raise ValueError("No valid reference point found. Adjust threshold or check input data.")

    # Randomly select one valid pixel
    y, x = random.choice(valid_indices)
    
    # Flip y-axis if needed (assuming `y` is inverted)
    #y = mean_tcoh.shape[0] - 1 - y

    print(f'Selected reference pixel (y/x): {(y, x)}')
    
    return y, x
