import logging
import xarray as xr
import numpy as np
from pathlib import Path

from .io import write_geotiff

logger = logging.getLogger(__name__)

# Functions to run
def get_counts(array, label=0, reverse=False):
    if reverse:
        return np.sum(array != label, axis=0) 
    else:
        return np.sum(array == label, axis=0)
    
def func_stat(mode: str):
    # Mapping mode to corresponding statistical operation
    operations = {
        'mean': lambda x: x.mean(dim='time'),
        'median': lambda x: x.median(dim='time'),
        'max': lambda x: x.max(dim='time'),
        'min': lambda x: x.min(dim='time'),
        'std': lambda x: x.std(dim='time')
    }
    
    # Check if the mode is valid and return the corresponding function
    try:
        return operations[mode]
    except KeyError:
        raise ValueError((f"Unsupported mode: '{mode}'."
                          "Valid options are 'mean', 'median', 'max', 'min', 'std'."))

# Map_block template
def get_template(xr_df:xr.DataArray):
    # Get chunks (time, y, x)
    chunk_y = xr_df.chunks[1][0] # use first layer
    chunk_x = xr_df.chunks[2][0] 

    # Get shape
    shape = (xr_df.sizes['y'], xr_df.sizes['x'])

    # Create an empty DataArray (NaNs by default)
    empty_data = np.full(shape, np.nan, dtype=np.float32)

    # Create xarray DataArray without 'time'
    empty_xarray = xr.DataArray(
        empty_data,
        dims=["y", "x"],
        coords={"y": xr_df.y, "x": xr_df.x},
        )
    return empty_xarray.chunk(chunks={'y':chunk_y,'x':chunk_x})

# Quality metrics
def get_ps_percentage(stack_xr:xr.Dataset):
    # Get template
    logger.info('Get percentage of PS')
    template = get_template(stack_xr.persistent_scatterer_mask)

    data = stack_xr.persistent_scatterer_mask.map_blocks(get_counts,
                                            kwargs={'label':1},
                                            template=template)
    pct = data.values / np.int64(stack_xr.time.size) * 100
    return pct

def get_mask_percentage(stack_xr:xr.Dataset):
    # Get template
    logger.info('Get percentage of recommended mask')
    template = get_template(stack_xr.recommended_mask)

    data = stack_xr.recommended_mask.map_blocks(get_counts,
                                            kwargs={'label':0},
                                            template=template)
    pct = data.values / np.int64(stack_xr.time.size) * 100
    return pct

def get_conncomp_percentage(stack_xr:xr.Dataset, reverse=True):
    # Get template
    if reverse:
        logger.info('Get percentage of valid connected components') 
    else:
        logger.info('Get percentage of connected component 0')
    template = get_template(stack_xr.connected_component_labels)

    data = stack_xr.connected_component_labels.map_blocks(get_counts,
                                            kwargs={'label':0, 'reverse':reverse},
                                            template=template)
    pct = data.values / np.int64(stack_xr.time.size) * 100
    return pct

def get_mean_phasesim(stack_xr:xr.Dataset, mode='mean'):
    # Get template
    logger.info(f'Get {mode} phase similarity') 
    template = get_template(stack_xr.phase_similarity)

    stat_func = func_stat(mode)

    data = stack_xr.phase_similarity.map_blocks(stat_func,
                                                template=template)
    return data.values

def get_mean_tcoh(stack_xr:xr.Dataset, mode='mean'):
    # Get template
    logger.info(f'Get {mode} temporal coherence') 
    template = get_template(stack_xr.temporal_coherence)

    stat_func = func_stat(mode)

    data = stack_xr.temporal_coherence.map_blocks(stat_func,
                                                template=template)
    return data.values

def get_mean_phasecorr(stack_xr:xr.Dataset, mode='mean'):
    # Get template
    logger.info(f'Get {mode}  estimated_phase_quality') 
    template = get_template(stack_xr.estimated_phase_quality)

    stat_func = func_stat(mode)

    data = stack_xr.estimated_phase_quality.map_blocks(stat_func,
                                                template=template)
    return data.values
