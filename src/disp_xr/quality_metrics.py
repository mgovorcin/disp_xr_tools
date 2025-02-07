import logging
import xarray as xr
import numpy as np
from pathlib import Path

from .io import write_geotiff

logger = logging.getLogger(__name__)

def get_counts(array, label=0, reverse=False):
    if reverse:
        return np.sum(array != label, axis=0) 
    else:
        return np.sum(array == label, axis=0)

def get_mean_quality_layers(stack_xr: xr.Dataset, output_dir: str,
                             n_workers: int = 10, n_threads: int = 10,
                             memory_limit: str = '4GB') -> None:
    from dask.distributed import Client


    # Initialize the Dask client
    client = Client(n_workers=n_workers, threads_per_worker=n_threads, memory_limit=memory_limit)
    logger.info(f'Dask client: {client.dashboard_link}')

    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Get the mean quality layers
    layers =['temporal_coherence', 'phase_similarity', 'estimated_phase_quality']
    for lyr in layers:
        logger.info(f'Get mean {lyr}')
        avg_data = stack_xr[lyr].mean(dim='time')
        avg_data = avg_data.values
        output = out_dir / f'mean_{lyr}.tif'
        logger.info(output)
        write_geotiff(output,
                       avg_data, stack_xr.rio.bounds(),
                       stack_xr.rio.crs.to_epsg())
        
    # Get the mean connected component labels
    logger.info('Get mean connected component labels')
    ccomp = stack_xr.apply(get_counts)
    zero_count = ccomp.connected_component_labels.values
    zero_count = (zero_count  / np.int64(stack_xr.time.size)) * 100
    output = out_dir / 'pct_conncomp0.tif'
    logger.info(output)
    write_geotiff(output,
                   zero_count, stack_xr.rio.bounds(),
                   stack_xr.rio.crs.to_epsg())

    print('Get mean recommended mask')
    mask_count = ccomp.recommended_mask.values
    mask_count = (mask_count  / np.int64(stack_xr.time.size)) * 100
    output = out_dir / 'pct_mask.tif'
    logger.info(output, '|')
    write_geotiff(output,
                    mask_count, stack_xr.rio.bounds(),
                    stack_xr.rio.crs.to_epsg())
    client.close()
