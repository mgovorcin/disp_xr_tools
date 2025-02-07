import logging
import re
import numpy as np
from pathlib import Path

from .io import write_geotiff
from .stack import combine_disp_product
from .product import get_disp_info, _get_ministacks
from .quality_metrics import get_counts
from mintpy_utils import timeseries, geometry, utils


logger = logging.getLogger(__name__)



def get_mean_quality_layers(stack_xr: xr.Dataset, output_dir: str,
                             n_workers: int = 10, n_threads: int = 4,
                             memory_limit: str = '4GB'):
    from dask.distributed import Client
    def get_ccomp_counts(array, conn_label=0):
        return np.sum(array == conn_label, axis=0)

    # Initialize the Dask client
    client = Client(n_workers=n_workers,
                    threads_per_worker=n_threads,
                    memory_limit=memory_limit)
    logger.info(f'Dask client: {client.dashboard_link}')

    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Get the mean quality layers
    layers =['temporal_coherence',
             'phase_similarity',
             'estimated_phase_quality']
    for lyr in layers:
        logger.info(f'Get mean {lyr}')
        avg_data = stack_xr[lyr].mean(dim='time')
        avg_data = avg_data.values
        output = out_dir / f'mean_{lyr}.tif'
        logger.info(' ', output)
        write_geotiff(output,
                       avg_data, stack_xr.rio.bounds(),
                       stack_xr.rio.crs.to_epsg())
        
    # Get the mean connected component labels
    logger.info('Get mean connected component labels')
    ccomp = stack_xr.apply(get_counts)
    zero_count = ccomp.connected_component_labels.values
    zero_count = (zero_count / np.int64(stack_xr.time.size)) * 100
    output = out_dir / f'pct_conncomp0.tif'
    logger.info(' ', output)
    write_geotiff(output,
                   zero_count, stack_xr.rio.bounds(),
                   stack_xr.rio.crs.to_epsg())

    logger.info('Get mean recommended mask')
    mask_count = ccomp.recommended_mask.values
    mask_count = (mask_count  / np.int64(stack_xr.time.size)) * 100
    output = out_dir / f'pct_mask.tif'
    logger.info(' ', output)
    write_geotiff(output,
                    mask_count, stack_xr.rio.bounds(),
                    stack_xr.rio.crs.to_epsg())
    client.close()


# Fix the mintpy workflow
def create_timeseries_h5(products_path:str, output_dir: str, ref_lalo:list=None,
         mask:bool = True, drop_nan:bool=True, num_threads:int=4):
    # Get filename information from the DISP products
    disp_df = get_disp_info(products_path)

    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get reference dates
    ministacks, ref_dates = _get_ministacks(disp_df)
    print(' Ref dates:', list(ref_dates)) 

    # Get metadata from the DISP NetCDF file
    meta = utils.get_metadata(disp_df.path.iloc[0], 
                        disp_df.start_date.min().strftime('%Y%m%d'))

    # Load stacks 
    stack = combine_disp_product(disp_df)

    # Get mean quality layers
    stack.rio.write_crs(meta['crs'].to_epsg(), inplace=True)
    get_mean_quality_layers(stack, output_dir / 'mean_quality_layers',
                             n_workers= 10, n_threads = 10, memory_limit= '4GB')

    # Get reference point
    if ref_lalo is None:
        y,x = utils._find_reference_point(output_dir / 'mean_quality_layers')
        ref_utm = ut.coordinate(meta).yx2lalo(y,x)
        ref_lalo = list(ut.utm2latlon(meta, *reversed(ref_utm)))
        print(f'random select pixel\nlat/lon: {(ref_lalo[0], ref_lalo[1])}')

    # Get geometry
    burst_ids = (meta['DISP_BURSTS_ID'].decode()).split(' ')
    burst_ids =[meta['TRACK'] + '_' + burst for burst in burst_ids]
    burst_ids =[re.sub(r'-', '_', burst) for burst in burst_ids]
    geometry.prepare(meta, burst_ids, output_dir)

    # Get this size of one chunk
    chunk_mb = stack.displacement[utils._get_chunks_indices(stack)[0]].nbytes / (1024 ** 2)
    print(f'Chunk size: {chunk_mb} MB')
    print(f'Size of parallel run: {chunk_mb * num_threads} MB')


    # Call the write_mintpy_container_parallel function
    timeseries.write_mintpy_container_parallel(stack, meta, ref_lalo=ref_lalo,
                                    output=output_dir /'timeseries.h5',
                                    mask=mask, drop_nan=drop_nan, num_threads=num_threads)