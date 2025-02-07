import logging
from typing import List
import xarray as xr
import pandas as pd

from .product import _get_ministacks

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS = {'time':-1, 'x':512, 'y':512} 

def combine_disp_product(disp_df: pd.DataFrame, chunks: dict = None) -> xr.Dataset:
    """Stacks displacement products over time.

    Args:
        disp_df (pd.DataFrame): DataFrame containing displacement file paths with 'date1' and 'date2'.
        chunks (dict, optional): Chunking configuration for xarray. Defaults to DEFAULT_CHUNKS.

    Returns:
        xr.Dataset: Stacked displacement dataset.
    """
    logger.info('Stacking ministack into common stack')
    chunks = {**DEFAULT_CHUNKS, **(chunks or {})}  # Merge default chunks with user-defined chunks
    logger.info(f' Chunk blocks: {chunks}')
    # Get ministack and reference dates
    mini_stacks, reference_dates = _get_ministacks(disp_df)

    stacks = []
    for ix, date in enumerate(reference_dates):
        stack_files = mini_stacks.loc[date].sort_index().path.to_list()
        stack = xr.open_mfdataset(stack_files, chunks=chunks)
        
        # Append first epcoh of new ministack to last epochs of previous
        if ix > 0:
            stack['displacement'] += stacks[ix - 1].displacement.isel(time=-1)

        stacks.append(stack)
    
    return xr.concat(stacks, dim='time')


