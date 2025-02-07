import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def get_chunks_indices(xr_array: xr.Dataset) -> list:
    """
    Get the indices for chunked slices of an xarray Dataset.
    
    Parameters:
        xr_array (xr.Dataset): The input xarray Dataset.

    Returns:
        list: A list of slice objects representing the chunked slices.

    """
    chunks = xr_array.chunks
    _, iy, ix = chunks['time'], chunks['y'], chunks['x']

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