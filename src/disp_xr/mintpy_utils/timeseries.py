
import h5py
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import _get_chunks_indices
from mintpy.utils import writefile, utils as ut

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