import logging 
import pandas as pd
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

def get_disp_info(products_path: Path) ->pd.DataFrame:
    """
    Get filename information from OPERA DISP products.

    Args:
        products_path (Path): The path to the OPERA DISP products.

    Returns:
        pd.DataFrame: A DataFrame containing the filename information.
        list: List of duplicate DISP products
        reference_dates: List of reference dates

    """
    # Get all OPERA DISP products in the specified path
    disp_products = list(Path(products_path).rglob('*.nc'))
    logger.info(f'Found OPERA DISP: {len(disp_products)} products')

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
    logger.info(f' Starting date: {disp_df.start_date.min()}')
    logger.info(f' Ending date: {disp_df.end_date.max()}')

    # Get number of reference dates
    logger.info(f' Number of reference dates: {len(_get_reference_dates(disp_df)[1])}')

    return _find_duplicates(disp_df)[0].sort_values(by='date12', ignore_index=True)

def _get_reference_dates(df: pd.DataFrame) -> Union[pd.DataFrame, List]:
    substacks = df.groupby(['date1', 'date2']).apply(lambda x: x, include_groups=False)
    reference_dates = substacks.index.get_level_values(0).unique()
    return substacks, reference_dates

def _find_duplicates(input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    
    logger.info(f' Skip {len(duplicate_list)} duplicates')

    return input_df2, duplicate_list