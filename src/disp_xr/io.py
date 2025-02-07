import rasterio
from collections import namedtuple

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

def write_geotiff(output_file, data, bounds, epsg=4326):
    #min_x, min_y, max_x, max_y = stack.rio.bounds()
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