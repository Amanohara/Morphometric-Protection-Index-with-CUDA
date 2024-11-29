import numpy as np
from numba import cuda, float32, int32
import math
import rasterio
import argparse
from tqdm import tqdm

@cuda.jit
def compute_protection_index(dem, protection_index, radius, cellsize_x, cellsize_y, no_data_value, x_offset, y_offset, nx, ny, iDifX, iDifY):
    y_global, x_global = cuda.grid(2)
    y = y_global + y_offset
    x = x_global + x_offset

    if x >= nx or y >= ny:
        return

    center_elevation = dem[y, x]

    if center_elevation == no_data_value:
        protection_index[y, x] = no_data_value
        return

    dProtectionIndex = 0.0
    # ローカルメモリ上に配列を宣言
    aAngle = cuda.local.array(8, dtype=float32)
    for idx in range(8):
        aAngle[idx] = 0.0

    for i in range(8):
        j = 1
        dx = iDifX[i]
        dy = iDifY[i]

        dist_step = math.hypot(dx * cellsize_x, dy * cellsize_y)
        dDist = dist_step * j

        while dDist < radius:
            new_x = x + dx * j
            new_y = y + dy * j

            if 0 <= new_x < nx and 0 <= new_y < ny:
                new_x_int = int(new_x)
                new_y_int = int(new_y)

                new_elevation = dem[new_y_int, new_x_int]

                if new_elevation == no_data_value:
                    protection_index[y, x] = no_data_value
                    return

                dDifHeight = new_elevation - center_elevation
            else:
                protection_index[y, x] = no_data_value
                return

            if dDist != 0.0:
                dAngle = math.atan(dDifHeight / dDist)
            else:
                dAngle = 0.0

            if dAngle > aAngle[i]:
                aAngle[i] = dAngle

            j += 1
            dDist = dist_step * j

    # 8方向の角度の合計を計算
    for idx in range(8):
        dProtectionIndex += aAngle[idx]

    dProtectionIndex = dProtectionIndex / 8.0
    protection_index[y, x] = dProtectionIndex

def main():
    parser = argparse.ArgumentParser(description='Morphometric Protection Index Calculator')
    parser.add_argument('input_dem', help='Input DEM GeoTiff file path')
    parser.add_argument('output_file', help='Output Protection Index GeoTiff file path')
    parser.add_argument('--radius', type=float, default=2000.0, help='Radius in map units')
    parser.add_argument('--tile_size', type=int, default=256, help='Tile size for processing')
    args = parser.parse_args()

    input_dem_file = args.input_dem
    output_file = args.output_file
    radius = args.radius
    tile_size = args.tile_size

    # Read DEM data
    with rasterio.open(input_dem_file) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        no_data_value = src.nodata
        profile = src.profile

    ny, nx = dem.shape

    # Replace NoData values with a specific value
    if np.isnan(no_data_value):
        dem = np.nan_to_num(dem, nan=-9999.0)
        no_data_value = -9999.0
    else:
        dem[dem == no_data_value] = -9999.0
        no_data_value = -9999.0

    # Get cell size from transform
    cellsize_x = abs(transform.a)
    cellsize_y = abs(transform.e)

    # Prepare output array
    protection_index = np.full((ny, nx), fill_value=no_data_value, dtype=np.float32)

    # Transfer data to device
    dem_device = cuda.to_device(dem)
    protection_index_device = cuda.to_device(protection_index)

    # Define direction arrays and transfer to device
    iDifX = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
    iDifY = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)

    iDifX_device = cuda.to_device(iDifX)
    iDifY_device = cuda.to_device(iDifY)

    # Determine the number of tiles
    tiles_y = math.ceil(ny / tile_size)
    tiles_x = math.ceil(nx / tile_size)

    # CUDA kernel execution configuration
    threadsperblock = (16, 16)

    total_tiles = tiles_y * tiles_x

    # Process each tile and display progress
    with tqdm(total=total_tiles, desc='Processing Tiles') as pbar:
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                y_offset = ty * tile_size
                x_offset = tx * tile_size

                # Calculate the size of the current tile
                tile_height = min(tile_size, ny - y_offset)
                tile_width = min(tile_size, nx - x_offset)

                # Define blocks per grid based on tile size
                blockspergrid_x = int(math.ceil(tile_width / threadsperblock[1]))
                blockspergrid_y = int(math.ceil(tile_height / threadsperblock[0]))
                blockspergrid = (blockspergrid_y, blockspergrid_x)

                # Run the CUDA kernel for the current tile
                compute_protection_index[blockspergrid, threadsperblock](
                    dem_device, protection_index_device, radius, cellsize_x, cellsize_y,
                    no_data_value, x_offset, y_offset, nx, ny, iDifX_device, iDifY_device
                )

                pbar.update(1)

    # Copy result back to host
    protection_index = protection_index_device.copy_to_host()

    # Update profile for output
    profile.update(dtype=rasterio.float32, count=1, nodata=no_data_value)

    # Write output
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(protection_index, 1)

if __name__ == '__main__':
    main()
