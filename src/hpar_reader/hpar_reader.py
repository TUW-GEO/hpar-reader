import argparse
import gc
import os
from datetime import datetime

import numpy as np
from geopathfinder.folder_naming import build_smarttree
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from veranda.raster.native.geotiff import GeoTiffFile
from yeoda.datacube import DataCubeReader

from hpar_reader.output import ChunkedGeoTiff

# K-value of the harmonic parameters
# to be moved to the config
HM_KVALUE = 3

# Define variable names of the harmonic parameters
HM_VAR_FOLDER = 'SIG0-HPAR'
HM_STD_NAME = 'SIG0-HPAR-STD'
HM_MEAN_NAME = 'SIG0-HPAR-M0'
HM_NOBS_NAME = 'SIG0-HPAR-NOBS'
HM_SINE_NAME = 'SIG0-HPAR-S{}'
HM_COSINE_NAME = 'SIG0-HPAR-C{}'

folder_hierarchy = ["sensor", "parameter", "data_version", "subgrid_name", "tile_name"]


def loadDC(base_path, sensor='S1_CSAR_IWGRDH', parameter='SIG0-HPAR', data_version='V0M2R1', res=20, continent=None,
        tiles=None, folder_hierarchy=folder_hierarchy):

    if continent:
        subgrid = 'EQUI7_' + '{}{:03d}M'.format(continent, res)
    else:
        subgrid = None

    if sensor is not None:
        base_path = os.path.join(base_path, sensor)
        folder_hierarchy.remove("sensor")

        if parameter is not None:
            base_path = os.path.join(base_path, parameter)
            folder_hierarchy.remove("parameter")

            if data_version is not None:
                base_path = os.path.join(base_path, data_version)
                folder_hierarchy.remove("data_version")

                if subgrid is not None:
                    base_path = os.path.join(base_path, subgrid)
                    folder_hierarchy.remove("subgrid_name")

                    if tiles is not None and len(tiles) == 1:
                        base_path = os.path.join(base_path, tiles[0])
                        folder_hierarchy.remove("tile_name")

    dir_tree = build_smarttree(base_path + '/', folder_hierarchy, register_file_pattern="^[^Q].*.tif")
    filepaths = dir_tree.file_register

    dimensions = ["datetime_1", "datetime_2", "tile_name", "grid_name", "extra_field", "var_name", "sensor_field",
                  "data_version", "band", "creator"]
    dc_reader = DataCubeReader.from_filepaths(filepaths, fn_class=YeodaFilename, dimensions=dimensions,
                                              stack_dimension="var_name", tile_dimension="tile_name")

    print(dc_reader.file_register)

    var_names = [HM_STD_NAME, HM_MEAN_NAME, HM_NOBS_NAME]
    for n in range(1, HM_KVALUE + 1):
        var_names.append(HM_COSINE_NAME.format(str(n)))
        var_names.append(HM_SINE_NAME.format(str(n)))

    dc_reader.select_by_dimension(lambda v: v.isin(var_names), name="var_name", inplace=True)

    if tiles is not None and len(tiles) > 1:
        dc_reader.select_by_dimension(lambda t: t.isin(tiles), name="tile_name", inplace=True)

    '''
    # rename dimensions to filenaming independent names
    if 'datetime_1' in dimensions:
        dc_reader.rename_dimensions({'datetime_1': 'time'}, inplace=True)
    if 'datetime_2' in dimensions:
        dc_reader.rename_dimensions({'datetime_2': 'time2'}, inplace=True)
    if 'var_name' in dimensions:
        dc_reader.rename_dimensions({'var_name': 'variable'}, inplace=True)
    if 'tile_name' in dimensions:
        dc_reader.rename_dimensions({'tile_name': 'tile'}, inplace=True)
    if 'grid_name' in dimensions:
        dc_reader.rename_dimensions({'grid_name': 'grid'}, inplace=True)
    if 'band' in dimensions:
        dc_reader.rename_dimensions({'band': 'pol'}, inplace=True)
    if 'data_version' in dimensions:
        dc_reader.rename_dimensions({'data_version': 'version'}, inplace=True)
    if 'extra_field' in dimensions:
        dc_reader.rename_dimensions({'extra_field': 'orbit'}, inplace=True)
    '''

    return dc_reader


def isValidOrbit(orbit):
    """checks if relative orbit is valid"""
    valid = False

    if orbit[0] == 'A' or orbit[0] == 'D':
        num = int(orbit[1:4])
        if num > 1 and num <= 175:
            valid = True

    return valid


def computeDoY(dc, dtime, decoder, chunkSize=2500):
    """computes DoY from single Tile-Orbit HM datacube"""

    w = np.pi * 2 / 365
    t = dtime.timetuple().tm_yday

    num_vars = len(dc.file_register)

    # calcualte k value from available harmonic parameters
    if HM_NOBS_NAME in list(dc.file_register['var_name']):
        nx = num_vars - 2
    else:
        nx = num_vars - 1

    if (nx % 2) == 0:
        raise Exception('The number of harmonic parameters has to be odd. Current number is: ' + str(nx))

    k = int((nx - 1)/2)
    if k < 1 or k > 6:
        raise Exception('The k value of the harmonic analysis has to be between 1 and 6!')

    sample_tile = list(dc.file_register['tile_name'])[0]
    sample_grid = list(dc.file_register['grid_name'])[0]

    tileSize = int((int(sample_tile[-1]) * 100000) / int(sample_grid[2:5]))

    DoY = np.empty((1, tileSize, tileSize))
    DoY[:] = np.nan  # place holder with nans

    if chunkSize > tileSize:
        print('adjusting chunk size')
        chunkSize = tileSize

    for i in range(0, tileSize, chunkSize):
        if i + chunkSize < tileSize:
            numRows = chunkSize
        else:
            numRows = tileSize - i

        for j in range(0, tileSize, chunkSize):
            if j + chunkSize < tileSize:
                numCols = chunkSize
            else:
                numCols = tileSize - j

            dc_sub = dc.select_px_window(i, j, width=numRows, height=numCols, inplace=False)
            dc_sub.read(decoder=decoder)
            x_arr = dc_sub.data_view

            mean_arr = x_arr.sel(var_name='SIG0-HPAR-M0').to_array().to_numpy()

            for l in range(1, k + 1):
                sin_coeff = x_arr.sel(var_name='SIG0-HPAR-S{}'.format(l)).to_array().to_numpy()
                cos_coeff = x_arr.sel(var_name='SIG0-HPAR-C{}'.format(l)).to_array().to_numpy()

                mean_arr = mean_arr + sin_coeff * np.sin(i * w * t)
                mean_arr = mean_arr + cos_coeff * np.cos(i * w * t)

                del sin_coeff, cos_coeff
                gc.collect()

                # replace probematic pixels
                idx = np.isfinite(mean_arr)
                mean_arr[~idx] = np.nan

            DoY[0, i:i+numRows, j:j+numCols] = mean_arr
            #print(DoY[0, i:i+numRows, j:j+numCols])

    return DoY

def decoder(data, nodataval=-9999, scale_factor=0.10, **kwargs):
    data = data * scale_factor
    data[data == nodataval * scale_factor] = np.nan
    return data

def encoder(data, nodataval=-9999, scale_factor=0.10, dtype='int16', **kwargs):
    data = (data / scale_factor)
    data[np.isnan(data)] = nodataval
    return data.astype(dtype)


class HPAR_DC_Reader:
    """convenience class to wrap a datacube for harmonic parameter datacube reading and manipulation"""

    def __init__(self, base_path, sensor='S1_CSAR_IWGRDH', parameter='SIG0-HPAR', data_version='V0M2R1', res=20,
                continent=None, tiles=None, folder_hierarchy=folder_hierarchy):
        """HPARReader constructor"""

        self.dc = loadDC(base_path, sensor=sensor, parameter=parameter, data_version=data_version, res=res,
                        continent=continent, tiles=tiles, folder_hierarchy=folder_hierarchy)

        #if len(tiles) == 1: change as pandas check, make a function
        #    self.isSingleTile = True

    def filter_orbit(self, orbit):
        """Filters HM datacube to a given orbit"""

        if isValidOrbit(orbit):
            filt_dc = self.dc.select_by_dimension(lambda o: o == orbit, name="extra_field", inplace=False)
            if filt_dc.is_empty:
                raise RuntimeError('No HPARs selected')
        else:
            raise ValueError('Orbit is not valid')

        return filt_dc

    def filter_tiles(self, tiles):

        filt_dc = self.dc.select_by_dimension(lambda t: t.isin(tiles), name="tile_name", inplace=False)

        if filt_dc.is_empty:
            raise RuntimeError('No HPARs selected')

        if len(tiles) == 1: #to be chenged as pandas check on tiles names
            self.isSingleTile = True

        return filt_dc

    def convert_amp_ph(self, orbit, tile, out_filepath=''):

        filt_dc = self.dc.select_by_dimension(lambda o: o == orbit, name="extra_field", inplace=False)
        filt_dc = filt_dc.select_by_dimension(lambda t: t.isin([tile]), name="tile_name", inplace=False)

        sample_row = filt_dc.file_register.iloc[[0]].to_dict('r')[0]  # dc row data entry in dictionary format'
        sample_filepath = sample_row['filepath']
        del sample_row['filepath']

        #read meta information
        with GeoTiffFile(sample_filepath, mode='r') as src:
            gt = src.geotrans
            sref = src.sref_wkt
            meta = src.metadata
            raster_shape = src.raster_shape
            dt = src.dtypes[0]
            nd = src.nodatavals[0]
            sf = src.scale_factors[0]

        sample_tile = sample_row['tile_name']
        sample_grid = sample_row['grid_name']
        sample_row['datetime_1'] = sample_row['datetime_1'].date()
        sample_row['datetime_2'] = sample_row['datetime_2'].date()

        k = int(meta['k_value'])

        tileSize = int((int(sample_tile[-1]) * 100000) / int(sample_grid[2:5]))
        chunkSize = 1500

        for l in range(1, k + 1):
            # prepare chunked geotiffs to write result on:

            sample_row['var_name'] = 'SIG0-HPAR-A{}'.format(l)
            out_filepath_a = os.path.join(out_filepath, str(YeodaFilename(sample_row)))

            meta['creator'] = 'hpar_user'
            meta['date_creation'] = str(datetime.now().date())
            meta['date_modification'] = str(datetime.now().date())
            meta['software_version'] = 'V0.0.0'  # get from git
            meta['software_name'] = 'hpar_reader'  # get from git

            meta['harmonic_parameter'] = 'SIG0-HPAR-A{}'.format(l)

            AcGt = ChunkedGeoTiff(dst_file=out_filepath_a, sref=sref, gt=gt, n_cols=tileSize, n_rows=tileSize,
                                  nodata=nd, scale=sf, data_type=dt, metadata=meta, overwrite=True)

            sample_row['var_name'] = 'SIG0-HPAR-P{}'.format(l)
            out_filepath_p = os.path.join(out_filepath, str(YeodaFilename(sample_row)))

            meta['harmonic_parameter'] = 'SIG0-HPAR-P{}'.format(l)

            PcGt = ChunkedGeoTiff(dst_file=out_filepath_p, sref=sref, gt=gt, n_cols=tileSize, n_rows=tileSize,
                                  nodata=nd, scale=sf, data_type=dt, metadata=meta, overwrite=True)


            if chunkSize > tileSize:
                print('adjusting chunk size')
                chunkSize = tileSize

            for i in range(0, tileSize, chunkSize):
                if i + chunkSize < tileSize:
                    numRows = chunkSize
                else:
                    numRows = tileSize - i

                for j in range(0, tileSize, chunkSize):
                    if j + chunkSize < tileSize:
                        numCols = chunkSize
                    else:
                        numCols = tileSize - j

                    dc_sub = filt_dc.select_px_window(i, j, width=numRows, height=numCols, inplace=False)
                    dc_sub.read(decoder=decoder)
                    x_arr = dc_sub.data_view

                    sin_coeff = x_arr.sel(var_name='SIG0-HPAR-S{}'.format(l)).to_array().to_numpy()
                    cos_coeff = x_arr.sel(var_name='SIG0-HPAR-C{}'.format(l)).to_array().to_numpy()

                    A = np.sqrt(sin_coeff ** 2 + cos_coeff ** 2)
                    P = np.arctan( cos_coeff / sin_coeff)

                    # replace probematic pixels
                    idx = np.isfinite(A)
                    A[~idx] = np.nan

                    idx = np.isfinite(P)
                    P[~idx] = np.nan

                    AcGt.write_chunk(encoder(A[0, :, :], nodataval=nd, scale_factor=sf), min_row=i, min_col=j)
                    PcGt.write_chunk(encoder(P[0, :, :], nodataval=nd, scale_factor=sf), min_row=i, min_col=j)

            AcGt.close()
            PcGt.close()

    def compute_DoYE(self, orbit, tile, dtime=datetime.now(), out_filepath=None, chunkSize=5000):

        filt_dc = self.dc.select_by_dimension(lambda o: o == orbit, name="extra_field", inplace=False)
        filt_dc = filt_dc.select_by_dimension(lambda t: t.isin([tile]), name="tile_name", inplace=False)

        sample_row = filt_dc.file_register.iloc[[0]].to_dict('r')[0]  # dc row data entry in dictionary format'
        sample_filepath = sample_row['filepath']

        with GeoTiffFile(sample_filepath, mode='r') as src:
            gt = src.geotrans
            sref = src.sref_wkt
            meta = src.metadata
            raster_shape = src.raster_shape
            dt = src.dtypes[0]
            nd = src.nodatavals[0]
            sf = src.scale_factors[0]

        dc_decoder = lambda x: decoder(x, nodataval=nd, scale_factor=sf)
        # print(dc_decoder(np.array([1000, -9999, 10])))
        # setup chunked output
        if out_filepath is None:

            del sample_row['filepath']
            sample_row['var_name'] = 'SIG0-HPAR-DoYE'
            sample_row['datetime_1'] = dtime.date()
            del sample_row['datetime_2']

            out_filepath = str(YeodaFilename(sample_row))

        meta['creator'] = 'hpar_user'
        meta['harmonic_parameter'] = 'DoYE'
        meta['date_creation'] = str(datetime.now().date())
        meta['date_modification'] = str(datetime.now().date())
        meta['software_version'] = 'V0.0.0'  # get from git
        meta['software_name'] = 'hpar_reader'  # get from git
        meta['date estimated'] = dtime.date()

        sample_grid = sample_row['grid_name']
        tileSize = int((int(tile[-1]) * 100000) / int(sample_grid[2:5]))

        DcGt = ChunkedGeoTiff(dst_file=out_filepath, sref=sref, gt=gt, n_cols=tileSize, n_rows=tileSize,
                                  nodata=nd, scale=sf, data_type='int16', metadata=meta, overwrite=True)

        # start calculations
        w = np.pi * 2 / 365
        t = dtime.timetuple().tm_yday
        num_vars = len(filt_dc.file_register)

        # calcualte k value from available harmonic parameters
        if HM_NOBS_NAME in list(filt_dc.file_register['var_name']):
            nx = num_vars - 2
        else:
            nx = num_vars - 1

        if (nx % 2) == 0:
            raise Exception('The number of harmonic parameters has to be odd. Current number is: ' + str(nx))

        k = int((nx - 1) / 2)
        if k < 1 or k > 6:
            raise Exception('The k value of the harmonic analysis has to be between 1 and 6!')

        if chunkSize > tileSize:
            print('adjusting chunk size')
            chunkSize = tileSize

        for i in range(0, tileSize, chunkSize):
            if i + chunkSize < tileSize:
                numRows = chunkSize
            else:
                numRows = tileSize - i

            for j in range(0, tileSize, chunkSize):
                if j + chunkSize < tileSize:
                    numCols = chunkSize
                else:
                    numCols = tileSize - j

                dc_sub = filt_dc.select_px_window(i, j, width=numRows, height=numCols, inplace=False)
                dc_sub.read()
                x_arr = dc_sub.data_view
                mean_arr = dc_decoder(x_arr.sel(var_name='SIG0-HPAR-M0').to_array().to_numpy())

                for l in range(1, k + 1):
                    sin_coeff = dc_decoder(x_arr.sel(var_name='SIG0-HPAR-S{}'.format(l)).to_array().to_numpy())
                    cos_coeff = dc_decoder(x_arr.sel(var_name='SIG0-HPAR-C{}'.format(l)).to_array().to_numpy())

                    mean_arr = mean_arr + sin_coeff * np.sin(i * w * t)
                    mean_arr = mean_arr + cos_coeff * np.cos(i * w * t)

                    del sin_coeff, cos_coeff
                    gc.collect()

                    # replace probematic pixels
                    idx = np.isfinite(mean_arr)
                    mean_arr[~idx] = np.nan

                    DcGt.write_chunk(encoder(mean_arr[0, :, :], nodataval=nd, scale_factor=sf), min_row=i, min_col=j)

        DcGt.close()



def run():
    """Entry point for console_scripts"""

    parser = argparse.ArgumentParser(description="TUWien HPAR Reader. Computes DoY estiamte from TU HPARS")
    parser.add_argument("-path", "--hpar_path", help="Base directory of the TU Wien HPAR files.",
                        required=True, default='/home/marxt/tuw_s1_hpar', type=str)
    parser.add_argument("-sg", "--subgrid", help="Subgrid. Indentified using 2 character continent code e.g. EU",
                        required=True, type=str)
    parser.add_argument("-t", "--tiles",
                        help="tile names. Equi7grid tile names",
                        required=True, nargs="+", type=str)
    parser.add_argument("-o", "--orbit", help="Sentinel-1 Relative Orbit. ",
                        required=True, default=None, type=str)
    parser.add_argument("-d", "--date", help="Date to be estimated. ",
                        required=True, default=None, type=str)

    args = parser.parse_args()
    base_path = args.hpar_path
    subgrid = args.subgrid
    tiles = args.tiles
    orbit = args.orbit
    date = args.date

    if date is not None:
        date = datetime.strptime(date, '%Y%m%d')

    hm_dc_reader = HPAR_DC_Reader(base_path, continent=subgrid, tiles=tiles)
    hm_dc_reader.filter_orbit(orbit)
    hm_dc_reader.filter_tiles(tiles)

    hm_dc_reader.compute_DoYE(orbit, tiles[0], date)
    #hm_dc_reader.convert_amp_ph(orbit, tiles[0])


if __name__ == "__main__":
    run()
