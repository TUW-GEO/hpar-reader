import math
import os

from osgeo import gdal


class ChunkedGeoTiff:
    """Class to write a GeoTIFF file chunk-wise to disk."""
    def __init__(self, dst_file, sref, gt, n_cols, n_rows, nodata=-9999, scale=10, ct=None, compress_type="LZW",
                 data_type='int16', metadata=None, overwrite=False):
        """
        Write chuncked data continuously to single-band GeoTIFF file.

        Parameters
        ----------

        dst_file: str
            Full file path of output file.

        nodata: int, optional
            General nodata value for all bands (default: -9999).
        scale: int, optional
            Scale factor, which the output data is multiplied with (default: 10).
        ct: gdal colortable, optional
            If set, the colortable will be attached to GeoTIFF file (default: None).
        compress_type: str, optional
            GeoTIFF compression type (default: "LZW").
        data_type: str, optional
            Data type of the array to be written (default: 'int16').
        metadata : dict, optional
            Metadata dictionary (default: None).
        overwrite : bool
            If true and the given file path exists, the file will be deleted and re-created (default is False).

        """
        if os.path.exists(dst_file) and overwrite:
            os.remove(dst_file)

        _numpy2gdal_dtype = {"bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 3,
                             "uint32": 4, "int32": 5, "float32": 6, "float64": 7,
                             "complex64": 10, "complex128": 11}

        gdal_dtype = _numpy2gdal_dtype[data_type]

        # define gdal options
        opt = ["COMPRESS={0}".format(compress_type)]
        tilesize = int(512)
        # make sure the tilesize is exponent of 2
        tilesize = 2 ** int(round(math.log(tilesize, 2)))
        opt.append("TILED=YES")
        opt.append("BLOCKXSIZE={:d}".format(tilesize))
        opt.append("BLOCKYSIZE={:d}".format(tilesize))

        # create geotiff driver
        driver = gdal.GetDriverByName('GTiff')
        dst_data_create = driver.Create(dst_file, n_cols, n_rows, 1, gdal_dtype, opt)
        dst_data_create = None

        # open file-pointer
        self.dst_data = gdal.Open(dst_file, gdal.GA_Update)

        # set geoinformation
        self.dst_data.SetProjection(sref)
        self.dst_data.SetGeoTransform(gt)
        self.dst_data.GetRasterBand(1).SetNoDataValue(nodata)
        self.dst_data.GetRasterBand(1).Fill(nodata)
        self.dst_data.GetRasterBand(1).SetScale(scale)
        self.dst_data.GetRasterBand(1).SetOffset(0)

        # set color table
        if ct is not None:
            self.dst_data.GetRasterBand(1).SetRasterColorTable(ct)

        # attach metadata
        if metadata is not None:
            self.dst_data.SetMetadata(metadata)

    def write_chunk(self, data, min_row, min_col):
        """
        Writes given chunk to file.

        Parameters
        ----------
        data: numpy.array
            Data array which will be written to file.
        min_row: int
            Write array starting with this row index.
        min_col: int
            Write array starting with this column index.

        """

        self.dst_data.GetRasterBand(1).WriteArray(data, xoff=min_col, yoff=min_row)

    def close(self):
        """Deletes data which was not written to file and removed file-pointer."""
        self.dst_data.FlushCache()
        self.dst_data = None
