# Run with the Profiler environment from the vscode debugger
# visualize with snakeviz: snakeviz /tmp/tmp.prof
import concurrent.futures
import cProfile
import logging
import pstats
import sys
import tempfile
import time
import typing
from datetime import datetime
from pathlib import Path
from pprint import pprint
from random import randint
from time import sleep
from time import time as timer
from typing import Tuple
from urllib.parse import urlparse

import boto3
import geopandas as gpd
import pandas
import pystac_client
import requests
import shapely.geometry
from botocore.exceptions import ClientError
from threadpoolctl import threadpool_info
from tqdm.auto import tqdm

package_path='/workspaces/acolite_workflow'
# package_repo='https://dev.azure.com/csiro-easi/easi-hub-partners/_git/easi-workflows'
# repo = Path(package_path) / package_repo.split('/')[-1]
repo = Path(package_path) / 'easi-workflows'
sys.path.insert(1, str(repo))

from tasks.utils.earthdatasession import (AuthorizationError, EarthdataSession,
                                     FileNotFoundError, FileSizeError,
                                     ServiceBusyError, earthdata_download_file)
from tasks.utils.utils import elapsed_time, set_logger
from tasks.modis_l2_easi_prepare import main as modis_l2_easi_prepare
from tasks.modis_l2_to_cog import preprocess


def fetch_s3_obj(entry):
    path, uri, requester_pays = entry
    uu = urlparse(uri)
    bucket = uu.netloc
    key = uu.path.lstrip("/")
    requestPayer = 'requester' if requester_pays else ''
    session = boto3.session.Session() # Create a new session for each thread for thread-safety
    s3_client = session.client('s3')

    s3_client.download_file(bucket,
                            key,
                            str(path),
                            ExtraArgs = {'RequestPayer': requestPayer})
    return path


def s3_upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # boto3
    if 's3' not in locals():
        s3 = boto3.client('s3')
    # Upload the file
    try:
        s3.upload_file(file_name, bucket, object_name)  # Config=config
    except (ClientError, FileNotFoundError) as e:
        logging.error(e)
        return False
    return True

def fetch_http_file(entry: Tuple) -> Path:
    url, local_path, username, password = entry
    retries = 0
    downloaded = False
    MAX_RETRIES=5

    target = Path(url.rsplit('/', 1)[1])
    if local_path is not None:
        target = Path(local_path) / target

    # Get a valid EarthdataSession
    try:
        session = EarthdataSession(username, password)
    except Exception as e:
        return False, f"{e}"
    # Download with retries
    while not downloaded:
        # Download and handle repsonses
        try:
            if target.exists():
                target.unlink()
            logging.info(f'Downloading {url}')
            start = time.time()
            output_file = earthdata_download_file(session, url, str(target))
            logging.info(f'Download time: {elapsed_time(time.time()-start)}')
            downloaded = True
        except (FileSizeError, ServiceBusyError) as e:
            # Retry
            if retries < MAX_RETRIES:
                waiting = randint(5,30)
                logging.warning(f'Retrying download in {waiting} seconds: {e}')
                time.sleep(waiting)
                retries += 1
            else:
                return False, f'Unsuccessful attempts ({retries+1}): {e}'
        except FileNotFoundError as e:
            return False, f'Could not download {target}: {e}'
        except Exception as e:
            # Catch all other exceptions
            return False, f'Could not download {target}: {e}'

    return True, output_file

def download_and_process(stac_item, product_yaml: str, username: str, password: str, userid, scratch_bucket, s3_prefix):
    logger = set_logger("download_and_process", logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.debug("download_and_process")

    # Each scene will have all of its assets download into a sub-directory with it's id.
    with tempfile.TemporaryDirectory(prefix=stac_item['id']) as inputs_path:
        try:
            assets = stac_item["assets"]

            # download
            logger.debug("download items")
            files = []
            # files = [ (v['href'],inputs_path, username, password) for k,v in assets.items() ]
            for k,v in assets.items():
                url = urlparse(v['href'])
                if url.netloc=='cmr.earthdata.nasa.gov': # No auth requried for cmr urls
                    files.append((v['href'],inputs_path, None, None))
                else:
                    files.append((v['href'],inputs_path, username, password))
            source_assets = [] # Stash these for future deletion of downloaded files
            # for file in files:
            #     status, out = fetch_http_file(file)
            #     if not status:
            #         raise Exception(f'{out}') # out is error message
            #     source_assets.append(out) # out is file name
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for out in executor.map(fetch_http_file, files):
                    status, value = out
                    if not status:
                        raise Exception(f'{value}') # value is error message
                    source_assets.append(value) # value is file name
                    pass

            # Cogify
            logger.debug("Convert Netcdf to COGS")
            s,r = preprocess(Path(inputs_path), Path(product_yaml), True)
            if not s:
                logger.error(r)

            # Metadata
            logger.debug("Prepare metadata")
            metadata_file = modis_l2_easi_prepare(inputs_path, product_yaml)
            logger.info(f'Wrote metadta to {metadata_file}')
            # Clean up source assets and upload
            for f in source_assets:
                Path(f).unlink()

            # Store outputs in user scratch
            root = Path(inputs_path)
            for file in root.iterdir():
                logging.debug(str(file.relative_to(root)))
                # res = s3_upload_file(
                #     str(file),
                #     scratch_bucket,
                #     f'{userid}/{s3_outputs_prefix}/{stac_item["id"]}/{str(file.relative_to(root))}',
                # )
                # if not res:
                #     raise Exception(f'FAILED: {stac_item["id"]}')
        except (FileExistsError) as e:
            return f"{e}"
        except Exception as e:
            return f"{e}"

    return f's3://{scratch_bucket}/{userid}/{s3_outputs_prefix}/{stac_item["id"]}'

## Main workflow

now = datetime.now()
current_time = now.strftime("%H_%M_%S")


# x, y = (146.061975, -42.815608)  # Center point of a query
# km2deg = 1.0 / 111
# r = 25 * km2deg
# bbox = (x - r, y - r, x + r, y + r)

bbox = (134.8505,-32.373035,140.2338,-32.2587)
catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/OB_DAAC")

# query = catalog.search(collections=['MODISA_L2_OC.vR2022.0'], datetime="2023-01-01/2023-01-05", max_items=2, limit=10, bbox=bbox) #, query=["platform=LANDSAT_9", "landsat:collection_category=T1"])
query = catalog.search(collections=['MODISA_L2_SST.vR2019.0'], datetime="2023-01-01/2023-01-05", max_items=2, limit=10, bbox=bbox) #, query=["platform=LANDSAT_9", "landsat:collection_category=T1"])
# # Looking through collections
# for collection in catalog.get_collections():
#     print(collection.id)

# do something with a collections items
# PySTAC ItemCollection
items = query.get_all_items()
print(len(items))

# Dictionary (GeoJSON FeatureCollection)
items_json = items.to_dict()

# Grab the userid from AWS so we can place outputs in the Scratch Bucket
# under the users prefix from our dask workers.
# s3 = boto3.client('s3')
# scratch_bucket = "adias-prod-user-scratch"
# userid = boto3.client('sts').get_caller_identity()['UserId']
scratch_bucket = ""
userid = ""
# Set this prefix to place all the data from this run under it in s3
s3_outputs_prefix = 'tasmania'

product_yaml = Path(package_path+'/easi-workflows/products/obdaac/nasa_aqua_l2_sst.yaml')

# profiler = cProfile.Profile()
# profiler.enable()

for item in items_json['features']:
    result = download_and_process(item, product_yaml, "auser", "apassword", userid, scratch_bucket, s3_outputs_prefix)
    print(result)

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.dump_stats(f'./{current_time}_stats_file.dat')
# stats.strip_dirs()
# stats.print_stats(15).sort_stats('cumtime')

# scene = "LC08_L1TP_114079_20220130_20220204_02_T1"
# # Each scene will have all of its assets download into a sub-directory with it's id.
# inputs_path = Path(f'/tmp/sample_data/{scene}/inputs')
# outputs_path = Path(f"/tmp/sample_data/{scene}/outputs")
# inputs_path.mkdir(parents=True, exist_ok=True)

# settings = {
#     "inputfile": str(inputs_path),
#     "output": str(outputs_path),
#     # polygon=
#     # limit=-29.9,152.5,-29.2,154.0
#     "l2w_parameters": "Rrs_*",
#     "rgb_rhot": False,
#     "rgb_rhos": False,
#     # "map_l2w": False,
#     # "merge_zones": False,
#     # "dsf_residual_glint_correction": True,
#     "l2w_export_geotiff": True,
#     "l1r_delete_netcdf" : True,
#     "l2r_delete_netcdf": True,
#     "l2t_delete_netcdf": True,
#     "l2w_delete_netcdf": True,
#     # "dsf_interface_reflectance": False, # False is the default
#     "ancillary_data": False, # If you set this to True you must supply a username and password for EARTHDATA
#     # "EARTHDATA_u": "",
#     # "EARTHDATA_p": "",
#     # "verbosity": 5
# }

# profiler = cProfile.Profile()
# profiler.enable()

# ac.acolite.acolite_run(settings=settings)

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.dump_stats(f'./{current_time}_stats_file.dat')
# stats.sort_stats('cumtime')
# stats.strip_dirs()
# stats.print_stats(15)