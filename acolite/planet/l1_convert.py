## def l1_convert
## converts Planet data to l1r NetCDF for acolite-gen
## written by Quinten Vanhellemont, RBINS
## 2021-02-24
## modifications: 2021-12-08 (QV) added nc_projection
##                2021-12-31 (QV) new handling of settings
##                2022-01-04 (QV) added netcdf compression
##                2022-02-21 (QV) added Skysat
##                2022-08-12 (QV) added reprojection of unrectified data with RPC
##                2022-10-26 (QV) changed handling of multi file bundles
##                2022-11-14 (QV) added support for outputting Planet SR data
##                2023-04-18 (QV) added support for NTF files
##                2023-05-08 (QV) added support for composite files
##                2023-07-12 (QV) removed netcdf_compression settings from nc_write call

def l1_convert(inputfile, output = None, settings = {},

                percentiles_compute = True,
                percentiles = (0,1,5,10,25,50,75,90,95,99,100),

                check_sensor = True,
                check_time = True,
                max_merge_time = 600, # seconds

                from_radiance = False,

                gains = False,
                gains_toa = None,

                verbosity = 0, vname = ''):

    import os, zipfile, shutil, json
    import dateutil.parser, time, copy
    import numpy as np
    import acolite as ac

    if 'verbosity' in settings: verbosity = settings['verbosity']

    ## parse inputfile
    if type(inputfile) != list:
        if type(inputfile) == str:
            inputfile = inputfile.split(',')
        else:
            inputfile = list(inputfile)
    nscenes = len(inputfile)
    if verbosity > 1: print('Starting conversion of {} scenes'.format(nscenes))

    ## start with last file in time
    inputfile.sort()
    inputfile.reverse()

    new = True
    warp_to = None

    ofile = None
    ofiles = []
    setu = {}
    ## track files if there are multiple in the bundle
    ## e.g. extracted Planet zip file
    ifiles = []
    for bundle in inputfile:
        t0 = time.time()
        ## unzip files if needed
        ## they should have been unzipped by idendify_bundle
        ## so this section can be removed?
        zipped = False
        if bundle[-4:] == '.zip':
            zipped = True
            bundle_orig = '{}'.format(bundle)
            bundle,ext = os.path.splitext(bundle_orig)
            zip_ref = zipfile.ZipFile(bundle_orig, 'r')
            for z in zip_ref.infolist():
                z.filename = os.path.basename(z.filename)
                zip_ref.extract(z, bundle)
            zip_ref.close()

        ## test files
        files = ac.planet.bundle_test(bundle)
        print('Found {} scenes in {}'.format(len(files), bundle))
        for fk in files: ifiles.append((bundle, files[fk], fk))

    ## run through the found files
    for ifile in ifiles:
        bundle, files, fk = ifile
        ## composite image copy metadata path from the scene
        ## matching acquisition time and satellite
        if ('composite' in files) & ('metadata_json' in files):
            ## read json metadata
            with open(files['metadata_json']['path'], 'r') as f:
                md = json.load(f)
            ## get filler metadata
            for fi in ifiles:
                if 'metadata' not in fi[1]: continue
                bn = os.path.basename(fi[1]['metadata']['path'])
                dt = dateutil.parser.parse(md['properties']['acquired'])
                if (md['properties']['satellite_id'] in bn) & (dt.strftime('%Y%m%d') in bn) & (dt.strftime('%H%M%S') in bn):
                    files['metadata'] = copy.copy(fi[1]['metadata'])
        ## end find composite metadata

        ## check if we can process this scene
        if (not ((('metadata' in files) | ('metadata_json' in files)) & ('analytic' in files))) &\
           (not ((('metadata' in files) | ('metadata_json' in files)) & ('analytic_ntf' in files))) &\
           (not ((('metadata' in files) | ('metadata_json' in files)) & ('pansharpened' in files))) &\
           (not ('metadata' in files) & ('composite' in files)) &\
           (not ((('metadata' in files) | ('metadata_json' in files)) & ('sr' in files))):
            print('Bundle {} {} not recognised'.format(bundle, fk))
            continue

        metafile = None
        image_file = None
        if 'metadata' in files:
            metafile = files['metadata']['path']
        elif 'metadata_json' in files:
            metafile = files['metadata_json']['path']

        if 'analytic' in files:
            image_file = files['analytic']['path']
        elif 'analytic_ntf' in files:
            image_file = files['analytic_ntf']['path']
        elif 'pansharpened' in files:
            image_file = files['pansharpened']['path']
        elif 'composite' in files:
            image_file = files['composite']['path']
        image_file_original = '{}'.format(image_file)

        sr_image_file = None
        if 'sr' in files: sr_image_file = files['sr']['path']

        if image_file is None:
            print('No TOA radiance file found in {}'.format(bundle))
            if (sr_image_file is None):
                print('No SR file found in {}'.format(bundle))
                continue

        if verbosity > 1: print('Image file {}, metadata file {}'.format(image_file, metafile))

        ## read meta data
        if verbosity > 1: print('Importing metadata from {}'.format(bundle))
        meta = ac.planet.metadata_parse(metafile)
        dtime = dateutil.parser.parse(meta['isotime'])
        doy = dtime.strftime('%j')
        se_distance = ac.shared.distance_se(doy)
        isodate = dtime.isoformat()

        ## merge sensor specific settings
        if new:
            setu = ac.acolite.settings.parse(meta['sensor'], settings=settings)
            verbosity = setu['verbosity']

            ## get other settings
            limit = setu['limit']
            output_geolocation = setu['output_geolocation']
            output_geometry = setu['output_geometry']
            output_xy = setu['output_xy']
            netcdf_projection = setu['netcdf_projection']

            vname = setu['region_name']
            gains = setu['gains']
            gains_toa = setu['gains_toa']
            if output is None: output = setu['output']

            merge_tiles = setu['merge_tiles']
            merge_zones = setu['merge_zones']
            extend_region = setu['extend_region']

            ## check if ROI polygon is given
            if setu['polylakes']:
                poly = ac.shared.polylakes(setu['polylakes_database'])
                setu['polygon_limit'] = False
            else:
                poly = setu['polygon']
            clip, clip_mask = False, None
            if poly is not None:
                if os.path.exists(poly):
                    try:
                        limit = ac.shared.polygon_limit(poly)
                        if setu['polygon_limit']:
                            print('Using limit from polygon envelope: {}'.format(limit))
                        else:
                            limit = setu['limit']
                        clip = True
                    except:
                        print('Failed to import polygon {}'.format(poly))

            if merge_tiles:
                if (limit is None):
                    if verbosity > 0: print("Merging tiles without ROI limit, merging to first tile extent")
                merge_zones = True
                extend_region = True
        sub = None

        ## read rsr
        rsrf = ac.path+'/data/RSR/{}.txt'.format(meta['sensor'])
        rsr, rsr_bands = ac.shared.rsr_read(rsrf)
        waves = np.arange(250, 2500)/1000
        waves_mu = ac.shared.rsr_convolute_dict(waves, waves, rsr)
        waves_names = {'{}'.format(b):'{:.0f}'.format(waves_mu[b]*1000) for b in waves_mu}

        ## get F0 - not stricty necessary if using USGS reflectance
        f0 = ac.shared.f0_get(f0_dataset=setu['solar_irradiance_reference'])
        f0_b = ac.shared.rsr_convolute_dict(np.asarray(f0['wave'])/1000, np.asarray(f0['data'])*10, rsr)

        ## gains
        gains_dict = None
        if gains & (gains_toa is not None):
            if len(gains_toa) == len(rsr_bands):
                gains_dict = {b: float(gains_toa[ib]) for ib, b in enumerate(rsr_bands)}

        gatts = {'sensor':meta['sensor'], 'satellite_sensor':meta['satellite_sensor'],
                 'isodate':isodate, #'global_dims':global_dims,
                 'sza':meta['sza'], 'vza':meta['vza'], 'raa':meta['raa'], 'se_distance': se_distance,
                 'mus': np.cos(meta['sza']*(np.pi/180.))}

        if image_file is not None:
            gatts['acolite_file_type'] = 'L1R'
        else:
            gatts['acolite_file_type'] = 'SR'

        stime = dateutil.parser.parse(gatts['isodate'])
        oname = '{}_{}{}'.format(gatts['satellite_sensor'], stime.strftime('%Y_%m_%d_%H_%M_%S'), '_merged' if merge_tiles else '')
        if vname != '': oname+='_{}'.format(vname)

        ## output file information
        if (merge_tiles is False) | (ofile is None):
            ofile = '{}/{}_{}.nc'.format(output, oname, gatts['acolite_file_type'])
            gatts['oname'] = oname
            gatts['ofile'] = ofile
        elif (merge_tiles) & (ofile is None):
            ofile = '{}/{}_{}.nc'.format(output, oname, gatts['acolite_file_type'])
            gatts['oname'] = oname
            gatts['ofile'] = ofile

        ## check if we should merge these tiles
        if (merge_tiles) & (not new) & (os.path.exists(ofile)):
                fgatts = ac.shared.nc_gatts(ofile)
                if (check_sensor) & (fgatts['satellite_sensor'] != gatts['satellite_sensor']):
                    print('Sensors do not match, skipping {}'.format(bundle))
                    continue
                if check_time:
                    tdiff = dateutil.parser.parse(fgatts['isodate'])-dateutil.parser.parse(gatts['isodate'])
                    tdiff = abs(tdiff.days*86400 + tdiff.seconds)
                    if (tdiff > max_merge_time):
                        print('Time difference too large, skipping {}'.format(bundle))
                        continue

        ## add band info to gatts
        for b in rsr_bands:
            gatts['{}_wave'.format(b)] = waves_mu[b]*1000
            gatts['{}_name'.format(b)] = waves_names[b]
            gatts['{}_f0'.format(b)] = f0_b[b]

        ## try to read projection of image file
        try:
            if image_file is not None:
                dct = ac.shared.projection_read(image_file)
            elif sr_image_file is not None:
                dct = ac.shared.projection_read(sr_image_file)
        except:
            ## else reproject image to default resolution
            ## if limit not set then gdal will be used to set up projection
            ## to be improved using RPC info
            print('Cannot determine image projection of {}, reprojecting.'.format(image_file))
            ret = ac.shared.warp_and_merge(image_file_original, output = output,
                                     limit = limit, resolution = setu['default_projection_resolution'])
            if len(ret) == 3:
                image_file = ret[0]
                dct = ret[1]
                dct_limit = ret[2]
            else:
                print('Image projection unsuccesful.')
                continue

        gatts['scene_xrange'] = dct['xrange']
        gatts['scene_yrange'] = dct['yrange']
        gatts['scene_proj4_string'] = dct['proj4_string']
        gatts['scene_pixel_size'] = dct['pixel_size']
        gatts['scene_dims'] = dct['dimensions']
        if 'zone' in dct: gatts['scene_zone'] = dct['zone']

        ## check crop
        if (sub is None) & (limit is not None):
            dct_sub = ac.shared.projection_sub(dct, limit, four_corners=True)
            if dct_sub['out_lon']:
                if verbosity > 1: print('Longitude limits outside {}'.format(bundle))
                continue
            if dct_sub['out_lat']:
                if verbosity > 1: print('Latitude limits outside {}'.format(bundle))
                continue
            sub = dct_sub['sub']
        else:
            if extend_region:
                print("Can't extend region if no ROI limits given")
                extend_region = False

        ##
        if ((merge_tiles is False) & (merge_zones is False)): warp_to = None
        if sub is None:
            if ((merge_zones) & (warp_to is not None)):
                if dct_prj != dct: ## target projection differs from this tile, need to set bounds
                    if dct['proj4_string'] != dct_prj['proj4_string']:
                        ## if the prj does not match, project current scene bounds to lat/lon
                        lonr, latr = dct['p'](dct['xrange'], dct['yrange'], inverse=True)
                        ## then to target projection
                        xrange_raw, yrange_raw = dct_prj['p'](lonr, (latr[1], latr[0]))
                        ## fix to nearest full pixel
                        pixel_size = dct_prj['pixel_size']
                        dct_prj['xrange'] = [xrange_raw[0] - (xrange_raw[0] % pixel_size[0]), xrange_raw[1]+pixel_size[0]-(xrange_raw[1] % pixel_size[0])]
                        dct_prj['yrange'] = [yrange_raw[1]+pixel_size[1]-(yrange_raw[1] % pixel_size[1]), yrange_raw[0] - (yrange_raw[0] % pixel_size[1])]
                        ## need to add new dimensions
                        dct_prj['xdim'] = int((dct_prj['xrange'][1]-dct_prj['xrange'][0])/pixel_size[0])+1
                        dct_prj['ydim'] = int((dct_prj['yrange'][1]-dct_prj['yrange'][0])/pixel_size[1])+1
                        dct_prj['dimensions'] = [dct_prj['xdim'], dct_prj['ydim']]
                    else:
                        ## if the projection matches just use the current scene projection
                        dct_prj = {k:dct[k] for k in dct}
            elif (warp_to is None):
                dct_prj = {k:dct[k] for k in dct}
        else:
            gatts['sub'] = sub
            gatts['limit'] = limit
            ## get the target NetCDF dimensions and dataset offset
            if (warp_to is None):
                if (extend_region): ## include part of the roi not covered by the scene
                    dct_prj = {k:dct_sub['region'][k] for k in dct_sub['region']}
                else: ## just include roi that is covered by the scene
                    dct_prj = {k:dct_sub[k] for k in dct_sub}
        ## end cropped

        ## get projection info for netcdf
        if netcdf_projection:
            nc_projection = ac.shared.projection_netcdf(dct_prj, add_half_pixel=True)
        else:
            nc_projection = None

        ## save projection keys in gatts
        pkeys = ['xrange', 'yrange', 'proj4_string', 'pixel_size', 'zone']
        for k in pkeys:
            if k in dct_prj: gatts[k] = dct_prj[k]

        ## warp settings for read_band
        ## updated 2021-10-28
        xyr = [min(dct_prj['xrange']),
               min(dct_prj['yrange']),
               max(dct_prj['xrange']),
               max(dct_prj['yrange']),
               dct_prj['proj4_string']]

        res_method = 'average'
        warp_to = (dct_prj['proj4_string'], xyr, dct_prj['pixel_size'][0],dct_prj['pixel_size'][1], res_method)

        ## store scene and output dimensions
        gatts['scene_dims'] = dct['ydim'], dct['xdim']
        gatts['global_dims'] = dct_prj['dimensions']

        ## new file for every bundle if not merging
        if (merge_tiles is False):
            new = True
            new_pan = True

        ## if we are clipping to a given polygon get the clip_mask here
        if clip:
            clip_mask = ac.shared.polygon_crop(dct_prj, poly, return_sub=False)
            clip_mask = clip_mask.astype(bool) == False

        ## write lat/lon
        if (output_geolocation):
            if (os.path.exists(ofile) & (not new)):
                datasets = ac.shared.nc_datasets(ofile)
            else:
                datasets = []
            if ('lat' not in datasets) or ('lon' not in datasets):
                if verbosity > 1: print('Writing geolocation lon/lat')
                lon, lat = ac.shared.projection_geo(dct_prj, add_half_pixel=True)
                ac.output.nc_write(ofile, 'lon', lon, attributes=gatts, new=new, nc_projection=nc_projection)
                if verbosity > 1: print('Wrote lon ({})'.format(lon.shape))
                lon = None
                ac.output.nc_write(ofile, 'lat', lat)
                if verbosity > 1: print('Wrote lat ({})'.format(lat.shape))
                lat = None
                new=False

        ## write x/y
        if (output_xy):
            if os.path.exists(ofile) & (not new):
                datasets = ac.shared.nc_datasets(ofile)
            else:
                datasets = []
            if ('x' not in datasets) or ('y' not in datasets):
                if verbosity > 1: print('Writing geolocation x/y')
                x, y = ac.shared.projection_geo(dct_prj, xy=True, add_half_pixel=True)
                ac.output.nc_write(ofile, 'xm', x, new=new)
                if verbosity > 1: print('Wrote xm ({})'.format(x.shape))
                x = None
                ac.output.nc_write(ofile, 'ym', y)
                if verbosity > 1: print('Wrote ym ({})'.format(y.shape))
                y = None
                new=False

        ## convert bands TOA
        for b in rsr_bands:
            if image_file is None: continue
            if b in ['PAN']: continue
            idx = int(meta['{}-band_idx'.format(b)])

            ## read data
            md, data = ac.shared.read_band(image_file, idx=idx, warp_to=warp_to, gdal_meta = True)
            nodata = data == np.uint16(0)

            if 'Skysat' in meta['sensor']:
                ## get reflectance scaling from tiff tags
                try:
                    prop = json.loads(md['TIFFTAG_IMAGEDESCRIPTION'])['properties']
                except:
                    prop = {}

                if 'reflectance_coefficients' in prop:
                    ## convert to toa radiance & mask
                    bi = idx - 1
                    data = data.astype(float) * prop['reflectance_coefficients'][bi]
                    data[nodata] = np.nan
                else:
                    print('Using fixed 0.01 factor to convert Skysat DN to TOA radiance')
                    ## convert to toa radiance & mask
                    data = data.astype(float) * 0.01

                    ## convert to toa reflectance
                    f0 = gatts['{}_f0'.format(b)]/10
                    data *= (np.pi * gatts['se_distance']**2) / (f0 * gatts['mus'])
            else:
                ## convert from radiance
                if  (meta['sensor'] == 'RapidEye') | (from_radiance):
                    data = data.astype(float) * float(meta['{}-{}'.format(b,'to_radiance')])
                    f0 = gatts['{}_f0'.format(b)]/10
                    data *= (np.pi * gatts['se_distance']**2) / (f0 * gatts['mus'])
                else:
                    data = data.astype(float) * float(meta['{}-{}'.format(b,'to_reflectance')])
            data[nodata] = np.nan
            print(data.shape)

            ## clip to poly
            if clip: data[clip_mask] = np.nan

            ds = 'rhot_{}'.format(waves_names[b])
            ds_att = {'wavelength':waves_mu[b]*1000}

            if gains & (gains_dict is not None):
                ds_att['toa_gain'] = gains_dict[b]
                data *= ds_att['toa_gain']
                if verbosity > 1: print('Converting bands: Applied TOA gain {} to {}'.format(ds_att['toa_gain'], ds))

            if percentiles_compute:
                ds_att['percentiles'] = percentiles
                ds_att['percentiles_data'] = np.nanpercentile(data, percentiles)

            ## write to netcdf file
            ac.output.nc_write(ofile, ds, data, replace_nan=True, attributes=gatts,
                                new=new, dataset_attributes = ds_att, nc_projection=nc_projection)
            new = False
            if verbosity > 1: print('Converting bands: Wrote {} ({})'.format(ds, data.shape))

        ## convert bands SR
        if (sr_image_file is not None) & (setu['planet_store_sr']):
            md = ac.shared.read_gdal_meta(sr_image_file)
            if 'TIFFTAG_IMAGEDESCRIPTION' in md:
                md_dict = json.loads(md['TIFFTAG_IMAGEDESCRIPTION'])
                if 'atmospheric_correction' in md_dict:
                    for k in md_dict['atmospheric_correction']:
                        gatts['planet_sr_{}'.format(k)] = md_dict['atmospheric_correction'][k]
            update_attributes = True
            for b in rsr_bands:
                if b in ['PAN']: continue
                idx = int(meta['{}-band_idx'.format(b)])

                ## read data
                md, data = ac.shared.read_band(sr_image_file, idx=idx, warp_to=warp_to, gdal_meta = True)
                nodata = data == np.uint16(0)

                ## DN to rhos is 1/10000 according to Planet Docs
                data = data.astype(float) / 10000
                data[nodata] = np.nan

                ## clip to poly
                if clip: data[clip_mask] = np.nan

                ds = 'rhos_sr_{}'.format(waves_names[b])
                ds_att = {'wavelength':waves_mu[b]*1000}

                if percentiles_compute:
                    ds_att['percentiles'] = percentiles
                    ds_att['percentiles_data'] = np.nanpercentile(data, percentiles)

                ## write to netcdf file
                ac.output.nc_write(ofile, ds, data, replace_nan=True, attributes=gatts, update_attributes=update_attributes,
                                    new=new, dataset_attributes = ds_att, nc_projection=nc_projection)
                new = False
                update_attributes = False
                if verbosity > 1: print('Converting bands: Wrote {} ({})'.format(ds, data.shape))

        if verbosity > 1:
            print('Conversion took {:.1f} seconds'.format(time.time()-t0))
            print('Created {}'.format(ofile))

        if limit is not None: sub = None
        if ofile not in ofiles: ofiles.append(ofile)

        ## remove the extracted bundle
        if zipped:
             shutil.rmtree(bundle)
             bundle = '{}'.format(bundle_orig)

    return(ofiles, setu)
