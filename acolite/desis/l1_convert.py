## def l1_convert
## converts DESIS file to l1r NetCDF for acolite-gen
## written by Quinten Vanhellemont, RBINS
## 2021-08-10
## modifications: 2021-12-31 (QV) new handling of settings
##                2022-01-04 (QV) added netcdf compression
##                2022-03-28 (QV) added masking using QL data, updated crop subsetting
##                2022-04-15 (QV) fixed polygon masking
##                2022-09-24 (QV) fixed DESIS band naming
##                                aligned DESIS RSR handling with other hyperspectral sensors
##                2023-07-12 (QV) removed netcdf_compression settings from nc_write call

def l1_convert(inputfile, output = None, settings = {}, verbosity = 5):
    import numpy as np
    import datetime, dateutil.parser, os, copy
    import acolite as ac

    ## parse sensor specific settings
    setu = ac.acolite.settings.parse('DESIS_HSI', settings=settings)
    vname = setu['region_name']
    output_lt = setu['output_lt']
    if output is None: output = setu['output']
    verbosity = setu['verbosity']
    poly = setu['polygon']
    limit = setu['limit']

    ## parse inputfile
    if type(inputfile) != list:
        if type(inputfile) == str:
            inputfile = inputfile.split(',')
        else:
            inputfile = list(inputfile)
    nscenes = len(inputfile)
    if verbosity > 1: print('Starting conversion of {} scenes'.format(nscenes))

    ## check if ROI polygon is given
    clip, clip_mask = False, None
    if poly is not None:
        if os.path.exists(poly):
            try:
                limit = ac.shared.polygon_limit(poly)
                print('Using limit from polygon envelope: {}'.format(limit))
                clip = True
            except:
                print('Failed to import polygon {}'.format(poly))

    ## get F0 for radiance -> reflectance computation
    f0 = ac.shared.f0_get(f0_dataset=setu['solar_irradiance_reference'])

    ofiles = []
    for bundle in inputfile:
        ## identify files
        metafile, imagefile = ac.desis.bundle_test(bundle)
        dn = os.path.dirname(imagefile)
        bn = os.path.basename(imagefile)
        headerfile = '{}/{}'.format(dn, bn.replace('.tif', '.hdr'))
        qlfile = '{}/{}'.format(dn, bn.replace('SPECTRAL_IMAGE', 'QL_QUALITY'))

        ## read metadata
        header = ac.shared.hdr(headerfile)
        meta = ac.desis.metadata(metafile)

        ## set up projection
        warp_to, dct_prj, sub = None, None, None
        try:
            ## get projection from image
            dct = ac.shared.projection_read(imagefile)
        except:
            print('Could not determine image projection')
            dct = None

        ## find crop
        if (limit is not None) and (dct is not None):
            dct_sub = ac.shared.projection_sub(dct, limit)
            if dct_sub['out_lon']:
                if verbosity > 1: print('Longitude limits outside {}'.format(bundle))
                continue
            if dct_sub['out_lat']:
                if verbosity > 1: print('Latitude limits outside {}'.format(bundle))
                continue
            sub = dct_sub['sub']

        if dct is not None:
            if sub is None:
                dct_prj = {k:dct[k] for k in dct}
            else:
                dct_prj = {k:dct_sub[k] for k in dct_sub}

                ## updated 2022-03-28
                xyr = [min(dct_prj['xrange']),
                       min(dct_prj['yrange']),
                       max(dct_prj['xrange']),
                       max(dct_prj['yrange']),
                       dct_prj['proj4_string']]

                ## warp settings for read_band
                res_method = 'near'
                warp_to = (dct_prj['proj4_string'], xyr, dct_prj['pixel_size'][0],dct_prj['pixel_size'][1], res_method)

        ## date and time
        stime = dateutil.parser.parse(meta['startTime'])
        etime = dateutil.parser.parse(meta['endTime'])
        otime = (etime-stime).seconds
        time = stime + datetime.timedelta(seconds=otime/2)
        doy = time.strftime('%j')
        se_distance = ac.shared.distance_se(doy)

        ## collect global attributes
        gatts = {}
        gatts['isodate'] = time.isoformat()
        gatts['sensor'] = '{}_{}'.format(meta['mission'], meta['sensor'])
        gatts['version'] = meta['version']
        gatts['doy'] = doy
        gatts['se_distance'] = se_distance
        # obase  = '{}_{}_L1R'.format(gatts['sensor'],  time.strftime('%Y_%m_%d_%H_%M_%S'))
        obase = '{}_{}_{}_L1R'.format(gatts['sensor'], meta['tileID'],time.strftime('%Y_%m_%d_%H_%M_%S'))
        gatts['obase'] = obase

        ## add band info
        gatts['band_waves'] = header['wavelength']
        gatts['band_widths'] = header['fwhm']
        #gatts['band_names'] = header['band names']

        ## add projection info
        if dct_prj is not None:
            pkeys = ['xrange', 'yrange', 'proj4_string', 'pixel_size', 'zone']
            for k in pkeys:
                if k in dct_prj: gatts[k] = copy.copy(dct_prj[k])

            ## if we are clipping to a given polygon get the clip_mask here
            if clip:
                clip_mask = ac.shared.polygon_crop(dct_prj, poly, return_sub=False)
                clip_mask = clip_mask.astype(bool) == False

        ## make rsr and bands dataset
        rsr = ac.shared.rsr_hyper(gatts['band_waves'], gatts['band_widths'], step=0.1)
        rsrd = ac.shared.rsr_dict(rsrd={gatts['sensor']:{'rsr':rsr}})
        band_rsr = rsrd[gatts['sensor']]['rsr']
        f0d = ac.shared.rsr_convolute_dict(f0['wave']/1000, f0['data'], band_rsr)

        ## make bands dataset
        bands = {}
        for bi, b in enumerate(band_rsr):
            cwave = rsrd[gatts['sensor']]['wave_nm'][b]
            swave = '{:.0f}'.format(cwave)
            bands[b]= {'wave':cwave, 'wavelength':cwave, 'wave_mu':cwave/1000.,
                       'wave_name':'{:.0f}'.format(cwave),
                       'width': gatts['band_widths'][bi],
                       'rsr': band_rsr[b],'f0': f0d[b]}

        # gatts['saa'] = float(meta['sunAzimuthAngle'])
        # gatts['sza'] = float(meta['sunZenithAngle'])
        # gatts['vaa'] = float(meta['sceneAzimuthAngle'])
        # gatts['vza'] = float(meta['sceneIncidenceAngle'])
        gatts['saa'] = meta['saa']
        gatts['sza'] = meta['sza']
        gatts['vaa'] = meta['vaa']
        gatts['vza'] = meta['vza']
        gatts['raa'] = meta['raa']

        # if 'raa' not in gatts:
        #     raa_ave = abs(gatts['saa'] - gatts['vaa'])
        #     while raa_ave >= 180: raa_ave = abs(raa_ave-360)
        #     gatts['raa'] = raa_ave

        mu0 = np.cos(gatts['sza']*(np.pi/180))
        muv = np.cos(gatts['vza']*(np.pi/180))

        if output is None:
            odir = os.path.dirname(imagefile)
        else:
            odir = output
        if not os.path.exists(odir): os.makedirs(odir)
        ofile = '{}/{}.nc'.format(odir, obase)

        new = True
        if dct_prj is not None:
            print('Computing and writing lat/lon')
            ## offset half pixels to compute center pixel lat/lon
            dct_prj['xrange'] = dct_prj['xrange'][0]+dct_prj['pixel_size'][0]/2, dct_prj['xrange'][1]-dct_prj['pixel_size'][0]/2
            dct_prj['yrange'] = dct_prj['yrange'][0]+dct_prj['pixel_size'][1]/2, dct_prj['yrange'][1]-dct_prj['pixel_size'][1]/2
            ## compute lat/lon
            lon, lat = ac.shared.projection_geo(dct_prj, add_half_pixel = False)
            print(lat.shape)
            ac.output.nc_write(ofile, 'lat', lat, new = new, attributes = gatts)
            lat = None
            ac.output.nc_write(ofile, 'lon', lon)
            lon = None
            new = False

        ## read data cube (faster)
        read_cube = True
        if read_cube:
            print('Reading DESIS image cube')
            cube = ac.shared.read_band(imagefile, sub = sub, warp_to = warp_to).astype(np.float32)
            cube[cube == header['data ignore value']] = np.nan
            print(cube.shape)
            if setu['desis_mask_ql']:
                ## read QL data
                mask_cube = ac.shared.read_band(qlfile, sub = sub, warp_to = warp_to)
                ## mask cube data, assume any non zero is bad
                cube[mask_cube > 0] = np.nan

        ## write TOA data
        for bi, b in enumerate(bands):
            print('Computing rhot_{} for {}'.format(bands[b]['wave_name'], gatts['obase']))
            ds_att = {k: bands[b][k] for k in bands[b] if k not in ['rsr']}

            ## read data
            if read_cube:
                cdata_radiance = 1.0 * cube[bi, :, :]
            else:
                cdata_radiance = ac.shared.read_band(imagefile, bi+1, sub=sub, warp_to = warp_to).astype(np.float32)
                cdata_radiance[cdata_radiance == header['data ignore value']] = np.nan
                if setu['desis_mask_ql']:
                    ## read QL data
                    mask_data = ac.shared.read_band(qlfile, bi+1, sub = sub, warp_to = warp_to)
                    ## mask cube data, assume any non zero is bad
                    cdata_radiance[mask_data > 0] = np.nan

            ## compute radiance
            cdata_radiance = cdata_radiance.astype(np.float32) * header['data gain values'][bi]
            cdata_radiance += header['data offset values'][bi]

            if (clip) & (clip_mask is not None): cdata_radiance[clip_mask] = np.nan

            if output_lt:
                ## write toa radiance
                ac.output.nc_write(ofile, 'Lt_{}'.format(bands[b]['wave_name']), cdata_radiance,
                                            attributes = gatts, dataset_attributes = ds_att, new = new)
                new = False

            ## compute reflectance
            cdata = cdata_radiance * (np.pi * gatts['se_distance'] * gatts['se_distance']) / (bands[b]['f0']/10 * mu0)
            cdata_radiance = None

            ac.output.nc_write(ofile, 'rhot_{}'.format(bands[b]['wave_name']), cdata,\
                                            attributes = gatts, dataset_attributes = ds_att, new = new)
            cdata = None
            new = False
        cube = None

        ofiles.append(ofile)
    return(ofiles, setu)
