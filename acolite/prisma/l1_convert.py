## def l1_convert
## converts PRISMA HDF file to l1r NetCDF for acolite-gen
## written by Quinten Vanhellemont, RBINS
## 2021-07-14
## modifications: 2021-12-31 (QV) new handling of settings
##                2022-01-04 (QV) added netcdf compression
##                2022-02-23 (QV) added option to output L2C reflectances
##                2023-05-09 (QV) added option to crop
##                2023-07-12 (QV) removed netcdf_compression settings from nc_write call

def l1_convert(inputfile, output=None, settings = {}, verbosity=0):
    import numpy as np
    import h5py, dateutil.parser, os
    import acolite as ac

    ## parse settings
    sensor = 'PRISMA'
    setu = ac.acolite.settings.parse(sensor, settings=settings)
    verbosity = setu['verbosity']
    if output is None: output = setu['output']
    output_lt = setu['output_lt']
    vname = setu['region_name']
    limit = setu['limit']
    store_l2c = setu['prisma_store_l2c']
    store_l2c_separate_file = setu['prisma_store_l2c_separate_file']

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
    ## end ROI polygon

    ## parse inputfile
    if type(inputfile) != list:
        if type(inputfile) == str:
            inputfile = inputfile.split(',')
        else:
            inputfile = list(inputfile)
    nscenes = len(inputfile)
    if verbosity > 1: print('Starting conversion of {} scenes'.format(nscenes))

    ## get F0 for radiance -> reflectance computation
    f0 = ac.shared.f0_get(f0_dataset=setu['solar_irradiance_reference'])

    ofiles = []
    for file in inputfile:
        #f = h5py.File(file, mode='r')
        h5_gatts = ac.prisma.attributes(file)

        waves_vnir = h5_gatts['List_Cw_Vnir']
        bands_vnir = ['{:.0f}'.format(w) for w in waves_vnir]
        fwhm_vnir = h5_gatts['List_Fwhm_Vnir']
        n_vnir = len(waves_vnir)

        waves_swir = h5_gatts['List_Cw_Swir']
        bands_swir = ['{:.0f}'.format(w) for w in waves_swir]
        fwhm_swir = h5_gatts['List_Fwhm_Swir']
        n_swir = len(waves_swir)

        waves = [w for w in waves_vnir] + [w for w in waves_swir]
        fwhm = [f for f in fwhm_vnir] + [f for f in fwhm_swir]
        waves_names = ['{:.0f}'.format(w) for w in waves]
        instrument = ['vnir']*n_vnir + ['swir']*n_swir
        band_index = [i for i in range(n_vnir)] + [i for i in range(n_swir)]

        band_names_vnir = ['vnir_{}'.format(b) for b in range(0, n_vnir)]
        band_names_swir = ['swir_{}'.format(b) for b in range(0, n_swir)]

        rsr_vnir = {'vnir_{}'.format(b): ac.shared.gauss_response(waves_vnir[b], fwhm_vnir[b], step=0.1) for b in range(0, n_vnir)}
        rsr_swir = {'swir_{}'.format(b): ac.shared.gauss_response(waves_swir[b], fwhm_swir[b], step=0.1) for b in range(0, n_swir)}

        band_names = band_names_vnir + band_names_swir
        band_rsr = {}
        for b in rsr_vnir: band_rsr[b] = {'wave': rsr_vnir[b][0]/1000, 'response': rsr_vnir[b][1]}
        for b in rsr_swir: band_rsr[b] = {'wave': rsr_swir[b][0]/1000, 'response': rsr_swir[b][1]}

        ## use same rsr as acolite_l2r
        #rsr = ac.shared.rsr_hyper(gatts['band_waves'], gatts['band_widths'], step=0.1)
        # rsrd = ac.shared.rsr_dict(rsrd={sensor:{'rsr':band_rsr}})
        # waves = [rsrd[sensor]['wave_nm'][b] for b in band_names]
        # waves_names = [rsrd[sensor]['wave_name'][b] for b in band_names]

        idx = np.argsort(waves)
        f0d = ac.shared.rsr_convolute_dict(f0['wave']/1000, f0['data'], band_rsr)

        bands = {}
        for i in idx:
            cwave = waves[i]
            if cwave == 0: continue
            swave = '{:.0f}'.format(cwave)
            bands[swave]= {'wave':cwave, 'wavelength':cwave, 'wave_mu':cwave/1000.,
                           'wave_name':waves_names[i],
                           'width': fwhm[i],
                           'i':i, 'index':band_index[i],
                           'rsr': band_rsr[band_names[i]],
                           'f0': f0d[band_names[i]],
                           'instrument':instrument[i],}

        # print(rsrd[sensor]['wave_name'])
        # print(bands)
        # stop

        gatts = {}

        isotime = h5_gatts['Product_StartTime']
        time = dateutil.parser.parse(isotime)

        doy = int(time.strftime('%j'))
        d = ac.shared.distance_se(doy)

        ## lon and lat keys
        lat_key = 'Latitude_SWIR'
        lon_key = 'Longitude_SWIR'
        if 'PRS_L1G_STD_OFFL_' in os.path.basename(file):
            lat_key = 'Latitude'
            lon_key = 'Longitude'

        ## mask for L1G format
        mask_value = 65535
        dem = None

        ## reading settings
        src = 'HCO' ## coregistered radiance cube
        read_cube = True

        ## get geometry from l2 file if present
        if ac.settings['run']['l2cfile'] is not None:
            l2file = ac.settings['run']['l2cfile']
            print('Using user specified L2C file {}'.format(l2file))
        else:
            l2file = os.path.dirname(file) + os.path.sep + os.path.basename(file).replace('PRS_L1_STD_OFFL_', 'PRS_L2C_STD_')
        if not os.path.exists(l2file):
            print('PRISMA processing only supported when L2 geometry is present.')
            print('Please put {} in the same directory as {}'.format(os.path.basename(l2file), os.path.basename(file)))
            continue

        ## read geolocation
        with h5py.File(file, mode='r') as f:
            lat = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Geolocation Fields'][lat_key][:]
            lon = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Geolocation Fields'][lon_key][:]
            lat[lat>=mask_value] = np.nan
            lon[lon>=mask_value] = np.nan
        sub = None
        if limit is not None:
            sub = ac.shared.geolocation_sub(lat, lon, limit)
            if sub is None:
                print('Limit outside of scene {}'.format(file))
                continue
            ## crop to sub
            lat = lat[sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
            lon = lon[sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
        ## end read geolocation

        ## read geometry
        vza, vaa, sza, saa, raa = None, None, None, None, None
        with h5py.File(l2file, mode='r') as f:
            ## L1G format
            if 'PRS_L1G_STD_OFFL_' in os.path.basename(file):
                #lat_key = 'Latitude'
                #lon_key = 'Longitude'
                if sub is None:
                    vza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Zenith_Angle'][:]
                    vaa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Azimuth_Angle'][:]
                    sza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Zenith_Angle'][:]
                    saa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Azimuth_Angle'][:]
                else:
                    vza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Zenith_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
                    vaa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Sensor_Azimuth_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
                    sza = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Zenith_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
                    saa = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Geometric Fields']['Solar_Azimuth_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]

                ## apply mask
                vza[vza>=mask_value] = np.nan
                sza[sza>=mask_value] = np.nan
                saa[saa>=mask_value] = np.nan
                vaa[vaa>=mask_value] = np.nan

                ## compute relative azimuth
                raa = np.abs(saa - vaa)
                raa[raa>180] = 360 - raa[raa>180]

                ## get DEM data
                if sub is None:
                    dem = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Terrain Fields']['DEM'][:]
                else:
                    dem = f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Terrain Fields']['DEM'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
                dem[dem>=mask_value] = np.nan
            else:
                if sub is None:
                    vza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Observing_Angle'][:]
                    raa = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Rel_Azimuth_Angle'][:]
                    sza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Solar_Zenith_Angle'][:]
                else:
                    vza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Observing_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
                    raa = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Rel_Azimuth_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]
                    sza = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']['Solar_Zenith_Angle'][sub[1]:sub[1]+sub[3], sub[0]:sub[0]+sub[2]]

        gatts['vza'] = np.nanmean(np.abs(vza))
        gatts['raa'] = np.nanmean(np.abs(raa))
        gatts['sza'] = np.nanmean(np.abs(sza))

        with h5py.File(file, mode='r') as f:
            ## read bands in spectral order
            if read_cube:
                if sub is None:
                    vnir_data = h5_gatts['Offset_Vnir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][:]/h5_gatts['ScaleFactor_Vnir']
                    swir_data = h5_gatts['Offset_Swir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][:]/h5_gatts['ScaleFactor_Swir']
                else:
                    vnir_data = h5_gatts['Offset_Vnir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Vnir']
                    swir_data = h5_gatts['Offset_Swir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Swir']

                vnir_data[vnir_data>=mask_value] = np.nan
                swir_data[swir_data>=mask_value] = np.nan

            ## read LOS vectors
            x_ = f['KDP_AUX']['LOS_Vnir'][:,0]
            y_ = f['KDP_AUX']['LOS_Vnir'][:,1]
            z_ = f['KDP_AUX']['LOS_Vnir'][:,2]

        ## get vza/vaa
        #dtor = np.pi/180
        #vza = np.arctan2(y_,x_)/dtor
        #vaa = np.arctan2(z_,np.sqrt(x_**2+y_**2))/dtor
        #vza_ave = np.nanmean(np.abs(vza))
        #vaa_ave = np.nanmean(np.abs(vaa))
        sza_ave = h5_gatts['Sun_zenith_angle']
        saa_ave = h5_gatts['Sun_azimuth_angle']

        if setu['prisma_rhot_per_pixel_sza']:
            cossza = np.cos(np.radians(sza))
        else:
            cossza = np.cos(np.radians(sza_ave))

        vza_ave = 0
        vaa_ave = 0

        if 'sza' not in gatts: gatts['sza'] = sza_ave
        if 'vza' not in gatts: gatts['vza'] = vza_ave
        if 'saa' not in gatts: gatts['saa'] = saa_ave
        if 'vaa' not in gatts: gatts['vaa'] = vaa_ave

        if 'raa' not in gatts:
            raa_ave = abs(gatts['saa'] - gatts['vaa'])
            while raa_ave >= 180: raa_ave = abs(raa_ave-360)
            gatts['raa'] = raa_ave

        mu0 = np.cos(gatts['sza']*(np.pi/180))
        muv = np.cos(gatts['vza']*(np.pi/180))

        if output is None:
            odir = os.path.dirname(file)
        else:
            odir = output

        gatts['sensor'] = sensor
        gatts['isodate'] = time.isoformat()

        obase  = '{}_{}_L1R'.format(gatts['sensor'],  time.strftime('%Y_%m_%d_%H_%M_%S'))
        if not os.path.exists(odir): os.makedirs(odir)
        ofile = '{}/{}.nc'.format(odir, obase)

        gatts['obase'] = obase

        gatts['band_waves'] = [bands[w]['wave'] for w in bands]
        gatts['band_widths'] = [bands[w]['width'] for w in bands]

        new = True
        if (setu['output_geolocation']) & (new):
            if verbosity > 1: print('Writing geolocation lon/lat')
            ac.output.nc_write(ofile, 'lon', np.flip(np.rot90(lon)), new=new, attributes=gatts)
            if verbosity > 1: print('Wrote lon ({})'.format(lon.shape))
            new = False
            if not (store_l2c & store_l2c_separate_file): lon = None

            ac.output.nc_write(ofile, 'lat', np.flip(np.rot90(lat)), new=new, attributes=gatts)
            if verbosity > 1: print('Wrote lat ({})'.format(lat.shape))
            if not (store_l2c & store_l2c_separate_file): lat = None

        ## write geometry
        if os.path.exists(l2file):
            if (setu['output_geometry']):
                if verbosity > 1: print('Writing geometry')
                ac.output.nc_write(ofile, 'vza', np.flip(np.rot90(vza)), attributes=gatts, new=new)
                if verbosity > 1: print('Wrote vza ({})'.format(vza.shape))
                vza = None
                new = False
                if vaa is not None:
                    ac.output.nc_write(ofile, 'vaa', np.flip(np.rot90(vaa)), attributes=gatts, new=new)
                    if verbosity > 1: print('Wrote vaa ({})'.format(vaa.shape))
                    vaa = None

                ac.output.nc_write(ofile, 'sza', np.flip(np.rot90(sza)), attributes=gatts, new=new)
                if verbosity > 1: print('Wrote sza ({})'.format(sza.shape))

                if saa is not None:
                    ac.output.nc_write(ofile, 'saa', np.flip(np.rot90(saa)), attributes=gatts, new=new)
                    if verbosity > 1: print('Wrote saa ({})'.format(saa.shape))
                    saa = None

                ac.output.nc_write(ofile, 'raa', np.flip(np.rot90(raa)), attributes=gatts, new=new)
                if verbosity > 1: print('Wrote raa ({})'.format(raa.shape))
                raa = None

            if dem is not None:
                ac.output.nc_write(ofile, 'dem', np.flip(np.rot90(dem)))

        ## store l2c data
        if store_l2c & read_cube:
            if store_l2c_separate_file:
                obase_l2c  = '{}_{}_converted_L2C'.format('PRISMA',  time.strftime('%Y_%m_%d_%H_%M_%S'))
                ofile_l2c = '{}/{}.nc'.format(odir, obase_l2c)
                ac.output.nc_write(ofile_l2c, 'lat', np.flip(np.rot90(lat)), new=True, attributes=gatts)
                lat = None
                ac.output.nc_write(ofile_l2c, 'lon', np.flip(np.rot90(lon)))
                lon = None
            else:
                ofile_l2c = '{}'.format(ofile)

            ## get l2c details for reflectance conversion
            h5_l2c_gatts = ac.prisma.attributes(l2file)
            scale_max = h5_l2c_gatts['L2ScaleVnirMax']
            scale_min = h5_l2c_gatts['L2ScaleVnirMin']

            ##  read in data cube
            with h5py.File(l2file, mode='r') as f:
                if sub is None:
                    vnir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['VNIR_Cube'][:]
                    swir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['SWIR_Cube'][:]
                else:
                    vnir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['VNIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]
                    swir_l2c_data = f['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Data Fields']['SWIR_Cube'][sub[1]:sub[1]+sub[3], :, sub[0]:sub[0]+sub[2]]

        ## write TOA data
        for bi, b in enumerate(bands):
            wi = bands[b]['index']
            i = bands[b]['i']
            print('Reading rhot_{}'.format(bands[b]['wave_name']))

            if bands[b]['instrument'] == 'vnir':
                if read_cube:
                    cdata_radiance = vnir_data[:,wi,:]
                    cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)
                    if store_l2c:
                        cdata_l2c = scale_min + (vnir_l2c_data[:, wi, :] * (scale_max - scale_min)) / 65535
                else:
                    if sub is None:
                        cdata_radiance = h5_gatts['Offset_Vnir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][:,i,:]/h5_gatts['ScaleFactor_Vnir']
                    else:
                        cdata_radiance = h5_gatts['Offset_Vnir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['VNIR_Cube'][sub[1]:sub[1]+sub[3], i, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Vnir']
                    cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)

            if bands[b]['instrument'] == 'swir':
                if read_cube:
                    cdata_radiance = swir_data[:,wi,:]
                    cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)
                    if store_l2c:
                        cdata_l2c = scale_min + (swir_l2c_data[:, wi, :] * (scale_max - scale_min)) / 65535
                else:
                    if sub is None:
                        cdata_radiance = h5_gatts['Offset_Swir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][:,i,:]/h5_gatts['ScaleFactor_Swir']
                    else:
                        cdata_radiance = h5_gatts['Offset_Swir'] + \
                                f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(src)]['Data Fields']['SWIR_Cube'][sub[1]:sub[1]+sub[3], i, sub[0]:sub[0]+sub[2]]/h5_gatts['ScaleFactor_Swir']
                    cdata = cdata_radiance * (np.pi * d * d) / (bands[b]['f0'] * cossza)

            ds_att = {k:bands[b][k] for k in bands[b] if k not in ['rsr']}

            if output_lt:
                ## write toa radiance
                ac.output.nc_write(ofile, 'Lt_{}'.format(bands[b]['wave_name']),
                                    np.flip(np.rot90(cdata_radiance)),dataset_attributes = ds_att)
                cdata_radiance = None

            ## write toa reflectance
            ac.output.nc_write(ofile, 'rhot_{}'.format(bands[b]['wave_name']),
                                    np.flip(np.rot90(cdata)), dataset_attributes = ds_att)
            cdata = None
            print('Wrote rhot_{}'.format(bands[b]['wave_name']))

            ## store L2C data
            if store_l2c & read_cube:
                ac.output.nc_write(ofile_l2c, 'rhos_l2c_{}'.format(bands[b]['wave_name']),
                                    np.flip(np.rot90(cdata_l2c)),dataset_attributes = ds_att)
                ofile_l2c_new = False
                cdata_l2c = None
                print('Wrote rhos_l2c_{}'.format(bands[b]['wave_name']))

        ## output PAN
        if setu['prisma_output_pan']:
            psrc = src.replace('H', 'P')
            with h5py.File(file, mode='r') as f:
                if sub is None:
                    pan = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Data Fields']['Cube'][:]
                    plat = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Latitude'][:]
                    plon = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Longitude'][:]
                else:
                    psub = [s*6 for s in sub]
                    pan = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Data Fields']['Cube'][psub[1]:psub[1]+psub[3], psub[0]:psub[0]+psub[2]]
                    plat = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Latitude'][psub[1]:psub[1]+psub[3], psub[0]:psub[0]+psub[2]]
                    plon = f['HDFEOS']['SWATHS']['PRS_L1_{}'.format(psrc)]['Geolocation Fields']['Longitude'][psub[1]:psub[1]+psub[3], psub[0]:psub[0]+psub[2]]

            ## convert to radiance
            pan = h5_gatts['Offset_Pan'] + pan / h5_gatts['ScaleFactor_Pan']

            ## output netcdf
            ofile_pan = '{}/{}_pan.nc'.format(odir, obase)
            ac.output.nc_write(ofile_pan, 'lon', np.flip(np.rot90(plon)),new = True) #dataset_attributes = ds_att,
            plon = None
            ac.output.nc_write(ofile_pan, 'lat', np.flip(np.rot90(plat))) #dataset_attributes = ds_att,
            plat = None
            ac.output.nc_write(ofile_pan, 'pan', np.flip(np.rot90(pan))) #dataset_attributes = ds_att,
            pan = None
         ## end PAN

        ofiles.append(ofile)
    return(ofiles, setu)
