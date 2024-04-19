## separate flagging function for ACOLITE NetCDF files
## flagging steps copied from acolite_l2w
##
## written by Quinten Vanhellemont, RBINS
## 2024-01-26
## modifications:
##

import concurrent.futures
from threading import Lock

def acolite_flags(gem, create_flags_dataset=True, write_flags_dataset=False, return_flags_dataset=True):
    import acolite as ac
    import numpy as np
    import scipy.ndimage

    # WIP Lock to handle access to the gem object which has single threaded file IO
    lock = Lock()

    ## read gem file if NetCDF
    if type(gem) is str:
        gemf = '{}'.format(gem)
        gem = ac.gem.gem(gem)
    gemf = gem.file
    if ac.settings['run']['verbosity'] > 0: print('Running ACOLITE flagging function for {}.'.format(gemf))

    ## get rhot and rhos wavelengths
    rhot_ds = [ds for ds in gem.datasets if 'rhot_' in ds]
    rhot_waves = [int(ds.split('_')[-1]) for ds in rhot_ds]
    if len(rhot_waves) == 0: print('{} is probably not an ACOLITE L2R file: {} rhot datasets.'.format(gemf, len(rhot_ds)))
    #print(rhot_ds, rhot_waves)

    rhos_ds = [ds for ds in gem.datasets if 'rhos_' in ds]
    rhos_waves = [int(ds.split('_')[-1]) for ds in rhos_ds]
    if len(rhos_waves) == 0: print('{} is probably not an ACOLITE L2R file: {} rhos datasets.'.format(gemf, len(rhos_ds)))
    #print(rhos_ds, rhos_waves)

    ## source or output flags name
    ## only used when writing here
    flags_name = 'l2_flags'
    #flags_name = '{}_flags'.format(gem.gatts['acolite_file_type'][0:2].lower())

    ## create l2_flags dataset
    flags_att = {}
    if (flags_name not in gem.datasets) | (create_flags_dataset):
        flags = np.zeros(gem.data('lon').shape,np.int32)
    else:
        flags = gem.data(flags_name)

    ## compute flags
    ####
    ## non water/swir threshold
    if ac.settings['run']['verbosity'] > 3: print('Computing non water threshold mask.')
    cidx,cwave = ac.shared.closest_idx(rhot_waves, ac.settings['run']['l2w_mask_wave'])
    ## use M bands for masking
    if ('VIIRS' in gem.gatts['sensor']) & (ac.settings['run']['viirs_mask_mband']):
        rhot_waves_m = [int(ds.split('_')[-1]) for ds in rhot_ds if 'M' in ds]
        cidx,cwave = ac.shared.closest_idx(rhot_waves_m, ac.settings['run']['l2w_mask_wave'])
    cur_par = 'rhot_{}'.format(cwave)
    cur_par = [ds for ds in rhot_ds if ('{:.0f}'.format(cwave) in ds)][0]
    if ac.settings['run']['verbosity'] > 3: print('Computing non water threshold mask from {} > {}.'.format(cur_par, ac.settings['run']['l2w_mask_threshold']))
    cur_data = gem.data(cur_par)
    if ac.settings['run']['l2w_mask_smooth']:
        cur_data = ac.shared.fillnan(cur_data)
        cur_data = scipy.ndimage.gaussian_filter(cur_data, ac.settings['run']['l2w_mask_smooth_sigma'], mode='reflect')
    cur_mask = cur_data > ac.settings['run']['l2w_mask_threshold']
    cur_data = None
    flags = cur_mask.astype(np.int32)*(2**ac.settings['run']['flag_exponent_swir'])
    cur_mask = None
    ## end non water/swir threshold
    ####

    ####
    ## cirrus masking
    if ac.settings['run']['verbosity'] > 3: print('Computing cirrus mask.')
    cidx,cwave = ac.shared.closest_idx(rhot_waves, ac.settings['run']['l2w_mask_cirrus_wave'])
    if np.abs(cwave - ac.settings['run']['l2w_mask_cirrus_wave']) < 5:
        cur_par = 'rhot_{}'.format(cwave)
        cur_par = [ds for ds in rhot_ds if ('{:.0f}'.format(cwave) in ds)][0]

        if ac.settings['run']['verbosity'] > 3: print('Computing cirrus mask from {} > {}.'.format(cur_par, ac.settings['run']['l2w_mask_cirrus_threshold']))
        cur_data = gem.data(cur_par)
        if ac.settings['run']['l2w_mask_smooth']:
            cur_data = ac.shared.fillnan(cur_data)
            cur_data = scipy.ndimage.gaussian_filter(cur_data, ac.settings['run']['l2w_mask_smooth_sigma'], mode='reflect')
        cirrus_mask = cur_data > ac.settings['run']['l2w_mask_cirrus_threshold']
        cirrus = None
        flags = (flags) | (cirrus_mask.astype(np.int32)*(2**ac.settings['run']['flag_exponent_cirrus']))
        cirrus_mask = None
    else:
        if ac.settings['run']['verbosity'] > 2: print('No suitable band found for cirrus masking.')
    ## end cirrus masking
    ####

    ####
    ## TOA out of limit
    if ac.settings['run']['verbosity'] > 3: print('Computing TOA limit mask.')
    toa_mask = None
    outmask = None
    # for ci, cur_par in enumerate(rhot_ds):
    def compute_toa_limit(ci, cur_par, rhot_waves, ac, gem, lock):
        if rhot_waves[ci]<ac.settings['run']['l2w_mask_high_toa_wave_range'][0]: return ci,None, None
        if rhot_waves[ci]>ac.settings['run']['l2w_mask_high_toa_wave_range'][1]: return ci,None, None
        if ac.settings['run']['verbosity'] > 3: print('Computing TOA limit mask from {} > {}.'.format(cur_par, ac.settings['run']['l2w_mask_high_toa_threshold']))
        cwave = rhot_waves[ci]
        cur_par = [ds for ds in rhot_ds if ('{:.0f}'.format(cwave) in ds)][0]
        with lock:
            cur_data = gem.data(cur_par)
        outmask = np.isnan(cur_data)
        if ac.settings['run']['l2w_mask_smooth']:
            cur_data = ac.shared.fillnan(cur_data)
            cur_data = scipy.ndimage.gaussian_filter(cur_data, ac.settings['run']['l2w_mask_smooth_sigma'], mode='reflect')
        toa_mask = np.zeros(cur_data.shape).astype(bool)
        toa_mask = (cur_data > ac.settings['run']['l2w_mask_high_toa_threshold'])

        return ci, toa_mask, outmask

    with  concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_to_cur_par = {
            executor.submit(compute_toa_limit, cur_par[0], cur_par[1], rhot_waves, ac, gem, lock) : cur_par for cur_par in enumerate(rhot_ds)
        }
        for future in concurrent.futures.as_completed(future_to_cur_par):
            try:
                ci, toa_mask_partial, outmask_partial = future.result()
                if toa_mask_partial is not None:
                    if toa_mask is None: toa_mask = np.zeros(toa_mask_partial.shape).astype(bool)
                    toa_mask = (toa_mask) | toa_mask_partial
                if outmask_partial is not None:
                    if outmask is None: outmask = np.zeros(outmask_partial.shape).astype(bool)
                    outmask = (outmask) | outmask_partial
                print('compute_toa_limit: Band %r completed' % (ci))
            except Exception as exc:
                print('compute_toa_limit: generated an exception: %s' % (exc))

    flags = (flags) | (toa_mask.astype(np.int32)*(2**ac.settings['run']['flag_exponent_toa']))
    toa_mask = None
    flags = (flags) | (outmask.astype(np.int32)*(2**ac.settings['run']['flag_exponent_outofscene']))
    outmask = None
    ## end TOA out of limit
    ####

    ####
    ## negative rhos
    if ac.settings['run']['verbosity'] > 3: print('Computing negative reflectance mask.')
    neg_mask = None
    for ci, cur_par in enumerate(rhos_ds):
        if rhos_waves[ci]<ac.settings['run']['l2w_mask_negative_wave_range'][0]: continue
        if rhos_waves[ci]>ac.settings['run']['l2w_mask_negative_wave_range'][1]: continue
        if ac.settings['run']['verbosity'] > 3: print('Computing negative reflectance mask from {}.'.format(cur_par))
        cwave = rhos_waves[ci]
        cur_par = [ds for ds in rhos_ds if ('{:.0f}'.format(cwave) in ds)][0]
        cur_data = gem.data(cur_par)
        #if setu['l2w_mask_smooth']: cur_data = scipy.ndimage.gaussian_filter(cur_data, setu['l2w_mask_smooth_sigma'])
        if neg_mask is None: neg_mask = np.zeros(cur_data.shape).astype(bool)
        neg_mask = (neg_mask) | (cur_data < 0)
    flags = (flags) | (neg_mask.astype(np.int32)*(2**ac.settings['run']['flag_exponent_negative']))
    neg_mask = None
    ## end negative rhos
    ####

    ####
    ## mixed pixels mask
    if ('VIIRS' in gem.gatts['sensor']) & (ac.settings['run']['viirs_mask_immixed']):
        if ac.settings['run']['verbosity'] > 3: print('Finding mixed pixels using VIIRS I and M bands.')
        mix_mask = None
        if type(ac.settings['run']['viirs_mask_immixed_bands']) is not list:
            ac.settings['run']['viirs_mask_immixed_bands'] = [ac.settings['run']['viirs_mask_immixed_bands']]
        for imc in ac.settings['run']['viirs_mask_immixed_bands']:
            ib, mb = imc.split('/')
            ds0 = [ds for ds in rhot_ds if ib in ds][0]
            ds1 = [ds for ds in rhot_ds if mb in ds][0]
            cur_data0 = gem.data(ds0)
            cur_data1 = gem.data(ds1)
            if mix_mask is None: mix_mask = np.zeros(cur_data0.shape).astype(bool)
            if ac.settings['run']['viirs_mask_immixed_rat']:
                mix_mask = (mix_mask) | (np.abs(1-(cur_data0/cur_data1)) > ac.settings['run']['viirs_mask_immixed_maxrat'])
            if ac.settings['run']['viirs_mask_immixed_dif']:
                mix_mask = (mix_mask) | (np.abs(cur_data0-cur_data1) > ac.settings['run']['viirs_mask_immixed_maxdif'])
        flags = (flags) | (mix_mask.astype(np.int32)*(2**ac.settings['run']['flag_exponent_mixed']))
        mix_mask = None
    ## end mixed pixels mask
    ####

    ####
    ## dem shadow mask
    if ac.settings['run']['dem_shadow_mask']:
        if ac.settings['run']['verbosity'] > 3:
            print('Computing DEM shadow mask.')
            for k in ac.settings['run']:
                if 'dem_shadow' in k: print(k, ac.settings['run'][k])
        ## add gem version of dem_shadow_mask_nc?
        shade = ac.masking.dem_shadow_mask_nc(gemf)
        flags += shade.astype(np.int32)*(2**ac.settings['run']['flag_exponent_dem_shadow'])
    ## end dem shadow mask
    ####

    ## write flags dataset to netcdf
    if (write_flags_dataset): gem.write(flags_name, flags, ds_att = flags_att)

    ## return flags dataset
    if (return_flags_dataset): return(flags)

    return(gem)
