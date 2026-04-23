## def acolite_luts
## function to retrieve & test required LUTs, e.g. to set up Docker
## written by Quinten Vanhellemont, RBINS
## 2021-07-26
## modifications:
##                2021-10-24 (QV) added LUT identifiers and pressures, get_remote keyword
##                2022-04-12 (QV) add par parameter to import_luts
##                2025-02-11 (QV) add sensor settings parsing and reverse_lut_sensors list
##                2025-02-15 (QV) fix for rsr_version

def acolite_luts(sensor = None, hyper = False,
                 get_remote = True, compute_reverse = True,
                 pressures = [500, 750, 1013, 1100],
                 base_luts = ['ACOLITE-LUT-202110-MOD1', 'ACOLITE-LUT-202110-MOD2'],
                 rsky_lut = 'ACOLITE-RSKY-202102-82W',
                 pars = ['romix', 'romix+rsky_t']):
    import acolite as ac

    ## get all RSR if no sensor(s) specified
    if type(sensor) is str:
        if sensor.upper() == 'NONE': sensor = None

    if (sensor is None):
        rsrd = ac.shared.rsr_dict()
    else:
        if type(sensor) is not list:
            if ',' in sensor:
                sensor = [s.strip() for s in sensor.split(',')]
            elif ' ' in sensor:
                sensor = [s for s in sensor.split(' ') if len(s) > 0]
            else:
                sensor = [sensor]
        rsrd = {}
        for s in sensor:
            if (s.upper() in ['CHRIS', 'PRISMA', 'HYPER']) or ('DESIS' in s.upper()):
                hyper = True
                continue

            ## get sensor settings
            setd = ac.acolite.settings.parse(s)
            lut_sensor = '{}'.format(s)
            if 'rsr_version' in setd:
                if setd['rsr_version'] is not None:
                    lut_sensor = '{}_{}'.format(s, setd['rsr_version'])

            rd = ac.shared.rsr_dict(sensor=lut_sensor)
            if len(rd) == 0:
                print('Sensor {} ({}) not recognised.'.format(s, lut_sensor))
                continue
            rsrd[lut_sensor] = rd[lut_sensor]

    sensors = list(rsrd.keys())
    sensors.sort()

    ## Add "None" to sensors to retrieve generic LUT
    if hyper: sensors += [None]

    ## prefetch all required LUT files in parallel before the per-sensor loop.
    ## the existing import path below will then find every file in cache and skip
    ## the network. URL/local-path conventions live in the per-LUT helpers so the
    ## sequential code path remains the source of truth.
    if get_remote:
        from acolite.aerlut.import_lut import _remote_paths_lut
        from acolite.aerlut.import_rsky_lut import _remote_paths_rsky
        from acolite.aerlut.reverse_lut import _remote_paths_reverse

        prefetch_jobs = []
        skip_sensors = {'L5_TM_B6', 'L7_ETM_B6', 'L8_TIRS', 'L9_TIRS',
                        'EO1_ALI_ORANGE', 'L8_OLI_ORANGE', 'L9_OLI_ORANGE'}

        def _is_skipped(name):
            if name is None: return False
            if name in skip_sensors: return True
            if '_CONTRA' in name: return True
            if 'DESIS' in name: return True
            return False

        for s in sensors:
            if _is_skipped(s): continue

            ## sensor LUTs across base x pressure
            for base_lut in base_luts:
                for pr in pressures:
                    lutid = '{}-{}mb'.format(base_lut, '{}'.format(pr).zfill(4))
                    lutdir = '{}/{}'.format(ac.config['lut_dir'], '-'.join(lutid.split('-')[0:3]))
                    url, path = _remote_paths_lut(lutid, lutdir, sensor = s)
                    prefetch_jobs.append((url, path))

            ## RSKY LUTs (MOD1 + MOD2)
            for model in (1, 2):
                url, path = _remote_paths_rsky(model, lutbase = rsky_lut, sensor = s)
                prefetch_jobs.append((url, path))

            ## reverse LUTs - only when computed and supported
            if compute_reverse and (s is not None) and (s in ac.config['reverse_lut_sensors']):
                rsr_file = ac.config['data_dir'] + '/RSR/{}.txt'.format(s)
                try:
                    _, rsr_bands = ac.shared.rsr_read(rsr_file)
                except Exception:
                    rsr_bands = []
                for base_lut in base_luts:
                    for par in pars:
                        for b in rsr_bands:
                            url, path = _remote_paths_reverse(s, base_lut, par, b)
                            prefetch_jobs.append((url, path))

        if prefetch_jobs:
            print('Prefetching up to {} LUT file(s) in parallel'.format(len(prefetch_jobs)))
            ac.shared.download_files(prefetch_jobs, verbosity = 1)

    ## run through wanted sensors
    for s in sensors:
        if s is not None:
            if s in ['L5_TM_B6', 'L7_ETM_B6', 'L8_TIRS', 'L9_TIRS']: continue ## skip thermals
            if s in ['EO1_ALI_ORANGE', 'L8_OLI_ORANGE', 'L9_OLI_ORANGE']: continue ## skip contrabands
            if '_CONTRA' in s: continue ## skip contrabands
            if 'DESIS' in s: continue ## skip DESIS

        print('Testing {}'.format('sensor {}'.format(s) if s is not None else 'generic LUT'))

        ## try getting gas transmittance
        tg_dict = ac.ac.gas_transmittance(0, 0, uoz=0.3, uwv=1.6, rsr=None if s is None else rsrd[s]['rsr'])

        ## get sensor LUT
        tmp = ac.aerlut.import_luts(sensor = s, get_remote = get_remote, pressures = pressures, par=pars[-1],
                                    base_luts = base_luts, rsky_lut = rsky_lut)

        ## get reverse LUT
        if (compute_reverse) & (s is not None) & (s in ac.config['reverse_lut_sensors']):
            for par in pars:
                revl = ac.aerlut.reverse_lut(s, get_remote = get_remote, par=par, pressures = pressures,
                                            base_luts = base_luts, rsky_lut = rsky_lut)
