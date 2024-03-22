## def parse
## parses acolite settings files and merges with defaults
## written by Quinten Vanhellemont, RBINS
## 2021-03-09
## modifications: 2022-03-28 (QV) moved int and float lists to external files
##                2023-01-02 (QV) test which is the lowest level of output path that needs to be created
##                2023-10-31 (QV) check if luts_pressures is a list

def parse(sensor, settings=None, merge=True):
    import os, time
    import numpy as np
    import scipy.ndimage
    import acolite as ac

    ## read default settings for sensor
    if (sensor is not None) | (merge):
        setu = ac.acolite.settings.load(sensor)
    else:
        setu = {}

    ## add user settings
    if settings is not None:
        if type(settings) is str:
            sets = ac.acolite.settings.read(settings)
            if merge:
                for k in sets: setu[k] = sets[k]
            else:
                setu = {k:sets[k] for k in sets}
        elif type(settings) is dict:
            if merge:
                for k in settings: setu[k] = settings[k]
            else:
                setu = {k:settings[k] for k in settings}

    ## make sure luts setting is a list
    if 'luts' in setu:
        if type(setu['luts']) is not list: setu['luts'] = [setu['luts']]
    if 'luts_pressures' in setu:
        if type(setu['luts_pressures']) is not list: setu['luts_pressures'] = [setu['luts_pressures']]

    ## import settings that need to be converted to ints and floats
    int_list = ac.acolite.settings.read_list(ac.config['data_dir']+'/ACOLITE/settings_int.txt')
    float_list = ac.acolite.settings.read_list(ac.config['data_dir']+'/ACOLITE/settings_float.txt')

    ## convert values to numbers
    for k in setu:
        if k not in setu: continue
        if setu[k] is None: continue

        if type(setu[k]) is list:
            if k in int_list: setu[k] = [int(i) for i in setu[k]]
            if k in float_list: setu[k] = [float(i) for i in setu[k]]
        else:
            if k in int_list: setu[k] = int(setu[k])
            if k in float_list: setu[k] = float(setu[k])

    ## default pressure
    if 'pressure' in setu:
        setu['pressure'] = float(setu['pressure_default']) if setu['pressure'] is None else float(setu['pressure'])

    ## test which new directory levels will be created
    if 'output' in setu:
        if setu['output'] is not None:
            output = os.path.abspath(setu['output'])
            output_split = output.split(os.path.sep)

            test_path = ''
            new_path = None
            for l in output_split:
                test_path += l+os.path.sep
                if os.path.exists(test_path): continue
                new_path = test_path
                break
            setu['new_path'] = new_path

    return(setu)
