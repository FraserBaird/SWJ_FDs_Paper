import numpy as np

def apply_corrections(data, operator, station_rigidity, data_keys):
    """
    main function for the corrections module. Takes data and an operator string, finds what corrections to apply and
    applies them
    :param station_rigidity: cutoff rigidity if GV of station taking measurements
    :param data: dataframe containing data to be corrected and correction data
    :param operator: string specifying the network operator
    :return: dataframe containing corrected data
    """
    # get the keys of data to correct
    correction_key_dict = set_corr_keys(operator)
    # get the correction factors
    correction_factors = get_corr_factors(data, correction_key_dict, station_rigidity)
    # correct the data
    for key in data_keys:
        data[key] = data[key] * correction_factors['p_corr'] * correction_factors['h_corr']
    return data


def set_corr_keys(operator):
    """
    function to accept an operator string and return the dataframe keys in a dictionary
    :param operator: string specifying the operator, must be 'COSMOS-UK' or 'COSMOS-US'
    :return: dictionary containing the pressure and humidity correction keys

    """
    if "COSMOS-US" == operator:
        key_dict = {'p_corr': 'PRESS', 'h_corr': None}
    elif "COSMOS-UK" == operator:
        key_dict = {'p_corr': 'PA', 'h_corr': 'Q'}
    else:
        raise KeyError('%s is not a valid operator key' % operator)

    return key_dict


def get_corr_factors(data, key_dict, station_rigidity):
    """
    takes some data and an operator and returns correction factors
    :param station_rigidity: cutoff rigidity of station taking measurements (GV)
    :param data: dataframe containing data and correction data
    :param key_dict: dictionary containing correction keys
    :return: dictionary containing the correction factors
    """
    if key_dict['p_corr'] is not None:
        p_corr = pressure_correction(data[key_dict['p_corr']], station_rigidity)
    else:
        p_corr = np.full(len(data), 1)

    if key_dict['h_corr'] is not None:
        h_corr = humidity_correction(data[key_dict['h_corr']])
    else:
        h_corr = np.full(len(data), 1)

    ret_dict = {'p_corr': p_corr, 'h_corr': h_corr}

    return ret_dict


def pressure_correction(pressure, rigidity):
    """
    function to get pressure correction factors, given a pressure time series and rigidity value for the station
    :param pressure: time series of pressure values over the time of the data observations
    :param rigidity: cut-off rigidity of the station making the observations
    :return: series of correction factors
    """
    p_0 = np.nanmean(pressure)

    pressure_diff = pressure - p_0
    # g cm^-2. See Desilets & Zreda 2003
    mass_attenuation_length = attenuation_length(p_0, rigidity)

    exponent = pressure_diff * mass_attenuation_length

    pressure_corr = np.exp(exponent)

    return pressure_corr


def humidity_correction(humidity):
    """

    :param humidity:
    :return:
    """

    mean_hum = np.nanmean(humidity)
    hum_change = humidity - mean_hum

    hum_corr = 1 + (0.0054 * hum_change)

    return hum_corr


def attenuation_length(pressure, rigidity):
    """

    :param pressure:
    :param rigidity:
    :return:
    """
    # constants from Desilets & Zreda 2003, Earth & Planetary Science Letters 206
    c_0 = 5.4196 * (10 ** (-3))
    c_1 = 2.2082 * (10 ** (-4))
    c_2 = -5.1952 * (10 ** (-7))
    c_3 = 7.2062 * (10 ** (-6))
    c_4 = -1.9702 * (10 ** (-6))
    c_5 = -9.8334 * (10 ** (-9))
    c_6 = 3.4201 * (10 ** (-9))
    c_7 = 4.9898 * (10 ** (-12))
    c_8 = -1.7192 * (10 ** (-12))
    depth = pressure * 0.981
    beta = c_0 \
           + c_1 * rigidity \
           + c_2 * rigidity ** 2 \
           + (c_3 + c_4 * rigidity) * depth \
           + (c_5 + (c_6 * rigidity)) * depth ** 2 \
           + (c_7 + c_8 * rigidity) * (depth ** 3)

    return beta
