import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from datahandling.import_data import import_neutron_data
from coscal.correct_data import apply_corrections


def average_neutron_data(folder_path, operator, start_date, stop_date, rigidity_range={'min': 0, 'max': 20},
                         original_frequency='3600s', new_frequency='3600s', excluded_stations=[]):
    """

    :param excluded_stations:
    :param folder_path: string containing system path to folder containing data to be averaged. Must also contain a
    meta-data file called 'something_to_pick_soon.txt'
    :param operator: string specifying the operator of the network supplying the data: must be 'COSMOS-UK', 'COSMOS-US'
    or 'NMDB'
    :param start_date: pandas datetime containing the start date of the range of data to be averaged
    :param stop_date: pandas datetime containing the end date of the range of data to be averaged
    :param rigidity_range: list containing floats specifying the lower and upper range of rigidities for data to be
     accepted [lower_limit, upper_limit]
    :param frequency: the time resolution of the averaged data
    :return: averaged data as a pandas dataframe
    """
    metafile_name = 'station_info.txt'
    # check that the folder at the location exists - if not ask user to correct or exit
    folder_path = check_path_exists(folder_path)
    # navigate to the directory
    os.chdir(folder_path)
    other_keys = get_other_keys(operator)
    metafile_name = check_path_exists(metafile_name)
    # read the meta data file
    station_info = pd.read_table(metafile_name, sep=other_keys['meta_sep'])
    # extract the station names from the file
    names = station_info[other_keys['name_column']].values
    rigidity_list = station_info['CutoffRigidity'].values
    # initialise array of data to be averaged
    data_keys = get_data_keys(operator)
    error_keys = get_error_keys(operator)
    contributing_stations = {'name': [], 'rigidity': []}

    length = (stop_date - start_date)/pd.Timedelta(new_frequency) + other_keys['length_mod']
    # loop through every file in the folder
    first_station = True
    for i in tqdm(range(0, len(names))):
        # skip if station is in the excluded list
        if names[i] in excluded_stations:
            continue
        # create the filename
        filename = names[i] + other_keys['extension']
        rigidity = rigidity_list[i]
        # open the file
        if not np.logical_and(rigidity >= rigidity_range['min'], rigidity <= rigidity_range['max']):
            continue
        station_data, valid = import_neutron_data(filename, operator, start_date, stop_date)

        # if the data is valid
        if valid and not station_data.empty:
            # resample the data
            station_data = resample_data(station_data, operator, original_frequency, new_frequency)
            if len(station_data.index) != length:
                continue
            # if the UK is the operator then carry out QC check
            if "COSMOS-UK" == operator:
                station_data = qc_check_data(station_data)
            # correct data - this needs updated!
            station_data = coscal.apply_corrections(station_data, operator, rigidity, data_keys)
            # remove outlying data points
            station_data = handle_outliers_interp(station_data, data_keys, 1, 97)

            contributing_stations['name'].append(names[i])
            contributing_stations['rigidity'].append(rigidity)

            # add data to the array to be averaged - this probably needs a new function too because the difference in
            # moderated and unmoderated counts from different networks
            if first_station:
                all_data = make_data_dict(station_data, data_keys)
                first_station = False
                index = station_data.index
            else:

                for key in data_keys:
                    all_data[key] = np.vstack((all_data[key], station_data[key].values))

    average = average_each_key(all_data, data_keys)
    rel_change = dict_to_rel_change(average, data_keys)
    rel_change['Time'] = index

    final_df = pd.DataFrame(rel_change)
    final_df.set_index('Time', inplace=True)

    return final_df, contributing_stations


def check_path_exists(path):
    """
    quick function to make sure a path exists. if not ask the user to enter a new one
    :param path: file path to check if it exists
    :return: a valid filepath
    """
    ret_path = path
    while True:
        if os.path.exists(path):
            break
        else:
            ret_path = input("Path \"%s\" does not exist. Please enter a new path, or \"exit\" to terminate: " % path)
            if path == 'exit':
                exit(2)

    return ret_path


def get_other_keys(operator):
    """
    function to return a bunch of keys for opening metafiles and the like for a specified operator
    :param operator: valid operators string: one on 'COSMOS-UK' and 'COSMOS-US'
    :return: dictionary of key strings
    """

    ret_dict = {}
    if "COSMOS-US" == operator:
        ret_dict['meta_sep'] = '\t'
        ret_dict['name_column'] = 'SiteName'
        ret_dict['length_mod'] = 0
        ret_dict['extension'] = '.txt'

    elif "COSMOS-UK" == operator:
        ret_dict['meta_sep'] = ','
        ret_dict['name_column'] = 'SITE_ID'
        ret_dict['length_mod'] = 1
        ret_dict['extension'] = '.csv'

    return ret_dict


def get_data_keys(operator):
    """
    function to return keys for data columns for the
    :param operator: one of 'COSMOS-US', 'COSMOS-UK', or 'NMDB'
    :return:
    """
    # get the keys based on the network operator
    if "COSMOS-US" == operator:
        keys = ['MOD', 'UNMO']
    elif "COSMOS-UK" == operator:
        keys = ['CTS_MOD']
    elif "NMDB" == operator:
        keys = ['counts']
    else:
        raise KeyError('%s is not a valid operator' % operator)

    return keys


def resample_data(data, operator, original_frequency, new_frequency):
    """

    :param original_frequency: string containing the original frequency of the data
    :param new_frequency: string specifying the frequency to which the data is to be resampled
    :param data: the data to be resampled
    :param operator: string containing the operator
    :return: dataframe containing data resampled to the frequency specified
    """
    original_frequency_td = pd.Timedelta(original_frequency)
    new_frequency_td = pd.Timedelta(new_frequency)
    # if the new frequency is equal to the old frequency
    if original_frequency_td == new_frequency_td:
        # resample by moving data points to the nearest regular value (will be on the hour)
        resampled_data = data.resample(new_frequency).nearest()

    # otherwise if the new frequency is larger than the old data
    elif new_frequency > original_frequency:
        # check if it is a multiple of the old data
        if is_multiple(new_frequency_td, original_frequency_td):
            # if so, pass the operator string to a function to create a dictionary containing resampling instructions
            resampler_dict = get_resampler_dict(operator)
            # resample the data
            resampled_data = data.resample(new_frequency).agg(resampler_dict)
        # otherwise raise an error
        else:
            raise ValueError('New frequency must be a multiple of the original frequency')
    # otherwise raise an error
    else:
        raise ValueError('New frequency must be = n * original frequency for n an int > 0')

    # return the resampled data
    return resampled_data


def is_multiple(large_td, small_td):
    """
    function to find if a large timedelta is a multiple of a smaller timedelta
    :param large_td: the larger timedelta
    :param small_td: the smaller timedelta
    :return:
    """
    return large_td % small_td == pd.Timedelta('0T')


def get_resampler_dict(operator):
    """
    function to return a dictionary containing resample methods for different pandas dataframe keys given an operator
    assume that the resampling is for a downsampling to a frequency which is a multiple of the oringial frequency
    probably shouldn't use this for long times
    :param operator: string specifying the operator
    :return:
    """
    if operator == 'NMDB':
        ret_dict = {'counts': 'sum'}
    elif operator == 'COSMOS-US':
        ret_dict = {'MOD': 'sum',
                    'UNMO': 'sum',
                    'PRESS': 'mean',
                    'TEM': 'mean',
                    'RH': 'mean',
                    'BATT': 'mean'
                    }

    elif operator == 'COSMOS-UK':
        # this drops a LOT of data - can be edited to pick up more later
        ret_dict = {'CTS_MOD': 'sum',
                    'CTS_MOD2': 'sum',
                    'CTS_BARE': 'sum',
                    'PA': 'mean',
                    'Q': 'mean',
                    'CTS_MOD_QCFLAG': 'sum',
                    'CTS_MOD2_QCFLAG': 'sum',
                    'CTS_BARE_QCFLAG': 'sum',
                    'PA_QCFLAG': 'sum',
                    'Q_QCFLAG': 'sum'
                    }

    return ret_dict


def get_error_keys(operator):
    """
    function to return a dictionary containing error keys for an operator
    :param operator: string specifying the operator
    :return:
    """
    # TODO consider merging with get_data_keys
    if "COSMOS-US" == operator:
        keys = ['E_MOD', 'E_UNMO']
    elif "COSMOS-UK" == operator:
        keys = ['E_CTS_MOD']

    else:
        raise KeyError('%s is not a valid operator' % operator)

    return keys


def qc_check_data(data):
    """
    function to accept data and check QC flags on it. This only works if the operator is comsos-uk.
    If a QC flag is non-zero then the corresponding value will be set to nan.
    :param data: data frame holding data and QC flags
    :return:
    """
    qc_dict = get_qc_dict()

    for key in qc_dict:
        data[key] = data.mask(data[qc_dict[key]] > 0)[key]

    return data


def get_qc_dict():
    """
    function to get a dicitonary containing the uk cosmos qc flags and the corresponding data columm
    :return:
    """

    qc_dict = {
        'CTS_MOD': 'CTS_MOD_QCFLAG',
        'CTS_MOD2': 'CTS_MOD2_QCFLAG',
        'CTS_BARE': 'CTS_BARE_QCFLAG',
        'PA': 'PA_QCFLAG',
        'Q': 'Q_QCFLAG'
    }
    return qc_dict


def handle_outliers_interp(data, keys, min_percentile, max_percentile):
    """
    function to accept a dataframe, key list, maximum and minimum percentile values. Removes values outwith the
    range [max_percentile, min_percentile] and replaces them with linearly interpolated values.
    :param data: pandas dataframe
    :param keys: list containing keys for the columns to have the operation carried out
    :param min_percentile: minimum percentile. Values that are below this percentile will be removed and replaced by
    interpolated values
    :param max_percentile: maximum percentile. Values that are above this percentile will be removed and replaced by
    interpolated values
    :return: dataframe with outliers removed and replaced by interpolated values
    """

    for key in keys:
        # convert the outliers to NaN values
        data[key] = outliers_to_nans(data[key].values, min_percentile, max_percentile)
        data[key] = interp_nans(data[key].values)

    return data


def interp_nans(data):
    """
    function to find nans in a data array and replace with interpolated values.
    NOTE: for NaNs at the beginning or end of an array it will replace these with the first/last valid point
    :param data: array of data to have nans interpolated over
    :return: array of data with nans replaced by interpolated values
    """
    # define anonymous function to find nans
    is_nan = lambda a: np.where(np.isnan(a))
    # define another to find where the values are not nans
    is_not_nan = lambda a: np.where(~np.isnan(a))
    # make an array of indices (consider to be x-values, and data to be y-values)
    x_array = np.arange(0, len(data))

    # bit of black magic going on here. The first arg of interp is the x-location of the values which are to be
    # calculated by interpolation. The second is the x-location of the data that is to be used to perform the
    # interpolation and the third is the values of that data. Assign the result to the location of nan values in the
    # data
    data[is_nan(data)] = np.interp(x_array[is_nan(data)], x_array[is_not_nan(data)], data[is_not_nan(data)])

    return data


def outliers_to_nans(data, min_percentile, max_percentile):
    # function to replace data outside specified percentile range with linearly interpolated values

    # calculate the corresponding data values for each percentile
    min_value = np.percentile(data, min_percentile)
    max_value = np.percentile(data, max_percentile)

    # set values outwith the acceptable range to be NaN
    data[np.logical_or(np.greater(data, max_value),np.less(data, min_value))] = np.nan

    return data


def make_data_dict(data, keys):
    """
    function to make a dictionary containing empty arrays for data to be averaged, with different
    :param data:
    :param keys: list of strings containing data keys
    :return: dictionary containing initialised arrays
    """

    # initialise the dictionary
    data_dict = {}

    # loop through the keys adding an empty array to the dictionary for each
    for key in keys:
        data_dict[key] = data[key].values

    return data_dict


def calculate_poisson_percentage(data_dict, error_keys, data_keys):
    """

    :param data_dict:
    :param error_keys:
    :param data_keys:
    :return:
    """
    error_dict = {}
    for i in range(len(error_keys)):
        summed_data = np.nansum(data_dict[data_keys[i]], axis=0)
        error_dict[error_keys[i]] = np.sqrt(summed_data)/summed_data * 100

    return error_dict


def average_each_key(data_dict, keys):
    """
    function to average the data in a dict, key by key
    :param data_dict: dict containing data 
    :param keys: data keys 
    :return: dict containing the averaged data
    """
    average = {}
    for key in keys:
        # sum up data
        summed = np.nansum(data_dict[key], axis=0)
        # find the number of non-nan contributions to each point
        num_not_nan = np.count_nonzero(~np.isnan(data_dict[key]), axis=0)
        # calculate percentage poisson error
        average['E_' + key] = np.multiply(np.divide(np.sqrt(summed), summed), 100)
        average[key] = np.divide(summed, num_not_nan)

    return average


def dict_to_rel_change(data_dict, keys):
    """
    convert data in a dict to relative change from counts
    :param data_dict: dictionary containing the data
    :param keys: data keys
    :return:
    """
    rel_change = {}
    for key in keys:
        rel_change[key] = convert_to_rel_change(data_dict[key])
        rel_change[key + '_std'] = normalise_std(data_dict[key + '_std'], data_dict[key])
    return rel_change


def convert_to_rel_change(data):
    # one day I'll introduce docstrings and validation checks. One day.

    # find the data mean
    data_mean = np.nanmean(data[0:48])

    # find the difference from the mean at each data point
    change = data - data_mean

    # express this as a percentage of the mean
    rel_change = (change / data_mean) * 100

    # return the new data expressed as a relative change from the mean
    return rel_change


def normalise_std(std_array, data_array):

    normal_std = std_array/data_array * 100

    return normal_std
