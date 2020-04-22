import pandas as pd


def import_neutron_data(filename, operator, start=None, stop=None):
    """
    function to import COSMOS data depending on the operator, within a specified time
    :param filename: string specifying the path to the file
    :param operator: string specifying the operator of the network - either UK or US
    :param start: string specifying the start date of the data to be imported
    :param stop: string specifying the end date of the data to be imported
    :return data: cosmos data as a pandas dataframe
    :return validity: boolean variable, true if data is valid - false otherwise
    """

    # set the data frame keys depending on the operator of the network
    import_dict = set_keys_and_parser(operator)

    # read the data
    data = pd.read_table(filename, sep=import_dict['separator'], parse_dates=import_dict['ind'])
    # sort the values by data, necessary for some of the UK data which is a little jumbled
    data.sort_values(by=[import_dict['date_key']], inplace=True)
    # check the data frame isn't empty. This is a common error in the US network - yet to establish why
    if data.empty:
        return data, False
    # set the datetime column to be the index
    data.set_index(import_dict['date_key'], inplace=True)

    # validate the data for the specified time range if one has been specified
    if start is not None:
        data, validity = slice_data_for_dates(data, start, stop)
        return data, validity

    return data, True


def set_keys_and_parser(operator):
    """
    function to return the keys used to access data frames and the indices to parse dates
    :param operator: string, MUST be 'UK' or 'US'
    :return: dictionary containing the date key, list of date indexes and separator string
    """
    ret_dict = {}

    if operator == 'COSMOS-UK':

        ret_dict['date_key'] = 'DATE_TIME'
        ret_dict['ind'] = [2]
        ret_dict['separator'] = ','

    elif operator == 'COSMOS-US':

        ret_dict['date_key'] = 'Date_Time'
        ret_dict['ind'] = [[0, 1]]
        ret_dict['separator'] = '\s+'

    elif operator == 'NMDB':

        ret_dict['date_key'] = 'DateTime'
        ret_dict['ind'] = [0]
        ret_dict['separator'] = ';'

    elif operator == 'COSMOS-Av':
        ret_dict['date_key'] = 'Time'
        ret_dict['ind'] = [0]
        ret_dict['separator'] = ','

    else:
        raise KeyError('%s is an invalid operator. Operator keys must be one of \"COSMOS-UK\", \"COSMOS-US\", '
                       '\"NMDB\", or \"COSMOS-Av\"')

    return ret_dict


def slice_data_for_dates(data, start, stop):
    """
    function to accept data and a date range then check the data exists within that range, return the data within that
     range
    :param data: pandas dataframe with a datetime axis to be sliced for dates
    :param start: start date as a string
    :param stop: stop date as a string
    :return: pandas dataframe containing the data within the date range
    """

    # convert strings to datetime variables
    start_date = pd.to_datetime(start)
    stop_date = pd.to_datetime(stop)

    # run the validity check
    if date_valid(data, start, stop):
        # on success return the sliced data and the validity
        sliced_data = data[start_date:stop_date]
        if sliced_data.empty:
            return data, False
        else:
            return sliced_data, True
    else:
        # on failure return the original data and validity
        return data, False


def date_valid(data, start_date, stop_date):
    """
    function to check that the dates passed are within the range of the dataframe
    :param data: pandas dataframe with a datetime axis to be sliced for dates
    :param start_date: start date as a pandas datetime
    :param stop_date: stop date as a pandas datetime
    :return: True if dates are both in range, or false otherwise
    """
    # TODO change to np.logical_and or just generally find a better way of doing this
    if data.index[0] <= start_date and data.index[-1] >= stop_date:
        return True
    else:
        return False


def import_soho_data(filename):
    """
    function to import data from the soho satellite
    :param filename: string specifying the location of the file
    :return:
    """
    # read in the table using pandas
    data = pd.read_table(filename, sep='\s+')
    # convert the numbers in the year column into a string
    data['YY'] = data['YY'].apply(str)
    # convert the numbers in the day of year etc column into a string
    data['DOY:HH:MM:SS'] = data['DOY:HH:MM:SS'].apply(str)
    # concatenate these two columns with a space inbetween
    data['YY'] += ' ' + data['DOY:HH:MM:SS']
    # convert the column into a datetime column
    data['YY'] = pd.to_datetime(data['YY'], format='%y %j:%H:%M:%S')
    # make the datetime column the index
    data.set_index('YY', inplace=True)
    # get rid of now defunct datetime specifiers
    del data['MON'], data['DY'], data['DOY:HH:MM:SS']

    return data
