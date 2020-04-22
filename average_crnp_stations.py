from datahandling.average_data import average_neutron_data
import pandas as pd
import sys

event_folder = sys.argv[1]
data_folder = sys.argv[2]
operator = sys.argv[3]
if len(sys.argv) == 6:
    rig_min = int(sys.argv[4])
    rig_max = int(sys.argv[5])
elif len(sys.argv) == 7:
    rig_min = int(sys.argv[4])
    rig_max = int(sys.argv[5])
    comment = sys.argv[6]
else:
    rig_min = 0
    rig_max = 18
run_info_folder = event_folder + 'RunInfo/'
result_folder = event_folder + 'AverageResponse/'
excluded_station_file = run_info_folder + 'ExcludedStations.txt'
start_stop_file = run_info_folder + 'Range.txt'

excluded_station_list = pd.read_csv(excluded_station_file).values
start_stop = pd.read_csv(start_stop_file)
start = start_stop['Start'][0]
stop = start_stop['Stop'][0]
start_datetime = pd.to_datetime(start)
stop_datetime = pd.to_datetime(stop)

start_date = start.split(' ')[0]
stop_date = stop.split(' ')[0]


average, contributing_stations = average_neutron_data(data_folder, operator, start_datetime,
                                                         stop_datetime, excluded_stations=excluded_station_list,
                                                         rigidity_range={'min': rig_min, 'max': rig_max})

save_name = result_folder + '%s-%s_%s-%s_%s.csv' % (start_date, stop_date, rig_min, rig_max, comment)
average.to_csv(save_name)

cs_df = pd.DataFrame(contributing_stations)
cs_df.to_csv(run_info_folder + 'contributing_stations_%s-%s_%s-%s_%s..csv' % (start_date, stop_date, rig_min, rig_max,
                                                                              comment))

exit()
