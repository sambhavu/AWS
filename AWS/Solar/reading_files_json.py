import json
import urllib
from urllib.request import urlopen
import matplotlib.pyplot as plt


"""
Solar Cycle Datasets SWPC 
"""

f10_flux_smoothed_json = 'https://services.swpc.noaa.gov/json/solar-cycle/f10-7cm-flux-smoothed.json'
f10_flux_json = 'https://services.swpc.noaa.gov/json/solar-cycle/f10-7cm-flux.json'
observed_solar_cycle_json = 'https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json'
predicted_solar_cycle_json = 'https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json'
f10_flux_prediction_json = 'https://services.swpc.noaa.gov/json/solar-cycle/solar-cycle-25-f10-7cm-flux-predicted-high-low.json'
predicted_solar_cycle_25_json = 'https://services.swpc.noaa.gov/json/solar-cycle/solar-cycle-25-predicted.json'
predicted_solar_cycle_25_high_low_json = 'https://services.swpc.noaa.gov/json/solar-cycle/solar-cycle-25-ssn-predicted-high-low.json'
sunspots_smoothed_json = 'https://services.swpc.noaa.gov/json/solar-cycle/sunspots-smoothed.json'
sunspots_json = 'https://services.swpc.noaa.gov/json/solar-cycle/sunspots.json'
swpc_observed_ssn_json = 'https://services.swpc.noaa.gov/json/solar-cycle/swpc_observed_ssn.json'




def get_json_data(json_url):
    output = json.loads(urlopen(json_url).read())
    return output

data = get_json_data(json_url)


"""

time_tag = []
smoothed_ssn = []
f10 = []

for i in range(len(data)):
    time_tag.append(data[i]['time-tag'])

for i in range(len(data)):
    smoothed_ssn.append(data[i]['smoothed_ssn_minus_6mo'])

for i in range(len(data)):
    f10.append(data[i]['smoothed_ssn_minus10_minus6mo'])



plt.plot(smoothed_ssn)
plt.plot(f10)

plt.show()


#"time_tag"
#"smoothed_ssn"
#"f10.7"
"""
