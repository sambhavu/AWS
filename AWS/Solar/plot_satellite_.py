import ephem
from pyorbital.orbital import Orbital
from datetime import datetime, timedelta
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import numpy as np
import skyfield
from skyfield.api import EarthSatellite, load
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv


AU = 149597871


desktop = "/users/satish/Desktop/"

base = datetime(2019, 1, 1)
times2 = np.array([base + timedelta(hours=i) for i in range(24)])
times = np.arange(datetime(2016,1,1), datetime(2021,1,1), timedelta(days=1)).astype(datetime)

#fig = plt.figure(figsize=(15,15))
fig = plt.figure()
ax = fig.gca(projection = '3d' )


class Solar():
    def __init__(self):
        pass

    def plot_Show_2d(self):
        plt.legend()
        plt.show()

    def plot_Show_3d(self):
        ax.legend()
        plt.show()

    def plot_satellite_2d(self, lat, long, alt, satellite_name):
        plt.plot(lat, label = 'Latitude')
        plt.plot(long, label = 'Longitude')
        plt.plot(alt, label = 'Altitude')


    def plot_satellite_3d(self, lat, long, alt, satellite_name):

        ax.plot( long, lat, alt, label = satellite_name, lw = .2 )


    def get_satellite_position_skyfield(self, time, orb):
        orb.get_position(time)
        return orb.get_lonlatalt(time)

    def get_satellite_position_sgp4(self, sat, time):
        position, velocity = sat.propagate(time.year, time.month, time.day, time.minute)

        lat = position[0]
        long = position[1]
        alt = position[2]

        return lat, long, alt



    def plot_TLE_data_Skyfield(self, name, line1, line2):
            latitude = []
            longitude = []
            altitude = []

            """
            #skyfield package
            ts = load.timescale()
            t = ts.utc(2001, 5, 21, 18, 22, 50)
            satellite = EarthSatellite(line1, line2)
            # Geocentric
            geometry = satellite.at(t)
            # Geographic point beneath satellite
            subpoint = geometry.subpoint()
            latitude_ = subpoint.latitude.degrees
            longitude_ = subpoint.longitude.degrees
            elevation_ = float(subpoint.elevation)
            print(latitude_)
            print(longitude_)
            print(elevation_)
            exit(1)
            """

            #EPHEM Package
            #tle_rec = ephem.readtle(name, line1, line2);
            #tle_rec.compute(times[0]);
            #print(tle_rec.sublong, tle_rec.sublat)


            try:
                orb = Orbital(name, tle_file=None, line1=line1, line2=line2)
                for time in times:

                    data = self.get_satellite_position_skyfield(time, orb)

                    lat = data[0]
                    long = data[1]
                    alt = data[2]

                    latitude.append(lat)
                    longitude.append(long)
                    altitude.append(alt)

                    print("Name: ", name, "Time: ", time, "Lat: ", lat, "Long: ", long, "Alt: ", alt )

                self.plot_satellite_2d(latitude, longitude, altitude, name)

            except Exception as e:
                print("Failed Skyfield...")
                print(e)





    def plot_TLE_data_SGP4(self, name, line1, line2):
        latitude = []
        longitude = []
        altitude = []

        try:
            #sg4 package
            satellite = twoline2rv(line1, line2, wgs72)
            for time in times:

                lat, long, alt = self.get_satellite_position_sgp4(satellite, time)

                latitude.append(lat)
                longitude.append(long)
                altitude.append(alt)

                print("Name: ", name, "Time: ", time, "Lat: ", lat, "Long: ", long, "Alt: ", alt )
            self.plot_satellite_3d(latitude, longitude, altitude, name)


        except Exception as e:
            print("Failed SGP4...")
            print(e)

def main():


    #Example TLE DATA
    TLE_LINE0 = "CALSPHERE 1"


    TLE_LINE1 = "1 43013U 17073A   21065.76936730  .00000007  00000-0  24183-4 0  9993"
    TLE_LINE2 = "2 43013  98.7452   5.9775 0000941  71.3410 288.7868 14.19552110170870"

    Galaxy_15 = "GALAXY 15 (G-15)"
    Galaxy_15_Line1 = "1 27820U 03024A   21065.61756845 -.00000237  00000-0  00000-0 0  9997"
    Galaxy_15_line2 = "2 27820   3.0978  85.0877 0019064 290.4677 180.5695  0.99033950 64897"


    Galaxyafda_15 = "GALAXY 15 (G-15)"
    Galaxydaf_15_Line1 = "1 27820U 03024A   21065.61756845 -.00000237  00000-0  00000-0 0  9997"
    Galaxyadsf_15_line2 = "2 27820   3.0978  85.0877 0019064 290.4677 180.5695  0.99033950 64897"

    Galaxy_15 = "GALAXY 15 (G-15)"
    Galaxy_15_Line1 = "1 28884U 05041A   21065.54809044  .00000063  00000-0  00000-0 0  9997"
    Galaxy_15_Line2 = "2 28884   0.0519 273.8575 0001781  89.8859 225.0308  1.00272216 56303"

    Syracuse = "SYRACUSE 3A"
    Syracuse_Line1 = "1 28885U 05041B   21065.78253187  .00000113  00000-0  00000+0 0  9990"
    Syracuse_Line2 = "2 28885   0.0133 146.0524 0002905 209.0068 138.3470  1.00270148 56395"

    Ariane = "ARIANE 5 R/B"
    Ariane_Line1 = "1 28886U 05041C   21065.56347196  .00000022  00000-0  66626-3 0  9990"
    Ariane_Line2 = "2 28886   7.5782 211.3285 7181829  22.1583 107.1628  2.24338452126155"


    Ariane_DEB = "ARIANE 5 DEB [SYLDA]"
    Ariane_DEB_Line1 = "1 28887U 05041D   21065.38435364 -.00000084  00000-0 -82870-3 0  9995"
    Ariane_DEB_Line2 = "2 28887   6.7481 178.9474 7179752  93.3169 343.2828  2.26511697127188"


    AMC = "AMC-9 (GE-12)"
    AMC_Line1 = "1 27820U 03024A   21066.45216240 -.00000257  00000-0  00000-0 0  9992"
    AMC_Line2 = "2 27820   3.0999  85.0797 0019054 290.6628 117.9389  0.99034081 64900"



    SPACEWAY = "SPACEWAY 2"
    SPACEWAY_Line1 = "1 28903U 05046B   21066.52507348  .00000085  00000-0  00000-0 0  9999"
    SPACEWAY_Line2 = "2 28903   0.8102  92.0138 0000440 334.4638 149.0748  1.00272508 32011"


    g = "GSAT0221 (GALILEO 25)"
    gline1 = "1 43564U 18060A   21065.96020128 -.00000068  00000-0  00000-0 0  9994"
    gline2 = "2 43564  56.9660  33.8748 0005803 287.7672  72.1543  1.70475566 16295"

    iss = "ISS (ZARYA)"
    issline1 = "1 25544U 98067A   21066.62808894  .00001399  00000-0  33502-4 0  9999"
    issline2 = "2 25544  51.6449 125.0413 0003365  80.3323  78.7298 15.49048400272853"

    mms = "MMS 1"
    mmsline1 = "1 40482U 15011A   21057.39583333 -.00001653  00000-0  00000+0 0  9993"
    mms_line2 = "2 40482  32.6102 118.2910 8454071  29.2447 281.5793  0.28559530 11940"



    sat = Solar()

    sat.plot_TLE_data_SGP4(Galaxy_15, Galaxy_15_Line1, Galaxy_15_Line2)
    #sat.plot_TLE_data_SGP4(Syracuse, Syracuse_Line1, Syracuse_Line2)
    sat.plot_TLE_data_SGP4(AMC, AMC_Line1, AMC_Line2)

    #sat.plot_TLE_data_SGP4(g, gline1, gline2)
    #sat.plot_TLE_data_SGP4(iss, issline1, issline2)
    #sat.plot_TLE_data_SGP4(mms, mmsline1, mms_line2)

    sat.plot_Show_3d()


 #   sat.plot_TLE_data_Skyfield(iss, issline1, issline2)
  #  sat.plot_Show_2d()



main()


"""

https://space.stackexchange.com/questions/25958/how-can-i-plot-a-satellites-orbit-in-3d-from-a-tle-using-python-and-skyfield
https://pyorbital.readthedocs.io/en/latest/
https://github.com/Elucidation/OrbitalElements/blob/master/orbit.py
https://www.studytonight.com/matplotlib/matplotlib-3d-plotting-line-and-scatter-plot
https://www.space-track.org/auth/login

TLE Datasets:
https://celestrak.com/NORAD/archives/

Workshop:
https://www.swpc.noaa.gov/news/virtual-2021-space-weather-workshop-april-20-22-2021

"""
