import ephem
from pyorbital.orbital import Orbital
from datetime import datetime
import time
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd

desktop = "/users/satish/Desktop/"

fig = plt.figure()
ax = fig.gca(projection= '3d' )

lat = pd.read_csv(desktop + "lat.txt")
long = pd.read_csv(desktop + "long.txt")
alt = pd.read_csv(desktop + "alt.txt")


lat = np.ravel(np.asarray(lat))
long = np.ravel(np.asarray(long))
alt = np.ravel(np.asarray(alt))

print(lat.shape)
print(long.shape)
print(alt.shape)

#ax.plot(lat, long, alt, label= 'parametric curve')
#ax.legend()
#plt.show()







TLE_LINE0 = "CALSPHERE 1"
TLE_LINE1 = "1 00900U 64063C   21054.73634469  .00000273  00000-0  28281-3 0  9998"
TLE_LINE2 = "2 00900  90.1597  32.5108 0028780  25.0413  96.3426 13.73487676804948"



tle_rec = ephem.readtle(TLE_LINE0, TLE_LINE1, TLE_LINE2)
tle_rec.compute()

print(tle_rec.sublong, tle_rec.sublat)


# Use current TLEs from the internet:
orb = Orbital("Suomi NPP")

# Get normalized position and velocity of the satellite:


# Get longitude, latitude and altitude of the satellite:
now = datetime(2018, 6, 1)
print("Date: ", now)
orb.get_position(now)
print("Position: ", orb.get_lonlatalt(now) )

exit(1)

while True:
    time.sleep(2)
    now = datetime.datetime(2018, 6, 1)
    print(now)
    orb.get_position(now)
    print( orb.get_lonlatalt(now) )



