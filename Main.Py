import numpy as np
import matplotlib.pyplot as plt

(import os 
import math
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from numpy import random
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from prettytable import PrettyTable

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score,roc_curve,auc, f1_score, roc_auc_score,confusion_matrix, accuracy_score, classification_report

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from pylab import rcParams
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier)



z, t, flux = np.loadtxt("Data/TIC-KIC.dat",unpack=True,skiprows=)
plt.figure(figsize=(4,3))
plt.plot(t,flux,'k.',markersize=2)
plt.xlabel("time (days)")
plt.ylabel("Flux")
plt.tight_layout()
ndata=200
t2=np.sort(5.*np.random.rand(ndata))
y=np.zeros(ndata,dtype=float)
gradient=-20.
for i in range (0,ndata):
    if (t2[i]%1>=0.4) and (t2[i]%1 <=0.45):
        y[i]=0.+gradient*(t2[i]%1-0.4)
    elif (t2[i]%1>=0.55) and (t2[i]%1 <=0.60):
        y[i]=0.+gradient*(0.6-t2[i]%1)
    elif (t2[i]%1>0.45) and (t2[i]%1 <=0.55):
        y[i]=-1.
    else:
        y[i]=0.
plt.figure(figsize=(4,4))
plt.plot(t2,y,'b.')
plt.figure(figsize=(4,4))
plt.plot(t2,y,'b.')
plt.plot(t2,y,'r:')
import numpy as np
import matplotlib.pyplot as plt

# Assuming t2 and y are already defined, and P is the period

P = 1.0
tPhaseFolded = t2 % P  # Phase folding the time

plt.figure(figsize=(4, 4))
plt.plot(tPhaseFolded, y, 'b.')
plt.xlabel(r"$t\:\mathrm{mod}\: P$")  # Use raw string to avoid the invalid escape sequence
plt.ylabel("$y$")
plt.tight_layout()
plt.show()

for i in [0.015,0.03,0.0,-0.015,-0.03]:
    tPhaseFolded=t2%(P+i)

    plt.figure(figsize=(4,4))
    plt.plot(tPhaseFolded,y,'b.')
    #plt.xlabel("$t\:\mathrm{mod}\: P$"+"("+str(i)+")")
    xstring="t mod (P+"+str(i)+")"
    plt.xlabel(xstring)
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.show()


tPhaseFolded=(t2%P)-P/2. # subtract or add to X to move your function along the x axis
plt.figure(figsize=(4,4))
plt.plot(tPhaseFolded,y,'b.')
plt.xlabel(r"$(t\:\mathrm{mod}\: P) - P/2.$")
plt.ylabel("$y$")
plt.tight_layout()
plt.show()

t_pf, flux2 = np.loadtxt("Data/KIC-TIC_phaseFold.dat",unpack=True)
plt.figure(figsize=(4,4))
plt.plot(t_pf,flux2,'b.',markersize=2)
plt.xlabel("(time % Period) [days]")
plt.ylabel("$L$")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with actual data)
t = np.linspace(0, 20, 1000)  # Time in days
flux = 1 - 0.01 * (np.abs(np.sin(2 * np.pi * t / 5)) < 0.1)  # Simple model of a transit every 5 days

# Plot the light curve
plt.figure(figsize=(10, 4))
plt.plot(t, flux, 'k.')
plt.xlabel("Time (days)")
plt.ylabel("Flux")
plt.title("Light Curve with Transits")
plt.xlim(0, 20)  # Zoom in on the relevant part of the data
plt.show()


# Given estimates
t0 = 2.5  # Midpoint of the first transit
P = 5.0   # Orbital period

# Phase-fold the data
phase = ((t - t0) % P) / P  # Phase is between 0 and 1
phase = phase - (phase > 0.5)  # Shift phase to range -0.5 to 0.5



("""Instead of phase fold, If the data is also heavily imbalanced. To deal with this issue, SMOTE algorithm can be use!

A synthetic minority oversampling technique (SMOTE) does as it says, by oversampling the minority class of the data, it can balance out the data classes more evenly. as smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)""")

# Plot the phase-folded light curve
plt.figure(figsize=(10, 4))
plt.plot(phase, flux, 'k.')
plt.xlabel("Phase")
plt.ylabel("Flux")
plt.title("Phase-Folded Light Curve")
plt.xlim(-0.5, 0.5)
plt.show()

import numpy as np

# Constants
G = 6.67e-11  # Gravitational constant in N m^2 / kg^2
M_sun = 1.989e30  # Solar mass in kg
AU = 1.496e11  # 1 Astronomical Unit in meters

def semi_major_axis(T_days, M_star_Msun):
    """
    Calculate the semi-major axis of a planet's orbit in AU.
    
    Parameters:
    T_days (float): Orbital period in days.
    M_star_Msun (float): Mass of the star in solar masses.
    
    Returns:
    float: Semi-major axis in AU.
    """
    # Convert orbital period from days to seconds
    T_seconds = T_days * 24 * 3600  # 1 day = 86400 seconds
    
    # Convert stellar mass from solar masses to kilograms
    M_star = M_star_Msun * M_sun
    
    # Calculate the semi-major axis in meters
    a_meters = ((G * M_star * T_seconds**2) / (4 * np.pi**2))**(1/3)
    
    # Convert semi-major axis from meters to AU
    a_AU = a_meters / AU
    
    return a_AU

# Example: Calculate for KIC006922244 (Kepler-8 b)
T_days = 3.5225  # Orbital period in days
M_star_Msun = 1.132  # Stellar mass in solar masses

a = semi_major_axis(T_days, M_star_Msun)
print(f"Semi-major axis: {a:.5f} AU")


import numpy as np

# Constants (example values)
R_star = 1.489 * 6.955e8  # Stellar radius in meters (for KIC006922244)
P_orb = 3.5225 * 24 * 3600  # Orbital period in seconds (for KIC006922244)
a = semi_major_axis(3.5225, 1.132) * 1.496e11  # Semi-major axis in meters
T_dur = 0.1 * P_orb  # Transit duration in seconds (estimate)

# Function to calculate impact parameter b
def calculate_impact_parameter(a, R_star, T_dur, P_orb):
    T_tot = P_orb
    b = np.sqrt((a / R_star)**2 * (1 - (T_dur / T_tot)**2))
    return b

# Function to calculate orbital inclination i
def calculate_inclination(a, b, R_star):
    cos_i = b * R_star / a
    i_rad = np.arccos(cos_i)
    i_deg = np.degrees(i_rad)
    return i_deg

# Calculate b and i
b = calculate_impact_parameter(a, R_star, T_dur, P_orb)
i = calculate_inclination(a, b, R_star)

print(f"Impact Parameter (b): {b:.4f}")
print(f"Orbital Inclination (i): {i:.4f} degrees")



import matplotlib.pyplot as plt
import numpy as np

# Load your data (replace with your data file)
t = np.linspace(0, 100, 1000)  # Example time data
flux = 1 - 0.01 * (np.abs(np.sin(2 * np.pi * t / 10)) < 0.1)  # Example flux data

# Plot the light curve
plt.figure(figsize=(10, 4))
plt.plot(t, flux, 'k.')
plt.xlabel("Time (days)")
plt.ylabel("Flux")
plt.title("Light Curve")
plt.show()

# Estimate orbital period and other parameters as before
# Compare with NASA's Exoplanet Archive data




