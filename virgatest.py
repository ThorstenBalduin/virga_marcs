from bokeh.io import output_notebook
from bokeh.plotting import show, figure
from bokeh.palettes import Colorblind
output_notebook()
import numpy as np
import pandas as pd
import astropy.units as u

#here is pyeddy
import virga.justdoit as jdi
import virga.justplotit as jpi

input_data_marcs='/groups/astro/mhundrup/MARCS/models/K2_18b/test_400K/marcs_output.dat'

TP = np.genfromtxt(input_data_marcs, skip_header = 93, max_rows= 53)
chf = np.genfromtxt(input_data_marcs, skip_header = 39, max_rows= 53)[:,5]
layer = TP[:,0]
T = TP[:,4]
P = TP[:,6]*1e-6
#T = 300
#P = 1e0

K2_18b = pd.DataFrame({'pressure': P, 'temperature': T, 'chf': chf})
print(K2_18b)
print(np.shape(K2_18b))

mieff_dir =  '/groups/astro/mhundrup/virga/virga'

metallicity = 100 #atmospheric metallicity relative to Solar
mean_molecular_weight = 2.2 # atmospheric mean molecular weight

recommended_gases = jdi.recommend_gas(P, T,
                                     metallicity,mean_molecular_weight)

print(recommended_gases)

#set the run
a_100 = jdi.Atmosphere(['H2O'],
                  fsed=0.1,mh=metallicity,
                 mmw = mean_molecular_weight)
a_1 = jdi.Atmosphere(['H2O'],
                  fsed=0.1,mh=1,
                 mmw = mean_molecular_weight)

#set the planet gravity
a_100.gravity(gravity=2.95, gravity_unit=u.Unit('m/(s**2)'))
a_1.gravity(gravity=2.95, gravity_unit=u.Unit('m/(s**2)'))

#Get preset pt profile for testing
#a.ptk(df = jdi.hot_jupiter())
a_100.ptk(df = K2_18b)
a_1.ptk(df = K2_18b)

#get full dictionary output
all_out_100 = jdi.compute(a_100, as_dict=True,
                      directory=mieff_dir)
all_out_1 = jdi.compute(a_1, as_dict=True,
                      directory=mieff_dir)

#show(jpi.pt(all_out_100,plot_height=450))
#show(jpi.pt(all_out_1,plot_height=450))
#show(jpi.opd_by_gas(all_out_100))