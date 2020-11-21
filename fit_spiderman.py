import os
import matplotlib.pyplot as plt
import pickle
import juliet

plt.style.use('ggplot')

import spiderman as sp
import pymultinest
import numpy as np
import utils
import json

# Load brigthness map:
spider_params = sp.ModelParams(brightness_model="zhang")
spider_params.n_layers= 5

# Wavelength arrays:
wlows = np.array([2.87, 3.27, 3.83, 4.23, 4.63])
whighs = np.array([3.27, 3.67, 4.23, 4.63, 5.03])

# Pre-calculated "stellar" grid array:
stellar_grids = [[], [], [], [], []]

# Load datasets: 
t1, f1, f1err = np.loadtxt('atmosphere_1000_0_3.07um.dat',unpack=True, usecols = (0, 1, 2))
t2, f2, f2err = np.loadtxt('atmosphere_1000_0_3.47um.dat',unpack=True, usecols = (0, 1, 2))
t3, f3, f3err = np.loadtxt('atmosphere_1000_0_4.03um.dat',unpack=True, usecols = (0, 1, 2))
t4, f4, f4err = np.loadtxt('atmosphere_1000_0_4.43um.dat',unpack=True, usecols = (0, 1, 2))
t5, f5, f5err = np.loadtxt('atmosphere_1000_0_4.83um.dat',unpack=True, usecols = (0, 1, 2))
all_t = [t1,t2,t3,t4,t5]
all_f = [f1,f2,f3,f4,f5]
all_ferr = [f1err,f2err,f3err,f4err,f5err]
# Bin datasets for easier fitting; 10-minute cadence seems okay:
for i in range(len(all_t)):
    all_t[i], all_f[i], all_ferr[i] = juliet.utils.bin_data(all_t[i],all_f[i],60)
    # Average the errors, they are all the same anyways:
    all_ferr[i] = np.median(all_ferr[i])
    stellar_grids[i] = sp.stellar_grid.gen_grid(wlows[i]*1e-6,whighs[i]*1e-6,logg=4.5, stellar_model='blackbody')

ndata = len(all_t[0])

# Define fixed parameters:
P = 0.2803244
spider_params.t0= P/2. # Central time of PRIMARY transit [days] (P/2) so eclipse time is at 0.
spider_params.per= P       # Period [days]
spider_params.a_abs= 0.0072586911252        # The absolute value of the semi-major axis [AU]
spider_params.inc= 86.3            # Inclination [degrees]
spider_params.ecc= 0.0              # Eccentricity
spider_params.w= 90                 # Argument of periastron
spider_params.rp= 0.02037            # Planet to star radius ratio
spider_params.a= 2.292              # Semi-major axis scaled by stellar radius
spider_params.p_u1= 0               # Planetary limb darkening parameter
spider_params.p_u2= 0               # Planetary limb darkening parameter
spider_params.T_s = 4599. # stellar temp

# Define the prior (you have to transform your parameters, that come from the unit cube,
# to the prior you want):
def prior(cube, ndim, nparams):
    # Prior on "txi" is uniform between -1 and 1. txi is a transform of xi such that xi = abs(txi) so thing is symmetric around 0.
    cube[0] = utils.transform_uniform(cube[0],0.,1.)
    # Prior on nightside temperature is uniform from 0 to 3000:
    cube[1] = utils.transform_uniform(cube[1],0.,3000.)
    # Prior on delta temperature goes from 0 to 3000 as well:
    cube[2] = utils.transform_uniform(cube[1],0.,3000.)

# Define the likelihood:
def loglike(cube, ndim, nparams):
    # Extract parameters:
    xi,night_T,deltaT = cube[0], cube[1], cube[2]

    # Generate model:
    spider_params.xi= np.abs(xi)
    spider_params.T_n = night_T
    spider_params.delta_T = deltaT

    # Initialize log-like:
    loglikelihood = 0.

    # Iterate through the lightcurves, add up the log-likelihood:
    for i in range(5):
        t, f = all_t[i], all_f[i]
        spider_params.l1 = wlows[i]*1e-6
        spider_params.l2 = whighs[i]*1e-6
        # Evaluate lightcurve:
        model = spider_params.lightcurve(t, stellar_grid = stellar_grids[i])
        # Evaluate and add-up the log-likelihood:
        loglikelihood += -0.5*ndata*np.log(2.*np.pi*all_ferr[i]**2) + (-0.5 * ((model - f) / all_ferr[i])**2).sum()
    return loglikelihood

n_params = 3
out_file = 'out_multinest'


if not os.path.exists('fit.pkl'):
    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points = 500,outputfiles_basename=out_file, resume = False, verbose = True)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
    # Save matrix:
    pickle.dump(posterior_samples,open('fit.pkl','wb'))
    print('Done! Run again to plot')
    sys.exit()
else:
    posterior_samples = pickle.load(open('fit.pkl','rb'))

xi = posterior_samples[:,0]
T_n = posterior_samples[:,1]
delta_T = posterior_samples[:,2]

for i in range(5):
    t, f = all_t[i], all_f[i]
    plt.errorbar(t, f + i*(300*1e-6), yerr = np.ones(len(t))*all_ferr[i],fmt = '.', alpha = 0.2)

for (txi, night_T, deltaT) in posterior_samples[-300:,:]:
    spider_params.xi= np.abs(txi)
    spider_params.T_n = night_T
    spider_params.delta_T = deltaT
    for i in range(5):
        spider_params.l1 = wlows[i]*1e-6
        spider_params.l2 = whighs[i]*1e-6
        # eval:
        model = spider_params.lightcurve(all_t[i],stellar_grid = stellar_grids[i])
        plt.plot(all_t[i], model + i*(300*1e-6), '-', color='blue', alpha=0.01)
plt.show()

import corner
names = ['$\tau_\textrm{rad}/\tau_\textrm{adv}$', '$T_{N}$', '$\Delta T$']
figure = corner.corner(posterior_samples, labels = names, color='cornflowerblue')
plt.show()
