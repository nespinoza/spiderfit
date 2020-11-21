import os
import matplotlib.pyplot as plt
import pickle
import juliet

#plt.style.use('ggplot')

import spiderman as sp
import pymultinest
import numpy as np
import utils
import json

import seaborn as sns
sns.set_style('ticks')
pal = sns.color_palette("magma", n_colors = 5)

ifit = 0
use_phoenix = False

if use_phoenix:
    # Load brigthness map:
    spider_params = sp.ModelParams(brightness_model="spherical", stellar_model = 'k2141_star.dat')
else:
    spider_params = sp.ModelParams(brightness_model="spherical")
spider_params.n_layers= 5

# Wavelength arrays:
wlows = np.array([2.87, 3.27, 3.83, 4.23, 4.63])
whighs = np.array([3.27, 3.67, 4.23, 4.63, 5.03])

# Pre-calculated "stellar" grid array:
stellar_grids = [[], [], [], [], []]

# Load datasets: 
t1, f1, f1err = np.loadtxt('atmosphere_basaltic_3.07um.dat',unpack=True, usecols = (0, 1, 2))
t2, f2, f2err = np.loadtxt('atmosphere_basaltic_3.47um.dat',unpack=True, usecols = (0, 1, 2))
t3, f3, f3err = np.loadtxt('atmosphere_basaltic_4.03um.dat',unpack=True, usecols = (0, 1, 2))
t4, f4, f4err = np.loadtxt('atmosphere_basaltic_4.43um.dat',unpack=True, usecols = (0, 1, 2))
t5, f5, f5err = np.loadtxt('atmosphere_basaltic_4.83um.dat',unpack=True, usecols = (0, 1, 2))
#for (laO, loO, l_c1) in posterior_samples[-300:,:]:
all_t = [t1,t2,t3,t4,t5]
all_f = [f1,f2,f3,f4,f5]
all_ferr = [f1err,f2err,f3err,f4err,f5err]
# Bin datasets for easier fitting; 10-minute cadence seems okay:
for i in range(len(all_t)):
    all_t[i], all_f[i], all_ferr[i] = juliet.utils.bin_data(all_t[i],all_f[i],60)
    # Average the errors, they are all the same anyways:
    all_ferr[i] = np.median(all_ferr[i])
    if use_phoenix:
        stellar_grids[i] = sp.stellar_grid.gen_grid(wlows[i]*1e-6,whighs[i]*1e-6,logg=4.5,stellar_model = 'k2141_star.dat')
    else:
        stellar_grids[i] = sp.stellar_grid.gen_grid(wlows[i]*1e-6,whighs[i]*1e-6,logg=4.5,stellar_model = 'blackbody')

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
spider_params.degree = 2

# Define the prior (you have to transform your parameters, that come from the unit cube,
# to the prior you want):
def prior(cube, ndim, nparams):
    # Prior on la0
    cube[0] = utils.transform_uniform(cube[0],-np.pi,np.pi)
    #cube[0] = utils.transform_uniform(cube[0],-180.,180.)
    # Prior on lo0:
    #cube[1] = utils.transform_uniform(cube[1],-90.,90.)
    cube[1] = utils.transform_uniform(cube[1],-np.pi/2.,np.pi/2.)
    # Log10 of spherical harmonic coefficient:
    cube[2] = utils.transform_uniform(cube[2],-6, -2.)
    cube[3] = utils.transform_uniform(cube[3],-6, -2.)
    #cube[4] = utils.transform_uniform(cube[4],-6, -2.)
    #cube[5] = utils.transform_uniform(cube[5],-6, -2.)

# Define the likelihood:
spider_params.l1 = wlows[ifit]*1e-6
spider_params.l2 = whighs[ifit]*1e-6
def loglike(cube, ndim, nparams):
    # Extract parameters:
    laO, loO, l_c1, l_c4 = cube[0], cube[1], cube[2], cube[3]
    #laO, loO, l_c1, l_c2, l_c3, l_c4 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]
    #laO, loO, l_c1 = cube[0], cube[1], cube[2]
    #txi,night_T,deltaT = cube[0], cube[1], cube[2]
    #xi = np.abs(txi)
    # Generate model:
    spider_params.la0= laO
    spider_params.lo0 = loO
    #spider_params.sph = [10**(l_c1), 10**(l_c2), 10**(l_c3), 10**(l_c4)]
    spider_params.sph = [10**(l_c1), 0., 0., 10**(l_c4)]

    # Iterate through the lightcurves, add up the log-likelihood:
    t, f = all_t[ifit], all_f[ifit]
    # Evaluate lightcurve:
    model = spider_params.lightcurve(t, stellar_grid = stellar_grids[ifit])
    idx = np.where(model<0)[0]
    if len(idx) != 0:
        return -np.inf
    else:
        # Evaluate and add-up the log-likelihood:
        return -0.5*ndata*np.log(2.*np.pi*all_ferr[ifit]**2) + (-0.5 * ((model - f) / all_ferr[ifit])**2).sum()

n_params = 4
out_file = 'out_multinest'


if not os.path.exists('fit-sph-'+str(ifit)+'.pkl'):
    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points = 500,outputfiles_basename=out_file, resume = False, verbose = True)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
    # Save matrix:
    pickle.dump(posterior_samples,open('fit-sph-'+str(ifit)+'.pkl','wb'))
    print('Done! Run again to plot')
    sys.exit()
else:
    posterior_samples = pickle.load(open('fit-sph-'+str(ifit)+'.pkl','rb'))

t, f = all_t[ifit], all_f[ifit]
plt.errorbar(t, f, yerr = np.ones(len(t))*all_ferr[ifit],fmt = '.', alpha = 0.2)

#for (laO, loO, l_c1, l_c2, l_c3, l_c4) in posterior_samples[-300:,:]:
for (laO, loO, l_c1, l_c4) in posterior_samples[-300:,:]:
    spider_params.la0= laO #np.abs(txi)
    spider_params.lo0= loO #night_T
    #spider_params.sph = [10**(l_c1), 10**(l_c2), 10**(l_c3), 10**(l_c4)]#deltaT
    spider_params.sph = [10**(l_c1),0.,0., 10**(l_c4)]
    # eval:
    model = spider_params.lightcurve(all_t[ifit],stellar_grid = stellar_grids[ifit])
    plt.plot(all_t[ifit], model, '-', color='blue', alpha=0.01)
plt.show()

import corner
names = [r'$Latitude offset$', r'$Longitude offset$', r'$l=0, m=0$', 'r$l=1$ $m=1$']
figure = corner.corner(posterior_samples, labels = names, titles = names, bins=15,plot_datapoints='False',quantiles=[.16,0.5,.84],show_titles='True',plot_contours='True',levels=(1.-np.exp(-(1)**2/2.),1.-np.exp(-(2)**2/2.),1.-np.exp(-(3)**2/2.)),color=pal[0])
plt.show()
