import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import gauss
from scipy.optimize import curve_fit
import copy


def resample_array(current, current_std):
    new = []
    for s in range(0, len(current)):
        if current_std[s] > 0:
            new.append(gauss(current[s], current_std[s]))
        else:
            new.append(current[s])
    return np.asarray(new)

 # Monte-Carlo to resample all values in their uncertainties, which gives us lists of the params to evaluate quantiles
def mc_fit(func, x:list|np.ndarray, y:list|np.ndarray, x_err:list|np.ndarray=None, y_err:list|np.ndarray=None, iterations:int=1000, **curve_fit_kwargs):
    """Function that does Monte Carlo fitting"""
    res_list = []
    sigmas_list = []
    x = np.array(x)
    y = np.array(y)
    for i in tqdm(range(iterations)):
        if x_err is not None:
            resampled_x = resample_array(x, x_err)
        else:
            resampled_x = x
        if y_err is not None:
            resampled_y = resample_array(y, y_err)
        else:
            resampled_y = y
        res, cov_matrix = curve_fit(func, resampled_x, resampled_y, **curve_fit_kwargs)
        sigmas = np.sqrt(np.diag(cov_matrix))
        res_list.append(res)
        sigmas_list.append(sigmas)
    res_list = np.array(res_list)
    res = np.median(res_list, axis=0)
    stat_sig = np.median(sigmas_list, axis=0)
    fits = []
    fit_param, fit_param_unc = np.median(res_list, axis=0),np.std(res_list, axis=0)
    for i in tqdm(range(iterations)):
        params = resample_array(fit_param, np.sqrt(fit_param_unc**2+stat_sig**2)) # Resample the fitted parameters in their uncertainties
        fits.append(func(x, *params))
    params, params_unc = np.median(res_list, axis=0), np.std(res_list, axis=0)
    return np.median(fits), np.quantile(fits, 0.16), np.quantile(fits, 0.84), params, params_unc


import emcee
from tqdm import tqdm

def resolution_theorique(energies):
    return ((260**2 - 120**2 + 2440*energies)**(1/2))/1000

def racine_carre(energies, p):
    A, k = p[0], p[1]
    return ((k + (A*energies))**(1/2))

def chi2(p, y, x):
    model = racine_carre(x,p)
    return np.sum((y-model)**2)

def fit_func(energie, fwhm, p0:dict=None, bounds:dict=None, nwalkers=32, nsteps=5000, burnin=1000):
    """
    Function to fit a spectrum segment.
    
    """
    default_p0 = {"A": 2000,
                  "k": 100,
                  }
    if p0 is not None:
        for key in p0.keys():
            default_p0[key] = p0[key] # For every key that was specified, change the value from the default one.
    p0 = default_p0 # Assign the changed default initial values.
    p0 = np.array([p0[i] for i in p0.keys()]) # Convert to numpy array

    if bounds is None: bounds = {}
    bounds["A"] = bounds.get("A", (-np.inf,np.inf))
    bounds["k"] = bounds.get("k", (-np.inf,np.inf))
    # Define likelihood function we want ot minimize
    def neg_log_ll(p, y, x):
        ll = np.sum((y-racine_carre(x,p))**2/2)
        if np.any(~np.isfinite(p)):
            return np.inf
        if not (bounds["A"][0] < p[0] < bounds["A"][1]): return np.inf # alpha bounds
        if not (bounds["k"][0] < p[1] < bounds["k"][1]): return np.inf # sig_n bounds
        if np.isnan(ll): return np.inf
        return ll
    def log_prob(params):
        return -neg_log_ll(params,fwhm,energie)
    ndim = len(p0)
    # --- initialize walkers ---
    pos = p0 + 1e-2* np.random.randn(nwalkers, ndim)
    # --- run sampler ---
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=None)
    sampler.run_mcmc(pos, nsteps, progress=True)
    # --- discard burn-in and flatten ---
    flat_samples = sampler.get_chain(discard=burnin, thin=10, flat=True)
    # take median (or MAP) as best fit
    best_fit = np.median(flat_samples, axis=0)
    lls = [neg_log_ll(flat_samples[i],fwhm,energie) for i in range(len(flat_samples))]
    best_fit = flat_samples[lls.index(min(lls))]

    return best_fit, neg_log_ll(best_fit,fwhm,energie), flat_samples, chi2(best_fit,fwhm,energie)

energies = np.array([22.16, 24.94, 8.05, 8.91, 6.4, 7.06, 9.18, 10.55, 12.61, 14.76])
fwhms = ([0.4292508, 0.37489338, 0.27805385, 0.34902806, 0.28922925, 0.26359529, 0.33061002, 0.33061002, 0.37026958, 0.32813065])
E = np.sort(energies)

latex_labels = [r"$A$", r"$k$", r"$c$"]
coeffs, ll, flat_samples, _ = fit_func(energies, fwhms, nsteps=5000, burnin=500,
                                       bounds={"c":(-10,10)}
                                       )
print(ll)


import corner
corner_fig = corner.corner(flat_samples, labels=latex_labels,
                        title_fmt='.2f',
                        quantiles=[0.16, 0.5, 0.84], truths=coeffs, show_titles=True,
                        title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                        levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels
plt.show()

fits = np.array([
    racine_carre(E, p)
    for p in flat_samples
])
fit_med = np.median(fits, axis=0)
fit_lo  = np.quantile(fits, 0.16, axis=0)
fit_hi  = np.quantile(fits, 0.84, axis=0)



plt.plot(energies, fwhms, marker="o", ls="None", color="darkred", label="data")
plt.plot(E, racine_carre(E, coeffs), marker="None", ls="-", color="blue", label="Best fit")
plt.plot(E, fit_med, marker="None", ls="--", color="blue", label="Median fit")
plt.fill_between(E, fit_lo, fit_hi, color="blue", alpha=0.3)
plt.legend(fontsize=12)
plt.show()