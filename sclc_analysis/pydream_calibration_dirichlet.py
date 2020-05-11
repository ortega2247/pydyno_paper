from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm, uniform, dirichlet
from three_state_sclc_NEv2toNonNE_noCC import model
import signal


class TimeoutException(RuntimeError):
    """ Time out occurred! """
    pass


def handler(signum, frame):
    print('forever is over!')
    raise TimeoutException()


# Register the signal function handler
signal.signal(signal.SIGALRM, handler)

# Initialize PySB solver object for running simulations.  Simulation timespan should match experimental data.
tspan = np.linspace(0, 60, 1001)
dt = tspan[1] - tspan[0]
solver = ScipyOdeSimulator(model, integrator='lsoda', compiler='cython')
param_values = np.array([p.value for p in model.parameters])
rates_mask = [p not in model.parameters_initial_conditions() for p in model.parameters]

sampled_params_list = []

like_pct_data = dirichlet([0.24937233, 2.35457684, 5.48145111])

like_steady_state = norm(0, 10)

# PRIOR
sp_k_NE_div_0 = SampledParam(norm, loc=np.log10(.428), scale=.25)
sampled_params_list.append(sp_k_NE_div_0)
sp_k_NE_div_x = SampledParam(norm, loc=np.log10(1.05), scale=1)
sampled_params_list.append(sp_k_NE_div_x)
sp_KD_Kx_NE_div = SampledParam(norm, loc=np.log10(1000), scale=1)
sampled_params_list.append(sp_KD_Kx_NE_div)

sp_k_NE_die_0 = SampledParam(norm, loc=np.log10(0.365), scale=.5)
sampled_params_list.append(sp_k_NE_die_0)
sp_k_NE_die_x = SampledParam(norm, loc=np.log10(0.95), scale=1)
sampled_params_list.append(sp_k_NE_die_x)
sp_KD_Kx_NE_die = SampledParam(norm, loc=np.log10(1000), scale=1)
sampled_params_list.append(sp_KD_Kx_NE_die)

sp_k_NEv2_div_0 = SampledParam(norm, loc=np.log10(.428), scale=.25)
sampled_params_list.append(sp_k_NEv2_div_0)
sp_k_NEv2_div_x = SampledParam(norm, loc=np.log10(1.05), scale=1)
sampled_params_list.append(sp_k_NEv2_div_x)
sp_KD_Kx_NEv2_div = SampledParam(norm, loc=np.log10(1000), scale=1)
sampled_params_list.append(sp_KD_Kx_NEv2_div)

sp_k_NEv2_die_0 = SampledParam(norm, loc=np.log10(0.365), scale=.5)
sampled_params_list.append(sp_k_NEv2_die_0)
sp_k_NEv2_die_x = SampledParam(norm, loc=np.log10(0.95), scale=1)
sampled_params_list.append(sp_k_NEv2_die_x)
sp_KD_Kx_NEv2_die = SampledParam(norm, loc=np.log10(1000), scale=1)
sampled_params_list.append(sp_KD_Kx_NEv2_die)

sp_k_nonNE_div_0 = SampledParam(norm, loc=np.log10(.428), scale=.5)
sampled_params_list.append(sp_k_nonNE_div_0)
sp_k_nonNE_div_x = SampledParam(norm, loc=np.log10(0.95), scale=1)
sampled_params_list.append(sp_k_nonNE_div_x)
sp_KD_Kx_nonNE_div = SampledParam(norm, loc=np.log10(1000), scale=1)
sampled_params_list.append(sp_KD_Kx_nonNE_div)

sp_k_nonNe_die = SampledParam(norm, loc=np.log10(0.365), scale=.5)
sampled_params_list.append(sp_k_nonNe_die)

sp_kf_diff_ne_nev2 = SampledParam(norm, loc=np.log10(0.05), scale=2.5)
sampled_params_list.append(sp_kf_diff_ne_nev2)
sp_kr_diff_ne_nev2 = SampledParam(norm, loc=np.log10(0.05), scale=2.5)
sampled_params_list.append(sp_kr_diff_ne_nev2)

sp_kf_diff_nev2_nonNe = SampledParam(norm, loc=np.log10(0.05), scale=2.5)
sampled_params_list.append(sp_kf_diff_nev2_nonNe)

# Likelihood function


# USER must define a likelihood function!
def likelihood(position):
    Y = np.copy(position)
    param_values[rates_mask] = 10 ** Y
    signal.alarm(30)
    try:
        sim = solver.run(param_values=param_values, tspan=tspan).species
        sim_data = np.array(sim)
    except TimeoutException as exc:
        return -np.inf
    except ZeroDivisionError:
        return -np.inf
    else:
        signal.alarm(0)
    all_lessthan1 = np.any(sim_data[-1, :] < 1)

    # if there aren't enough cells, or if the end of the sim gets to NaNs (because it grew too fast)
    end_point_total_cells = np.sum(sim_data[-1, :])
    if end_point_total_cells < 1000000:
        print('not enough cells ' + str(end_point_total_cells))
        return -np.inf
    elif np.isnan(end_point_total_cells):
        print('nans in simulations ')
        return -np.inf
    elif all_lessthan1: #smallest size in SCLC allografts (Lim et al) 1cm^3 (~10^8 cells), largest ~3.5cm^3 (~4*10^8)
        print('less than 1 cell in at lead one species')
        return -np.inf
    else:
        # Return -inf if any species trajectory hasn't reached steady state
        e2 = 0
        for sp in range(len(model.species)):
            sp_tr = sim_data[:, sp]
            derivative = np.diff(sp_tr) / dt
            equilibrated = np.isclose(derivative[-50:], 0, atol=1)
            if not np.any(equilibrated):
                # print(derivative[-50:])
                print('not equilibrated')
                e2 += np.size(equilibrated) - np.count_nonzero(equilibrated)

        # Obtain percentages of last time points to compare to data
        species_pctg = sim_data[-1, :] / end_point_total_cells
        # Score
        total_cost = np.sum(like_pct_data.logpdf(species_pctg[::-1])) + like_steady_state.logpdf(e2)
        if np.isnan(total_cost):
            total_cost = -np.inf
        return total_cost

# Run pydream


niterations = 100000
nchains = 5
converged = False
if __name__ == '__main__':
    sampled_params, log_ps = run_dream(parameters=sampled_params_list,
                                       likelihood=likelihood,
                                       niterations=niterations,
                                       nchains=nchains,
                                       multitry=False,
                                       gamma_levels=6,
                                       nCR=6,
                                       snooker_=0.4,
                                       adapt_gamma=False,
                                       history_thin=1,
                                       model_name='dreamzs_5chain_NEv2_Sage_NM',
                                       verbose=True)
    total_iterations = niterations
    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('dreamzs_5chain_NEv2_Sage_NM_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('dreamzs_5chain_NEv2_Sage_NM_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    np.savetxt('dreamzs_5chain_NEv2_Sage_NM_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)
