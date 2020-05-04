from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import norm, uniform
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

# Data distributions
TKO_pct = {
'NE_obs':0.670866848446898,
'NEv2_obs':0.294235217878072,
'NonNE_obs':0.033886762908045
}
TKO_stdev = {
'NE_obs':0.152576600276884,
'NEv2_obs':0.147927688661517,
'NonNE_obs':0.031684600453998
}

TKO_ne_nev2_nonne_mean = [0.670866848446898, 0.294235217878072, 0.033886762908045]
TKO_ne_nev2_nonne_std = [0.152576600276884, 0.147927688661517, 0.031684600453998]

like_pct_data = norm(loc=TKO_ne_nev2_nonne_mean, scale=TKO_ne_nev2_nonne_std)

obs_list = ['NE_obs','NEv2_obs','NonNE_obs']

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

sp_kf_diff_ne_nev2 = SampledParam(uniform, loc=np.log10(0.05), scale=2.5)
sampled_params_list.append(sp_kf_diff_ne_nev2)
sp_kr_diff_ne_nev2 = SampledParam(uniform, loc=np.log10(0.05), scale=2.5)
sampled_params_list.append(sp_kr_diff_ne_nev2)

sp_kf_diff_nev2_nonNe = SampledParam(uniform, loc=np.log10(0.05), scale=2.5)
sampled_params_list.append(sp_kf_diff_nev2_nonNe)

# Likelihood function

TOLERANCE = 1e-4
# USER must define a likelihood function!
def likelihood(position):
    Y = np.copy(position)
    param_values[rates_mask] = 10 ** Y
    signal.alarm(300)
    try:
        sim = solver.run(param_values=param_values, tspan=tspan).species
        sim_data = np.array(sim)
    except TimeoutException as exc:
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
        return np.inf*-1
    else:
        # Return -inf if any species trajectory hasn't reached steady state
        for sp in range(len(model.species)):
            sp_tr = sim_data[:, sp]
            derivative = np.diff(sp_tr) / dt
            equilibrated = np.allclose(derivative[-50:], 0, atol=1e-4)
            if not equilibrated:
                print(derivative[-50:])
                print('not equilibrated')
                return -np.inf
        # Obtain percentages of last time points to compare to data
        species_pctg = sim_data[-1, :] / end_point_total_cells
        # Score
        total_cost = np.sum(like_pct_data.logpdf(species_pctg))
        if np.isnan(total_cost):
            total_cost = -np.inf
        return total_cost

# Run pydream


niterations = 100000
nchains = 4
converged = False

starts = []
log10_original_values = np.log10(param_values[rates_mask])
for chain in range(nchains):
    start_position = log10_original_values + np.random.uniform(-0.25, 0.25, size=np.shape(log10_original_values))
    starts.append(start_position)


if __name__ == '__main__':
    sampled_params, log_ps = run_dream(parameters=sampled_params_list,
                                       likelihood=likelihood,
                                       start=starts,
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
