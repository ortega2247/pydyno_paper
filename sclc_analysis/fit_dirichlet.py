from pydream.core import run_dream
import pandas as pd
from scipy.stats import dirichlet, norm, truncnorm, uniform
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
import numpy as np

data = pd.read_csv('cibersort_data/cibersort_data_Sage.csv')

# data_4states = data[['ML', 'MLH', 'NEH', 'NE']].values
data_4states = data[['ML', 'MLH', 'NEH', 'NE']].values
# Add NEH values to MLH to have only three states and keep the sum to 1
data_4states[:, 1] = data_4states[:, 1] + data_4states[:, 2]
data_3states = data_4states[:, [0, 1, 3]]
# Add small number because dirichlet has problems when the data is zero in a sample
data_3states += 1e-10
dirichlet_likelihood = dirichlet([0.4, 5, 15])

myclip_a = 0
myclip_b = 10
my_mean = 0.5
my_std = 0.3

a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

a0 = SampledParam(uniform, loc=0, scale=60)
a1 = SampledParam(uniform, loc=0, scale=60)
a2 = SampledParam(uniform, loc=0, scale=60)
# a3 = SampledParam(uniform, loc=0, scale=60)

sampled_params_list = [a0, a1, a2]

def likelihood(position):
    pars = np.copy(position)
    # pars = 10 ** pars
    print(pars)
    try:
        cost = np.sum([dirichlet.logpdf(sample, pars) for sample in data_3states])
        print('cost', cost)
    except ValueError as e:
        print(e)
        cost = -np.inf
    return cost

# Run pydream


niterations = 10000
nchains = 4
converged = False
if __name__ == '__main__':
    sampled_params, log_ps = run_dream(parameters=sampled_params_list,
                                       likelihood=likelihood,
                                       niterations=niterations,
                                       nchains=nchains,
                                       multitry=False,
                                       gamma_levels=6,
                                       nCR=6,
                                       hardboundaries=True,
                                       snooker_=0.4,
                                       adapt_gamma=False,
                                       history_thin=1,
                                       model_name='dreamzs_5chain_dirichlet',
                                       verbose=True)
    total_iterations = niterations
    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('dreamzs_5chain_dirichlet_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('dreamzs_5chain_dirichletlogps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    np.savetxt('dreamzs_5chain_dirichlet_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)

