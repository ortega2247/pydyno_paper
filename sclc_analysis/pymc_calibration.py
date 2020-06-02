import sys
sys.path.append('../')
import re
import os
from three_state_sclc_NEv2toNonNE_noCC import model
from pysb.bng import generate_equations
import sympy
import numpy as np
import itertools
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from pysb import Parameter
import pandas as pd

# Experimental Data
data = pd.read_csv('cibersort_data/cibersort_data_Sage.csv')

# data_4states = data[['ML', 'MLH', 'NEH', 'NE']].values
data_4states = data[['ML', 'MLH', 'NEH', 'NE']].values
# Add NEH values to MLH to have only three states and keep the sum to 1
data_4states[:, 1] = data_4states[:, 1] + data_4states[:, 2]
data_3states = data_4states[:, [0, 1, 3]]
# Add small number because dirichlet has problems when the data is zero in a sample
data_3states += 1e-10

generate_equations(model)

eqn_subs = {e: e.expand_expr(expand_observables=True) for
            e in model.expressions}
eqn_subs.update({e: e.expand_expr(expand_observables=True) for
                 e in model._derived_expressions})
ode_mat = sympy.Matrix(model.odes).subs(eqn_subs)


def _eqn_substitutions(eqns, _model):
         """String substitutions on the sympy C code for the ODE RHS and
         Jacobian functions to use appropriate terms for variables and
         parameters."""
         # Substitute 'y[i]' for 'si'
         eqns = re.sub(r'\b__s(\d+)\b',
                       lambda m: 'y[%s]' % (int(m.group(1))),
                       eqns)

         # Substitute 'p[i]' for any named parameters
         for i, p in enumerate(_model.parameters):
             eqns = re.sub(r'\b(%s)\b' % p.name, 'p[%d]' % i, eqns)
         for i, p in enumerate(_model._derived_parameters):
             eqns = re.sub(r'\b(%s)\b' % p.name,
                           'p[%d]' % (i + len(_model.parameters)), eqns)
         return eqns

eqn_repr=sympy.ccode
code_eqs = '\n\t'.join(['ydot[%d] = %s;' %
                                   (i, eqn_repr(o))
                                   for i, o in enumerate(ode_mat)])
code_eqs = str(_eqn_substitutions(code_eqs, model))
code_eqs = 'def rhs(y,t,p):\n\tydot=[0]*3\n\t' + code_eqs + '\n\treturn ydot'
print(code_eqs)
exec(code_eqs)

tspan = np.linspace(0, 100, 501)
rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rates_mask = [p not in model.parameters_initial_conditions() for p in model.parameters]


def _sp_initial(model, sp):
    """
    Get initial condition of a species
    Parameters
    ----------
    sp: pysb.ComplexPattern, pysb species

    Returns
    -------

    """
    sp_0 = 0
    for spInitial in model.initials:
        if spInitial.pattern.is_equivalent_to(sp):
            if isinstance(spInitial.value, Parameter):
                sp_0 = spInitial.value.get_value()
            else:
                sp_0 = float(spInitial.value.get_value())
            break
    return sp_0


initials = [_sp_initial(model, i) for i in model.species]


def rhs(y, t, p):
    ydot = [0] * 3
    ydot[0] = p[18] * y[1] + y[0] * (p[3] * p[1] + p[2] * y[2]) / (p[3] + y[2]) - y[0] * (p[6] * p[4] + p[5] * y[2]) / (
                p[6] + y[2]) + (-1) * (p[17] * y[0]);
    ydot[1] = p[17] * y[0] + y[1] * (p[9] * p[7] + p[8] * y[2]) / (p[9] + y[2]) - y[1] * (
                p[12] * p[10] + p[11] * y[2]) / (p[12] + y[2]) + (-1) * (p[19] * y[1]) + (-1) * (p[18] * y[1]);
    ydot[2] = p[19] * y[1] + y[2] * (p[15] * p[13] + p[14] * (y[0] + y[1])) / (p[15] + y[0] + y[1]) + (-1) * (
                p[16] * y[2]);
    return ydot


if __name__ ==  '__main__':

    with pm.Model() as pysb_model:

        sampled_params_list = list()

        sp_k_NE_div_0 = pm.Normal('sp_k_NE_div_0', mu=np.log10(.428), sigma=.25)
        sampled_params_list.append(sp_k_NE_div_0)
        sp_k_NE_div_x = pm.Normal('sp_k_NE_div_x', mu=np.log10(1.05), sigma=1)
        sampled_params_list.append(sp_k_NE_div_x)
        sp_KD_Kx_NE_div = pm.Normal('sp_KD_Kx_NE_div', mu=np.log10(1000), sigma=1)
        sampled_params_list.append(sp_KD_Kx_NE_div)

        sp_k_NE_die_0 = pm.Normal('sp_k_NE_die_0', mu=np.log10(0.365), sigma=.5)
        sampled_params_list.append(sp_k_NE_die_0)
        sp_k_NE_die_x = pm.Normal('sp_k_NE_die_x', mu=np.log10(0.95), sigma=1)
        sampled_params_list.append(sp_k_NE_die_x)
        sp_KD_Kx_NE_die = pm.Normal('sp_KD_Kx_NE_die', mu=np.log10(1000), sigma=1)
        sampled_params_list.append(sp_KD_Kx_NE_die)

        sp_k_NEv2_div_0 = pm.Normal('sp_k_NEv2_div_0', mu=np.log10(.428), sigma=.25)
        sampled_params_list.append(sp_k_NEv2_div_0)
        sp_k_NEv2_div_x = pm.Normal('sp_k_NEv2_div_x', mu=np.log10(1.05), sigma=1)
        sampled_params_list.append(sp_k_NEv2_div_x)
        sp_KD_Kx_NEv2_div = pm.Normal('sp_KD_Kx_NEv2_div', mu=np.log10(1000), sigma=1)
        sampled_params_list.append(sp_KD_Kx_NEv2_div)

        sp_k_NEv2_die_0 = pm.Normal('sp_k_NEv2_die_0', mu=np.log10(0.365), sigma=.5)
        sampled_params_list.append(sp_k_NEv2_die_0)
        sp_k_NEv2_die_x = pm.Normal('sp_k_NEv2_die_x', mu=np.log10(0.95), sigma=1)
        sampled_params_list.append(sp_k_NEv2_die_x)
        sp_KD_Kx_NEv2_die = pm.Normal('sp_KD_Kx_NEv2_die', mu=np.log10(1000), sigma=1)
        sampled_params_list.append(sp_KD_Kx_NEv2_die)

        sp_k_nonNE_div_0 = pm.Normal('sp_k_nonNE_div_0', mu=np.log10(.428), sigma=.5)
        sampled_params_list.append(sp_k_nonNE_div_0)
        sp_k_nonNE_div_x = pm.Normal('sp_k_nonNE_div_x', mu=np.log10(0.95), sigma=1)
        sampled_params_list.append(sp_k_nonNE_div_x)
        sp_KD_Kx_nonNE_div = pm.Normal('sp_KD_Kx_nonNE_div', mu=np.log10(1000), sigma=1)
        sampled_params_list.append(sp_KD_Kx_nonNE_div)

        sp_k_nonNe_die = pm.Normal('sp_k_nonNe_die', mu=np.log10(0.365), sigma=.5)
        sampled_params_list.append(sp_k_nonNe_die)

        sp_kf_diff_ne_nev2 = pm.Uniform('sp_kf_diff_ne_nev2', lower=np.log10(0.05), upper=2.5)
        sampled_params_list.append(sp_kf_diff_ne_nev2)
        sp_kr_diff_ne_nev2 = pm.Uniform('sp_kr_diff_ne_nev2', lower=np.log10(0.05), upper=2.5)
        sampled_params_list.append(sp_kr_diff_ne_nev2)

        sp_kf_diff_nev2_nonNe = pm.Uniform('sp_kf_diff_nev2_nonNe', lower=np.log10(0.05), upper=2.5)
        sampled_params_list.append(sp_kf_diff_nev2_nonNe)

        y_hat = pm.ode.DifferentialEquation(
            func=rhs,
            times=tspan,
            n_states=len(model.species),
            n_theta=len(model.parameters),
            t0=0
        )(
            y0=initials, theta=sampled_params_list
        )
        ne = y_hat.T[0][-1]
        nev2 = y_hat.T[1][-1]
        nonne = y_hat.T[2][-1]
        a0 = ne + nev2 + nonne

        e1 = pm.Dirichlet('percentages', a=pm.math.stack([nonne/a0, nev2/a0, ne/a0]), shape=3, observed=data_3states)

    #     prior = pm.sample_prior_predictive()
        trace = pm.sample(20, tune=10, cores=4,  init='adapt_diag')
    #     posterior_predictive = pm.sample_posterior_predictive(trace)

# def SIR(y, t, p):
#     ds = -p[0] * y[0] * y[1]
#     di = p[0] * y[0] * y[1] - p[1] * y[1]
#     return [ds, di]
#
#
# times = np.arange(0, 5, 0.25)
#
# beta, gamma = 4, 1.0
# # Create true curves
# y = odeint(SIR, t=times, y0=[0.99, 0.01], args=((beta, gamma),), rtol=1e-8)
# # Observational model.  Lognormal likelihood isn't appropriate, but we'll do it anyway
# yobs = np.random.lognormal(mean=np.log(y[1::]), sigma=[0.2, 0.3])
#
# sir_model = DifferentialEquation(
#     func=SIR,
#     times=np.arange(0.25, 5, 0.25),
#     n_states=2,
#     n_theta=2,
#     t0=0,
# )
#
# with pm.Model() as model4:
#     sigma = pm.HalfCauchy('sigma', 1, shape=2)
#
#     # R0 is bounded below by 1 because we see an epidemic has occured
#     R0 = pm.Bound(pm.Normal, lower=1)('R0', 2, 3)
#     lam = pm.Lognormal('lambda', pm.math.log(2), 2)
#     beta = pm.Deterministic('beta', lam * R0)
#
#     sir_curves = sir_model(y0=[0.99, 0.01], theta=[beta, lam])
#
#     Y = pm.Lognormal('Y', mu=pm.math.log(sir_curves), sd=sigma, observed=yobs)
#
#     prior = pm.sample_prior_predictive()
#     trace = pm.sample(2000, tune=1000, target_accept=0.9, cores=1)
#     posterior_predictive = pm.sample_posterior_predictive(trace)
#
#     # data = az.from_pymc3(trace=trace, prior = prior, posterior_predictive = posterior_predictive)
