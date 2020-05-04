from pysb import *


def k_fate(ename, k_fate_0, k_fate_x, KD_Kx_fate, effector_cell_obs):
    return Expression(ename, (k_fate_0*KD_Kx_fate + k_fate_x*effector_cell_obs) / (KD_Kx_fate + effector_cell_obs))


Model()

Monomer('NE')
Monomer('NEv2')
Monomer('NonNE')

Parameter('NE_init', 100)
Initial(NE(), NE_init)

Observable('NE_obs', NE())
Observable('NEv2_obs', NEv2())
Observable('NonNE_obs', NonNE())
Observable('NE_all', NE() + NEv2())

# Parameter('k_ne_div', 1)
Parameter('k_NE_div_0', 1) # TPCs divide approximately once per day in culture
Parameter('k_NE_div_x', 2)
Parameter('KD_Kx_NE_div', 1000)
k_fate('k_NE_div', k_NE_div_0, k_NE_div_x, KD_Kx_NE_div, NonNE_obs)
Rule('NE_div', NE() >> NE() + NE(), k_NE_div)

# Parameter('k_ne_die', 0.9)
Parameter('k_NE_die_0', 0.9)
Parameter('k_NE_die_x', 0.1)
Parameter('KD_Kx_NE_die', 1000)
k_fate('k_NE_die', k_NE_die_0, k_NE_die_x, KD_Kx_NE_die, NonNE_obs)
Rule('NE_die', NE() >> None, k_NE_die)

# Parameter('k_nev2_div', 1)
Parameter('k_NEv2_div_0', 1) # TPCs divide approximately once per day in culture
Parameter('k_NEv2_div_x', 2)
Parameter('KD_Kx_NEv2_div', 1000)
k_fate('k_NEv2_div', k_NEv2_div_0, k_NEv2_div_x, KD_Kx_NEv2_div, NonNE_obs)
Rule('NEv2_div', NEv2() >> NEv2() + NEv2(), k_NEv2_div)

# Parameter('k_nev2_die', 0.9)
Parameter('k_NEv2_die_0', 0.9)
Parameter('k_NEv2_die_x', 0.1)
Parameter('KD_Kx_NEv2_die', 1000)
k_fate('k_NEv2_die', k_NEv2_die_0, k_NEv2_die_x, KD_Kx_NEv2_die, NonNE_obs)
Rule('NEv2_die', NEv2() >> None, k_NEv2_die)

#Expression('k_NEv2_CC',(k_NEv2_div-k_NEv2_die) / carrying_capacity_each_subtype)
#Rule('NEv2_CC', NEv2() + NEv2() >> NEv2(), k_NEv2_CC)

# Parameter('k_nonNe_div', 0.9)
Parameter('k_nonNE_div_0', 1.1)
Parameter('k_nonNE_div_x', 0.9)
Parameter('KD_Kx_nonNE_div', 1000)
k_fate('k_nonNE_div', k_nonNE_div_0, k_nonNE_div_x, KD_Kx_nonNE_div, NE_all)
Rule('NonNE_div', NonNE() >> NonNE() + NonNE(), k_nonNE_div)

Parameter('k_nonNe_die', 0.1)
Rule('NonNE_die', NonNE() >> None, k_nonNe_die)

Parameter('kf_diff_ne_nev2', 0.1)
Parameter('kr_diff_ne_nev2', 0.1)
Rule('NE_diff_NEv2', NE() | NEv2(), kf_diff_ne_nev2, kr_diff_ne_nev2)

Parameter('kf_diff_nev2_nonNe', 0.1)
Rule('NEv2_diff_NonNE', NEv2() >> NonNE(), kf_diff_nev2_nonNe)

'''
tspan = np.linspace(0, 100, 1001)

sim = ScipyOdeSimulator(model, verbose=True)
x = sim.run(tspan)

plt.figure()
for obs in model.observables[:4]:
    label = obs.name[:obs.name.find('_')]
    plt.plot(tspan, x.all[obs.name], lw=3, label=label)
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc=0)
plt.tight_layout()

# cell_tot = np.array([sum(x.observables[i]) for i in range(len(x.observables))])
cell_tot = sum(x.all[obs.name] for obs in [NE_obs, NEv1_obs, NEv2_obs, NonNE_obs])

plt.figure()
label = [obs.name[:obs.name.find('_')] for obs in model.observables[:4]]
plt.fill_between(tspan, x.all[model.observables[0].name] / cell_tot, label=label[0])
sum_prev = x.all[model.observables[0].name]
for i in range(1,len(model.observables[:4])-1):
    plt.fill_between(tspan, (x.all[model.observables[i].name] + sum_prev) / cell_tot, sum_prev / cell_tot, label=label[i])
    sum_prev += x.all[model.observables[i].name]
plt.fill_between(tspan, [1]*len(tspan), sum_prev / cell_tot, label=label[-1])
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell fraction', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=(0.75,0.6), framealpha=1)
plt.tight_layout()

plt.show()
    

'''





