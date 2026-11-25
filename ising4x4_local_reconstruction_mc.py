#!/usr/bin/env python3
"""Paired Monte Carlo local-field reconstruction benchmark, Ising 4x4."""
import json, time
import numpy as np
from scipy.sparse.linalg import expm_multiply
import ising4x4_first_controls as b

SEED=20261112
NSAMP=20000
T_REF=1.0
T_CTRL=1.0
H0=2.0
rng=np.random.default_rng(SEED)
mvals=np.arange(-16,17,2)

def build_map():
    Q=b.generator(0.0)
    cond=np.empty((b.NS,len(mvals)))
    for k,m in enumerate(mvals):
        cond[:,k]=expm_multiply(Q*T_REF,(b.M==m).astype(float))
    joint=b.p0[:,None]*np.clip(cond,0,1)
    # deterministic tie break: first state index among exact posterior maxima
    map_states=np.argmax(joint,axis=0)
    return map_states

def gibbs_state():
    return int(rng.choice(b.NS,p=b.p0))

def spins_of(state):
    return (2*((state >> np.arange(b.N)) & 1)-1).astype(np.int8)

def simulate(state, duration, hvec):
    """Gillespie single-spin-flip CTMC under fixed site fields hvec."""
    t=0.0
    while True:
        sp=spins_of(state)
        rates=np.empty(b.N,dtype=float)
        for i in range(b.N):
            loc=int(sp[b.neighbors[i]].sum())
            de=2.0*sp[i]*(b.J*loc+float(hvec[i]))
            z=np.clip(b.beta*de,-700,700)
            rates[i]=b.gamma/(1.0+np.exp(z))
        total=float(rates.sum())
        wait=rng.exponential(1.0/total)
        if t+wait>duration:
            break
        r=rng.random()*total
        i=int(np.searchsorted(np.cumsum(rates),r,side="right"))
        if i>=b.N: i=b.N-1
        state ^= (1<<i)
        t += wait
    return state

def wilson(k,n,z=1.959963984540054):
    p=k/n
    den=1+z*z/n
    ctr=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(ctr-half),float(ctr+half)]

t0=time.time()
map_states=build_map()
# Paired samples: same X0 and same uncontrolled Glauber trajectory XT for both controllers.
work_no=np.empty(NSAMP); work_yes=np.empty(NSAMP)
success_no=np.zeros(NSAMP,dtype=bool); success_yes=np.zeros(NSAMP,dtype=bool)
m_obs=np.empty(NSAMP,dtype=int)
for n in range(NSAMP):
    x0=gibbs_state()
    xt=simulate(x0,T_REF,np.zeros(b.N))
    m=int(spins_of(xt).sum())
    m_obs[n]=m
    target_no=map_states[(m+16)//2]
    target_yes=x0
    sxt=spins_of(xt)
    h_no=H0*spins_of(target_no).astype(float)
    h_yes=H0*spins_of(target_yes).astype(float)
    # Instantaneous switch 0 -> h_i: delta work on system = -sum_i s_i dh_i.
    work_no[n]=-float(np.dot(sxt,h_no))
    work_yes[n]=-float(np.dot(sxt,h_yes))
    xf_no=simulate(xt,T_CTRL,h_no)
    xf_yes=simulate(xt,T_CTRL,h_yes)
    success_no[n]=(xf_no==x0)
    success_yes[n]=(xf_yes==x0)

kn=int(success_no.sum()); ky=int(success_yes.sum())
mean_no=float(work_no.mean()); mean_yes=float(work_yes.mean())
se_no=float(work_no.std(ddof=1)/np.sqrt(NSAMP)); se_yes=float(work_yes.std(ddof=1)/np.sqrt(NSAMP))
# Paired difference is the cleanest comparison.
wd=work_yes-work_no
# Conditional information from exact Krylov result.
Hcond=6.854905134675901
Hprior=8.273491832934397
beta=b.beta
kBT=1.0/beta
landauer_cond=kBT*np.log(2)*Hcond
landauer_prior=kBT*np.log(2)*Hprior
out={
 "protocol":{
  "lattice":"4x4 periodic Ising, J=1, beta=0.4, gamma=1",
  "prior":"exact Gibbs beta=0.4, h=0",
  "reference":"Gillespie Glauber at h=0 for T_ref=1",
  "observation":"M_T only for no-auxiliary controller",
  "actuation":"site fields h_i=H0*target_spin_i switched instantaneously from 0 to target, then held fixed during Glauber T_ctrl=1",
  "H0":H0,"T_ctrl":T_CTRL,"samples":NSAMP,"seed":SEED,
  "controller_no_info":"target is exact Bayesian MAP state conditional on M_T; no access to X0",
  "controller_with_info":"target is actual X0; has access to all 16 initial spin values",
  "paired":"same sampled X0 and same uncontrolled X_T for both controllers; independent post-switch stochastic trajectories"
 },
 "no_auxiliary":{
  "exact_successes":kn,"fidelity":float(success_no.mean()),"wilson_95":wilson(kn,NSAMP),
  "mean_actuator_work":mean_no,"work_SE":se_no
 },
 "with_X0":{
  "exact_successes":ky,"fidelity":float(success_yes.mean()),"wilson_95":wilson(ky,NSAMP),
  "mean_actuator_work":mean_yes,"work_SE":se_yes
 },
 "paired_comparison":{
  "work_difference_with_minus_without":float(wd.mean()),
  "work_difference_SE":float(wd.std(ddof=1)/np.sqrt(NSAMP)),
  "fidelity_difference":float(success_yes.mean()-success_no.mean())
 },
 "memory_accounting":{
  "H_X0_given_M_T_bits":Hcond,
  "H_X0_bits":Hprior,
  "ideal_conditional_reset_lower_bound_kBTln2Hcond":float(np.log(2)*Hcond),
  "ideal_conditional_reset_lower_bound_energy_units_at_beta_0p4":float(landauer_cond),
  "ideal_unconditional_reset_lower_bound_energy_units_at_beta_0p4":float(landauer_prior),
  "note":"These are ideal Landauer lower bounds for resetting a memory register after use, not measured acquisition/storage work and not automatically additive to actuator work. Conditional reset assumes M_T remains available as side information. Actual memory architecture, measurement, writing, and reset protocol are not simulated."
 },
 "runtime_seconds":time.time()-t0,
 "limitations":[
  "This is a fixed-amplitude open-loop local-field protocol, not an optimized minimum-work controller.",
  "The field is switched once and held; switching work is measured exactly from the pre-switch microstate, while no work is done during the hold.",
  "The post-switch CTMC is sampled by Gillespie; confidence intervals quantify Monte Carlo uncertainty, not model uncertainty.",
  "The no-information controller uses M_T only to choose a MAP target and does not observe X_T microstate.",
  "The information controller is given X0 as a classical side-information register; its thermodynamic acquisition cost is not included."
 ]
}
with open("ising4x4_local_reconstruction_mc.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
