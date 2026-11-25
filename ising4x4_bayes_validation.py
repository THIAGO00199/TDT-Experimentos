#!/usr/bin/env python3
import json, time
import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.stats import poisson
from scipy.sparse import diags
import ising4x4_first_controls as base

T=1.0
Q=base.generator(0.0)
mvals=np.arange(-16,17,2)
# Reference: scipy Krylov action, one backward observable per magnetization.
t0=time.time()
cond_k=np.empty((base.NS,len(mvals)))
for k,m in enumerate(mvals):
    cond_k[:,k]=expm_multiply(Q*T,(base.M==m).astype(float))
cond_k=np.clip(cond_k,0,1)
# Independent uniformization: exp(Qt)=exp(-lambda*t) sum (lambda*t)^n/n! P^n.
lam=float(np.max(-Q.diagonal()))
P=diags(np.ones(base.NS),format="csr")+Q/lam
mu=lam*T
nmax=int(poisson.ppf(1-1e-14,mu))
weights=poisson.pmf(np.arange(nmax+1),mu)
cond_u=np.empty_like(cond_k)
for k,m in enumerate(mvals):
    v=(base.M==m).astype(float)
    acc=weights[0]*v
    for n in range(1,nmax+1):
        v=P @ v
        acc += weights[n]*v
    cond_u[:,k]=acc
# Gibbs prior as in the baseline.
p0=base.p0
def metrics(cond):
    joint=p0[:,None]*cond
    pm=joint.sum(axis=0)
    map_success=float(np.max(joint,axis=0).sum())
    posterior=np.divide(joint,pm[None,:],out=np.zeros_like(joint),where=pm[None,:]>0)
    h=0.0
    for k in range(len(mvals)):
        z=posterior[:,k]>0
        h += pm[k]*(-np.sum(posterior[z,k]*np.log2(posterior[z,k])))
    return map_success,float(h),float(np.max(np.abs(cond.sum(axis=1)-1))),pm
mk=metrics(cond_k); mu_=metrics(cond_u)
out={
 "model":{"L":4,"N":16,"states":base.NS,"J":1.0,"beta":0.4,"gamma":1.0,"T":T,"h":0.0},
 "krylov":{"MAP_success":mk[0],"H_X0_given_M_bits":mk[1],"partition_error":mk[2]},
 "uniformization":{"lambda_max_exit":lam,"poisson_mean":mu,"n_terms":nmax+1,"poisson_tail_bound":float(poisson.sf(nmax,mu)),"MAP_success":mu_[0],"H_X0_given_M_bits":mu_[1],"partition_error":mu_[2]},
 "cross_method":{"max_abs_conditional_probability_difference":float(np.max(np.abs(cond_k-cond_u))),"MAP_absolute_difference":abs(mk[0]-mu_[0]),"conditional_entropy_abs_difference_bits":abs(mk[1]-mu_[1]),"magnetization_marginal_max_abs_difference":float(np.max(np.abs(mk[3]-mu_[3])))},
 "runtime_seconds":time.time()-t0,
 "interpretation":"Inference-only MAP benchmark; not physical reconstruction. Krylov and independent uniformization both evaluate the same continuous-time CTMC at T=1."
}
with open("ising4x4_bayes_validation.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
