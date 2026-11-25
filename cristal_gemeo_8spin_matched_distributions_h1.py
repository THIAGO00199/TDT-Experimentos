#!/usr/bin/env python3
import json
import numpy as np
from scipy.optimize import root

N=8; NS=2**N; J=1.0; KT=1.0; BETA_A=1.0; HF=1.0
ids=np.arange(NS)
bits=((ids[:,None]>>np.arange(N))&1)
S=(2*bits-1).astype(float)
M=S.sum(axis=1)
bonds=np.array([S[:,i]*S[:,(i+1)%N] for i in range(N)]).T
B=bonds.sum(axis=1)
Eint=-J*B
Hf=Eint-HF*M
# Local correlation marker: one selected bond, not the translational average.
C=bonds[:,0]
def normalize_logw(logw):
    z=logw-np.max(logw)
    p=np.exp(z); return p/p.sum()
def gibbs_A():
    return normalize_logw(-BETA_A*Hf/KT)
def dist_B(theta,alpha):
    beta,lam=theta
    return normalize_logw(-beta*Eint/KT+lam*M+alpha*C)
pA=gibbs_A()
targetE=float(pA@Hf); targetM=float(pA@M)
def obs(p):
    return np.array([p@Hf,p@M])
def solve_alpha(alpha):
    def f(theta):
        return (obs(dist_B(theta,alpha))-np.array([targetE,targetM]))/np.array([8.0,8.0])
    sol=root(f,np.array([BETA_A, BETA_A*HF/KT]),method="hybr",options={"xtol":1e-11})
    p=dist_B(sol.x,alpha)
    return sol,p,float(np.linalg.norm(obs(p)-[targetE,targetM]))
def entropy(p):
    return -float(np.sum(p[p>0]*np.log(p[p>0])))
def summarize(p):
    corr=(p@bonds)
    return {
      "mean_energy_final_H":float(p@Hf),
      "mean_magnetization":float(p@M),
      "mean_bond_correlations":corr.tolist(),
      "selected_bond_correlation":float(corr[0]),
      "other_bond_correlation_mean":float(np.mean(corr[1:])),
      "entropy_nats":entropy(p),
      "Fneq":float(p@Hf-KT*entropy(p)),
      "distribution":p.tolist()
    }
A=summarize(pA)
results=[]
for alpha in [0.25,0.5,1.0,1.5,2.0]:
    sol,p,err=solve_alpha(alpha)
    results.append({
      "alpha_local_correlation_bias":alpha,
      "solver_success":bool(sol.success),
      "solver_message":str(sol.message),
      "lagrange_parameters_beta_lambda":sol.x.tolist(),
      "constraint_residual_norm_scaled":err,
      "delta_E":float(p@Hf-targetE),
      "delta_M":float(p@M-targetM),
      "L1_distance_from_A":float(np.abs(p-pA).sum()),
      "delta_Fneq":float((p@Hf-KT*entropy(p))-(pA@Hf-KT*entropy(pA))),
      "selected_minus_other_bond_corr":float((p@bonds)[0]-np.mean((p@bonds)[1:])),
      "distribution":p.tolist()
    })
out={
 "model":{"N":N,"states":NS,"geometry":"periodic 8-spin Ising ring","J":J,"kBT":KT,
          "final_field_hf":HF,"A":"exact Gibbs distribution at beta=1,h=1",
          "B":"exponential-family nonequilibrium distribution with a local bond-correlation bias alpha; beta and magnetization multiplier are re-solved to match A's mean final energy and magnetization"},
 "history_A":A,
 "history_B_candidates":results,
 "interpretation":"This constructs exact matched (mean E, mean M) distributions with a tunable spatial correlation difference. It does not yet prove reachability by a specified localized-field pulse and does not calculate reversal work or R.",
 "R_status":"NOT_COMPUTED: a physically reachable preparation pulse, common reversal target, admissible control bounds, horizon, and entropy-production convention must be fixed before optimization."
}
path="/home/iknowyou/TDT_Experimentos/cristal_gemeo_8spin_matched_distributions_h1.json"
with open(path,"w") as f: json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items() if k not in ("history_A","history_B_candidates")},indent=2))
print("A_SUMMARY",json.dumps({k:v for k,v in A.items() if k!="distribution"},indent=2))
for r in results:
 print("B_CANDIDATE",json.dumps({k:v for k,v in r.items() if k!="distribution"},indent=2))
