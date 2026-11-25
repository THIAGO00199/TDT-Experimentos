#!/usr/bin/env python3
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

N=4; NS=16; J=1.0; KT=1.0; BETA=1.0; GAMMA=1.0; HF=4.0
ids=np.arange(NS)
bits=((ids[:,None]>>np.arange(N))&1)
S=(2*bits-1).astype(float)
M=S.sum(axis=1)
Eint=np.array([-J*sum(S[x,i]*S[x,(i+1)%N] for i in range(N)) for x in range(NS)])
def energy(h): return Eint-h*M
def gibbs(h):
    e=energy(h); z=np.exp(-(e-e.min())/KT); return z/z.sum()
def Q(h):
    e=energy(h); q=np.zeros((NS,NS))
    for x in range(NS):
        for i in range(N):
            y=x^(1<<i)
            de=e[y]-e[x]
            q[x,y]=GAMMA/(1+np.exp(np.clip(de/KT,-700,700)))
        q[x,x]=-q[x].sum()
    return q
def stats(p,h):
    return {"M":float(p@M),"E":float(p@energy(h)),
            "p":p.tolist(),
            "Fneq":float(p@energy(h)+KT*np.sum(p[p>0]*np.log(p[p>0])))}
pA=gibbs(HF)
p0=gibbs(0.0)
qf=Q(HF)
targetM=float(pA@M); targetE=float(pA@energy(HF))
def pB(t): return p0@expm(qf*t)
def loss(t):
    p=pB(t)
    return ((p@M-targetM)/4.0)**2+((p@energy(HF)-targetE)/8.0)**2
opt=minimize_scalar(loss,bounds=(0,100),method="bounded",options={"xatol":1e-13})
ts=[0,0.01,0.1,0.25,0.5,1,2,4,8,16,32,64,100]
rows=[]
for t in ts:
 p=pB(t); rows.append({"t":t,"M":float(p@M),"E":float(p@energy(HF)),
                       "dM":float(p@M-targetM),"dE":float(p@energy(HF)-targetE),
                       "L1_to_A":float(np.abs(p-pA).sum())})
out={
 "model":{"N":N,"microstates":NS,"geometry":"4-spin periodic ring","J":J,"kBT":KT,
          "beta":BETA,"gamma":GAMMA,"field_final":HF,
          "rates":"single-spin Glauber w(x->x^i)=gamma/[1+exp(DeltaE/kBT)]",
          "row_probability_convention":"p(t)=p(0) exp(Q t)"},
 "history_A":{"protocol":"quasi-static equilibrium at each field step; endpoint is exact Gibbs at h=4",
              **stats(pA,HF)},
 "history_B":{"protocol":"instantaneous quench h=0 to h=4; subsequent exact master-equation relaxation",
              "initial":stats(p0,0),"best_finite_time_match":stats(pB(float(opt.x)),HF),
              "best_tquench":float(opt.x),"normalized_squared_mismatch":float(opt.fun),
              "dM":float(pB(float(opt.x))@M-targetM),
              "dE":float(pB(float(opt.x))@energy(HF)-targetE),
              "L1_to_A":float(np.abs(pB(float(opt.x))-pA).sum())},
 "relaxation_scan":rows,
 "matching_conclusion":"finite-time exact simultaneous matching not assumed; report residual mismatch and distribution distance",
 "R_status":"NOT_COMPUTED: reversal control objective, endpoint/fidelity definition, admissible control family, and entropy-production accounting are not yet specified consistently for this history-matching experiment."
}
path="/home/iknowyou/TDT_Experimentos/cristal_gemeo_tdt_v1_hf4_exact16.json"
with open(path,"w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
