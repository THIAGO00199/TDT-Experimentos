#!/usr/bin/env python3
"""Constrained-fidelity amplitude-scan optimization for 4x4 Ising reconstruction."""
import json,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

L=4; N=16; NS=1<<N; J=1.; BETA=.4; GAMMA=1.
TOBS=1.; TCTRL=1.; NSTEPS=20
NSAMP=20000; SEED=20261113
FID_TARGET=.24
AMPS=[1.0,1.5]
ids=np.arange(NS,dtype=np.uint32)
bits=(ids[:,None] >> np.arange(N,dtype=np.uint32))&1
SPINS=(2*bits.astype(np.int8)-1).astype(np.int8)
MAG=SPINS.sum(1)
NEIGH=[]
for y in range(L):
 for x in range(L):
  NEIGH.append([y*L+(x-1)%L,y*L+(x+1)%L,((y-1)%L)*L+x,((y+1)%L)*L+x])
NEIGH=np.asarray(NEIGH)
bond=np.zeros(NS,dtype=np.int16)
for y in range(L):
 for x in range(L):
  i=y*L+x
  bond+=SPINS[:,i]*SPINS[:,y*L+(x+1)%L]
  bond+=SPINS[:,i]*SPINS[:,((y+1)%L)*L+x]
E=-J*bond
P0=np.exp(-BETA*(E-E.min())); P0/=P0.sum()
MVALS=np.arange(-16,17,2)

def make_map():
 rows=[]; cols=[]; vals=[]
 for i in range(N):
  de=2*SPINS[:,i]*SPINS[:,NEIGH[i]].sum(1)*J
  rate=GAMMA/(1+np.exp(np.clip(BETA*de,-700,700)))
  rows.append(ids.astype(np.int32)); cols.append((ids^np.uint32(1<<i)).astype(np.int32)); vals.append(rate)
 r=np.concatenate(rows); c=np.concatenate(cols); v=np.concatenate(vals)
 ex=np.bincount(r,weights=v,minlength=NS)
 Q=coo_matrix((np.r_[v,-ex],(np.r_[r,np.arange(NS)],np.r_[c,np.arange(NS)])),shape=(NS,NS)).tocsr()
 cond=np.empty((NS,len(MVALS)))
 for k,m in enumerate(MVALS):
  cond[:,k]=expm_multiply(Q*TOBS,(MAG==m).astype(float))
 joint=P0[:,None]*np.clip(cond,0,1)
 return np.argmax(joint,axis=0)

def rates(s,h):
 loc=s[NEIGH].sum(1)
 de=2*s*(J*loc+h)
 return GAMMA/(1+np.exp(np.clip(BETA*de,-700,700)))

def evolve(s,h,dt,rng):
 t=0.
 while True:
  rr=rates(s,h); total=float(rr.sum())
  if total<=0: break
  wait=rng.exponential(1/total)
  if t+wait>=dt: break
  z=rng.random()*total
  i=int(np.searchsorted(np.cumsum(rr),z,side="right"))
  if i>=N:i=N-1
  s[i]*=-1; t+=wait
 return s

def run_policy(x0,xt,m,map_ids,info,amp,seed):
 rng=np.random.default_rng(seed)
 w=np.zeros(len(x0)); ok=np.zeros(len(x0),dtype=bool)
 for n in range(len(x0)):
  s=xt[n].copy()
  if info: target=SPINS[x0[n]].astype(float)
  else: target=SPINS[map_ids[(int(m[n])+16)//2]].astype(float)
  for k in range(NSTEPS):
   old=amp*k/NSTEPS*target
   new=amp*(k+1)/NSTEPS*target
   w[n] -= np.dot(s,new-old)
   s=evolve(s,new,TCTRL/NSTEPS,rng)
  ok[n]=np.array_equal(s,SPINS[x0[n]])
 return float(w.mean()),float(w.std(ddof=1)/np.sqrt(len(w))),float(ok.mean()),float(np.sqrt(ok.mean()*(1-ok.mean())/len(ok)))

def main():
 t=time.time(); map_ids=make_map()
 rng=np.random.default_rng(SEED)
 x0=rng.choice(NS,size=NSAMP,p=P0)
 xt=np.empty((NSAMP,N),dtype=np.int8)
 for n in range(NSAMP):
  xt[n]=evolve(SPINS[x0[n]].copy(),np.zeros(N),TOBS,rng)
 m=xt.sum(1).astype(int)
 rows=[]
 for j,A in enumerate(AMPS):
  a=run_policy(x0,xt,m,map_ids,False,A,SEED+100+j)
  b=run_policy(x0,xt,m,map_ids,True,A,SEED+200+j)
  rows.append({"amplitude":A,
   "no_info":{"work":a[0],"work_se":a[1],"fidelity":a[2],"fidelity_se":a[3]},
   "with_X0":{"work":b[0],"work_se":b[1],"fidelity":b[2],"fidelity_se":b[3]}})
 selected={}
 for key in ["no_info","with_X0"]:
  feasible=[r for r in rows if r[key]["fidelity"]>=FID_TARGET]
  selected[key]=None if not feasible else min(feasible,key=lambda r:r["amplitude"])[key] | {"amplitude":min(feasible,key=lambda r:r["amplitude"])["amplitude"]}
 out={"model":{"L":L,"N":N,"J":J,"beta":BETA,"gamma":GAMMA,"prior":"Gibbs beta=.4 h=0",
  "observation_time":TOBS,"control_time":TCTRL,"steps":NSTEPS,"field_shape":"h_i(t)=a(t)*target_spin_i; linear ramp",
  "amplitude_grid":AMPS,"trajectories":NSAMP,"seed":SEED},
  "common_fidelity_target":FID_TARGET,
  "selection_rule":"Among scanned amplitudes with empirical exact fidelity >= target, choose lowest field amplitude (proxy for actuator effort). This is grid-constrained optimization, not a global optimum.",
  "results_by_amplitude":rows,"selected_grid_minima":selected,
  "paired_delta_at_selected":None,
  "limitations":"Monte Carlo estimates; separate trajectory random streams by policy and amplitude; common X0,XT sample set. The two selected amplitudes may differ. No time-shape optimization or confidence-bound constraint.",
  "runtime_seconds":time.time()-t}
 if selected["no_info"] and selected["with_X0"]:
  out["paired_delta_at_selected"]=selected["no_info"]["work"]-selected["with_X0"]["work"]
 with open("ising4x4_fidelity_constrained_scan.json","w") as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=="__main__":main()
