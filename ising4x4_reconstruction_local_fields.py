#!/usr/bin/env python3
"""Paired Monte Carlo physical reconstruction benchmark for TDT v3."""
import json, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

L=4; N=16; NS=1<<N; J=1.0; BETA=0.4; GAMMA=1.0
T_OBS=1.0; T_CTRL=1.0; NSTEPS=20; HMAX=3.0
NSAMP=20000; SEED=20261112
ids=np.arange(NS,dtype=np.uint32)
bits=((ids[:,None] >> np.arange(N,dtype=np.uint32)) & 1)
SPINS=(2*bits.astype(np.int8)-1).astype(np.int8)
MAG=SPINS.sum(axis=1).astype(np.int16)
NEIGH=[]
for y in range(L):
 for x in range(L):
  NEIGH.append([y*L+(x-1)%L,y*L+(x+1)%L,((y-1)%L)*L+x,((y+1)%L)*L+x])
NEIGH=np.asarray(NEIGH,dtype=np.int16)
bond_sum=np.zeros(NS,dtype=np.int16)
for y in range(L):
 for x in range(L):
  i=y*L+x
  bond_sum += SPINS[:,i]*SPINS[:,y*L+(x+1)%L]
  bond_sum += SPINS[:,i]*SPINS[:,((y+1)%L)*L+x]
E=-J*bond_sum
P0=np.exp(-BETA*(E-E.min())); P0/=P0.sum()
MVALS=np.arange(-16,17,2)

def generator_zero():
 rows=[]; cols=[]; vals=[]
 for i in range(N):
  loc=SPINS[:,NEIGH[i]].sum(1)
  de=2*SPINS[:,i]*(J*loc)
  rate=GAMMA/(1+np.exp(np.clip(BETA*de,-700,700)))
  rows.append(ids.astype(np.int32))
  cols.append((ids ^ np.uint32(1<<i)).astype(np.int32))
  vals.append(rate)
 r=np.concatenate(rows); c=np.concatenate(cols); v=np.concatenate(vals)
 ex=np.bincount(r,weights=v,minlength=NS)
 return coo_matrix((np.r_[v,-ex],(np.r_[r,np.arange(NS)],np.r_[c,np.arange(NS)])),shape=(NS,NS)).tocsr()

def map_targets():
 Q=generator_zero()
 cond=np.empty((NS,len(MVALS)))
 for k,m in enumerate(MVALS):
  cond[:,k]=expm_multiply(Q*T_OBS,(MAG==m).astype(float))
 cond=np.clip(cond,0,1)
 joint=P0[:,None]*cond
 pm=joint.sum(0)
 map_ids=np.argmax(joint,axis=0)
 hcond=0.0
 for k in range(len(MVALS)):
  if pm[k] <= 0: continue
  post=joint[:,k]/pm[k]
  z=post>0
  hcond += pm[k]*(-np.sum(post[z]*np.log2(post[z])))
 map_success=float(np.max(joint,axis=0).sum())
 return map_ids, hcond, map_success, float(np.max(np.abs(cond.sum(1)-1)))

def rates(spins, h):
 loc=spins[NEIGH].sum(axis=1)
 de=2.0*spins*(J*loc+h)
 return GAMMA/(1.0+np.exp(np.clip(BETA*de,-700,700)))

def evolve_gillespie(spins, h, duration, rng):
 t=0.0
 while True:
  rr=rates(spins,h)
  total=float(rr.sum())
  if total<=0: break
  wait=rng.exponential(1.0/total)
  if t+wait>=duration: break
  z=rng.random()*total
  idx=int(np.searchsorted(np.cumsum(rr),z,side="right"))
  if idx>=N: idx=N-1
  spins[idx]*=-1
  t+=wait
 return spins

def controller_run(x0, xt, m, map_ids, informed, seed):
 rng=np.random.default_rng(seed)
 works=np.zeros(len(x0),dtype=float)
 final=np.empty_like(xt)
 dh=HMAX/NSTEPS
 dt=T_CTRL/NSTEPS
 for n in range(len(x0)):
  s=xt[n].copy()
  target=SPINS[int(x0[n])] if informed else SPINS[int(map_ids[(int(m[n])+16)//2])]
  sign=target.astype(float)
  for k in range(NSTEPS):
   hnew=sign*(HMAX*(k+1)/NSTEPS)
   hold=sign*(HMAX*k/NSTEPS)
   works[n] += -float(np.dot(s,hnew-hold))
   s=evolve_gillespie(s,hnew,dt,rng)
  final[n]=s
 return works, final

def main():
 t0=time.time()
 map_ids,hcond,map_success,part_err=map_targets()
 rng=np.random.default_rng(SEED)
 x0=rng.choice(NS,size=NSAMP,p=P0)
 xt=np.empty((NSAMP,N),dtype=np.int8)
 for n in range(NSAMP):
  xt[n]=evolve_gillespie(SPINS[x0[n]].copy(),np.zeros(N),T_OBS,rng)
 m=xt.sum(axis=1).astype(int)
 w0,f0=controller_run(x0,xt,m,map_ids,False,SEED+1)
 w1,f1=controller_run(x0,xt,m,map_ids,True,SEED+2)
 success0=np.all(f0==SPINS[x0],axis=1)
 success1=np.all(f1==SPINS[x0],axis=1)
 delta=w0-w1
 # Memory erasure lower bound: kBT ln2 H(X0|M), with kB=1.
 kT=1.0/BETA
 memory_lb=kT*np.log(2.0)*hcond
 out={
  "model":{"L":L,"N":N,"microstates":NS,"boundary":"periodic","J":J,"beta":BETA,"gamma":GAMMA,
   "observation_time":T_OBS,"control_time":T_CTRL,"control_steps":NSTEPS,"local_field_max_abs":HMAX,
   "prior":"Gibbs at beta=0.4, h=0","dynamics":"continuous-time single-spin-flip Glauber via Gillespie"},
  "sampling":{"trajectories":NSAMP,"seed":SEED,"paired_same_X0_and_XT":True},
  "bayesian_reference":{"H_X0_given_M_T_bits":hcond,"exact_MAP_guess_probability":map_success,
   "conditional_partition_error":part_err},
  "no_auxiliary_controller":{"information":"M_T only; target is exact posterior MAP microstate for observed magnetization",
   "mean_work_actuator":float(w0.mean()),"se_work_actuator":float(w0.std(ddof=1)/np.sqrt(NSAMP)),
   "exact_reconstruction_fidelity":float(success0.mean()),
   "fidelity_se":float(np.sqrt(success0.mean()*(1-success0.mean())/NSAMP))},
  "with_X0_controller":{"information":"X0 and M_T; target is the original microstate X0",
   "mean_work_actuator":float(w1.mean()),"se_work_actuator":float(w1.std(ddof=1)/np.sqrt(NSAMP)),
   "exact_reconstruction_fidelity":float(success1.mean()),
   "fidelity_se":float(np.sqrt(success1.mean()*(1-success1.mean())/NSAMP))},
  "paired_comparison":{"delta_work_no_info_minus_with_info":float(delta.mean()),
   "delta_work_se":float(delta.std(ddof=1)/np.sqrt(NSAMP)),
   "mean_work_difference_over_kT_ln2_Hcond":float(delta.mean()/memory_lb),
   "memory_conditional_entropy_bits":hcond},
  "memory_accounting":{"conditional_memory_erasure_entropy_bits":hcond,
   "landauer_minimum_erasure_work_kT_ln2_H":float(memory_lb),
   "kT":kT,"kB":1.0,
   "warning":"This is an ideal reversible lower bound for erasing a memory of X0 when M_T is available. It is not the physical cost of writing, storing, or reading X0, and is not included in actuator work."},
  "protocol_warning":"Fixed open-loop local-field ramp, not optimized. Controller receives no observations during actuation. Monte Carlo finite-sample estimates. The MAP benchmark is an inference bound, not guaranteed physical fidelity.",
  "runtime_seconds":time.time()-t0
 }
 with open("ising4x4_reconstruction_local_fields.json","w") as f: json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=="__main__": main()
