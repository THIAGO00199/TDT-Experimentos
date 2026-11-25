#!/usr/bin/env python3
import json, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

L=4; N=16; NS=1<<N; J=1.; BETA=.4; GAMMA=1.
TOBS_LIST=[0.25,0.5,1.0,2.0,4.0]
TCTRL=1.; NSTEPS=20; NSAMP=5000; SEED=20261113
FID_TARGET=.24
AMPS=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
ids=np.arange(NS,dtype=np.uint32)
bits=(ids[:,None] >> np.arange(N,dtype=np.uint32))&1
SPINS=(2*bits.astype(np.int8)-1).astype(np.int8)
MAG=SPINS.sum(1).astype(np.int16)
NEIGH=[]
for y in range(L):
 for x in range(L):
  NEIGH.append([y*L+(x-1)%L,y*L+(x+1)%L,((y-1)%L)*L+x,((y+1)%L)*L+x])
NEIGH=np.asarray(NEIGH,dtype=np.int16)
bond=np.zeros(NS,dtype=np.int16)
for y in range(L):
 for x in range(L):
  i=y*L+x
  bond+=SPINS[:,i]*SPINS[:,y*L+(x+1)%L]
  bond+=SPINS[:,i]*SPINS[:,((y+1)%L)*L+x]
E=-J*bond
P0=np.exp(-BETA*(E-E.min())); P0/=P0.sum()
MVALS=np.arange(-16,17,2)

def make_generator():
 rows=[]; cols=[]; vals=[]
 for i in range(N):
  de=2*SPINS[:,i]*SPINS[:,NEIGH[i]].sum(1)*J
  rate=GAMMA/(1+np.exp(np.clip(BETA*de,-700,700)))
  rows.append(ids.astype(np.int32)); cols.append((ids^np.uint32(1<<i)).astype(np.int32)); vals.append(rate)
 r=np.concatenate(rows); c=np.concatenate(cols); v=np.concatenate(vals)
 ex=np.bincount(r,weights=v,minlength=NS)
 return coo_matrix((np.r_[v,-ex],(np.r_[r,np.arange(NS)],np.r_[c,np.arange(NS)])),shape=(NS,NS)).tocsr()

Q=make_generator()

def posterior_stats(tobs):
 indicators=np.column_stack([(MAG==m).astype(float) for m in MVALS])
 cond=expm_multiply(Q*tobs,indicators)
 cond=np.clip(cond,0.,1.)
 joint=P0[:,None]*cond
 pm=joint.sum(0)
 map_ids=np.argmax(joint,axis=0)
 h=0.
 for k in range(len(MVALS)):
  if pm[k]>0:
   post=joint[:,k]/pm[k]
   z=post>0
   h+=pm[k]*(-np.sum(post[z]*np.log2(post[z])))
 map_success=float(np.max(joint,axis=0).sum())
 part_err=float(np.max(np.abs(cond.sum(1)-1)))
 return map_ids,float(h),map_success,part_err

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

def observe(x0,tobs,rng):
 xt=np.empty((len(x0),N),dtype=np.int8)
 for n,x in enumerate(x0):
  xt[n]=evolve(SPINS[int(x)].copy(),np.zeros(N),tobs,rng)
 return xt

def run_policy(x0,xt,m,map_ids,info,amp,seed):
 rng=np.random.default_rng(seed)
 works=np.zeros(len(x0),dtype=np.float64)
 ok=np.zeros(len(x0),dtype=bool)
 dt=TCTRL/NSTEPS
 for n in range(len(x0)):
  s=xt[n].copy()
  target=SPINS[int(x0[n])] if info else SPINS[int(map_ids[(int(m[n])+16)//2])]
  target=target.astype(float)
  for k in range(NSTEPS):
   old=amp*k/NSTEPS*target
   new=amp*(k+1)/NSTEPS*target
   works[n]-=float(np.dot(s,new-old))
   s=evolve(s,new,dt,rng)
  ok[n]=np.array_equal(s,SPINS[int(x0[n])])
 return works,ok

def summary(x):
 p=float(x.mean()); se=float(np.sqrt(p*(1-p)/len(x)))
 return {"fidelity":p,"fidelity_se":se,"fidelity_ci95":[max(0.,p-1.96*se),min(1.,p+1.96*se)]}

def main():
 t0=time.time()
 out={"model":{"L":L,"N":N,"microstates":NS,"boundary":"periodic","J":J,"beta":BETA,"gamma":GAMMA,
  "prior":"Gibbs beta=.4, h=0","dynamics":"CTMC single-spin Glauber, Gillespie",
  "control":"local fields h_i(t)=a(t)*target_i, linear ramp; Tctrl=1; 20 steps",
  "fidelity_target":FID_TARGET,"amplitudes":AMPS,"observation_times":TOBS_LIST,
  "trajectories_per_time":NSAMP,"seed":SEED},
  "selection_rule":"For each controller independently, choose the smallest scanned amplitude whose approximate 95% Wilson-like Wald lower bound p-1.96*SE is >= target. Grid-constrained amplitude proxy, not global work optimization.",
  "results":[]}
 for ti,tobs in enumerate(TOBS_LIST):
  t1=time.time()
  map_ids,hcond,map_succ,part_err=posterior_stats(tobs)
  rng=np.random.default_rng(SEED+ti*1000)
  x0=rng.choice(NS,size=NSAMP,p=P0)
  xt=observe(x0,tobs,rng)
  m=xt.sum(1).astype(int)
  rows=[]
  samples={}
  for ai,A in enumerate(AMPS):
   seed=SEED+100000+ti*10000+ai*100
   w0,o0=run_policy(x0,xt,m,map_ids,False,A,seed)
   w1,o1=run_policy(x0,xt,m,map_ids,True,A,seed)
   rows.append({"amplitude":A,
    "no_info":{**summary(o0),"mean_work":float(w0.mean()),"work_se":float(w0.std(ddof=1)/np.sqrt(NSAMP))},
    "with_X0":{**summary(o1),"mean_work":float(w1.mean()),"work_se":float(w1.std(ddof=1)/np.sqrt(NSAMP))}})
   samples[A]=(w0,w1,o0,o1)
  selected={}
  for key,idx in [("no_info",0),("with_X0",1)]:
   feasible=[r for r in rows if r[key]["fidelity_ci95"][0]>=FID_TARGET]
   selected[key]=None if not feasible else min(feasible,key=lambda r:r["amplitude"])
  pair=None
  if selected["no_info"] and selected["with_X0"]:
   a0=selected["no_info"]["amplitude"]; a1=selected["with_X0"]["amplitude"]
   w0=samples[a0][0]; w1=samples[a1][1]
   d=w0-w1
   pair={"amplitude_no_info":a0,"amplitude_with_X0":a1,
    "delta_work_no_info_minus_with_X0":float(d.mean()),
    "delta_work_se_paired":float(d.std(ddof=1)/np.sqrt(NSAMP)),
    "delta_work_ci95":[float(d.mean()-1.96*d.std(ddof=1)/np.sqrt(NSAMP)),float(d.mean()+1.96*d.std(ddof=1)/np.sqrt(NSAMP))],
    "work_no_info":float(w0.mean()),"work_with_X0":float(w1.mean())}
  out["results"].append({"Tobs":tobs,"H_X0_given_M_T_bits":hcond,
   "exact_MAP_guess_probability":map_succ,"conditional_partition_error":part_err,
   "amplitude_scan":rows,"selected_common_fidelity":selected,
   "paired_work_comparison":pair,
   "runtime_seconds":time.time()-t1})
  with open("ising4x4_Tobs_curve_partial.json","w") as f: json.dump(out,f,indent=2)
  print(json.dumps({"Tobs":tobs,"Hcond":hcond,"selected":selected,"delta":pair}),flush=True)
 out["runtime_seconds"]=time.time()-t0
 with open("ising4x4_Tobs_curve.json","w") as f:json.dump(out,f,indent=2)
 print("FINAL_JSON")
 print(json.dumps(out,indent=2))
if __name__=="__main__":main()
