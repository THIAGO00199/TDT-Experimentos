#!/usr/bin/env python3
import json,time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, minimize_scalar
BETA=.4; KT=1/BETA; J=1.; GAMMA=1.; TOBS=1.; TCTRL=4.; NSTEPS=20
TARGET=.45
N=4; NS=16
ids=np.arange(NS)
bits=((ids[:,None]>>np.arange(N))&1)
S=(2*bits-1).astype(float)
MAG=S.sum(axis=1)
Ebase=np.array([-J*sum(S[x,i]*S[x,(i+1)%N] for i in range(N)) for x in range(NS)])
P0=np.exp(-BETA*(Ebase-Ebase.min())); P0/=P0.sum()
MVALS=np.array([-4,-2,0,2,4])
def rates(x,h):
 s=S[x]; out=[]
 for i in range(N):
  loc=s[(i-1)%N]+s[(i+1)%N]
  de=2*s[i]*(J*loc+h[i])
  out.append(GAMMA/(1+np.exp(np.clip(BETA*de,-700,700))))
 return np.array(out)
def Qmat(h):
 Q=np.zeros((NS,NS))
 for x in range(NS):
  rr=rates(x,h)
  for i in range(N): Q[x,x^(1<<i)]+=rr[i]
  Q[x,x]=-Q[x].sum()
 return Q
Q0=Qmat(np.zeros(N)); Pobs=expm(Q0*TOBS)
joint=np.zeros((NS,NS))
for x0 in range(NS): joint[x0,:]=P0[x0]*Pobs[x0,:]
pm=np.array([joint[:,MAG==m].sum() for m in MVALS])
post=np.zeros((NS,len(MVALS)))
for j,m in enumerate(MVALS):
 if pm[j]>0: post[:,j]=joint[:,MAG==m].sum(axis=1)/pm[j]
mapids=np.argmax(post,axis=0)
def H(p):
 p=p[p>0]; return float(-np.sum(p*np.log2(p)))
Hcond=sum(pm[j]*H(post[:,j]) for j in range(5) if pm[j]>0)
HM=H(pm)
# Each target group carries joint distribution (x0,current xt), normalized.
def groups(policy):
 out=[]
 if policy=="ref":
  for j,m in enumerate(MVALS):
   z=int(mapids[j]); q=np.zeros((NS,NS))
   q[:,MAG==m] = joint[:,MAG==m] # masked joint
   mass=q.sum()
   if mass>0: out.append((z,mass,q/mass))
 else:
  for z in range(NS):
   q=np.zeros((NS,NS))
   q[z,:]=joint[z,:]
   mass=q.sum()
   if mass>0: out.append((z,mass,q/mass))
 return out
GREF=groups("ref"); GOBS=groups("obs")
def run(policy,A,profile="linear"):
 gs=GREF if policy=="ref" else GOBS
 totalW=0.; totalF=0.; totalD=0.; weighted_initF=0.; weighted_finalF=0.
 for target,mass,q in gs:
  spin=S[target]
  pcur=q.sum(axis=0)
  # initial nonequilibrium free energy of current spin state conditional on target
  u0=Ebase
  F0=float(np.dot(pcur,u0)+KT*np.sum(pcur[pcur>0]*np.log(pcur[pcur>0])))
  W=0.
  dt=TCTRL/NSTEPS
  for k in range(NSTEPS):
   u=(k+1)/NSTEPS
   if profile=="ramp_hold": u=min(2*u,1.)
   h_old=A*spin*((k/NSTEPS) if profile=="linear" else min(2*k/NSTEPS,1.))
   h_new=A*spin*u
   dh=h_new-h_old
   pcur=q.sum(axis=0)
   W += -float(np.dot(pcur,S@dh))
   P=expm(Qmat(h_new)*dt)
   q=q@P
  pfin=q.sum(axis=0)
  Efin=Ebase-S@ (A*spin)
  Ffin=float(np.dot(pfin,Efin)+KT*np.sum(pfin[pfin>0]*np.log(pfin[pfin>0])))
  diss=W-(Ffin-F0)
  fid=float(sum(q[x,x] for x in range(NS)))
  totalW+=mass*W; totalF+=mass*fid; totalD+=mass*diss
  weighted_initF+=mass*F0; weighted_finalF+=mass*Ffin
 return {"fidelity":totalF,"mean_actuator_work":totalW,"dissipated_work":totalD,
         "delta_nonequilibrium_free_energy":weighted_finalF-weighted_initF}
def optimize(policy,profile):
 # Broad scan then identify feasible closest-to-target bracket; interpolate root on first crossing.
 grid=np.linspace(0,12,25)
 vals=[run(policy,float(a),profile) for a in grid]
 feasible=[i for i,v in enumerate(vals) if v["fidelity"]>=TARGET]
 if feasible:
  i=feasible[0]
  if i==0: aopt=grid[0]
  else:
   lo,hi=grid[i-1],grid[i]
   aopt=brentq(lambda a:run(policy,a,profile)["fidelity"]-TARGET,lo,hi,xtol=2e-5)
  res=run(policy,aopt,profile)
  res.update({"amplitude":float(aopt),"target_fidelity":TARGET,"target_reached":True,
              "scan_max_fidelity":max(v["fidelity"] for v in vals),"scan_max_amplitude":float(grid[int(np.argmax([v["fidelity"] for v in vals]))])})
 else:
  i=int(np.argmin([abs(v["fidelity"]-TARGET) for v in vals]))
  res=vals[i].copy(); res.update({"amplitude":float(grid[i]),"target_fidelity":TARGET,"target_reached":False,
       "scan_max_fidelity":max(v["fidelity"] for v in vals),"scan_max_amplitude":float(grid[int(np.argmax([v["fidelity"] for v in vals]))])})
 return res
def main():
 t=time.time()
 out={}
 for profile in ["linear","ramp_hold"]:
  out[profile]={}
  for pol in ["ref","obs"]: out[profile][pol]=optimize(pol,profile)
 # register reset: exact conditional Landauer plus previously validated finite-time dynamic reset
 out["memory"]={"H_X0_given_MT_bits":Hcond,"H_MT_bits":HM,
   "conditional_Landauer_bound":KT*np.log(2)*Hcond,
   "MT_reset_Landauer_bound_if_closed":KT*np.log(2)*HM,
   "dynamic_conditional_register_reset_tau1000":5.364432689279152,
   "dynamic_reset_includes_landscape_preparation":True,
   "write_read_detector_and_barrier_costs":"not modeled; memory total is therefore a lower bound/incomplete ledger"}
 for profile in ["linear","ramp_hold"]:
  r=out[profile]["ref"]; o=out[profile]["obs"]
  out[profile]["comparison"]={
   "fidelity_gap":r["fidelity"]-o["fidelity"],
   "dissipation_saving_before_memory":r["dissipated_work"]-o["dissipated_work"],
   "net_after_conditional_Landauer_bound":r["dissipated_work"]-o["dissipated_work"]-out["memory"]["conditional_Landauer_bound"],
   "net_after_dynamic_reset_tau1000":r["dissipated_work"]-o["dissipated_work"]-out["memory"]["dynamic_conditional_register_reset_tau1000"],
   "net_after_dynamic_reset_plus_MT_reset":r["dissipated_work"]-o["dissipated_work"]-out["memory"]["dynamic_conditional_register_reset_tau1000"]-out["memory"]["MT_reset_Landauer_bound_if_closed"]
  }
 out["method"]={"model":"exact deterministic master equation for 4-spin periodic Ising ring; 16 states; Gibbs prior; Glauber rates; 20-step piecewise-constant local-field protocol",
 "control_families":["linear ramp h_i=A*s_i(target)*t/T","ramp-hold h_i=A*s_i(target)*min(2t/T,1)"],
 "objective":"minimum dissipated work within one-parameter amplitude family subject to exact reconstruction fidelity=45%; root solved by brentq after grid scan",
 "dissipation":"W_actuator - [F_noneq(final,H_final)-F_noneq(initial,H_initial)] with target-conditioned ensemble; target register treated as retained external label",
 "important":"Deterministic ensemble calculation; no Monte Carlo confidence intervals. This is not a global optimum over arbitrary waveforms. The 45% condition is checked on exact master-equation probabilities."}
 out["runtime_seconds"]=time.time()-t
 with open("ising4state_matched_fidelity_optimization_T4.json","w") as f: json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=="__main__": main()
