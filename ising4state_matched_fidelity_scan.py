#!/usr/bin/env python3
import json, time
import numpy as np
import ising4state_reconstruction_4bit_register as base

NSAMP=12000
AMPS=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,8.0,10.0]
TARGET=0.45
def ctrl(x_start,target,A,rng):
    x=int(x_start); work=0.0; dt=base.T_CTRL/base.NSTEPS
    ts=base.SPINS[int(target)].astype(float)
    for k in range(base.NSTEPS):
        ho=ts*(A*k/base.NSTEPS); hn=ts*(A*(k+1)/base.NSTEPS)
        work += -float(np.dot(base.SPINS[x].astype(float),hn-ho))
        x=base.evolve_gillespie(x,hn,dt,rng)
    return work,x
def main():
    t0=time.time()
    _,_,_,pm,post,map_ids,*_=base.build_joint()
    rng=np.random.default_rng(base.SEED)
    x0s=rng.choice(base.NS,size=NSAMP,p=base.P0)
    xts=np.empty(NSAMP,dtype=int); ms=np.empty(NSAMP,dtype=int)
    for k,x0 in enumerate(x0s):
        xt=base.evolve_gillespie(int(x0),np.zeros(base.N),base.T_OBS,rng)
        xts[k]=xt; ms[k]=base.MAG[xt]
    results=[]
    for ai,A in enumerate(AMPS):
        rr=np.random.default_rng(base.SEED+100+ai)
        ro=np.random.default_rng(base.SEED+200+ai)
        fr=np.zeros(NSAMP,dtype=bool); fo=np.zeros(NSAMP,dtype=bool)
        wr=np.zeros(NSAMP); wo=np.zeros(NSAMP)
        for k,(x0,xt,m) in enumerate(zip(x0s,xts,ms)):
            j=int(np.where(base.MVALS==m)[0][0])
            wr[k],xr=ctrl(xt,int(map_ids[j]),A,rr)
            wo[k],xo=ctrl(xt,int(x0),A,ro)
            fr[k]=(xr==x0); fo[k]=(xo==x0)
        results.append({
          "amplitude":A,
          "reference":{"fidelity":float(fr.mean()),"fidelity_se":float(np.sqrt(fr.mean()*(1-fr.mean())/NSAMP)),"mean_signed_work":float(wr.mean()),"work_se":float(wr.std(ddof=1)/np.sqrt(NSAMP))},
          "with_history":{"fidelity":float(fo.mean()),"fidelity_se":float(np.sqrt(fo.mean()*(1-fo.mean())/NSAMP)),"mean_signed_work":float(wo.mean()),"work_se":float(wo.std(ddof=1)/np.sqrt(NSAMP))}
        })
    out={"title":"Matched-fidelity amplitude feasibility scan; not a dissipation optimization","n":NSAMP,"target_fidelity":TARGET,"amplitudes":AMPS,"results":results,"limitations":["The protocol ends with nonzero target-dependent fields, so signed actuator work is not W_diss.","The target is selected using information correlated with X0; a feedback-information correction is required for a thermodynamic dissipation inequality.","This scan only tests one scalar amplitude in a linear-ramp family; it is not global optimal control."]}
    with open("ising4state_matched_fidelity_scan.json","w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
    print("runtime_seconds",time.time()-t0)
if __name__=="__main__": main()
