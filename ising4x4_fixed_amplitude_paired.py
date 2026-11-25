import json, time, math
import numpy as np
import ising4x4_Tobs_curve as m
A=1.5
NSAMP=5000
out={"amplitude":A,"trajectories_per_horizon":NSAMP,"results":[]}
t0=time.time()
for ti,tobs in enumerate(m.TOBS_LIST):
    map_ids,hcond,map_succ,part_err=m.posterior_stats(tobs)
    rng=np.random.default_rng(m.SEED+ti*1000)
    x0=rng.choice(m.NS,size=NSAMP,p=m.P0)
    xt=m.observe(x0,tobs,rng)
    mag=xt.sum(1).astype(int)
    seedbase=m.SEED+800000+ti*10000
    w0,o0=m.run_policy(x0,xt,mag,map_ids,False,A,seedbase+1)
    w1,o1=m.run_policy(x0,xt,mag,map_ids,True,A,seedbase+2)
    d=w0-w1
    se=float(d.std(ddof=1)/np.sqrt(NSAMP))
    p0=float(o0.mean()); p1=float(o1.mean())
    out["results"].append({
      "Tobs":tobs,"H_X0_given_M_T_bits":hcond,
      "fidelity_no_info":p0,"fidelity_with_X0":p1,
      "work_no_info":float(w0.mean()),"work_with_X0":float(w1.mean()),
      "delta_work_no_info_minus_with_X0":float(d.mean()),
      "delta_work_paired_se":se,
      "delta_work_ci95":[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],
      "fidelity_se_no_info":float(np.sqrt(p0*(1-p0)/NSAMP)),
      "fidelity_se_with_X0":float(np.sqrt(p1*(1-p1)/NSAMP))
    })
    with open("ising4x4_fixed_amplitude_paired.json","w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out["results"][-1]),flush=True)
out["runtime_seconds"]=time.time()-t0
with open("ising4x4_fixed_amplitude_paired.json","w") as f: json.dump(out,f,indent=2)
print("FINAL",json.dumps(out,indent=2),flush=True)
