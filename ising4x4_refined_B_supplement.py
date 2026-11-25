#!/usr/bin/env python3
import json, math, time
import numpy as np
import ising4x4_Tobs_curve as m
from ising4x4_refined_B import eval_amp
NS=20000
SEED=20261127
CASES=[
 (0.5,"linear","no_info",[1.3]),
 (1.0,"linear","no_info",[2.6,2.7]),
 (1.0,"linear","with_X0",[1.8]),
 (1.0,"ramp_hold","no_info",[1.1]),
 (2.0,"linear","no_info",[3.2]),
 (2.0,"linear","with_X0",[4.0]),
 (2.0,"ramp_hold","no_info",[2.6]),
 (4.0,"linear","with_X0",[3.5]),
 (4.0,"ramp_hold","no_info",[3.2]),
]
def main():
 out={"n":NS,"target":.24,"results":[],"note":"Independent 20k validation at coarse boundary or best-in-grid candidates; not a continuous/global optimization."}
 for ci,(tobs,fam,pol,amps) in enumerate(CASES):
  map_ids,hcond,_,_=m.posterior_stats(tobs)
  rng=np.random.default_rng(SEED+ci*10000)
  x0=rng.choice(m.NS,size=NS,p=m.P0)
  xt=m.observe(x0,tobs,rng)
  mag=xt.sum(1).astype(int)
  for ai,a in enumerate(amps):
   r=eval_amp(x0,xt,mag,map_ids,pol=="with_X0",a,fam,SEED+ci*10000+ai+500)
   row={"Tobs":tobs,"H":hcond,"family":fam,"policy":pol,"amplitude":a,**r}
   out["results"].append(row)
   print(json.dumps(row),flush=True)
 with open("ising4x4_refined_B_supplement.json","w") as f: json.dump(out,f,indent=2)
 print("SUPPLEMENT_DONE",flush=True)
if __name__=="__main__": main()
