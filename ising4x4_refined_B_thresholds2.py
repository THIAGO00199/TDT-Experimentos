#!/usr/bin/env python3
import json
import numpy as np
import ising4x4_Tobs_curve as m
from ising4x4_refined_B import eval_amp
NS=20000
SEED=20261129
CASES=[
 (0.5,"linear","no_info",[1.0]),
 (1.0,"linear","no_info",[2.3]),
 (1.0,"linear","with_X0",[1.5]),
 (1.0,"ramp_hold","no_info",[1.0]),
 (2.0,"linear","with_X0",[3.8]),
 (2.0,"ramp_hold","with_X0",[2.3]),
 (4.0,"ramp_hold","with_X0",[3.0]),
]
def main():
 out={"n":NS,"target":.24,"results":[],"note":"Further downward threshold bracketing on the 0.1 amplitude grid; independent samples."}
 for ci,(tobs,fam,pol,amps) in enumerate(CASES):
  map_ids,hcond,_,_=m.posterior_stats(tobs)
  rng=np.random.default_rng(SEED+ci*10000)
  x0=rng.choice(m.NS,size=NS,p=m.P0); xt=m.observe(x0,tobs,rng); mag=xt.sum(1).astype(int)
  for ai,a in enumerate(amps):
   r=eval_amp(x0,xt,mag,map_ids,pol=="with_X0",a,fam,SEED+ci*10000+ai+500)
   row={"Tobs":tobs,"H":hcond,"family":fam,"policy":pol,"amplitude":a,**r}
   out["results"].append(row); print(json.dumps(row),flush=True)
 with open("ising4x4_refined_B_thresholds2.json","w") as f: json.dump(out,f,indent=2)
 print("THRESHOLD_SUPPLEMENT2_DONE",flush=True)
if __name__=="__main__": main()
