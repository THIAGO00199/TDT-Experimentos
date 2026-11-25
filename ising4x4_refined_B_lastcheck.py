#!/usr/bin/env python3
import json,numpy as np
import ising4x4_Tobs_curve as m
from ising4x4_refined_B import eval_amp
tobs=4.0; fam="linear"; pol="no_info"; a=3.5; n=20000
map_ids,h,_,_=m.posterior_stats(tobs)
rng=np.random.default_rng(20261201)
x0=rng.choice(m.NS,size=n,p=m.P0); xt=m.observe(x0,tobs,rng); mag=xt.sum(1).astype(int)
r=eval_amp(x0,xt,mag,map_ids,False,a,fam,20261202)
row={"Tobs":tobs,"H":h,"family":fam,"policy":pol,"amplitude":a,**r}
json.dump(row,open("ising4x4_refined_B_lastcheck.json","w"),indent=2)
print(json.dumps(row))
