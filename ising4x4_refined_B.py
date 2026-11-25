#!/usr/bin/env python3
import json, time, math
import numpy as np
import ising4x4_Tobs_curve as m

TOBS=m.TOBS_LIST[3:]
AMPS=np.round(np.arange(1.0,4.0001,0.1),2).tolist()
NS_COARSE=1000
NS_FINE=20000
TARGET=0.24
FAMILIES=("linear","ramp_hold")
SEED=20261120

def wilson(k,n,z=1.959963984540054):
    p=k/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]

def profile(fam):
    k=np.arange(1,m.NSTEPS+1,dtype=float)/m.NSTEPS
    if fam=="linear": return k
    return np.minimum(2*k,1.0)

def run_policy(x0,xt,mag,map_ids,info,amp,fam,seed):
    rng=np.random.default_rng(seed)
    works=np.zeros(len(x0),dtype=np.float64)
    ok=np.zeros(len(x0),dtype=np.bool_)
    g=profile(fam)
    dt=m.TCTRL/m.NSTEPS
    for n in range(len(x0)):
        s=xt[n].copy()
        target=m.SPINS[int(x0[n])] if info else m.SPINS[int(map_ids[(int(mag[n])+16)//2])]
        target=target.astype(float)
        for k in range(m.NSTEPS):
            old=amp*(0. if k==0 else g[k-1])*target
            new=amp*g[k]*target
            works[n]-=float(np.dot(s,new-old))
            rr=m.rates(s,new)
            total=float(rr.sum())
            t=0.
            while total>0:
                wait=rng.exponential(1/total)
                if t+wait>=dt: break
                z=rng.random()*total
                i=int(np.searchsorted(np.cumsum(rr),z,side="right"))
                if i>=m.N:i=m.N-1
                s[i]*=-1
                t+=wait
                rr=m.rates(s,new); total=float(rr.sum())
        ok[n]=np.array_equal(s,m.SPINS[int(x0[n])])
    return works,ok

def eval_amp(x0,xt,mag,map_ids,info,amp,fam,seed):
    w,o=run_policy(x0,xt,mag,map_ids,info,amp,fam,seed)
    k=int(o.sum()); n=len(o); p=k/n
    return {"k_success":k,"n":n,"fidelity":p,"fidelity_ci95_wilson":wilson(k,n),
            "mean_work":float(w.mean()),"work_se":float(w.std(ddof=1)/np.sqrt(n))}

def costs(amp,fam):
    g=profile(fam)
    c1=m.N*m.TCTRL*amp*float(np.mean(np.abs(g)))
    c2=m.N*m.TCTRL*amp*amp*float(np.mean(g*g))
    return {"C1_integral_abs_field":c1,"C2_integral_squared_field":c2}

def main():
    t0=time.time()
    out={"model":{"system":"2D Ising 4x4","prior":"Gibbs beta=.4, h=0",
      "dynamics":"single-spin Glauber CTMC, Gillespie","Tctrl":m.TCTRL,
      "fidelity_target":TARGET,"amplitude_grid":AMPS,"coarse_trajectories":NS_COARSE,
      "fine_trajectories":NS_FINE,"fine_rule":"20k independent trajectories at the coarse threshold and its adjacent +/-0.1 grid points",
      "families":{"linear":"g(t)=t","ramp_hold":"g(t)=min(2t,1), ramp during first half then hold"},
      "seed":SEED},
      "E0_status":"NOT PRE-REGISTERABLE FROM CURRENT TDT v3 SPECIFICATION: no independent derivation/value in cost-proxy units. No fitted E0 will be treated as confirmatory.",
      "results":json.load(open("ising4x4_refined_B_partial.json"))["results"] if __import__("os").path.exists("ising4x4_refined_B_partial.json") else []}
    for ti,tobs in enumerate(TOBS,start=3):
        map_ids,hcond,map_succ,part_err=m.posterior_stats(tobs)
        rng=np.random.default_rng(SEED+ti*10000)
        x0=rng.choice(m.NS,size=NS_COARSE,p=m.P0)
        xt=m.observe(x0,tobs,rng)
        mag=xt.sum(1).astype(int)
        # Fine data use an independently sampled, larger test cohort.
        rngf=np.random.default_rng(SEED+500000+ti*10000)
        x0f=rngf.choice(m.NS,size=NS_FINE,p=m.P0)
        xtf=m.observe(x0f,tobs,rngf)
        magf=xtf.sum(1).astype(int)
        entry={"Tobs":tobs,"H_X0_given_M_T_bits":hcond,
          "exact_MAP_guess_probability":map_succ,"partition_error":part_err,"families":{}}
        for fi,fam in enumerate(FAMILIES):
            coarse={}
            for ai,A in enumerate(AMPS):
                a0=eval_amp(x0,xt,mag,map_ids,False,A,fam,SEED+ti*1000000+fi*100000+ai*100+1)
                a1=eval_amp(x0,xt,mag,map_ids,True,A,fam,SEED+ti*1000000+fi*100000+ai*100+2)
                coarse[round(A,2)]={"no_info":a0,"with_X0":a1}
            selected={}
            fine={}
            for policy in ("no_info","with_X0"):
                feasible=[A for A in AMPS if coarse[round(A,2)][policy]["fidelity_ci95_wilson"][0]>=TARGET]
                a_star=min(feasible) if feasible else None
                selected[policy]={"coarse_min_feasible":a_star}
                if a_star is None:
                    selected[policy]["fine_min_feasible"]=None
                    fine[policy]=None
                    continue
                idx=AMPS.index(a_star)
                candidates=AMPS[max(0,idx-1):min(len(AMPS),idx+2)]
                fine[policy]={}
                for aj,A in enumerate(candidates):
                    res=eval_amp(x0f,xtf,magf,map_ids,policy=="with_X0",A,fam,
                       SEED+9000000+ti*1000000+fi*100000+ (0 if policy=="no_info" else 50000)+aj*100+1)
                    fine[policy][str(round(A,2))]={**res,**costs(A,fam)}
                fine_feasible=[float(a) for a,r in fine[policy].items() if r["fidelity_ci95_wilson"][0]>=TARGET]
                selected[policy]["fine_min_feasible"]=min(fine_feasible) if fine_feasible else None
            delta=None
            a0=selected["no_info"]["fine_min_feasible"]
            a1=selected["with_X0"]["fine_min_feasible"]
            if a0 is not None and a1 is not None:
                c0=costs(a0,fam); c1=costs(a1,fam)
                delta={"amplitude_no_info":a0,"amplitude_with_X0":a1,
                  "delta_C1_no_info_minus_with_X0":c0["C1_integral_abs_field"]-c1["C1_integral_abs_field"],
                  "delta_C2_no_info_minus_with_X0":c0["C2_integral_squared_field"]-c1["C2_integral_squared_field"],
                  "cost_no_info":c0,"cost_with_X0":c1}
            entry["families"][fam]={"coarse_scan":coarse,"selected":selected,"fine_validation":fine,"delta_cost":delta}
            print(json.dumps({"Tobs":tobs,"family":fam,"selected":selected,"delta_cost":delta}),flush=True)
        out["results"].append(entry)
        with open("ising4x4_refined_B_partial.json","w") as f: json.dump(out,f,indent=2)
    out["runtime_seconds"]=time.time()-t0
    with open("ising4x4_refined_B.json","w") as f: json.dump(out,f,indent=2)
    print("FINAL_SUMMARY")
    for r in out["results"]:
        print(json.dumps({"Tobs":r["Tobs"],"H":r["H_X0_given_M_T_bits"],
          "families":{k:{"selected":v["selected"],"delta_cost":v["delta_cost"]} for k,v in r["families"].items()}}),flush=True)
    print("RUNTIME",out["runtime_seconds"],flush=True)

if __name__=="__main__": main()
