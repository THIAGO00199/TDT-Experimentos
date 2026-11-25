#!/usr/bin/env python3
"""
TDT v2 vs stochastic thermodynamics benchmark.
Two-level system E0=0, E1=lambda; Glauber rates satisfy local detailed balance.
Exact deterministic master equation; no Monte Carlo.
"""
import json, math, os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

BETA = 1.0
KBT = 1.0
GAMMA = 1.0
LA, LB = -4.0, 4.0
NSEG = 32
DT = 0.25
HOLD = 20.0
NSTARTS = 4
EPS = 1e-14

def peq(lam):
    # Stable logistic 1/(1+exp(beta*lambda)).
    x = BETA * lam
    if x >= 0:
        e = math.exp(-x)
        return e/(1.0+e)
    e = math.exp(x)
    return 1.0/(1.0+e)

def free_energy(lam):
    return -KBT * float(np.logaddexp(0.0, -BETA*lam))

def relax(p0, lam, duration):
    pstar = peq(lam)
    p1 = pstar + (p0-pstar)*math.exp(-GAMMA*duration)
    return p1

def work_from_nodes(nodes, start_lam):
    p = peq(start_lam)
    prev = start_lam
    works = []
    probs_before = []
    for target in nodes:
        dw = p*(target-prev)
        works.append(dw)
        probs_before.append(p)
        p = relax(p, target, DT)
        prev = target
    return float(sum(works)), np.asarray(works), np.asarray(probs_before), p

def nodes_from_logits(z, start, end):
    logits = np.r_[np.asarray(z,dtype=float), 0.0]
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    increments = abs(end-start)*weights/np.sum(weights)
    direction = 1.0 if end > start else -1.0
    return start + direction*np.cumsum(increments)

def objective_logits(z, start, end):
    nodes = nodes_from_logits(z,start,end)
    w,_,_,_ = work_from_nodes(nodes,start)
    return w

def optimize_protocol(start, end, label):
    best = None
    rng = np.random.default_rng(20261113 + (0 if start < end else 77))
    starts = [np.zeros(NSEG-1)]
    for scale in (0.4, 1.0, 2.0):
        starts.append(rng.normal(0.0,scale,size=NSEG-1))
    records=[]
    for k,z0 in enumerate(starts[:NSTARTS]):
        res=minimize(objective_logits,z0,args=(start,end),
                     method="L-BFGS-B",bounds=[(-12.0,12.0)]*(NSEG-1),
                     options={"maxiter":5000,"maxfun":200000,"ftol":1e-14,"gtol":1e-9,"maxls":50})
        records.append({"start":k,"success":bool(res.success),"message":str(res.message),
                        "nit":int(res.nit),"fun":float(res.fun)})
        if best is None or res.fun<best.fun:
            best=res
    nodes=nodes_from_logits(best.x,start,end)
    w,works,probs,pdrive=work_from_nodes(nodes,start)
    return {"label":label,"nodes":nodes,"works":works,"work":w,
            "p_after_drive":pdrive,"optimizer":{"success":bool(best.success),
            "message":str(best.message),"nit":int(best.nit),"fun":float(best.fun),
            "multi_start":records}}

def ep_rate(p, lam):
    p=min(max(float(p),1e-300),1.0-1e-16)
    k01=GAMMA/(1.0+math.exp(BETA*lam))
    k10=GAMMA/(1.0+math.exp(-BETA*lam))
    flux01=k01*(1.0-p)
    flux10=k10*p
    j=flux01-flux10
    if flux01<=0 or flux10<=0:
        return 0.0
    return j*math.log(flux01/flux10)

def segment_ep(p0, lam, duration):
    pstar=peq(lam)
    def integrand(t):
        p=pstar+(p0-pstar)*math.exp(-GAMMA*t)
        return ep_rate(p,lam)
    value,err=quad(integrand,0.0,duration,epsabs=1e-13,epsrel=1e-11,limit=150)
    p1=relax(p0,lam,duration)
    return value,p1,err

def non_eq_free_energy(p,lam):
    p=min(max(float(p),1e-300),1.0-1e-16)
    s=-(p*math.log(p)+(1-p)*math.log1p(-p))
    return p*lam-KBT*s

def simulate_protocol(nodes,start,end,initial_hold=HOLD,final_hold=HOLD):
    p=peq(start)
    ep_total=0.0
    ep_drive=0.0
    ep_err=0.0
    works=[]
    p_before=[]
    prev=start
    # Initial equilibrium hold is included to define a time-symmetric endpoint.
    # Its entropy production is zero.
    for target in nodes:
        dw=p*(float(target)-prev)
        works.append(dw)
        p_before.append(p)
        ep,p,e=segment_ep(p,float(target),DT)
        ep_total+=ep
        ep_drive+=ep
        ep_err+=e
        prev=float(target)
    ep,p,e=segment_ep(p,end,final_hold)
    ep_total+=ep
    ep_err+=e
    W=float(sum(works))
    deltaF=free_energy(end)-free_energy(start)
    Fneq=non_eq_free_energy(p,end)
    deltaFneq=Fneq-free_energy(start)
    tdiss=W-deltaF
    tdiss_neq=W-deltaFneq
    balance_error=tdiss_neq-ep_total*KBT
    return {"work":W,"deltaF":deltaF,"dissipation":tdiss,
            "sigma":ep_total,"T_sigma":KBT*ep_total,
            "p_final":p,"p_eq_final":peq(end),
            "final_probability_error":p-peq(end),
            "deltaF_neq":deltaFneq,"dissipation_neq":tdiss_neq,
            "first_law_entropy_balance_residual":balance_error,
            "quad_error_estimate":ep_err,"works":np.asarray(works),
            "p_before":np.asarray(p_before)}

def tdt_pi(works, reference):
    # Explicit positive operationalization:
    # W_TDT(t)=W_ref + cumulative sum |delta W_control|.
    # Then integral [d ln W_TDT/dt]_+ dt = ln(W_TDT(final)/W_ref).
    tv=float(np.sum(np.abs(works)))
    return KBT*math.log1p(tv/reference),tv

def metrics_for_path(sim):
    refs=[0.1,1.0,10.0] # in kBT units, for regularization sensitivity
    out={}
    for ref in refs:
        pi,tv=tdt_pi(sim["works"],ref*KBT)
        out[str(ref)]={"Pi_TDT":pi,"total_variation_work":tv}
    return out

def main():
    opt_f=optimize_protocol(LA,LB,"forward_A_to_B")
    opt_r=optimize_protocol(LB,LA,"reverse_optimized_B_to_A")
    # Exact time reversal of the independently optimized forward schedule.
    reverse_of_forward=np.r_[opt_f["nodes"][-2::-1],LA]

    paths={}
    paths["forward_opt"]=simulate_protocol(opt_f["nodes"],LA,LB)
    paths["reverse_time_reversed"]=simulate_protocol(reverse_of_forward,LB,LA)
    paths["reverse_independently_optimized"]=simulate_protocol(opt_r["nodes"],LB,LA)

    # Forward W(t) trace and cumulative work, with the same exact dynamics.
    wf=paths["forward_opt"]["works"]
    cumulative=np.cumsum(wf)
    forward_curve=[]
    for i in range(len(wf)):
        forward_curve.append({"jump":i+1,"lambda":float(opt_f["nodes"][i]),
                              "p_before":float(paths["forward_opt"]["p_before"][i]),
                              "delta_work":float(wf[i]),
                              "cumulative_signed_work":float(cumulative[i]),
                              "cumulative_abs_work":float(np.sum(np.abs(wf[:i+1])))})

    # Classical exact identity checks.
    results={
      "model":{"beta":BETA,"kBT":KBT,"gamma":GAMMA,"lambda_A":LA,"lambda_B":LB,
               "segments":NSEG,"dt":DT,"drive_time":NSEG*DT,
               "endpoint_hold":HOLD,"rates":"k01=gamma/(1+exp(beta*lambda)); k10=gamma/(1+exp(-beta*lambda))",
               "initial_condition":"equilibrium at starting lambda",
               "energy_levels":"E0=0; E1=lambda"},
      "optimizer":{"forward":opt_f["optimizer"],"reverse":opt_r["optimizer"],
                   "forward_nodes":opt_f["nodes"].tolist(),
                   "reverse_nodes":opt_r["nodes"].tolist()},
      "paths":{},
      "tdt_regularization_reference_kBT":[0.1,1.0,10.0],
      "forward_work_trace":forward_curve
    }
    for name,path in paths.items():
        results["paths"][name]={k:v for k,v in path.items() if k not in ("works","p_before")}
        results["paths"][name]["tdt_v2_operationalization"]=metrics_for_path(path)
    # Compare the forward Pi against recovery dissipation as a literal TDT prediction.
    Pi_f=metrics_for_path(paths["forward_opt"])
    for rkey in Pi_f:
        Pi_f[rkey]["difference_vs_reverse_dissipation"] = (
            Pi_f[rkey]["Pi_TDT"]-paths["reverse_independently_optimized"]["dissipation"])
        Pi_f[rkey]["difference_vs_reverse_T_sigma"] = (
            Pi_f[rkey]["Pi_TDT"]-paths["reverse_independently_optimized"]["T_sigma"])
    results["forward_Pi_vs_optimal_recovery"]=Pi_f

    out="tdt_v2_two_level_optimal_control.json"
    with open(out,"w") as f: json.dump(results,f,indent=2)
    txt="tdt_v2_two_level_optimal_control.txt"
    with open(txt,"w") as f:
        f.write("TDT v2 / two-level exact master-equation benchmark\n")
        f.write("All energies are in kBT units (kBT=1). No Monte Carlo was used.\n")
        f.write(f"Model: beta={BETA}, gamma={GAMMA}, lambda A={LA}, lambda B={LB}, NSEG={NSEG}, dt={DT}, endpoint hold={HOLD}\n")
        f.write(f"Exact free-energy difference A->B: {free_energy(LB)-free_energy(LA):.12g}\n")
        for label,path in paths.items():
            f.write(f"\nPATH {label}\n")
            for key in ("work","deltaF","dissipation","sigma","T_sigma","p_final","p_eq_final",
                        "final_probability_error","deltaF_neq","dissipation_neq","first_law_entropy_balance_residual"):
                f.write(f"  {key}={path[key]:.15g}\n")
            f.write("  TDT operational Pi (reference in kBT):\n")
            for ref,val in metrics_for_path(path).items():
                f.write(f"    ref={ref}: Pi={val['Pi_TDT']:.12g}, total_variation_work={val['total_variation_work']:.12g}\n")
        f.write("\nOPTIMIZER diagnostics\n")
        f.write("  forward="+json.dumps(opt_f["optimizer"])+"\n")
        f.write("  reverse="+json.dumps(opt_r["optimizer"])+"\n")
        f.write("\nFORWARD Pi compared with independently optimized recovery dissipation\n")
        for ref,val in Pi_f.items():
            f.write(f"  W_ref={ref} kBT: Pi={val['Pi_TDT']:.12g}, recovery W-DeltaF={paths['reverse_independently_optimized']['dissipation']:.12g}, difference={val['difference_vs_reverse_dissipation']:.12g}\n")
        f.write("\nFIRST 10 FORWARD CONTROL POINTS\n")
        for row in forward_curve[:10]:
            f.write("  "+json.dumps(row)+"\n")
    print(open(txt).read())
    print(f"\nWrote {out} and {txt}")

if __name__=="__main__":
    main()
