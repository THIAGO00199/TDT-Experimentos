#!/usr/bin/env python3
"""
Demon da Manutencao — 8-spin Ising ring.
Exact 256-state Glauber master equation with time-dependent local fields u_i(t).
Controls vanish at t=0,T; Hamiltonian at both endpoints is the original hf=1 model.
Optimizes tracking KL + common field-amplitude/smoothness penalties.
Computes work from -sum_i dot(u_i)<s_i>, and Schnakenberg EP independently.
"""
import json, time
import numpy as np
from scipy.optimize import root, minimize
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

# ----------------------- model and initial distributions -----------------------
N=8; NS=2**N; J=1.0; KT=1.0; BETA=1.0; GAMMA=1.0; HF=1.0
T_HORIZON=4.0
N_SEG=8
N_SUB=20
DT_SEG=T_HORIZON/N_SEG
DT=DT_SEG/N_SUB
ALPHA=1.5
LAMBDA_U=1.0e-2
LAMBDA_DU=2.0e-4
U_BOUND=3.0

ids=np.arange(NS)
S=(2*((ids[:,None]>>np.arange(N))&1)-1).astype(float)
M=S.sum(axis=1)
bonds=np.stack([S[:,i]*S[:,(i+1)%N] for i in range(N)],axis=1)
Eint=-J*bonds.sum(axis=1)
H0=Eint-HF*M

def softmax_logw(x):
    x=np.asarray(x,dtype=float); x=x-np.max(x)
    p=np.exp(x); return p/p.sum()

pA=softmax_logw(-BETA*H0/KT)
targetE=float(pA@H0); targetM=float(pA@M)

def pB_from_alpha(alpha):
    def make(theta):
        beta_b, lam=theta
        return softmax_logw(-beta_b*Eint/KT+lam*M+alpha*bonds[:,0])
    def equations(theta):
        p=make(theta)
        return np.array([(p@H0-targetE)/8.0,(p@M-targetM)/8.0])
    sol=root(equations,[0.93,1.0],tol=1e-12)
    if not sol.success and np.linalg.norm(equations(sol.x))>1e-10:
        raise RuntimeError("B distribution constraints did not converge: "+str(sol.message))
    return make(sol.x),sol

pB, solB=pB_from_alpha(ALPHA)

def entropy(p):
    p=np.asarray(p)
    return -float(np.sum(p[p>0]*np.log(p[p>0])))

def fneq(p):
    return float(p@H0-KT*entropy(p))

def corr(p):
    return (p@bonds).tolist()

# JAX constants
S_j=jnp.asarray(S)
M_j=jnp.asarray(M)
H0_j=jnp.asarray(H0)
nei_j=jnp.asarray(ids[:,None] ^ (1<<np.arange(N))[None,:],dtype=jnp.int32)
site_j=jnp.arange(N,dtype=jnp.int32)
spin_j=jnp.asarray(S)
bond_neighbors_j=jnp.asarray(bonds)

def rates_for(u):
    # u_i is an additive local control field; total field hf+u_i.
    left=spin_j[:,(np.arange(N)-1)%N]
    right=spin_j[:,(np.arange(N)+1)%N]
    dE=2.0*spin_j*(J*(left+right)+HF+u[None,:])
    return GAMMA/(1.0+jnp.exp(jnp.clip(BETA*dE/KT,-60.0,60.0)))

def master_rhs(p,u):
    r=rates_for(u)
    r_from_neighbor=r[nei_j,site_j[None,:]]
    incoming=jnp.sum(p[nei_j]*r_from_neighbor,axis=1)
    outgoing=p*jnp.sum(r,axis=1)
    return incoming-outgoing

def sigma_dot(p,u):
    r=rates_for(u)
    pn=p[nei_j]
    rn=r[nei_j,site_j[None,:]]
    forward=p[:,None]*r
    backward=pn*rn
    # Every undirected spin-flip edge occurs twice in this directed sum.
    ratio=jnp.log(jnp.maximum(forward,1e-300)/jnp.maximum(backward,1e-300))
    return 0.5*jnp.sum((forward-backward)*ratio)

def kl_to(p,pt):
    return jnp.sum(p*jnp.log(jnp.maximum(p,1e-300)/jnp.maximum(pt,1e-300)))

def rhs_all(p,u,slope,pt):
    dp=master_rhs(p,u)
    m=p@S_j
    power=-jnp.dot(slope,m)  # work rate on system, H=H0-u.s
    sig=sigma_dot(p,u)
    kl=kl_to(p,pt)
    u2=jnp.sum(u*u)
    accum=jnp.stack([power,jnp.maximum(power,0.0),jnp.maximum(-power,0.0),sig,kl,u2])
    return dp,accum

def simulate(p0,pt,knots):
    # 9 control knots, with zero control at both endpoints.
    p0=jnp.asarray(p0); pt=jnp.asarray(pt); knots=jnp.asarray(knots)
    initial_acc=jnp.zeros((6,),dtype=jnp.float64)
    fractions=jnp.arange(N_SUB,dtype=jnp.float64)/N_SUB
    def segment(carry, pair):
        p,acc=carry
        u0,u1=pair
        slope=(u1-u0)/DT_SEG
        def step(carry2,frac):
            pnow,accnow=carry2
            ua=u0+frac*(u1-u0)
            um=u0+(frac+0.5/N_SUB)*(u1-u0)
            ub=u0+(frac+1.0/N_SUB)*(u1-u0)
            k1p,k1a=rhs_all(pnow,ua,slope,pt)
            k2p,k2a=rhs_all(pnow+0.5*DT*k1p,um,slope,pt)
            k3p,k3a=rhs_all(pnow+0.5*DT*k2p,um,slope,pt)
            k4p,k4a=rhs_all(pnow+DT*k3p,ub,slope,pt)
            pnext=pnow+(DT/6.0)*(k1p+2*k2p+2*k3p+k4p)
            anext=accnow+(DT/6.0)*(k1a+2*k2a+2*k3a+k4a)
            return (pnext,anext),None
        (p,acc),_=lax.scan(step,(p,acc),fractions)
        return (p,acc),None
    (pf,acc),_=lax.scan(segment,(p0,initial_acc),(knots[:-1],knots[1:]))
    w_net,w_in,w_out,sigma,kl_integral,u2_integral=acc
    du2_integral=jnp.sum(jnp.sum((knots[1:]-knots[:-1])**2,axis=1)/DT_SEG)
    klbar=kl_integral/T_HORIZON
    u2bar=u2_integral/T_HORIZON
    du2bar=du2_integral/T_HORIZON
    objective=klbar+LAMBDA_U*u2bar+LAMBDA_DU*du2bar
    return objective, (pf,acc,klbar,u2bar,du2bar)

def make_objective(p0,pt):
    def obj(x):
        knots=jnp.zeros((N_SEG+1,N),dtype=jnp.float64)
        interior=jnp.reshape(x,(N_SEG-1,N))
        knots=knots.at[1:-1,:].set(interior)
        val,_=simulate(p0,pt,knots)
        return val
    return jax.jit(jax.value_and_grad(obj)), jax.jit(obj)

def evaluate_control(p0,pt,knots):
    _,aux=simulate(p0,pt,jnp.asarray(knots))
    pf,acc,klbar,u2bar,du2bar=aux
    pfn=np.asarray(pf)
    wnet,win,wout,sigma,klint,u2int=np.asarray(acc,dtype=float)
    deltaF=fneq(pfn)-fneq(p0)
    bal_res=wnet-deltaF-KT*sigma
    return {
      "W_net":float(wnet),"W_input_gross":float(win),"W_recovered_gross":float(wout),
      "W_net_per_time":float(wnet/T_HORIZON),
      "sigma_Schnakenberg_nats":float(sigma),
      "delta_Fneq":float(deltaF),"classical_balance_residual":float(bal_res),
      "KL_time_average_nats":float(klbar),
      "KL_final_nats":float(np.sum(pfn*np.log(np.maximum(pfn,1e-300)/pt))),
      "L1_final_to_target":float(np.abs(pfn-pt).sum()),
      "final_mean_E_H0":float(pfn@H0),"final_mean_M":float(pfn@M),
      "final_bond_correlations":corr(pfn),
      "final_Fneq":fneq(pfn),
      "mean_field_square":float(u2bar),"mean_field_slew_square":float(du2bar),
      "final_distribution":pfn.tolist()
    }

def no_control_knots():
    return np.zeros((N_SEG+1,N),dtype=float)

def optimize_controller(label,p0,pt,maxiter=120):
    valgrad,obj=make_objective(p0,pt)
    x0=np.zeros((N_SEG-1)*N,dtype=float)
    def fun(x):
        v,g=valgrad(jnp.asarray(x))
        return float(v),np.asarray(g,dtype=float)
    result=minimize(fun,x0,jac=True,method="L-BFGS-B",
       bounds=[(-U_BOUND,U_BOUND)]*len(x0),
       options={"maxiter":maxiter,"ftol":1e-12,"gtol":2e-7,"maxls":30,"maxcor":12})
    knots=np.zeros((N_SEG+1,N),dtype=float)
    knots[1:-1]=result.x.reshape((N_SEG-1,N))
    met=evaluate_control(p0,pt,knots)
    met.update({"label":label,"optimizer_success":bool(result.success),
       "optimizer_message":str(result.message),"iterations":int(result.nit),
       "function_evaluations":int(result.nfev),"objective":float(result.fun),
       "knots":knots.tolist()})
    return met

def short(m):
    keys=["W_net","W_input_gross","W_recovered_gross","W_net_per_time",
          "sigma_Schnakenberg_nats","delta_Fneq","classical_balance_residual",
          "KL_time_average_nats","KL_final_nats","L1_final_to_target",
          "final_mean_E_H0","final_mean_M","mean_field_square","mean_field_slew_square",
          "optimizer_success","iterations","objective"]
    return {k:m[k] for k in keys if k in m}

def main():
    tic=time.time()
    # Uncontrolled trajectories: both A and B.
    z=no_control_knots()
    A0=evaluate_control(pA,pA,z)
    B0=evaluate_control(pB,pB,z)
    # Optimized controllers: same objective, horizon, bounds, and regularization.
    Aopt=optimize_controller("A_Gibbs",pA,pA,maxiter=60)
    Bopt=optimize_controller("B_correlated",pB,pB,maxiter=160)

    # Classical null for the differential work, computed independently as
    # Delta F_B - Delta F_A + kBT*(Sigma_B-Sigma_A), using directly integrated EP.
    Wclassic=(Bopt["delta_Fneq"]-Aopt["delta_Fneq"])+KT*(
        Bopt["sigma_Schnakenberg_nats"]-Aopt["sigma_Schnakenberg_nats"])
    Wdiff=Bopt["W_net"]-Aopt["W_net"]
    PiDin=Wdiff-Wclassic

    out={
      "model":{"N":N,"states":NS,"geometry":"periodic Ising ring",
        "J":J,"kBT":KT,"beta":BETA,"gamma":GAMMA,"base_field_hf":HF,
        "dynamics":"exact finite-state Glauber CTMC master equation",
        "control":"8 additive local fields u_i(t), piecewise-linear over 8 segments; u_i(0)=u_i(T)=0",
        "horizon":T_HORIZON,"segments":N_SEG,"rk4_substeps_per_segment":N_SUB,
        "max_abs_control_field":U_BOUND,
        "objective":"time-average KL(p(t)||p_target) + lambda_u*time-average sum(u_i^2) + lambda_du*time-average sum(dot(u_i)^2)",
        "lambda_u":LAMBDA_U,"lambda_du":LAMBDA_DU,
        "work_convention":"Wdot=-sum_i dot(u_i)<s_i>; positive work on system",
        "classical_null":"DeltaFneq difference + kBT times independently integrated Schnakenberg EP difference"},
      "initial_A":{"mean_E_H0":targetE,"mean_M":targetM,"Fneq":fneq(pA),
        "entropy_nats":entropy(pA),"bond_correlations":corr(pA)},
      "initial_B":{"alpha":ALPHA,"beta_B":float(solB.x[0]),"lambda_B":float(solB.x[1]),
        "mean_E_H0":float(pB@H0),"mean_M":float(pB@M),"Fneq":fneq(pB),
        "delta_Fneq_vs_A":fneq(pB)-fneq(pA),"KL_BA":float(np.sum(pB*np.log(pB/pA))),
        "L1_BA":float(np.abs(pB-pA).sum()),"entropy_nats":entropy(pB),
        "bond_correlations":corr(pB)},
      "uncontrolled_A":short(A0),"uncontrolled_B":short(B0),
      "optimized_A":short(Aopt),"optimized_B":short(Bopt),
      "work_difference_B_minus_A":float(Wdiff),
      "classical_work_difference_prediction":float(Wclassic),
      "PiDin_residual":float(PiDin),
      "full_runs":{"uncontrolled_A":A0,"uncontrolled_B":B0,
        "optimized_A":Aopt,"optimized_B":Bopt},
      "elapsed_seconds":time.time()-tic,
      "interpretation":"This is an exact-model simulation, not a physical pulse preparation. B is initialized as the exact mathematical correlated distribution. Any PiDin residual is a numerical/model-consistency residual, not independent evidence for TDT. Check whether the field-only controller can track B and how much net/gross work it uses."
    }
    path="/home/iknowyou/TDT_Experimentos/demon_manutencao_8spin_resultado.json"
    with open(path,"w") as f: json.dump(out,f,indent=2)
    txt="/home/iknowyou/TDT_Experimentos/demon_manutencao_8spin_resultado.txt"
    with open(txt,"w") as f:
      f.write("DEMONIO DA MANUTENCAO — RESULTADOS\n")
      f.write(json.dumps({k:v for k,v in out.items() if k!="full_runs"},indent=2))
      f.write("\n")
    print("INITIAL",json.dumps({"A":out["initial_A"],"B":out["initial_B"]},indent=2),flush=True)
    print("UNCONTROLLED_A",json.dumps(short(A0)),flush=True)
    print("UNCONTROLLED_B",json.dumps(short(B0)),flush=True)
    print("OPTIMIZED_A",json.dumps(short(Aopt)),flush=True)
    print("OPTIMIZED_B",json.dumps(short(Bopt)),flush=True)
    print("DIFFERENTIAL",json.dumps({"W_B_minus_W_A":Wdiff,"Wclassic":Wclassic,"PiDin":PiDin}),flush=True)
    print("SAVED",path,txt,"ELAPSED",time.time()-tic,flush=True)

if __name__=="__main__":
    main()
