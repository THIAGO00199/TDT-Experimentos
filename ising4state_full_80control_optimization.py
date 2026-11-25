import os
os.environ["JAX_ENABLE_X64"]="true"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import json,time
import numpy as np, jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
from jax import jit,grad,lax
from scipy.optimize import minimize
from scipy.linalg import expm
BETA=.4; KT=2.5; J=1.; GAMMA=1.; TOBS=1.; TCTRL=4.; NSTEPS=20; DT=TCTRL/NSTEPS; SUB=4; DS=DT/SUB
N=4; NS=16
ids=np.arange(NS); bits=((ids[:,None]>>np.arange(N))&1); S=(2*bits-1).astype(float); MAG=S.sum(1)
Ebase=np.array([-J*sum(S[x,i]*S[x,(i+1)%N] for i in range(N)) for x in range(NS)])
P0=np.exp(-BETA*(Ebase-Ebase.min())); P0/=P0.sum()
def qnp(h):
 Q=np.zeros((NS,NS))
 for x in range(NS):
  for i in range(4):
   loc=S[x,(i-1)%4]+S[x,(i+1)%4]; de=2*S[x,i]*(J*loc+h[i])
   Q[x,x^(1<<i)]+=GAMMA/(1+np.exp(np.clip(BETA*de,-700,700)))
  Q[x,x]=-Q[x].sum()
 return Q
joint=P0[:,None]*expm(qnp(np.zeros(4))*TOBS)
M=np.array([-4,-2,0,2,4]); pm=np.array([joint[:,MAG==m].sum() for m in M])
post=np.array([joint[:,MAG==m].sum(1)/pm[j] for j,m in enumerate(M)]); maps=np.argmax(post,axis=1)
def groups(pol):
 if pol=="ref": return maps,pm,np.array([joint[:,MAG==m].sum(1)/pm[j] for j,m in enumerate(M)])
 return np.arange(16),P0,np.array([joint[x]/P0[x] for x in range(16)])
def makef(pol):
 tg,w,pi=groups(pol); G=len(tg); tg=jnp.asarray(tg); w=jnp.asarray(w); pi=jnp.asarray(pi); Sj=jnp.asarray(S); Eb=jnp.asarray(Ebase)
 def qone(h):
  rr=[]
  for x in range(16):
   vals=[]
   for i in range(4):
    loc=Sj[x,(i-1)%4]+Sj[x,(i+1)%4]; de=2*Sj[x,i]*(J*loc+h[i])
    vals.append(GAMMA/(1+jnp.exp(BETA*de)))
   rr.append(jnp.stack(vals))
  rr=jnp.stack(rr); Q=jnp.zeros((16,16))
  for i in range(4): Q=Q.at[jnp.arange(16),jnp.arange(16)^(1<<i)].add(rr[:,i])
  return Q.at[jnp.arange(16),jnp.arange(16)].set(-jnp.sum(Q,axis=1))
 qb=jax.vmap(qone)
 def metrics(x):
  A=x.reshape((20,4))
  def step(carry,a):
   p,W,prev=carry
   h=a[None,:]*Sj[tg]; hp=prev[None,:]*Sj[tg]
   W=W-jnp.sum(w*jnp.sum((p@Sj)*(h-hp),axis=1))
   Qs=qb(h)
   def rk(_,pp):
    k1=jnp.einsum("gi,gij->gj",pp,Qs)
    k2=jnp.einsum("gi,gij->gj",pp+.5*DS*k1,Qs)
    k3=jnp.einsum("gi,gij->gj",pp+.5*DS*k2,Qs)
    k4=jnp.einsum("gi,gij->gj",pp+DS*k3,Qs)
    return pp+(DS/6)*(k1+2*k2+2*k3+k4)
   p=lax.fori_loop(0,SUB,rk,p)
   return (p,W,a),None
  (p,W,prev),_=lax.scan(step,(pi,jnp.array(0.),jnp.zeros(4)),A)
  W=W+jnp.sum(w*jnp.sum((p@Sj)*(prev[None,:]*Sj[tg]),axis=1))
  def nf(pp): return pp@Eb+KT*jnp.sum(jnp.where(pp>0,pp*jnp.log(jnp.maximum(pp,1e-300)),0),axis=1)
  dF=jnp.sum(w*(nf(p)-nf(pi)))
  return W-dF,jnp.sum(w*p[jnp.arange(G),tg]),W,dF
 @jit
 def obj(x):
  d,f,_,_=metrics(x); return d+5000*jnp.maximum(0.,.45-f)**2
 @jit
 def con(x): return metrics(x)[1]-.45
 return obj,con,metrics
def solve(pol,seed):
 obj,con,met=makef(pol); rng=np.random.default_rng(seed)
 starts=[np.zeros(80),rng.normal(0,.5,80)]
 out=[]
 for i,x0 in enumerate(starts):
  res=minimize(lambda x:float(obj(jnp.asarray(x))),x0,jac=lambda x:np.array(grad(obj)(jnp.asarray(x))),
   constraints=[{"type":"ineq","fun":lambda x:float(con(jnp.asarray(x))),"jac":lambda x:np.array(grad(con)(jnp.asarray(x)))}],
   method="SLSQP",bounds=[(-12,12)]*80,options={"maxiter":120,"ftol":1e-7,"disp":False})
  d,f,W,dF=map(float,met(jnp.asarray(res.x)))
  out.append({"start":i,"success":bool(res.success),"message":str(res.message),"iterations":int(res.nit),
   "dissipated_work":d,"fidelity":f,"actuator_work":W,"deltaF":dF,"max_abs_control":float(np.max(np.abs(res.x))),
   "controls":res.x.reshape(20,4).tolist()})
 feas=[z for z in out if z["fidelity"]>=.45-1e-7]
 best=min(feas,key=lambda z:z["dissipated_work"]) if feas else min(out,key=lambda z:abs(z["fidelity"]-.45))
 return {"best":best,"starts":out}
def main():
 t=time.time()
 out={"model":"4-spin periodic Ising ring; 16-state master equation integrated by RK4 (4 substeps per control interval); Glauber; Tobs=1; Tctrl=4; 20 intervals",
 "control":"80 optimized coefficients a[k,i], h_i(k)=a[k,i]*s_i(target), bounded [-12,12]; final field switched to zero",
 "objective":"SLSQP with JAX gradients; minimize W_diss subject to exact-in-integrator fidelity >=45%; two initializations per policy",
 "numerical_check":"RK4 substeps=4, dt_sub=0.05; validate best candidates with scipy expm in a separate high-accuracy check",
 "reference":solve("ref",20261113),"informed":solve("obs",20261114)}
 r=out["reference"]["best"]; o=out["informed"]["best"]; mem=5.364432689279152
 out["comparison"]={"saving_before_memory":r["dissipated_work"]-o["dissipated_work"],
  "net_after_dynamic_conditional_reset":r["dissipated_work"]-o["dissipated_work"]-mem,
  "conditional_Landauer":5.22354479168517,"dynamic_reset":mem,"MT_reset_if_closed":4.012151798428081,
  "net_if_MT_also_reset":r["dissipated_work"]-o["dissipated_work"]-mem-4.012151798428081}
 out["runtime_seconds"]=time.time()-t
 with open("ising4state_full_80control_optimization.json","w") as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=="__main__":main()
