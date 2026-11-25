import numpy as np
from scipy.integrate import solve_ivp
beta=0.4; kT=1/beta; nu=1.0; eps_f=15.0; B=20.0
k01=nu*np.exp(-beta*B)
tau_relax=1/(2*k01)
dF=kT*np.log(2)-kT*np.log(1+np.exp(-beta*eps_f))
def calc(eta):
    tau=eta*tau_relax
    deps=eps_f/tau
    def rhs(t,y):
        eps=deps*t
        k10=nu*np.exp(-beta*(B-eps))
        return [k01*(1-y[0])-k10*y[0]]
    ts=np.linspace(0,tau,20001)
    sol=solve_ivp(rhs,(0,tau),[.5],t_eval=ts,method="DOP853",rtol=1e-11,atol=1e-13)
    W=np.trapezoid(sol.y[0]*deps,ts)
    return W, W-dF, sol.y[0,-1], tau
print("REFINEMENT eta | W | W-DeltaF | p1_final | tau")
for eta in [0.01,1,10,100,1000,10000]:
    W,D,pf,tau=calc(eta)
    print(f"{eta:g} {W:.10f} {D:.10f} {pf:.10f} {tau:.6g}")
print(f"DeltaF={dF:.10f}; Landauer={kT*np.log(2):.10f}")
