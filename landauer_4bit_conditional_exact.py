import numpy as np, json
from scipy.linalg import expm

beta=0.4
kT=1/beta
J=1.0
h=0.0
gamma=1.0
T=1.0
n=4
N=2**n

def spins(x):
    return np.array([1 if (x>>i)&1 else -1 for i in range(n)], dtype=int)

S=np.array([spins(x) for x in range(N)])
E=np.array([-J*sum(S[x,i]*S[x,(i+1)%n] for i in range(n))-h*np.sum(S[x]) for x in range(N)])
p0=np.exp(-beta*E); p0/=p0.sum()

# Row-generator Q: Q[x,y] is rate x -> y.
Q=np.zeros((N,N))
for x in range(N):
    for i in range(n):
        y=x^(1<<i)
        dE=E[y]-E[x]
        rate=gamma/(1+np.exp(beta*dE))
        Q[x,y]+=rate
    Q[x,x]=-Q[x].sum()

P=expm(Q*T)
# Clamp only tiny roundoff; verify stochasticity.
P[np.abs(P)<1e-16]=0
row_err=float(np.max(np.abs(P.sum(axis=1)-1)))
minP=float(P.min())
if minP < -1e-12: raise RuntimeError(f"negative transition probability {minP}")
P=np.maximum(P,0); P/=P.sum(axis=1,keepdims=True)

# Joint distribution p(X0=x, M_T=m), with M_T final total magnetization.
mvals=np.array([int(np.sum(s)) for s in S])
unique_m=sorted(set(mvals.tolist()))
joint=np.zeros((N,len(unique_m)))
for x in range(N):
    for y in range(N):
        j=unique_m.index(int(mvals[y]))
        joint[x,j]+=p0[x]*P[x,y]

px=joint.sum(axis=1)
pm=joint.sum(axis=0)
def H(p):
    p=p[p>0]
    return float(-np.sum(p*np.log2(p)))
HX=H(px); HM=H(pm); Hjoint=H(joint.ravel())
Hcond=Hjoint-HM
I=HX-Hcond
# Conditional entropy calculated directly as cross-check.
Hcond_direct=0.0
for j in range(len(unique_m)):
    if pm[j]>0: Hcond_direct += pm[j]*H(joint[:,j]/pm[j])
# Ideal Landauer bounds, energy units.
W_no=kT*np.log(2)*HX
W_cond=kT*np.log(2)*Hcond
saving=kT*np.log(2)*I
W_M_erase=kT*np.log(2)*HM
net_after_M_reset=W_cond+W_M_erase
# Exact equilibrium free energy and detailed balance checks.
dbmax=0.0
for x in range(N):
    for y in range(N):
        if x!=y and Q[x,y]>0 and Q[y,x]>0:
            lhs=p0[x]*Q[x,y]
            rhs=p0[y]*Q[y,x]
            dbmax=max(dbmax,abs(lhs-rhs))
out={
 "model":"4-spin periodic Ising ring; X0 is 4-bit microstate; M_T is final total magnetization",
 "parameters":{"n_bits":n,"states":N,"beta":beta,"kT":kT,"J":J,"h":h,"gamma":gamma,"observation_time":T},
 "checks":{"max_transition_row_sum_error":row_err,"minimum_transition_probability":minP,"max_detailed_balance_absolute_residual":dbmax},
 "entropies_bits":{"H_X0":HX,"H_M_T":HM,"H_X0_given_M_T":Hcond,"H_X0_given_M_T_direct":Hcond_direct,"I_X0_M_T":I,"H_joint_X0_M_T":Hjoint},
 "ideal_landauer_energy_units":{"W_erase_without_side_information":W_no,"W_erase_with_M_T":W_cond,"maximum_saving":saving,"kTln2_times_H_cond":W_cond,"erase_M_T_separately":W_M_erase,"conditional_erase_plus_M_reset":net_after_M_reset,"net_saving_after_M_reset":W_no-net_after_M_reset},
 "identity_checks":{"saving_equals_kTln2_mutual_information":saving,"difference_of_erase_bounds":W_no-W_cond,"net_cost_identity":net_after_M_reset,"H_X0_equals_H_cond_plus_mutual_information":HX-(Hcond+I)},
 "distributions":{"M_T_values":unique_m,"P_M_T":pm.tolist()},
}
with open("landauer_4bit_conditional_exact.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
