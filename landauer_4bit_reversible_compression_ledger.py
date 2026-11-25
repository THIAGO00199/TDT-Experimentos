import numpy as np, json, heapq
from scipy.linalg import expm

# Exact 4-spin Ising ring and joint law p(X0, M_T).
beta, J, h, gamma, T = 0.4, 1.0, 0.0, 1.0, 1.0
kT = 1.0 / beta
n, N = 4, 16
S = np.array([[1 if (x >> i) & 1 else -1 for i in range(n)] for x in range(N)])
E = np.array([-J*sum(S[x,i]*S[x,(i+1)%n] for i in range(n))-h*np.sum(S[x]) for x in range(N)])
p0 = np.exp(-beta*E); p0 /= p0.sum()
Q = np.zeros((N,N))
for x in range(N):
    for i in range(n):
        y = x ^ (1 << i)
        Q[x,y] += gamma/(1+np.exp(beta*(E[y]-E[x])))
    Q[x,x] = -Q[x].sum()
P = expm(Q*T)
P[np.abs(P)<1e-16] = 0
P = np.maximum(P,0); P /= P.sum(axis=1,keepdims=True)
mvals = np.array([int(np.sum(s)) for s in S])
M = sorted(set(mvals.tolist()))
joint = np.zeros((N,len(M)))
for x in range(N):
    for y in range(N):
        joint[x,M.index(int(mvals[y]))] += p0[x]*P[x,y]
px, pm = joint.sum(axis=1), joint.sum(axis=0)

def H(p):
    p = np.asarray(p); p = p[p>0]
    return float(-np.sum(p*np.log2(p)))

def huffman_lengths(prob):
    # Exact binary Huffman lengths; symbols are indices with nonzero probability.
    heap = []
    for i,p in enumerate(prob):
        if p > 0: heapq.heappush(heap,(float(p),i, i))
    if len(heap) == 1: return {heap[0][2]:0}
    nodes = {}; nxt = len(prob)
    while len(heap)>1:
        p1,_,a = heapq.heappop(heap); p2,_,b = heapq.heappop(heap)
        nodes[nxt] = (a,b)
        heapq.heappush(heap,(p1+p2,nxt,nxt)); nxt += 1
    root = heap[0][2]; lengths = {}
    def walk(node,depth):
        if node < len(prob): lengths[node]=depth
        else:
            a,b=nodes[node]; walk(a,depth+1); walk(b,depth+1)
    walk(root,0)
    return lengths

def expected_length(prob):
    L = huffman_lengths(prob)
    return float(sum(float(p)*L[i] for i,p in enumerate(prob) if p>0)), L

HX, HM = H(px), H(pm)
Hjoint = H(joint.ravel())
Hcond = Hjoint-HM
I = HX-Hcond
Lx, Lx_codes = expected_length(px)
cond_codes = {}
Lcond = 0.0
for j,m in enumerate(M):
    pcond = joint[:,j]/pm[j]
    Lj, codes = expected_length(pcond)
    Lcond += pm[j]*Lj
    cond_codes[str(m)] = {"expected_length_bits":Lj, "code_lengths_by_X0_state":{str(k):v for k,v in codes.items()}}
unit = kT*np.log(2)
W_X = unit*HX
W_X_given_M = unit*Hcond
W_M = unit*HM
W_joint = unit*Hjoint
# The M register is a 3-bit binary register (5 possible symbols).
# Reversible write/read can have zero ideal work only with initialized blank ancillas;
# their initialization/reset and real measurement hardware are not inferred here.
out = {
 "model":"Exact 4-spin periodic Ising ring; X0 is 4-bit initial microstate; M_T is final magnetization",
 "parameters":{"beta":beta,"kT":kT,"J":J,"h":h,"gamma":gamma,"Tobs":T,"states_X0":N,"M_values":M},
 "entropies_bits":{"H_X0":HX,"H_M_T":HM,"H_X0_given_M_T":Hcond,"I_X0_M_T":I,"H_joint_X0_M_T":Hjoint},
 "reversible_prefix_compression":{
   "unconditional_huffman_expected_length_bits":Lx,
   "conditional_huffman_expected_length_bits_given_M_T":Lcond,
   "expected_code_length_saving_bits":Lx-Lcond,
   "conditional_code_lengths":cond_codes,
   "warning":"Huffman prefix encoding is lossless/injective and decodable given M_T. Variable-length code length is not itself physical work; finite-block reversible workspace/padding and reset must be included for a physical implementation."
 },
 "ideal_thermodynamic_ledger_energy_units":{
   "erase_X0_without_M":W_X,
   "erase_X0_conditioned_on_M":W_X_given_M,
   "gross_saving_on_X0_reset":W_X-W_X_given_M,
   "write_M_reversible_copy_given_preinitialized_blank_register":0.0,
   "read_M_reversible_non_destructive_interaction_ideal_limit":0.0,
   "erase_M_register_after_use":W_M,
   "total_reset_X0_and_M":W_X_given_M+W_M,
   "joint_entropy_reset_bound":W_joint,
   "net_extra_cost_vs_erasing_X0_alone":W_joint-W_X,
   "kTln2_per_bit":unit
 },
 "accounting_notes":[
   "The 3-bit M register must be initialized; in a cyclic device its preparation is supplied by the preceding reset, so do not count preparation and reset twice.",
   "Ideal reversible write/read are zero-work limits only when blank ancillas are already prepared and the interaction is quasistatic/reversible. Any copied readout register has its own reset cost.",
   "The M register reset is charged as kT ln(2) H(M_T), assuming optimal reversible compression before reset.",
   "The Ising evolution that produces M_T, the physical measurement apparatus, controller electronics, finite-time dissipation, and compression workspace are not included because no device Hamiltonian/protocol was specified.",
   "The exact joint reset lower bound is kT ln(2) H(X0,M_T), not merely the X0 conditional reset bound."
 ],
 "checks":{"entropy_identity_residual":HX-(Hcond+I),"joint_ledger_identity_residual":W_joint-(W_X_given_M+W_M)}
}
with open("landauer_4bit_reversible_compression_ledger.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
