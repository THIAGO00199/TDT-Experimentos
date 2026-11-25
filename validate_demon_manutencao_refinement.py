import sys,json
import numpy as np
sys.path.insert(0,"/home/iknowyou/TDT_Experimentos")
import demon_manutencao_8spin as m
m.N_SUB=40
m.DT=m.DT_SEG/m.N_SUB
p=json.load(open("/home/iknowyou/TDT_Experimentos/demon_manutencao_8spin_resultado.json"))
for label,key,p0,pt in [
 ("A_opt","optimized_A",m.pA,m.pA),
 ("B_uncontrolled","uncontrolled_B",m.pB,m.pB),
 ("B_opt","optimized_B",m.pB,m.pB)]:
    run=p["full_runs"][key]
    knots=np.zeros((m.N_SEG+1,m.N)) if key=="uncontrolled_B" else np.array(run["knots"])
    r=m.evaluate_control(p0,pt,knots)
    print(label,json.dumps({k:r[k] for k in ["W_net","W_input_gross","W_recovered_gross","sigma_Schnakenberg_nats","delta_Fneq","classical_balance_residual","KL_time_average_nats","KL_final_nats","L1_final_to_target"]}),flush=True)
