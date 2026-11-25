import sys,json,time
sys.path.insert(0,"/home/iknowyou/TDT_Experimentos")
import demon_manutencao_8spin as m
rows=[]
for lam in [0.001,0.01,0.1]:
    m.LAMBDA_U=lam
    m.LAMBDA_DU=2e-4
    r=m.optimize_controller("B_alpha1.5_lambdaU_"+str(lam),m.pB,m.pB,maxiter=120)
    rows.append({"lambda_u":lam,"lambda_du":m.LAMBDA_DU,
       "objective":r["objective"],"iterations":r["iterations"],
       "optimizer_message":r["optimizer_message"],
       "W_net":r["W_net"],"W_input_gross":r["W_input_gross"],
       "W_recovered_gross":r["W_recovered_gross"],
       "sigma":r["sigma_Schnakenberg_nats"],"delta_F":r["delta_Fneq"],
       "balance_residual":r["classical_balance_residual"],
       "KL_average":r["KL_time_average_nats"],"KL_final":r["KL_final_nats"],
       "L1_final":r["L1_final_to_target"],
       "mean_field_square":r["mean_field_square"],
       "mean_field_slew_square":r["mean_field_slew_square"],
       "max_abs_u":max(abs(x) for row in r["knots"] for x in row),
       "knots":r["knots"]})
    print("SWEEP",json.dumps({k:v for k,v in rows[-1].items() if k!="knots"}),flush=True)
out={"sweep":rows,"uncontrolled_B":m.evaluate_control(m.pB,m.pB,m.no_control_knots())}
with open("/home/iknowyou/TDT_Experimentos/demon_manutencao_8spin_sensitivity.json","w") as f: json.dump(out,f,indent=2)
print("SAVED /home/iknowyou/TDT_Experimentos/demon_manutencao_8spin_sensitivity.json",flush=True)
