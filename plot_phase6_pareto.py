import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path("/home/iknowyou/TDT_Experimentos/phase6_pareto_results.json")
d=json.loads(p.read_text())
rows=[r for r in d["runs"] if r["phase"]=="pareto"]
x=[r["endpoint_u_rms"] for r in rows]
y=[r["endpoint_q_rms"] for r in rows]
lam=[r["lambda_q"] for r in rows]
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(x,y,"o-",linewidth=1.5)
for a,b,l,r in zip(x,y,lam,rows):
    ax.annotate(f"λ={l:g}"+("" if r["success"] else " *"),(a,b),xytext=(5,5),textcoords="offset points",fontsize=8)
ax.set_xlabel("Endpoint RMS error in u")
ax.set_ylabel("Endpoint RMS error in q")
ax.set_title("TDT Phase 6 — weighted trade-off (local optimization)")
ax.grid(True,alpha=.3)
fig.tight_layout()
fig.savefig("/home/iknowyou/TDT_Experimentos/phase6_pareto_curve.png",dpi=180)
with open("/home/iknowyou/TDT_Experimentos/phase6_pareto_summary.csv","w") as f:
    f.write("lambda_q,u_rms,q_rms,success,iterations,max_force,max_rate\n")
    for r in rows:
        f.write(f'{r["lambda_q"]},{r["endpoint_u_rms"]},{r["endpoint_q_rms"]},{r["success"]},{r["iterations"]},{r["max_abs_f_space"]},{r["max_abs_dfdt_space"]}\n')
print("PLOT_AND_CSV_SAVED")
