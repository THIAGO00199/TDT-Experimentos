from pathlib import Path
p=Path("/home/iknowyou/TDT_Experimentos/demon_creutz_banho8.cpp")
s=p.read_text()
s=s.replace("NTRAIN=700, NVALID=12000","NTRAIN=700, NVALID=50000")
s=s.replace("array<double,NS-1> c0{},c1{};\n    double M0=0,M1=0;", "array<double,NS-1> c0{},c1{};\n    array<double,NB-1> bc0{},bc1{};\n    array<double,NS*NB> xc0{},xc1{};\n    double M0=0,M1=0;")
s=s.replace("out.M0=magSys(mask&255);for(int i=0;i<NS-1;i++)out.c0[i]=corrBond(mask&255,i);",
"""out.M0=magSys(mask&255);for(int i=0;i<NS-1;i++)out.c0[i]=corrBond(mask&255,i);
    for(int i=0;i<NB-1;i++)out.bc0[i]=double(spin(mask,NS+i)*spin(mask,NS+i+1));
    for(int i=0;i<NS;i++)for(int j=0;j<NB;j++)out.xc0[i*NB+j]=double(spin(mask,i)*spin(mask,NS+j));""")
s=s.replace("out.M1=magSys(mask&255);for(int i=0;i<NS-1;i++)out.c1[i]=corrBond(mask&255,i);\n    out.bal=", 
"""out.M1=magSys(mask&255);for(int i=0;i<NS-1;i++)out.c1[i]=corrBond(mask&255,i);
    for(int i=0;i<NB-1;i++)out.bc1[i]=double(spin(mask,NS+i)*spin(mask,NS+i+1));
    for(int i=0;i<NS;i++)for(int j=0;j<NB;j++)out.xc1[i*NB+j]=double(spin(mask,i)*spin(mask,NS+j));
    out.bal=""")
s=s.replace("array<double,NS-1> c0{},c1{};\n    array<double,6> corrSnaps{};", "array<double,NS-1> c0{},c1{};\n    array<double,NB-1> bc0{},bc1{};\n    array<double,NS*NB> xc0{},xc1{};\n    array<double,6> corrSnaps{};")
s=s.replace("for(int i=0;i<NS-1;i++){e.c0[i]+=q.c0[i];e.c1[i]+=q.c1[i];}\n        for(int z=0;z<6;z++)", 
"""for(int i=0;i<NS-1;i++){e.c0[i]+=q.c0[i];e.c1[i]+=q.c1[i];}
        for(int i=0;i<NB-1;i++){e.bc0[i]+=q.bc0[i];e.bc1[i]+=q.bc1[i];}
        for(int i=0;i<NS*NB;i++){e.xc0[i]+=q.xc0[i];e.xc1[i]+=q.xc1[i];}
        for(int z=0;z<6;z++)""")
s=s.replace("for(int i=0;i<NS-1;i++){e.c0[i]*=inv;e.c1[i]*=inv;}\n    for(int z=0;z<6;z++)", 
"""for(int i=0;i<NS-1;i++){e.c0[i]*=inv;e.c1[i]*=inv;}
    for(int i=0;i<NB-1;i++){e.bc0[i]*=inv;e.bc1[i]*=inv;}
    for(int i=0;i<NS*NB;i++){e.xc0[i]*=inv;e.xc1[i]*=inv;}
    for(int z=0;z<6;z++)""")
s=s.replace('f<<"],\\"corr_snapshots_C01\\":[";', 'f<<"],\\"corr_snapshots_C01\\":[";')
# Insert final bath and cross-correlation arrays after corr_final.
s=s.replace('f<<"],\\"corr_snapshots_C01\\":[";', 
"""f<<"],\\"bath_corr_initial\\":[";
    for(int i=0;i<NB-1;i++)f<<e.bc0[i]<<(i+1==NB-1?"":",");
    f<<"],\\"bath_corr_final\\":[";
    for(int i=0;i<NB-1;i++)f<<e.bc1[i]<<(i+1==NB-1?"":",");
    f<<"],\\"sys_bath_corr_initial\\":[";
    for(int i=0;i<NS*NB;i++)f<<e.xc0[i]<<(i+1==NS*NB?"":",");
    f<<"],\\"sys_bath_corr_final\\":[";
    for(int i=0;i<NS*NB;i++)f<<e.xc1[i]<<(i+1==NS*NB?"":",");
    f<<"],\\"corr_snapshots_C01\\":[";""")
# paired work/input/return difference vectors and estimates
s=s.replace("vector<double>diffs;diffs.reserve(NVALID);", "vector<double>diffs,dpos,dneg;diffs.reserve(NVALID);dpos.reserve(NVALID);dneg.reserve(NVALID);")
s=s.replace("diffs.push_back(qb.work-qa.work);", "diffs.push_back(qb.work-qa.work);dpos.push_back(qb.wpos-qa.wpos);dneg.push_back(qb.wneg-qa.wneg);")
s=s.replace("double seDiff=sqrt(vv/(NVALID-1)/NVALID);",
"""double seDiff=sqrt(vv/(NVALID-1)/NVALID);
    auto meanSe=[&](const vector<double>&v){
       double m=accumulate(v.begin(),v.end(),0.0)/v.size(),q=0;
       for(double x:v)q+=(x-m)*(x-m);
       return pair<double,double>{m,sqrt(q/(v.size()-1)/v.size())};
    };
    auto posDiff=meanSe(dpos),negDiff=meanSe(dneg);""")
s=s.replace('f<<"\\"paired_delta_W_B_minus_A\\":"<<dW<<",\\"paired_delta_W_SE\\":"<<seDiff<<",\\"paired_delta_W_seed_estimate\\":"<<md<<",\\n";',
"""f<<"\\"paired_delta_W_B_minus_A\\":"<<dW<<",\\"paired_delta_W_SE\\":"<<seDiff<<",\\"paired_delta_W_seed_estimate\\":"<<md
     <<",\\"paired_delta_W_input_B_minus_A\\":"<<posDiff.first<<",\\"paired_delta_W_input_SE\\":"<<posDiff.second
     <<",\\"paired_delta_W_return_B_minus_A\\":"<<negDiff.first<<",\\"paired_delta_W_return_SE\\":"<<negDiff.second<<",\\n";""")
p.write_text(s)
print("patched NVALID=50000; added final bath and system-bath correlations; paired gross-work differences")
