from pathlib import Path
p=Path("/home/iknowyou/TDT_Experimentos/demon_creutz_banho8.cpp")
s=p.read_text()
s=s.replace('vector<double>diffs,dpos,dneg;diffs.reserve(NVALID);dpos.reserve(NVALID);dneg.reserve(NVALID);',
'''vector<double>diffs,dpos,dneg;diffs.reserve(NVALID);dpos.reserve(NVALID);dneg.reserve(NVALID);
    array<double,NS*NB> xsum{},xsum2{};
    array<double,NB-1> bsum{},bsum2{};
    array<double,NS-1> ssum{},ssum2{};
    array<double,4> esum{},esum2{};''')
s=s.replace('diffs.push_back(qb.work-qa.work);dpos.push_back(qb.wpos-qa.wpos);dneg.push_back(qb.wneg-qa.wneg);',
'''diffs.push_back(qb.work-qa.work);dpos.push_back(qb.wpos-qa.wpos);dneg.push_back(qb.wneg-qa.wneg);
       for(int i=0;i<NS*NB;i++){double z=qb.xc1[i]-qa.xc1[i];xsum[i]+=z;xsum2[i]+=z*z;}
       for(int i=0;i<NB-1;i++){double z=qb.bc1[i]-qa.bc1[i];bsum[i]+=z;bsum2[i]+=z*z;}
       for(int i=0;i<NS-1;i++){double z=qb.c1[i]-qa.c1[i];ssum[i]+=z;ssum2[i]+=z*z;}
       array<double,4> de={
         (qb.sysE1-qb.sysE0)-(qa.sysE1-qa.sysE0),
         (qb.bathE1-qb.bathE0)-(qa.bathE1-qa.bathE0),
         (qb.intE1-qb.intE0)-(qa.intE1-qa.intE0),
         (qb.d1-qb.d0)-(qa.d1-qa.d0)};
       for(int i=0;i<4;i++){esum[i]+=de[i];esum2[i]+=de[i]*de[i];}''')
s=s.replace('auto posDiff=meanSe(dpos),negDiff=meanSe(dneg);',
'''auto posDiff=meanSe(dpos),negDiff=meanSe(dneg);
    auto pairedArrays=[&](const auto&sum,const auto&sum2,int n){
       vector<pair<double,double>> z;z.reserve(sum.size());
       for(size_t i=0;i<sum.size();i++){
          double m=sum[i]/n, v=max(0.0,(sum2[i]-n*m*m)/(n-1));
          z.push_back({m,sqrt(v/n)});
       }
       return z;
    };
    auto crossDiff=pairedArrays(xsum,xsum2,NVALID);
    auto bathDiff=pairedArrays(bsum,bsum2,NVALID);
    auto sysDiff=pairedArrays(ssum,ssum2,NVALID);
    auto energyDiff=pairedArrays(esum,esum2,NVALID);''')
s=s.replace('<<",\\"paired_delta_W_return_B_minus_A\\":"<<negDiff.first<<",\\"paired_delta_W_return_SE\\":"<<negDiff.second<<",\\n";',
'''<<",\\"paired_delta_W_return_B_minus_A\\":"<<negDiff.first<<",\\"paired_delta_W_return_SE\\":"<<negDiff.second;
    f<<",\\"paired_energy_component_deltas_B_minus_A\\":[";
    for(int i=0;i<4;i++)f<<"{\\"mean\\":"<<energyDiff[i].first<<",\\"se\\":"<<energyDiff[i].second<<"}"<<(i==3?"":",");
    f<<"],\\"paired_system_bond_corr_deltas\\":[";
    for(int i=0;i<NS-1;i++)f<<"{\\"mean\\":"<<sysDiff[i].first<<",\\"se\\":"<<sysDiff[i].second<<"}"<<(i==NS-2?"":",");
    f<<"],\\"paired_bath_bond_corr_deltas\\":[";
    for(int i=0;i<NB-1;i++)f<<"{\\"mean\\":"<<bathDiff[i].first<<",\\"se\\":"<<bathDiff[i].second<<"}"<<(i==NB-2?"":",");
    f<<"],\\"paired_system_bath_corr_deltas\\":[";
    for(int i=0;i<NS*NB;i++)f<<"{\\"mean\\":"<<crossDiff[i].first<<",\\"se\\":"<<crossDiff[i].second<<"}"<<(i==NS*NB-1?"":",");
    f<<"],\\n";''')
p.write_text(s)
print("patched paired uncertainty for system, bath, cross-correlations and energy transfers")
