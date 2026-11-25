#include <bits/stdc++.h>
using namespace std;
static constexpr int NS=8, EMAX=20, SWEEPS=20, MTRAJ=30000, BBOOT=300, CORE_MIN=20;
static constexpr uint64_t GOLD=0x9e3779b97f4a7c15ULL;

struct RNG {
    uint64_t x;
    explicit RNG(uint64_t s):x(s){}
    uint64_t next(){x+=GOLD;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
    int idx(int n){return int(next()%uint64_t(n));}
};
struct State {
    uint64_t lo=0,hi=0; uint8_t d2=0;
    bool operator==(const State&o)const{return lo==o.lo&&hi==o.hi&&d2==o.d2;}
};
static inline uint64_t mix64(uint64_t z){z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
struct StateHash {
    size_t operator()(const State&s)const noexcept{
        uint64_t h=mix64(s.lo^0x243f6a8885a308d3ULL);
        h^=mix64(s.hi^0x13198a2e03707344ULL);
        h^=mix64(uint64_t(s.d2)+0xa4093822299f31d0ULL);
        return size_t(mix64(h));
    }
};
struct Model {
    int nb,N; vector<int>J; State initial;
    Model(int b):nb(b),N(NS+b),J(N,1){
        for(int i=0;i<NS-1;i++)J[i]=2;
        string s="10101010";
        for(int i=0;i<NS;i++)setSpin(initial,i,s[i]=='1'?1:-1);
        RNG r(9918273);
        for(int i=NS;i<N;i++)setSpin(initial,i,(r.next()&1)?1:-1);
        initial.d2=0;
    }
    static void setSpin(State&s,int i,int v){
        uint64_t bit=uint64_t(1)<<(i<64?i:i-64);
        if(i<64){if(v>0)s.lo|=bit;else s.lo&=~bit;}
        else{if(v>0)s.hi|=bit;else s.hi&=~bit;}
    }
    static int spin(const State&s,int i){
        uint64_t b=(i<64?s.lo:s.hi)>>(i<64?i:i-64);
        return (b&1ULL)?1:-1;
    }
    int demon(const State&s)const{return 2*int(s.d2);}
    int deltaE(const State&s,int i)const{
        int l=(i+N-1)%N,rr=(i+1)%N;
        return 2*spin(s,i)*(J[l]*spin(s,l)+J[i]*spin(s,rr));
    }
    bool attempt(State&s,int i)const{
        int d=demon(s),de=deltaE(s,i);
        if(de<=d&&d-de<=EMAX){setSpin(s,i,-spin(s,i));s.d2=uint8_t((d-de)/2);return true;}
        return false;
    }
    State acceptedNeighbor(const State&s,int i,bool&ok)const{State t=s;ok=attempt(t,i);return t;}
    array<int,4> energyParts(const State&s)const{
        int es=0,eb=0,ei=0;
        for(int i=0;i<NS-1;i++)es-=J[i]*spin(s,i)*spin(s,i+1);
        for(int i=NS;i<N-1;i++)eb-=J[i]*spin(s,i)*spin(s,i+1);
        ei-=J[NS-1]*spin(s,NS-1)*spin(s,NS);
        ei-=J[N-1]*spin(s,N-1)*spin(s,0);
        return {es,eb,ei,demon(s)};
    }
};
struct Snapshot{int step=0;vector<State>states,next;};
using Edge=pair<int,int>;
static double entropyPlugin(const vector<int>&c,int M){
    double h=0;for(int n:c)if(n>0){double p=double(n)/M;h-=p*log(p);}return h;
}
static double sigmaEdges(const vector<int>&c,const vector<Edge>&edges,int N,int M){
    double s=0;for(auto e:edges){int a=c[e.first],b=c[e.second];if(a<=0||b<=0)continue;double pa=double(a)/M,pb=double(b)/M;s+=(pa-pb)*(log(pa)-log(pb))/double(N);}return s;
}
static double quantile(vector<double>&v,double q){
    if(v.empty())return NAN;sort(v.begin(),v.end());double x=q*(v.size()-1);size_t a=size_t(floor(x)),b=size_t(ceil(x));double f=x-a;return v[a]*(1-f)+v[b]*f;
}
static uint64_t pairKey(int a,int b){return(uint64_t(uint32_t(a))<<32)|uint32_t(b);}

static void analyzeSnapshot(const Model&m,const Snapshot&sn,int M,uint64_t bootSeed){
    unordered_map<State,int,StateHash> idA,idB;
    idA.reserve(size_t(M)*2);idB.reserve(size_t(M)*2);
    vector<State> ua,ub;vector<int> ca,cb,ia,ib;
    ia.reserve(M);ib.reserve(M);
    for(const State&s:sn.states){
        auto it=idA.find(s);
        if(it==idA.end()){int z=int(ua.size());idA.emplace(s,z);ua.push_back(s);ca.push_back(1);ia.push_back(z);}
        else{ca[it->second]++;ia.push_back(it->second);}
    }
    if(sn.next.size()==size_t(M)){
        for(const State&s:sn.next){
            auto it=idB.find(s);
            if(it==idB.end()){int z=int(ub.size());idB.emplace(s,z);ub.push_back(s);cb.push_back(1);ib.push_back(z);}
            else{cb[it->second]++;ib.push_back(it->second);}
        }
    }
    int U=int(ua.size());
    vector<Edge> edges,coreEdges;
    edges.reserve(size_t(U)*2);
    unordered_map<uint64_t,int> edgeId;edgeId.reserve(size_t(U)*2);
    long long reverseCheck=0,reverseBad=0;
    for(int a=0;a<U;a++){
        for(int i=0;i<m.N;i++){
            bool ok=false;State t=m.acceptedNeighbor(ua[a],i,ok);if(!ok)continue;
            auto it=idA.find(t);if(it==idA.end())continue;
            int b=it->second;if(b<=a)continue;
            bool back=false;(void)m.acceptedNeighbor(t,i,back);reverseCheck++;if(!back)reverseBad++;
            int e=int(edges.size());edges.emplace_back(a,b);edgeId.emplace(pairKey(a,b),e);
            if(ca[a]>=CORE_MIN&&ca[b]>=CORE_MIN)coreEdges.emplace_back(a,b);
        }
    }
    vector<long long> ab(edges.size(),0),ba(edges.size(),0);
    long long transitionsInside=0;
    if(ib.size()==size_t(M)){
        for(int k=0;k<M;k++){
            int a=ia[k],b=ib[k];if(a==b)continue;
            int x=min(a,b),y=max(a,b);auto it=edgeId.find(pairKey(x,y));if(it==edgeId.end())continue;
            if(a<b)ab[it->second]++;else ba[it->second]++;transitionsInside++;
        }
    }
    long long nAB=accumulate(ab.begin(),ab.end(),0LL),nBA=accumulate(ba.begin(),ba.end(),0LL);
    double H=entropyPlugin(ca,M),sigAll=sigmaEdges(ca,edges,m.N,M),sigCore=sigmaEdges(ca,coreEdges,m.N,M);
    double Hnext=NAN,dH=NAN;
    if(cb.size()>0){Hnext=entropyPlugin(cb,M);dH=Hnext-H;}

    double qAbs=0,qWeight=0;int qEdges=0;
    for(size_t e=0;e<edges.size();e++){
        int a=edges[e].first,b=edges[e].second;
        if(ca[a]<50||ca[b]<50)continue;
        double qa=double(ab[e])/ca[a],qb=double(ba[e])/ca[b],w=double(ca[a]+ca[b]);
        qAbs+=w*(fabs(qa-1.0/m.N)+fabs(qb-1.0/m.N))/2.0;qWeight+=w;qEdges++;
    }
    double qMAE=qWeight>0?qAbs/qWeight:NAN;

    vector<double> bootDH,bootCore;
    if(cb.size()>0){bootDH.reserve(BBOOT);bootCore.reserve(BBOOT);}
    RNG br(bootSeed);vector<int> baCounts(U),bbCounts(ub.size());
    if(cb.size()>0){
        for(int r=0;r<BBOOT;r++){
            fill(baCounts.begin(),baCounts.end(),0);fill(bbCounts.begin(),bbCounts.end(),0);
            for(int k=0;k<M;k++){int z=br.idx(M);baCounts[ia[z]]++;bbCounts[ib[z]]++;}
            bootDH.push_back(entropyPlugin(bbCounts,M)-entropyPlugin(baCounts,M));
            bootCore.push_back(sigmaEdges(baCounts,coreEdges,m.N,M));
        }
    }
    double dhLo=NAN,dhHi=NAN,scLo=NAN,scHi=NAN;
    if(!bootDH.empty()){dhLo=quantile(bootDH,0.025);dhHi=quantile(bootDH,0.975);scLo=quantile(bootCore,0.025);scHi=quantile(bootCore,0.975);}

    cout<<"DIST bath="<<m.nb<<" N="<<m.N<<" step="<<sn.step<<" sweeps="<<double(sn.step)/m.N
        <<" samples="<<M<<" unique="<<U<<" coverage="<<setprecision(6)<<double(U)/M
        <<" H_plugin_nats="<<setprecision(10)<<H
        <<" H_next_plugin_nats="<<Hnext<<" dH_total_discrete_nats="<<dH
        <<" dH_boot95=["<<dhLo<<","<<dhHi<<"]"
        <<" sigma_obs_allEdges_nats_attempt="<<sigAll
        <<" sigma_core_minCount"<<CORE_MIN<<"_nats_attempt="<<sigCore
        <<" sigma_core_boot95=["<<scLo<<","<<scHi<<"]"
        <<" observed_edges="<<edges.size()<<" core_edges="<<coreEdges.size()
        <<" reverse_check="<<reverseCheck<<" reverse_bad="<<reverseBad
        <<" q_exact_xy_yx="<<setprecision(10)<<1.0/m.N
        <<" qhat_weighted_MAE="<<qMAE<<" qhat_edges_n="<<qEdges
        <<" transitions_inside_support="<<transitionsInside<<" nxy="<<nAB<<" nyx="<<nBA<<"\n";
}

int main(){
    cout<<setprecision(10);
    for(int nb:{8,16,32,64}){
        Model m(nb);int maxStep=SWEEPS*m.N;
        vector<int>times={0,1,2,4,8,16,32,64,m.N,2*m.N,5*m.N,10*m.N,maxStep-1,maxStep};
        sort(times.begin(),times.end());times.erase(unique(times.begin(),times.end()),times.end());
        vector<int>snapAt(maxStep+1,-1);vector<Snapshot>snaps(times.size());
        for(size_t z=0;z<times.size();z++){snaps[z].step=times[z];snaps[z].states.reserve(MTRAJ);if(times[z]<maxStep)snaps[z].next.reserve(MTRAJ);snapAt[times[z]]=int(z);}
        long long ss[4]={0,0,0,0},se[4]={0,0,0,0},maxRes=0,exact=0;
        auto p0=m.energyParts(m.initial);int E0=p0[0]+p0[1]+p0[2]+p0[3];
        for(int j=0;j<MTRAJ;j++){
            RNG r(7000000ULL+uint64_t(nb)*100000ULL+uint64_t(j)*GOLD);State s=m.initial;
            for(int k=0;k<=maxStep;k++){
                int si=snapAt[k];if(si>=0)snaps[si].states.push_back(s);
                if(k<maxStep){int site=r.idx(m.N);m.attempt(s,site);if(si>=0)snaps[si].next.push_back(s);}
            }
            auto pe=m.energyParts(s);int Ef=pe[0]+pe[1]+pe[2]+pe[3];long long residual=Ef-E0;
            maxRes=max(maxRes,llabs(residual));if(residual==0)exact++;
            for(int c=0;c<4;c++){ss[c]+=p0[c];se[c]+=pe[c];}
        }
        cout<<"ENERGY bath="<<nb<<" N="<<m.N<<" trajectories="<<MTRAJ<<" E0_total="<<E0
            <<" start_ES="<<double(ss[0])/MTRAJ<<" end_ES="<<double(se[0])/MTRAJ<<" dES="<<double(se[0]-ss[0])/MTRAJ
            <<" start_EB="<<double(ss[1])/MTRAJ<<" end_EB="<<double(se[1])/MTRAJ<<" dEB="<<double(se[1]-ss[1])/MTRAJ
            <<" start_Eint="<<double(ss[2])/MTRAJ<<" end_Eint="<<double(se[2])/MTRAJ<<" dEint="<<double(se[2]-ss[2])/MTRAJ
            <<" start_ED="<<double(ss[3])/MTRAJ<<" end_ED="<<double(se[3])/MTRAJ<<" dED="<<double(se[3]-ss[3])/MTRAJ
            <<" dEtotal="<<double(se[0]+se[1]+se[2]+se[3]-ss[0]-ss[1]-ss[2]-ss[3])/MTRAJ
            <<" max_abs_residual="<<maxRes<<" exact_conserved="<<exact<<"/"<<MTRAJ<<" field_work=0\n";
        for(size_t z=0;z<snaps.size();z++)analyzeSnapshot(m,snaps[z],MTRAJ,0xabc123ULL+uint64_t(nb)*1000003ULL+uint64_t(snaps[z].step)*7919ULL);
        cout<<"END_BATH bath="<<nb<<"\n";cerr<<"finished bath="<<nb<<"\n";
    }
}
