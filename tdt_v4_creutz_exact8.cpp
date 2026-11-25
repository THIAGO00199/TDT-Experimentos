#include <bits/stdc++.h>
using namespace std;

static constexpr int NS=8, N=16, EMAX=20, NSTEPS=320, M=30000, BBOOT=200;
static constexpr uint64_t GOLD=0x9e3779b97f4a7c15ULL;

struct RNG {
    uint64_t x;
    explicit RNG(uint64_t s):x(s){}
    uint64_t next(){x+=GOLD;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
    int idx(int n){return int(next()%uint64_t(n));}
};

static inline int spin(uint16_t mask,int i){return ((mask>>i)&1U)?1:-1;}
static inline int energy(uint16_t mask,const array<int,N>&J){
    int e=0;
    for(int i=0;i<N;i++)e-=J[i]*spin(mask,i)*spin(mask,(i+1)%N);
    return e;
}
static inline int deltaE(uint16_t mask,int i,const array<int,N>&J){
    int l=(i+N-1)%N,r=(i+1)%N;
    return 2*spin(mask,i)*(J[l]*spin(mask,l)+J[i]*spin(mask,r));
}
static inline bool attempt(uint16_t&mask,int&d,int i,const array<int,N>&J){
    int de=deltaE(mask,i,J);
    if(de<=d && d-de<=EMAX){mask^=uint16_t(1U<<i);d-=de;return true;}
    return false;
}
static long double entropy(const vector<long double>&p){
    long double h=0;
    for(long double x:p)if(x>0)h-=x*logl(x);
    return h;
}
static double quantile(vector<double>&v,double q){
    if(v.empty())return NAN;
    sort(v.begin(),v.end());
    double z=q*double(v.size()-1);
    size_t a=size_t(floor(z)),b=size_t(ceil(z));
    double f=z-double(a);
    return v[a]*(1-f)+v[b]*f;
}
struct Edge{int a,b;};

static long double sigmaSchnakenberg(const vector<long double>&p,const vector<Edge>&edges){
    long double sig=0;
    for(const auto&e:edges){
        long double a=p[e.a],b=p[e.b];
        if(a==0 && b==0)continue;
        if(a==0 || b==0)return numeric_limits<long double>::infinity();
        sig+=(a-b)*logl(a/b)/N;
    }
    return sig;
}
static double sigmaEmpirical(const vector<int>&c,const vector<Edge>&edges){
    double sig=0;
    for(const auto&e:edges){
        int a=c[e.a],b=c[e.b];
        if(a==0 || b==0)continue; // matches the observed-edge-only MC estimator
        double pa=double(a)/M,pb=double(b)/M;
        sig+=(pa-pb)*log(pa/pb)/N;
    }
    return sig;
}
static long double discretePathEP(const vector<long double>&p,const vector<long double>&pn,
                                  const vector<vector<int>>&adj){
    // Exact KL between forward one-step paths p(x)P(x,y)
    // and reversed paths pn(y)P(y,x); P is symmetric.
    long double z=0;
    for(size_t x=0;x<p.size();x++){
        long double px=p[x];
        if(px==0)continue;
        int deg=int(adj[x].size());
        long double self=1.0L-(long double)deg/N;
        if(self>0 && pn[x]>0)
            z+=px*self*logl(px/pn[x]);
        for(int y:adj[x]){
            if(pn[y]<=0)continue;
            z+=px*(1.0L/N)*logl(px/pn[y]);
        }
    }
    return z;
}

int main(){
    cout<<setprecision(12);
    array<int,N>J{};for(int i=0;i<NS-1;i++)J[i]=2;for(int i=NS-1;i<N;i++)J[i]=1;

    // Reproduce the bath initialization in the original Creutz code.
    RNG initR(9918273ULL);
    string sys="10101010";
    uint16_t initial=0;
    for(int i=0;i<NS;i++)if(sys[i]=='1')initial|=uint16_t(1U<<i);
    for(int i=NS;i<N;i++)if(initR.next()&1ULL)initial|=uint16_t(1U<<i);

    const int E0=energy(initial,J); // initial demon energy is zero
    const int initialD=0;
    const int shellTotal=1<<N;
    vector<int> shellId(shellTotal,-1),demon(shellTotal,-1);
    vector<uint16_t> masks;
    for(int mm=0;mm<shellTotal;mm++){
        uint16_t m=uint16_t(mm);
        int d=E0-energy(m,J);
        if(d>=0 && d<=EMAX && (d%2)==0){
            shellId[mm]=int(masks.size());
            demon[mm]=d;
            masks.push_back(m);
        }
    }
    int startShell=shellId[initial];
    if(startShell<0){cerr<<"ERROR initial state not in energy shell\n";return 2;}

    // Construct the full accepted-flip graph on the exact energy shell.
    vector<vector<int>> shellAdj(masks.size());
    vector<Edge> shellEdges;
    for(size_t a=0;a<masks.size();a++){
        uint16_t m=masks[a];int d=demon[m];
        for(int i=0;i<N;i++){
            uint16_t t=m;int dd=d;
            if(attempt(t,dd,i,J)){
                int b=shellId[t];
                if(b<0 || demon[t]!=dd){cerr<<"ERROR transition left energy shell\n";return 3;}
                shellAdj[a].push_back(b);
                if(int(a)<b)shellEdges.push_back({int(a),b});
            }
        }
    }

    // Keep only the connected component actually reachable from the prescribed initial state.
    vector<char>seen(masks.size(),0);
    queue<int>q;q.push(startShell);seen[startShell]=1;
    vector<int>compOld;compOld.reserve(masks.size());
    while(!q.empty()){
        int a=q.front();q.pop();compOld.push_back(a);
        for(int b:shellAdj[a])if(!seen[b]){seen[b]=1;q.push(b);}
    }
    vector<int>oldToComp(masks.size(),-1);
    vector<uint16_t>cmasks(compOld.size());
    vector<int>cdemon(compOld.size());
    for(size_t i=0;i<compOld.size();i++){
        oldToComp[compOld[i]]=int(i);
        cmasks[i]=masks[compOld[i]];
        cdemon[i]=demon[cmasks[i]];
    }
    vector<vector<int>>adj(compOld.size());
    vector<Edge>edges;
    for(size_t a=0;a<compOld.size();a++){
        for(int oldb:shellAdj[compOld[a]]){
            int b=oldToComp[oldb];
            if(b>=0)adj[a].push_back(b);
            if(int(a)<b)edges.push_back({int(a),b});
        }
    }
    vector<int>maskToComp(shellTotal,-1);
    for(size_t i=0;i<cmasks.size();i++)maskToComp[cmasks[i]]=int(i);

    // The symmetric discrete-time transition kernel:
    // P(x,y)=1/N for accepted single-spin flips; rejected proposals are self-loops.
    vector<long double>p(cmasks.size(),0.0L),pn(cmasks.size(),0.0L);
    p[maskToComp[initial]]=1.0L;

    vector<int>times={0,1,2,4,8,16,32,64,80,160,319,320};
    vector<vector<int>>mcSamples(times.size(),vector<int>(M,-1));
    vector<int>timeIndex(NSTEPS+1,-1);
    for(size_t k=0;k<times.size();k++)timeIndex[times[k]]=int(k);

    // Exact propagation and matched Monte Carlo trajectories.
    vector<vector<long double>>exactP(times.size());
    for(size_t k=0;k<times.size();k++)exactP[k].reserve(cmasks.size());
    for(int j=0;j<M;j++){
        RNG r(7000000ULL+8ULL*100000ULL+uint64_t(j)*GOLD);
        uint16_t m=initial;int d=initialD;
        for(int step=0;step<=NSTEPS;step++){
            int ti=timeIndex[step];
            if(ti>=0)mcSamples[ti][j]=maskToComp[m];
            if(step<NSTEPS)attempt(m,d,r.idx(N),J);
        }
    }
    // Exact distribution p_t for every requested attempt count.
    for(int step=0;step<=NSTEPS;step++){
        int ti=timeIndex[step];
        if(ti>=0)exactP[ti]=p;
        if(step==NSTEPS)break;
        fill(pn.begin(),pn.end(),0.0L);
        for(size_t x=0;x<p.size();x++){
            int deg=int(adj[x].size());
            pn[x]+=p[x]*(1.0L-(long double)deg/N);
            for(int y:adj[x])pn[y]+=p[x]/N;
        }
        p.swap(pn);
    }

    cout<<"ENUM bath=8 spins_total=16 spin_configurations="<<shellTotal
        <<" conserved_energy="<<E0
        <<" allowed_energy_shell_states="<<masks.size()
        <<" reachable_component_states="<<cmasks.size()
        <<" reachable_undirected_edges="<<edges.size()
        <<" initial_mask="<<initial<<" initial_demon="<<initialD
        <<" demon_levels=11 q_xy_q_yx="<<(1.0/N)<<"\n";
    cout<<"KERNEL symmetric=yes row_stochastic=yes stationary_distribution=uniform_on_reachable_component\n";

    for(size_t ti=0;ti<times.size();ti++){
        const auto&pe=exactP[ti];
        vector<long double>peNext(cmasks.size(),0.0L);
        for(size_t x=0;x<pe.size();x++){
            int deg=int(adj[x].size());
            peNext[x]+=pe[x]*(1.0L-(long double)deg/N);
            for(int y:adj[x])peNext[y]+=pe[x]/N;
        }
        long double H=entropy(pe),Hn=entropy(peNext),dH=Hn-H;
        long double sig=sigmaSchnakenberg(pe,edges);
        long double path=discretePathEP(pe,peNext,adj);
        int support=0;long double minPositive=1.0L;
        for(long double x:pe)if(x>0){support++;minPositive=min(minPositive,x);}
        long double missingMass=0;
        vector<int>counts(cmasks.size(),0);
        for(int z:mcSamples[ti])if(z>=0)counts[z]++;
        int unique=0;long double exactMassUnseen=0;
        for(size_t x=0;x<counts.size();x++){
            if(counts[x]>0)unique++;
            else exactMassUnseen+=pe[x];
        }
        double sigMC=sigmaEmpirical(counts,edges);

        // Nonparametric bootstrap over independent trajectories, preserving the
        // exact same observed-edge-only estimator used for the MC point estimate.
        RNG br(0x5eed1234ULL+uint64_t(times[ti])*1009ULL);
        vector<double>boot;boot.reserve(BBOOT);
        vector<int>bc(cmasks.size());
        for(int b=0;b<BBOOT;b++){
            fill(bc.begin(),bc.end(),0);
            for(int z=0;z<M;z++)bc[mcSamples[ti][br.idx(M)]]++;
            boot.push_back(sigmaEmpirical(bc,edges));
        }
        double lo=quantile(boot,0.025),hi=quantile(boot,0.975);

        cout<<"TIME step="<<times[ti]<<" sweeps="<<double(times[ti])/N
            <<" exact_support="<<support<<"/"<<cmasks.size()
            <<" min_positive_p="<<setprecision(5)<<scientific<<double(minPositive)<<defaultfloat
            <<" H_exact_nats="<<setprecision(12)<<double(H)
            <<" deltaH_exact_nats="<<double(dH)
            <<" discrete_path_EP_exact_nats="<<double(path)
            <<" pathEP_minus_deltaH="<<double(path-dH)
            <<" Schnakenberg_same_time=";
        if(isinf(sig))cout<<"INF_support_boundary";
        else cout<<setprecision(12)<<double(sig);
        cout<<" sigma_exact_finite="<<(!isinf(sig)?"YES":"NO")
            <<" MC_unique="<<unique<<"/"<<cmasks.size()
            <<" MC_exact_mass_unseen="<<double(exactMassUnseen)
            <<" MC_sigma_truncated="<<setprecision(12)<<sigMC
            <<" MC_boot95=["<<lo<<","<<hi<<"]"
            <<" MC_bias_vs_exact=";
        if(isinf(sig))cout<<"NOT_DEFINED";
        else cout<<setprecision(12)<<(sigMC-double(sig));
        cout<<"\n";
    }

    // Verify total energy decomposition for each state in the accessible component.
    int maxResidual=0;
    for(size_t x=0;x<cmasks.size();x++){
        int total=energy(cmasks[x],J)+cdemon[x];
        maxResidual=max(maxResidual,abs(total-E0));
    }
    cout<<"ENERGY_CHECK states="<<cmasks.size()<<" max_abs_residual="<<maxResidual<<"\n";
}
