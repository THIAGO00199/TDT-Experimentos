#include <bits/stdc++.h>
using namespace std;

// Creutz finite bath maintenance pilot: 8 system spins + 8 bath spins + bounded demon.
static constexpr int NS=8, NB=8, N=16, NSYS=1<<NS, K=5;
static constexpr int EMAX=20, CTRL=50, ITERS=60, NTRAIN=700, NVALID=50000;
static constexpr double J=1.0, HF=1.0, ALPHA=1.5;
static constexpr double LAMBDA_U=0.01, LAMBDA_DU=0.0002, COEFF_BOUND=3.0;
static constexpr uint64_t GOLD=0x9e3779b97f4a7c15ULL;

struct RNG {
    uint64_t x;
    explicit RNG(uint64_t s):x(s){}
    uint64_t next(){x+=GOLD;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
    int idx(int n){return int(next()%uint64_t(n));}
    double uni(){return (next()>>11)*0x1.0p-53;}
};

static inline int spin(uint16_t m,int i){return ((m>>i)&1U)?1:-1;}
static inline double legendre(int k,double z){
    if(k==0)return 1.0;
    if(k==1)return z;
    if(k==2)return 0.5*(3*z*z-1);
    if(k==3)return 0.5*(5*z*z*z-3*z);
    return (35*z*z*z*z-30*z*z+3)/8.0;
}
static inline double basis(int k,double u){return sin(M_PI*u)*legendre(k,2*u-1);}
static inline double fieldAt(const vector<double>&a,int i,double u){
    double h=0.0;for(int k=0;k<K;k++)h+=a[i*K+k]*basis(k,u);return h;
}
struct Dist {
    vector<double> p,logp;
    double E=0,M=0,H=0,F=0;
    array<double,NS-1> corr{};
    Dist():p(NSYS),logp(NSYS){}
};

static double EintSys(int mask){
    double e=0;for(int i=0;i<NS-1;i++)e-=J*spin(uint16_t(mask),i)*spin(uint16_t(mask),i+1);
    return e;
}
static double magSys(int mask){
    double m=0;for(int i=0;i<NS;i++)m+=spin(uint16_t(mask),i);return m;
}
static double corrBond(int mask,int i){return double(spin(uint16_t(mask),i)*spin(uint16_t(mask),i+1));}
static Dist makeA(){
    Dist d;double mx=-1e300;
    for(int m=0;m<NSYS;m++){double Hs=EintSys(m)-HF*magSys(m);d.logp[m]=-Hs;mx=max(mx,d.logp[m]);}
    double z=0;for(int m=0;m<NSYS;m++){d.p[m]=exp(d.logp[m]-mx);z+=d.p[m];}
    for(double &v:d.p)v/=z;
    for(int m=0;m<NSYS;m++){
        double Hs=EintSys(m)-HF*magSys(m);d.E+=d.p[m]*Hs;d.M+=d.p[m]*magSys(m);
        d.H-=d.p[m]*log(d.p[m]);
        for(int i=0;i<NS-1;i++)d.corr[i]+=d.p[m]*corrBond(m,i);
    }
    d.F=d.E-d.H;for(int m=0;m<NSYS;m++)d.logp[m]=log(d.p[m]);
    return d;
}
static Dist makeB(const Dist&A){
    double beta=1.0,lambda=1.0;
    vector<double> lp(NSYS),p(NSYS);
    auto calc=[&](double b,double l){
        double mx=-1e300;
        for(int m=0;m<NSYS;m++) {lp[m]=-b*EintSys(m)+l*magSys(m)+ALPHA*corrBond(m,0);mx=max(mx,lp[m]);}
        double z=0;for(int m=0;m<NSYS;m++){p[m]=exp(lp[m]-mx);z+=p[m];}
        for(double &v:p)v/=z;
    };
    for(int it=0;it<100;it++){
        calc(beta,lambda);
        double e=0,mag=0,me=0,mm=0,ee=0,em=0,mm2=0;
        for(int m=0;m<NSYS;m++){
            double E=EintSys(m)-HF*magSys(m), Mv=magSys(m), Ei=EintSys(m);
            e+=p[m]*E;mag+=p[m]*Mv;me+=p[m]*Ei;mm+=p[m]*Mv;
        }
        double cEE=0,cEM=0,cMM=0,cHE=0,cHM=0,cME=0,cMMv=0;
        for(int m=0;m<NSYS;m++){
            double E=EintSys(m)-HF*magSys(m), Mv=magSys(m), Ei=EintSys(m);
            cHE+=p[m]*(E-e)*(Ei-me);
            cHM+=p[m]*(E-e)*(Mv-mm);
            cME+=p[m]*(Mv-mag)*(Ei-me);
            cMMv+=p[m]*(Mv-mag)*(Mv-mag);
        }
        double r1=e-A.E,r2=mag-A.M;
        if(max(abs(r1),abs(r2))<2e-13)break;
        // Jacobian wrt (beta,lambda): [-Cov(H,Eint), Cov(H,M); -Cov(M,Eint), Var(M)]
        double a=-cHE,b=cHM,c=-cME,d=cMMv,det=a*d-b*c;
        if(abs(det)<1e-14)throw runtime_error("singular B moment-matching Jacobian");
        double db=(-r1*d+b*r2)/det;
        double dl=(-a*r2+c*r1)/det;
        beta+=db;lambda+=dl;
    }
    calc(beta,lambda);
    Dist B; B.p=p;B.logp.resize(NSYS);
    for(int m=0;m<NSYS;m++){
        double Hs=EintSys(m)-HF*magSys(m);B.E+=p[m]*Hs;B.M+=p[m]*magSys(m);
        B.H-=p[m]*log(p[m]);for(int i=0;i<NS-1;i++)B.corr[i]+=p[m]*corrBond(m,i);
    }
    B.F=B.E-B.H;
    for(int m=0;m<NSYS;m++)B.logp[m]=log(B.p[m]);
    cerr<<setprecision(15)<<"B_PARAMS beta="<<beta<<" lambda="<<lambda<<"\n";
    return B;
}

struct Parts {double sys=0,bath=0,interface=0,demon=0,total=0;};
static Parts energyParts(uint16_t mask,double demon,const array<double,NS>&u){
    Parts e;
    for(int i=0;i<NS-1;i++)e.sys-=J*spin(mask,i)*spin(mask,i+1);
    for(int i=0;i<NS;i++)e.sys-=HF*spin(mask,i)+u[i]*spin(mask,i);
    for(int i=NS;i<N-1;i++)e.bath-=J*spin(mask,i)*spin(mask,i+1);
    e.interface=-J*spin(mask,NS-1)*spin(mask,NS)-J*spin(mask,N-1)*spin(mask,0);
    e.demon=demon;e.total=e.sys+e.bath+e.interface+e.demon;return e;
}
static inline double deltaE(uint16_t mask,int i,const array<double,NS>&u){
    int l=(i+N-1)%N,r=(i+1)%N;
    double local=J*spin(mask,l)+J*spin(mask,r);
    if(i<NS)local+=HF+u[i];
    return 2.0*spin(mask,i)*local;
}
static inline bool attempt(uint16_t &mask,double &d,int i,const array<double,NS>&u){
    double de=deltaE(mask,i,u);
    if(de<=d+1e-12 && d-de<=EMAX+1e-12){
        mask^=uint16_t(1U<<i);d-=de;return true;
    }
    return false;
}

struct Traj {
    double work=0,wpos=0,wneg=0,field2=0,slew2=0,klSum=0;
    double bal=0;
    Parts start,end;
    double sysE0=0,sysE1=0,bathE0=0,bathE1=0,intE0=0,intE1=0,d0=0,d1=0;
    array<double,NS-1> c0{},c1{};
    array<double,NB-1> bc0{},bc1{};
    array<double,NS*NB> xc0{},xc1{};
    double M0=0,M1=0;
    array<double,6> corrSnapshots{}; // contrast not stored here
    vector<double> klByStep;
};
static uint16_t sampleSys(const Dist&D,RNG&r){
    double x=r.uni(),cum=0;for(int m=0;m<NSYS-1;m++){cum+=D.p[m];if(x<=cum)return uint16_t(m);}
    return uint16_t(NSYS-1);
}
static Traj trajectory(const Dist&D,const vector<double>&a,uint64_t seed,bool keepSeries=false){
    RNG r(seed);
    uint16_t mask=sampleSys(D,r);
    for(int i=NS;i<N;i++)if(r.next()&1ULL)mask|=uint16_t(1U<<i);
    double demon=0.0;
    array<double,NS> u{},prev{};
    Traj out;out.klByStep.reserve(CTRL);
    out.start=energyParts(mask,demon,u);
    out.sysE0=EintSys(mask&255)-HF*magSys(mask&255);out.bathE0=out.start.bath;out.intE0=out.start.interface;out.d0=demon;
    out.M0=magSys(mask&255);for(int i=0;i<NS-1;i++)out.c0[i]=corrBond(mask&255,i);
    for(int i=0;i<NB-1;i++)out.bc0[i]=double(spin(mask,NS+i)*spin(mask,NS+i+1));
    for(int i=0;i<NS;i++)for(int j=0;j<NB;j++)out.xc0[i*NB+j]=double(spin(mask,i)*spin(mask,NS+j));
    double total0=out.start.total;
    const int snapSteps[6]={0,4,9,19,34,49};
    for(int st=0;st<CTRL;st++){
        double t=double(st)/(CTRL-1);
        for(int i=0;i<NS;i++){
            double h=fieldAt(a,i,t),dh=h-prev[i];
            double dw=-spin(mask,i)*dh;
            out.work+=dw;if(dw>0)out.wpos+=dw;else out.wneg-=dw;
            u[i]=h;prev[i]=h;
            out.field2+=h*h/(double(CTRL*NS));
            if(st>0)out.slew2+=(dh*dh)/(double((CTRL-1)*NS));
        }
        for(int z=0;z<N;z++)attempt(mask,demon,r.idx(N),u);
        int sm=mask&255;
        out.klSum+=-D.logp[sm]/CTRL;
        if(keepSeries){
            double stepKL=-D.logp[sm]-D.H;
            out.klByStep.push_back(stepKL);
        }
        for(int q=0;q<6;q++)if(st==snapSteps[q])out.corrSnapshots[q]=corrBond(sm,0);
    }
    out.end=energyParts(mask,demon,u);
    out.sysE1=EintSys(mask&255)-HF*magSys(mask&255);out.bathE1=out.end.bath;out.intE1=out.end.interface;out.d1=demon;
    out.M1=magSys(mask&255);for(int i=0;i<NS-1;i++)out.c1[i]=corrBond(mask&255,i);
    for(int i=0;i<NB-1;i++)out.bc1[i]=double(spin(mask,NS+i)*spin(mask,NS+i+1));
    for(int i=0;i<NS;i++)for(int j=0;j<NB;j++)out.xc1[i*NB+j]=double(spin(mask,i)*spin(mask,NS+j));
    out.bal=out.end.total-total0-out.work;
    return out;
}

struct Eval {
    double kl=0,W=0,Wpos=0,Wneg=0,field2=0,slew2=0,bal=0;
    double sysE0=0,sysE1=0,bathE0=0,bathE1=0,intE0=0,intE1=0,d0=0,d1=0,M0=0,M1=0;
    double w2=0,balMax=0;
    array<double,NS-1> c0{},c1{};
    array<double,NB-1> bc0{},bc1{};
    array<double,NS*NB> xc0{},xc1{};
    array<double,6> corrSnaps{};
    double Wse=0;
    vector<double> workSamples;
    vector<double> klSeries;
    double objective=0;
};
static Eval evaluate(const Dist&D,const vector<double>&a,int n,uint64_t seed0,bool keepSeries=false){
    Eval e;e.workSamples.reserve(n);if(keepSeries)e.klSeries.assign(CTRL,0.0);
    double sumW2=0;
    for(int j=0;j<n;j++){
        Traj q=trajectory(D,a,seed0+uint64_t(j)*GOLD,keepSeries);
        e.W+=q.work;e.Wpos+=q.wpos;e.Wneg+=q.wneg;e.field2+=q.field2;e.slew2+=q.slew2;
        e.kl+=q.klSum;e.bal+=q.bal;e.balMax=max(e.balMax,abs(q.bal));
        e.sysE0+=q.sysE0;e.sysE1+=q.sysE1;e.bathE0+=q.bathE0;e.bathE1+=q.bathE1;
        e.intE0+=q.intE0;e.intE1+=q.intE1;e.d0+=q.d0;e.d1+=q.d1;e.M0+=q.M0;e.M1+=q.M1;
        for(int i=0;i<NS-1;i++){e.c0[i]+=q.c0[i];e.c1[i]+=q.c1[i];}
        for(int i=0;i<NB-1;i++){e.bc0[i]+=q.bc0[i];e.bc1[i]+=q.bc1[i];}
        for(int i=0;i<NS*NB;i++){e.xc0[i]+=q.xc0[i];e.xc1[i]+=q.xc1[i];}
        for(int z=0;z<6;z++)e.corrSnaps[z]+=q.corrSnapshots[z];
        e.workSamples.push_back(q.work);sumW2+=q.work*q.work;
        if(keepSeries)for(int z=0;z<CTRL;z++)e.klSeries[z]+=q.klByStep[z];
    }
    double inv=1.0/n;
    e.W*=inv;e.Wpos*=inv;e.Wneg*=inv;e.field2*=inv;e.slew2*=inv;
    e.kl=e.kl*inv-D.H;
    e.bal*=inv;e.sysE0*=inv;e.sysE1*=inv;e.bathE0*=inv;e.bathE1*=inv;
    e.intE0*=inv;e.intE1*=inv;e.d0*=inv;e.d1*=inv;e.M0*=inv;e.M1*=inv;
    for(int i=0;i<NS-1;i++){e.c0[i]*=inv;e.c1[i]*=inv;}
    for(int i=0;i<NB-1;i++){e.bc0[i]*=inv;e.bc1[i]*=inv;}
    for(int i=0;i<NS*NB;i++){e.xc0[i]*=inv;e.xc1[i]*=inv;}
    for(int z=0;z<6;z++)e.corrSnaps[z]*=inv;
    e.Wse=sqrt(max(0.0,sumW2*inv-e.W*e.W)/n);
    if(keepSeries)for(double &v:e.klSeries)v=v*inv-D.H;
    e.objective=e.kl+LAMBDA_U*e.field2+LAMBDA_DU*e.slew2;
    return e;
}
static double objective(const Dist&D,const vector<double>&a,int n,uint64_t seed){
    return evaluate(D,a,n,seed,false).objective;
}
struct OptResult{vector<double>a;Eval train;double bestObj=1e100;int iters=0;};
static OptResult optimize(const Dist&D,uint64_t seed){
    int dim=NS*K;vector<double>a(dim,0.0),best=a;
    double bestObj=objective(D,a,NTRAIN,seed+0x111111ULL);
    RNG prng(seed^0x1234abcd9876ULL);
    for(int it=0;it<ITERS;it++){
        vector<double>delta(dim),ap=a,am=a;
        for(int i=0;i<dim;i++){delta[i]=(prng.next()&1ULL)?1.0:-1.0;}
        double ck=0.25/pow(double(it+1),0.101);
        for(int i=0;i<dim;i++){ap[i]+=ck*delta[i];am[i]-=ck*delta[i];}
        double jp=objective(D,ap,NTRAIN,seed+0x111111ULL);
        double jm=objective(D,am,NTRAIN,seed+0x111111ULL);
        double g=(jp-jm)/(2.0*ck);
        double ak=0.35/pow(double(it+21),0.602);
        for(int i=0;i<dim;i++)a[i]=clamp(a[i]-ak*g*delta[i],-COEFF_BOUND,COEFF_BOUND);
        if((it+1)%10==0){
            double jo=objective(D,a,NTRAIN,seed+0x222222ULL);
            if(jo<bestObj){bestObj=jo;best=a;}
            cerr<<"SPSA iter="<<(it+1)<<" kl_obj="<<jo<<" best="<<bestObj<<"\n";
        }
    }
    Eval ev=evaluate(D,best,NTRAIN,seed+0x222222ULL,false);
    return {best,ev,bestObj,ITERS};
}

static void printEval(const string&name,const Eval&e){
    cout<<setprecision(12)<<name
      <<" KLavg="<<e.kl<<" Wnet="<<e.W<<" Wpos="<<e.Wpos<<" Wreturn="<<e.Wneg
      <<" Wse="<<e.Wse<<" field2="<<e.field2<<" slew2="<<e.slew2
      <<" bal="<<e.bal<<" balMax="<<e.balMax
      <<" dEsys="<<(e.sysE1-e.sysE0)<<" dEbath="<<(e.bathE1-e.bathE0)
      <<" dEinterface="<<(e.intE1-e.intE0)<<" dEdemon="<<(e.d1-e.d0)
      <<" dM="<<(e.M1-e.M0)<<" corr0_init="<<e.c0[0]<<" corr0_final="<<e.c1[0]
      <<" obj="<<e.objective<<"\n";
}
static void printTarget(const string&name,const Dist&D){
    cout<<setprecision(12)<<"TARGET "<<name<<" E="<<D.E<<" M="<<D.M<<" H="<<D.H<<" Fneq="<<D.F
      <<" corr=";
    for(int i=0;i<NS-1;i++)cout<<D.corr[i]<<(i+1==NS-1?"":";");
    cout<<"\n";
}
static void jsonEval(ostream&f,const Eval&e){
    f<<"{\"KLavg\":"<<setprecision(17)<<e.kl<<",\"Wnet\":"<<e.W<<",\"Wpos\":"<<e.Wpos
     <<",\"Wreturn\":"<<e.Wneg<<",\"Wse\":"<<e.Wse<<",\"field2\":"<<e.field2
     <<",\"slew2\":"<<e.slew2<<",\"balance_mean\":"<<e.bal<<",\"balance_max_abs\":"<<e.balMax
     <<",\"sysE0\":"<<e.sysE0<<",\"sysE1\":"<<e.sysE1<<",\"bathE0\":"<<e.bathE0
     <<",\"bathE1\":"<<e.bathE1<<",\"interfaceE0\":"<<e.intE0<<",\"interfaceE1\":"<<e.intE1
     <<",\"demon0\":"<<e.d0<<",\"demon1\":"<<e.d1<<",\"M0\":"<<e.M0<<",\"M1\":"<<e.M1
     <<",\"corr0_init\":"<<e.c0[0]<<",\"corr0_final\":"<<e.c1[0]<<",\"corr_final\":[";
    for(int i=0;i<NS-1;i++)f<<e.c1[i]<<(i+1==NS-1?"":",");
    f<<"],\"bath_corr_initial\":[";
    for(int i=0;i<NB-1;i++)f<<e.bc0[i]<<(i+1==NB-1?"":",");
    f<<"],\"bath_corr_final\":[";
    for(int i=0;i<NB-1;i++)f<<e.bc1[i]<<(i+1==NB-1?"":",");
    f<<"],\"sys_bath_corr_initial\":[";
    for(int i=0;i<NS*NB;i++)f<<e.xc0[i]<<(i+1==NS*NB?"":",");
    f<<"],\"sys_bath_corr_final\":[";
    for(int i=0;i<NS*NB;i++)f<<e.xc1[i]<<(i+1==NS*NB?"":",");
    f<<"],\"corr_snapshots_C01\":[";
    for(int i=0;i<6;i++)f<<e.corrSnaps[i]<<(i==5?"":",");
    f<<"],\"KL_series\":[";
    for(size_t i=0;i<e.klSeries.size();i++)f<<e.klSeries[i]<<(i+1==e.klSeries.size()?"":",");
    f<<"]}";
}
static void jsonDist(ostream&f,const Dist&D){
    f<<"{\"E\":"<<setprecision(17)<<D.E<<",\"M\":"<<D.M<<",\"H\":"<<D.H<<",\"Fneq\":"<<D.F<<",\"corr\":[";
    for(int i=0;i<NS-1;i++)f<<D.corr[i]<<(i+1==NS-1?"":",");
    f<<"],\"probabilities\":[";
    for(int i=0;i<NSYS;i++)f<<D.p[i]<<(i+1==NSYS?"":",");
    f<<"]}";
}

int main(){
    cout<<setprecision(12);
    Dist A=makeA(),B=makeB(A);
    printTarget("A",A);printTarget("B",B);
    cout<<"INITIAL_MATCH dE="<<B.E-A.E<<" dM="<<B.M-A.M
        <<" KL_BA="<<accumulate(B.p.begin(),B.p.end(),0.0,[&](double s,double p){static int k=0;double q=p*log(B.p[k]/A.p[k]);k++;return s+q;})
        <<" L1="<<([](const Dist&x,const Dist&y){double z=0;for(int i=0;i<NSYS;i++)z+=abs(x.p[i]-y.p[i]);return z;})(A,B)<<"\n";
    vector<double>zero(NS*K,0.0);
    const uint64_t trainSeed=0x6a09e667f3bcc909ULL;
    Eval A0=evaluate(A,zero,NVALID,trainSeed+0x300000ULL,false);
    Eval B0=evaluate(B,zero,NVALID,trainSeed+0x300000ULL,true);
    printEval("NOCTRL_A",A0);printEval("NOCTRL_B",B0);

    cerr<<"OPTIMIZING A\n";
    OptResult OA=optimize(A,0x243f6a8885a308d3ULL);
    cerr<<"OPTIMIZING B\n";
    OptResult OB=optimize(B,0x13198a2e03707344ULL);

    Eval Av=evaluate(A,OA.a,NVALID,trainSeed+0x500000ULL,true);
    Eval Bv=evaluate(B,OB.a,NVALID,trainSeed+0x500000ULL,true);
    Eval A0v=evaluate(A,zero,NVALID,trainSeed+0x500000ULL,false);
    Eval B0v=evaluate(B,zero,NVALID,trainSeed+0x500000ULL,false);
    printEval("OPT_A",Av);printEval("OPT_B",Bv);
    double dW=Bv.W-Av.W;
    // Paired uncertainty by rerunning paired trajectories with saved controls.
    vector<double>diffs,dpos,dneg;diffs.reserve(NVALID);dpos.reserve(NVALID);dneg.reserve(NVALID);
    array<double,NS*NB> xsum{},xsum2{};
    array<double,NB-1> bsum{},bsum2{};
    array<double,NS-1> ssum{},ssum2{};
    array<double,4> esum{},esum2{};
    for(int j=0;j<NVALID;j++){
       Traj qa=trajectory(A,OA.a,trainSeed+0x700000ULL+uint64_t(j)*GOLD);
       Traj qb=trajectory(B,OB.a,trainSeed+0x700000ULL+uint64_t(j)*GOLD);
       diffs.push_back(qb.work-qa.work);dpos.push_back(qb.wpos-qa.wpos);dneg.push_back(qb.wneg-qa.wneg);
       for(int i=0;i<NS*NB;i++){double z=qb.xc1[i]-qa.xc1[i];xsum[i]+=z;xsum2[i]+=z*z;}
       for(int i=0;i<NB-1;i++){double z=qb.bc1[i]-qa.bc1[i];bsum[i]+=z;bsum2[i]+=z*z;}
       for(int i=0;i<NS-1;i++){double z=qb.c1[i]-qa.c1[i];ssum[i]+=z;ssum2[i]+=z*z;}
       array<double,4> de={
         (qb.sysE1-qb.sysE0)-(qa.sysE1-qa.sysE0),
         (qb.bathE1-qb.bathE0)-(qa.bathE1-qa.bathE0),
         (qb.intE1-qb.intE0)-(qa.intE1-qa.intE0),
         (qb.d1-qb.d0)-(qa.d1-qa.d0)};
       for(int i=0;i<4;i++){esum[i]+=de[i];esum2[i]+=de[i]*de[i];}
    }
    double md=accumulate(diffs.begin(),diffs.end(),0.0)/NVALID, vv=0;
    for(double x:diffs)vv+=(x-md)*(x-md);
    double seDiff=sqrt(vv/(NVALID-1)/NVALID);
    auto meanSe=[&](const vector<double>&v){
       double m=accumulate(v.begin(),v.end(),0.0)/v.size(),q=0;
       for(double x:v)q+=(x-m)*(x-m);
       return pair<double,double>{m,sqrt(q/(v.size()-1)/v.size())};
    };
    auto posDiff=meanSe(dpos),negDiff=meanSe(dneg);
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
    auto energyDiff=pairedArrays(esum,esum2,NVALID);

    string outPath="/home/iknowyou/TDT_Experimentos/demon_creutz_banho8_resultado.json";
    ofstream f(outPath);
    f<<"{\n\"model\":{\"system_spins\":8,\"bath_spins\":8,\"total_spins\":16,\"topology\":\"16-site periodic ring; all J=1 bonds; system is sites 0..7, bath 8..15; two system-bath boundary bonds\",\"system_field_hf\":1,\"kBT_reference\":1,\"demon_initial\":0,\"demon_bounds\":[0,20],\"dynamics\":\"Creutz microcanonical single-spin proposals, one sweep (16 attempts) per control knot\",\"control\":\"8 additive local fields on system spins only; 5 sin(pi*u)*Legendre basis coefficients per site; 50 knots over 50 sweeps; fields zero at endpoints\",\"work\":\"W increment = -sum_i s_i delta(u_i) at each field update; Creutz flips conserve spin Hamiltonian plus demon energy at fixed controls\",\"objective\":\"mean cross-entropy to target system distribution minus target entropy, plus lambda_u mean u^2 and lambda_du mean delta-u^2\",\"lambda_u\":";
    f<<LAMBDA_U<<",\"lambda_du\":"<<LAMBDA_DU<<",\"coefficient_bound\":"<<COEFF_BOUND<<",\"spsa_iterations\":"<<ITERS<<",\"training_trajectories\":"<<NTRAIN<<",\"validation_trajectories\":"<<NVALID<<"},\n";
    f<<"\"target_A\":";jsonDist(f,A);f<<",\n\"target_B\":";jsonDist(f,B);f<<",\n";
    f<<"\"uncontrolled_A\":";jsonEval(f,A0);f<<",\n\"uncontrolled_B\":";jsonEval(f,B0);f<<",\n";
    f<<"\"optimized_A\":";jsonEval(f,Av);f<<",\n\"optimized_B\":";jsonEval(f,Bv);f<<",\n";
    f<<"\"paired_delta_W_B_minus_A\":"<<dW<<",\"paired_delta_W_SE\":"<<seDiff<<",\"paired_delta_W_seed_estimate\":"<<md
     <<",\"paired_delta_W_input_B_minus_A\":"<<posDiff.first<<",\"paired_delta_W_input_SE\":"<<posDiff.second
     <<",\"paired_delta_W_return_B_minus_A\":"<<negDiff.first<<",\"paired_delta_W_return_SE\":"<<negDiff.second;
    f<<",\"paired_energy_component_deltas_B_minus_A\":[";
    for(int i=0;i<4;i++)f<<"{\"mean\":"<<energyDiff[i].first<<",\"se\":"<<energyDiff[i].second<<"}"<<(i==3?"":",");
    f<<"],\"paired_system_bond_corr_deltas\":[";
    for(int i=0;i<NS-1;i++)f<<"{\"mean\":"<<sysDiff[i].first<<",\"se\":"<<sysDiff[i].second<<"}"<<(i==NS-2?"":",");
    f<<"],\"paired_bath_bond_corr_deltas\":[";
    for(int i=0;i<NB-1;i++)f<<"{\"mean\":"<<bathDiff[i].first<<",\"se\":"<<bathDiff[i].second<<"}"<<(i==NB-2?"":",");
    f<<"],\"paired_system_bath_corr_deltas\":[";
    for(int i=0;i<NS*NB;i++)f<<"{\"mean\":"<<crossDiff[i].first<<",\"se\":"<<crossDiff[i].second<<"}"<<(i==NS*NB-1?"":",");
    f<<"],\n";
    f<<"\"control_A_coefficients\":[";for(int i=0;i<(int)OA.a.size();i++)f<<OA.a[i]<<(i+1==(int)OA.a.size()?"":",");f<<"],\n";
    f<<"\"control_B_coefficients\":[";for(int i=0;i<(int)OB.a.size();i++)f<<OB.a[i]<<(i+1==(int)OB.a.size()?"":",");f<<"],\n";
    f<<"\"interpretation\":\"This is a classical Creutz-model Monte Carlo prediction with a finite 8-spin bath and bounded demon. It does not measure a TDT-specific residual. The pathwise energy balance is checked independently. The initial B distribution is mathematical and not physically prepared by a pulse.\"\n}\n";
    f.close();

    ofstream txt("/home/iknowyou/TDT_Experimentos/demon_creutz_banho8_resultado.txt");
    txt<<setprecision(12);
    txt<<"DEMONIO DA MANUTENCAO — CREUTZ — BANHO DE 8 SPINS\n\n";
    txt<<"Modelo: 8 spins do sistema + 8 spins do banho, anel de 16 sitios, J=1; campo hf=1 apenas no sistema; demon inicial 0, limitado entre 0 e 20. A/B sao as distribuicoes exatas do sistema, amostradas independentemente do banho uniforme inicial. Controle local em oito campos, 50 sweeps, cinco coeficientes por sitio.\n\n";
    txt<<"A: E="<<A.E<<" M="<<A.M<<" Fneq="<<A.F<<" H="<<A.H<<"\n";
    txt<<"B: E="<<B.E<<" M="<<B.M<<" Fneq="<<B.F<<" H="<<B.H<<" KL(B||A)="<<(B.F-A.F)<<"\n";
    txt<<"No control A:\n";printEval("A0",A0);
    txt<<"No control B:\n";printEval("B0",B0);
    txt<<"Optimized A:\n";printEval("A",Av);
    txt<<"Optimized B:\n";printEval("B",Bv);
    txt<<"Paired dW(B-A)="<<dW<<" SE="<<seDiff<<"\n";
    txt<<"Pathwise mean energy balance residual A="<<Av.bal<<" B="<<Bv.bal<<"; max abs residual A="<<Av.balMax<<" B="<<Bv.balMax<<"\n";
    txt.close();

    cerr<<"SAVED "<<outPath<<"\n";
    return 0;
}
