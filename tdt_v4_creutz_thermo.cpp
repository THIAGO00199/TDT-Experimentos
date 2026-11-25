#include <bits/stdc++.h>
using namespace std;
static constexpr int NS=8, K=5, EMAX=20, CTRL=50, HREL=20;
struct RNG {
 uint64_t x;
 RNG(uint64_t s):x(s){}
 uint64_t next(){x+=0x9e3779b97f4a7c15ULL; uint64_t z=x; z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31);}
 int idx(int n){return int(next()%uint64_t(n));}
 double uni(){return (next()>>11)*0x1.0p-53;}
};
struct Model {
 int nb,N; vector<int> J; vector<int8_t> x0;
 Model(int b):nb(b),N(NS+b),J(N,1),x0(N){
  for(int i=0;i<NS-1;i++)J[i]=2;
  string s="10101010"; for(int i=0;i<NS;i++)x0[i]=(s[i]=='1'?1:-1);
  RNG r(9918273); for(int i=NS;i<N;i++)x0[i]=(r.next()&1)?1:-1;
 }
 double eBath(const vector<int8_t>&s) const {double e=0;for(int i=NS;i<N-1;i++)e-=J[i]*s[i]*s[i+1];return e;}
 double eInt(const vector<int8_t>&s) const {return -J[NS-1]*s[NS-1]*s[NS]-J[N-1]*s[N-1]*s[0];}
 double eSys(const vector<int8_t>&s) const {double e=0;for(int i=0;i<NS-1;i++)e-=J[i]*s[i]*s[i+1];return e;}
};
struct Stats {double p=0,c=0,w=0;};
static inline double basis(int k,double u){
 double z=2*u-1, p=1;
 if(k==1)p=z;
 if(k==2)p=(3*z*z-1)/2;
 if(k==3)p=(5*z*z*z-3*z)/2;
 if(k==4)p=(35*z*z*z*z-30*z*z+3)/8;
 return sin(M_PI*u)*p;
}
static inline double fieldAt(const vector<double>&a,int base,double u){
 double h=0; for(int k=0;k<K;k++)h+=a[base+k]*basis(k,u); return h;
}
struct One {bool hit; double cost,work;};
One trajectory(const Model&m,const vector<double>&a,bool bathControl,uint64_t seed,
               double *eb0=nullptr,double *eb1=nullptr,double *ei0=nullptr,double *ei1=nullptr,double *dout=nullptr){
 RNG r(seed); vector<int8_t>s=m.x0; double d=0;
 double b0=m.eBath(s), i0=m.eInt(s);
 for(int z=0;z<HREL*m.N;z++){
  int i=r.idx(m.N), l=(i+m.N-1)%m.N, rr=(i+1)%m.N;
  double de=2.0*s[i]*(m.J[l]*s[l]+m.J[i]*s[rr]);
  if(de<=d+1e-12 && d-de<=EMAX+1e-12){d-=de;s[i]=-s[i];}
 }
 double b1=m.eBath(s), i1=m.eInt(s);
 if(eb0)*eb0=b0;if(eb1)*eb1=b1;if(ei0)*ei0=i0;if(ei1)*ei1=i1;if(dout)*dout=d;
 double cost=0,work=0; vector<double> prev(m.N,0.0);
 for(int st=0;st<CTRL;st++){
  double u=double(st)/(CTRL-1), du=1.0/(CTRL-1);
  for(int i=0;i<m.N;i++){
   bool active=bathControl ? (i>=NS) : (i<NS);
   if(!active)continue;
   int base=bathControl ? (i-NS)*K : i*K;
   double h=fieldAt(a,base,u);
   double dh=h-prev[i];
   work-=double(s[i])*dh;
   cost+=h*h*du;
   prev[i]=h;
  }
  for(int z=0;z<m.N;z++){
   int i=r.idx(m.N), l=(i+m.N-1)%m.N, rr=(i+1)%m.N;
   double de=2.0*s[i]*(m.J[l]*s[l]+m.J[i]*s[rr]);
   bool active=bathControl ? (i>=NS) : (i<NS);
   if(active){int base=bathControl?(i-NS)*K:i*K; de+=2.0*fieldAt(a,base,u)*s[i];}
   if(de<=d+1e-12 && d-de<=EMAX+1e-12){d-=de;s[i]=-s[i];}
  }
 }
 int same=0;for(int i=0;i<NS;i++)if(s[i]==m.x0[i])same++;
 return {same==NS,cost,work};
}
Stats evaluate(const Model&m,const vector<double>&a,bool bathControl,int n,uint64_t seed0){
 long long hits=0; long double c=0,w=0;
 for(int j=0;j<n;j++){auto q=trajectory(m,a,bathControl,seed0+uint64_t(j)*0x9e3779b97f4a7c15ULL);hits+=q.hit;c+=q.cost;w+=q.work;}
 return {double(hits)/n,double(c/n),double(w/n)};
}
double objective(const Stats&s){
 double v=max(0.0,0.10-s.p);
 return s.c/100.0 + 500.0*v*v;
}
struct Opt {vector<double>a; Stats train; double bestObj; int iter;};
Opt optimize(const Model&m,bool bathControl,int iters,int ntrain,uint64_t seed){
 int dim=(bathControl?m.nb:NS)*K; vector<double>a(dim,0.0),best=a; double bestObj=1e100; Stats bestS;
 RNG prng(seed^0x1234abcd9876ULL);
 for(int it=0;it<iters;it++){
  vector<double>delta(dim); for(int i=0;i<dim;i++)delta[i]=(prng.next()&1)?1.0:-1.0;
  double ck=0.25/pow(double(it+1),0.101);
  vector<double>ap=a,am=a;
  for(int i=0;i<dim;i++){ap[i]+=ck*delta[i];am[i]-=ck*delta[i];}
  // Common random numbers: identical trajectory seeds for + and - evaluations.
  Stats sp=evaluate(m,ap,bathControl,ntrain,seed+0xabc000ULL);
  Stats sm=evaluate(m,am,bathControl,ntrain,seed+0xabc000ULL);
  double gp=(objective(sp)-objective(sm))/(2*ck);
  double ak=0.35/pow(double(it+21),0.602);
  for(int i=0;i<dim;i++)a[i]=clamp(a[i]-ak*gp*delta[i],-8.0,8.0);
  if((it+1)%10==0){
   Stats sc=evaluate(m,a,bathControl,ntrain,seed+0xabc000ULL);
   double ob=objective(sc);
   if(ob<bestObj){bestObj=ob;best=a;bestS=sc;}
   cerr<<"bath="<<m.nb<<" mode="<<(bathControl?"BATH":"SYSTEM")<<" iter="<<(it+1)<<" p="<<sc.p<<" C="<<sc.c<<" obj="<<ob<<"\n";
  }
 }
 return {best,bestS,bestObj,iters};
}
double chi2p_approx(double chi2,int df){
 // Wilson-Hilferty approximation to upper-tail chi-square probability.
 double z=(pow(max(chi2,1e-12)/df,1.0/3.0)-(1.0-2.0/(9.0*df)))/sqrt(2.0/(9.0*df));
 return 0.5*erfc(z/sqrt(2.0));
}
void demonReport(const Model&m,int n,uint64_t seed){
 const int B=11; vector<long long>hist(B,0); long double eb0=0,eb1=0,ei0=0,ei1=0,db=0,db2=0,deb=0,deb2=0,dei=0,dei2=0;
 for(int j=0;j<n;j++){
  double a,b,c,d,e; trajectory(m,vector<double>(NS*K,0.0),false,seed+uint64_t(j)*0x9e3779b97f4a7c15ULL,&a,&b,&c,&d,&e);
  eb0+=a;eb1+=b;ei0+=c;ei1+=d;db+=e;db2+=e*e;double xb=b-a,xi=d-c;deb+=xb;deb2+=xb*xb;dei+=xi;dei2+=xi*xi;
  int bin=min(B-1,max(0,int(floor(e/2.0))));hist[bin]++;
 }
 double meanD=double(db/n), varD=max(0.0,double(db2/n)-meanD*meanD), seD=sqrt(varD/n);double meanDEB=double(deb/n),seDEB=sqrt(max(0.0,double(deb2/n)-meanDEB*meanDEB)/n);double meanDEI=double(dei/n),seDEI=sqrt(max(0.0,double(dei2/n)-meanDEI*meanDEI)/n);
 // Fit a discrete truncated exponential over demon-energy bins 0..20 by MLE.
 auto meanFor=[&](double beta){long double z=0,mu=0;for(int j=0;j<B;j++){double w=exp(-beta*2.0*j);z+=w;mu+=(2.0*j)*w;}return double(mu/z);};
 double target=meanD,lo=-5,hi=5;
 for(int q=0;q<120;q++){double mid=(lo+hi)/2;if(meanFor(mid)>target)lo=mid;else hi=mid;}
 double beta=(lo+hi)/2, mu=meanFor(beta), second=0,z=0;
 for(int j=0;j<B;j++){double w=exp(-beta*2.0*j);z+=w;second+=(2.0*j)*(2.0*j)*w;}
 double var=second/z-mu*mu, betaSE=(var>1e-12)?1.0/sqrt(n*var):NAN;
 double chi=0;for(int j=0;j<B;j++){double expected=n*exp(-beta*2.0*j)/z;if(expected>0)chi+=(hist[j]-expected)*(hist[j]-expected)/expected;}
 int df=B-2;double pv=chi2p_approx(chi,df);
 cout<<"THERMO bath="<<m.nb<<" N="<<n
     <<" EB_before="<<double(eb0/n)<<" EB_after="<<double(eb1/n)<<" dEB="<<double((eb1-eb0)/n)
     <<" dEB_SE="<<seDEB<<" Eint_before="<<double(ei0/n)<<" Eint_after="<<double(ei1/n)<<" dEint="<<meanDEI<<" dEint_SE="<<seDEI
     <<" Ed_mean="<<meanD<<" Ed_SE="<<seD<<" beta_fit="<<beta<<" beta_SE="<<betaSE
     <<" chi2="<<chi<<" df="<<df<<" p_approx="<<pv
     <<" fit_valid="<<((pv>0.05)?"YES":"NO")<<" hist=";
 for(int j=0;j<B;j++)cout<<hist[j]<<(j+1==B?"":";");
 cout<<"\n";
}
int main(){cout<<setprecision(8);for(int nb: {8,16,32,64}){Model m(nb);demonReport(m,20000,7000000ULL+nb*100000ULL);}}
