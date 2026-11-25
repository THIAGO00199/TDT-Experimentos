#include <bits/stdc++.h>
using namespace std;
static uint64_t sm(uint64_t &x){uint64_t z=(x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
struct RNG{uint64_t s;RNG(uint64_t v):s(v){}uint64_t next(){return sm(s);}int idx(int n){return int(next()%n);}};
int main(){
 const int baths[]={8,16,32,64}, amps[]={12,20,40,80}, ctrls[]={5,10,20,50};
 const int NSAMP=10000,HREL=20,NS=8;
 cout<<setprecision(10)<<"Creutz joint amplitude-duration scan; t_relax=20; N=10000 per cell; system exact-state fidelity\n";
 for(int nb:baths){int N=NS+nb;vector<int>J(N,1);for(int i=0;i<NS-1;i++)J[i]=2;
  vector<int8_t>x0(N);string sys="10101010";for(int i=0;i<NS;i++)x0[i]=(sys[i]=='1'?1:-1);
  RNG brng(9918273);for(int i=NS;i<N;i++)x0[i]=(brng.next()&1)?1:-1;
  cout<<"\nBATH "<<nb<<" N "<<N<<"\n";
  for(int mode=0;mode<2;mode++){
   cout<<"MODE "<<(mode==0?"SYSTEM_ONLY":"SYSTEM_AND_BATH")<<"\n";
   for(int H:amps)for(int CTRL:ctrls){long long hit=0;long double ws=0,wb=0,ws2=0,wb2=0,dem=0;
    RNG rng(20261113ULL+nb*1000003ULL+H*10007ULL+CTRL*101ULL+mode*900000001ULL);
    for(int tr=0;tr<NSAMP;tr++){
     vector<int8_t>s=x0;int d=0;
     auto att=[&](int i,bool field){int l=(i+N-1)%N,r=(i+1)%N;int de=2*s[i]*(J[l]*s[l]+J[i]*s[r]);if(field&&(mode==1||i<NS))de+=2*H*x0[i]*s[i];if(de<=d){d-=de;s[i]=-s[i];}};
     for(int k=0;k<HREL*N;k++)att(rng.idx(N),false);
     int os=0,ob=0;for(int i=0;i<NS;i++)os+=x0[i]*s[i];for(int i=NS;i<N;i++)ob+=x0[i]*s[i];
     for(int k=0;k<CTRL*N;k++)att(rng.idx(N),true);
     int fs=0,fb=0;for(int i=0;i<NS;i++)fs+=x0[i]*s[i];for(int i=NS;i<N;i++)fb+=x0[i]*s[i];
     double a=H*(fs-os),b=(mode==1?H*(fb-ob):0);
     ws+=a;wb+=b;ws2+=a*a;wb2+=b*b;dem+=d;
     bool ok=true;for(int i=0;i<NS;i++)if(s[i]!=x0[i])ok=false;if(ok)hit++;
    }
    double ms=double(ws/NSAMP),mb=double(wb/NSAMP),p=double(hit)/NSAMP;
    double es=sqrt(max(0.0,double(ws2/NSAMP)-ms*ms)/NSAMP),eb=sqrt(max(0.0,double(wb2/NSAMP)-mb*mb)/NSAMP);
    cout<<"H="<<H<<" CTRL="<<CTRL<<" psys="<<p<<" hits="<<hit<<" Wsys="<<ms<<" SEsys="<<es<<" Wbath="<<mb<<" SEbath="<<eb<<" demon="<<double(dem/NSAMP)<<"\n";
   }
  }
 }
}