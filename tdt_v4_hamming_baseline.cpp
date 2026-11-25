#define main old_main
#include "tdt_v4_spsa_hamming.cpp"
#undef main
int main(){cout<<setprecision(8);for(int nb:{8,16,32,64}){Model m(nb);for(bool bc:{false,true}){int dim=(bc?nb:NS)*K;vector<double>a(dim,0.0);Stats s=evaluate(m,a,bc,12000,180000000ULL+nb*1000000ULL+(bc?777777ULL:0ULL));cout<<"BASELINE bath="<<nb<<" mode="<<(bc?"BATH":"SYSTEM")<<" p="<<s.p<<" H="<<s.h<<" C="<<s.c<<"\n";}}}
