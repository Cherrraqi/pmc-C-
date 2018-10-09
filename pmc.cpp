//09/12/2013 Cherraqi El Bouazzaoui
//définition des fonctions  utilisées dans  le PMC
#include"pmc.h"	
#define n 0.1
using namespace std;



void poidbiaiscc(float *poidbcc,float *biaiscc,int nncc)

	  {  srand(time(NULL));  for(int i=0;i<nncc;i++)
               {
              poidbcc[i]=-0.5+(float)rand()/RAND_MAX;
		biaiscc[i]=1;
                    }
           }



void poidbiaiscs(float* poidbcs,float* biaiscs,int nncs)
             {
               srand(time(NULL));
	for(int i=0;i<nncs;i++){
              poidbcs[i]=-0.5+(float)rand()/RAND_MAX;
		biaiscs[i]=1;  }

             }


void poidcc(float **poidcc,int nncc,int nnce)
		{
                for(int i=0;i<nncc;i++) for(int j=0;j<nnce;j++)
	
		poidcc[i][j]=-0.5+(float)rand()/RAND_MAX;
                }




void poidcs(float** poidcs,int nncs,int nncc)
	 {	
              for(int i=0;i<nncs;i++) for(int j=0;j<nncc;j++)

			poidcs[i][j]=-0.5+(float)rand()/RAND_MAX;	 
        }

float sigmoide(float y)         
	{
  	  return 1/(1 + exp(-(double)y));
	} 



void propagationavant(float* poidbcc,float** poidcc,float* biaiscc,float** poidcs,float* biaiscs,float* poidbcs,float* yc,float* x,float* ys ,int nnce,int nncc ,int nncs )
                           {                                                                 
                           for(int i=0;i<nncc;i++){		
                                  float sommac=0;  for(int j=0;j<nnce;j++){  sommac+=x[j]*poidcc[i][j];}
                           yc[i]=sigmoide(sommac+biaiscc[i]*poidbcc[i]);
                                                  }

		          for(int i=0;i<nncs;i++){
				     float sommas=0; for(int j=0;j<nncc;j++){

                                       sommas+=yc[j]*poidcs[i][j]; }

                                          ys[i]=sigmoide(sommas+biaiscs[i]*poidbcs[i]);
                                                 }
                             }




void erreurdesortie(float* es,float* ys,float* d,int nncs) 
               {  for(int i=0;i<nncs;i++){
                 
                         es[i]=d[i]-ys[i];}
                                             
                                                     
                }

void deltacs(float* dk,float* es ,float* ys ,int nncs)
                { for(int i=0;i<nncs;i++)           

                        dk[i]=es[i]*(ys[i]*(1-ys[i]));    
                } 





void actpoidcs(float** pcs,float* dk,float* yc,int nncc,int nncs)
                {
                     for(int i=0;i<nncs;i++)for(int j=0;j<nncc;j++)
                   {
		    pcs[i][j]=pcs[i][j]+(n*dk[i]*yc[j]);
                   }
		}		






void actbpcs(float *pbcs,float* dk,int nncs)
                  { 

                  for(int i=0;i<nncs;i++)   pbcs[i]=pbcs[i]+n*dk[i];
                   }



void deltacc(float **poidcs,float *dk,float *dj,float *yc,int nncc,int nncs) 
			{  
              for(int i=0;i<nncc;i++)
                           {float somme=0;
               for(int j=0;j<nncs;j++) {somme+=dk[j]*poidcs[j][i];
               dj[i]=yc[i]*(1-yc[i])*somme;}}
                        }




 void actpoidcc(float **poidcc,float *dj,float* x ,int nncc,int nnce){ 

			for(int i=0;i<nncc;i++){ for(int j=0;j<nnce;j++){
		poidcc[i][j]=poidcc[i][j]+n*dj[i]*x[j];}}

}



void actbpcc(float *pbcc,float *dj,int nncc)
                     {  

                  for(int i=0;i<nncc;i++)   pbcc[i]=pbcc[i]+n*dj[i];
                                   }

void permut(float **X, int nbf , int nnce)
                   {
                    for(int i=0;i<nbf-1;i++)for(int j=0;j<nnce;j++)
                                              {
                                       float tmp=X[i][j];
                                              X[i][j]=X[i+1][j]; 
                                                X[i+1][j]=tmp;
                                               }
                     }


//the winner takes all 	

int  wta(float *ys,int nncs){
               int max=0;
               for(int i=0;i<nncs;i++) {
                                
                                 if(ys[i]>ys[max]) max=i;}
            
                          return max;	}


