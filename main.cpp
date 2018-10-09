

#include "pmc.h"


using namespace std;
char base[20];
char wcc[20];
char test[20];
FILE *f,*f2,*f1,*f3;
float *poidbcc,*biaiscc,*poidbcs,*biaiscs,**pcc,**pcs,*ys,*yc,*x,*d,*es,*dk,*dj,*pbcc,*pbcs,**X,**Y,*dt;
int nncc;//nombre de neuron de la couche cachée
int nncs;//nombre de neuron de la couche de sortie 
int nnce;//nombre d'entrée 
int nbf,nbft;//nbf nombre de forme a reconnaitre nbft nombre de formes de test 
double eps,em,g;
int tmax;
int t=0;
int comp=0;
int compt=0;
int k,i,a;
clock_t t1=0,t2=0;
double s=0;

int main()
{                      system("clear");         
              



cout<<"            ______________________________________________________________________________________________"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                              RNA (PMC)                                     |"<<endl;
cout<<"            |                       régle d'apprentissage :la rétropropgation des erreurs                |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                              réalisé par :                                                 |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                      *CHERRAQI ELBOUAZZAOUI                                |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |                                                                                            |"<<endl;
cout<<"            |____________________________________________________________________________________________|"<<endl;
          




cout<<"Nom du fichier contenant les entrées et les valeurs désiréés du reseau--------------->"; cin>>base;
f=fopen(base,"rt");
cout<<"Le nombre d'exemple à apprendre ----------------------------------------------------->"; //les exemples à apprendre
cin>>nbf;
cout<<"Le nombre des éléments des vecteurs objets ------------------------------------------>"; 
cin>>nnce;
cout<<"Le nombre de nuerons de la couche cachée--------------------------------------------->";
cin>>nncc;
cout<<"Le nombre de nuerons de la couche de sortie------------------------------------------>";
cin>>nncs;
cout<<endl;				
				


pcc=new float*[nncc];  for(int i=0;i<nncc;i++) pcc[i]=new float[nnce];
pcs=new float*[nncs];  for(int i=0;i<nncs;i++) pcs[i]=new float[nncc];
srand(time(0));
poidbcc=new float [nncc];poidbcs=new float [nncs];
biaiscc=new float [nncc];biaiscs=new float [nncs];
yc=new float [nncc];ys=new float [nncs];x=new float [nnce];X=new float*[nbf];for(int i=0;i<nbf;i++) X[i]=new float [nnce+nncs];
d=new float [nncs];es=new float [nncs];dk=new float [nncs];Y=new float*[nbf];for(int i=0;i<nbf;i++) Y[i]=new float [nncs];
dj=new float [nncc];
dt=new float [nncs];
               
     
//préparation du reseau%



            
for(int i=0; i<nbf; i++){ for(int j=0; j<nnce; j++){ fscanf(f,"%f",&X[i][j]); } for(int m=0;m<nncs;m++)fscanf(f,"%f",&Y[i][m]);}//chargement des vecteurs d'objets et les sorties désirée correspondante    

poidbiaiscc(poidbcc,biaiscc,nncc);// poids correspondant au biais des neurones de la couche cachée
		
poidbiaiscs(poidbcs,biaiscs,nncs);//poids correspondant au biais des neurons de la couche  sortie
		
poidcc(pcc,nncc,nnce);//poids de couches cachée
		
poidcs(pcs,nncs,nncc);//poids de couches de sortie 
		

char p;
cout<<"pour faire l'apprentissage entrer p"<<endl;cin>>p;

    if (p=='p'|| p=='P'){  



                       cout<<"Le nombre d'itération maximum--------------------------------------------------------------->";
		       cin>>tmax;
		       cout<<"l'erreur quadratique tolérable observée sur l'ensemble des neurons de la couche de sortie--->";
		       cin>>eps;


  //Etape d'apprentissage  	

do {t1=clock(); t++;  em=0;   permut(X,nbf,nnce);permut(Y,nbf,nncs);

                for(int k=0;k<nbf;k++) {for(int j=0; j<nnce; j++){ x[j]=X[k][j];}  for(int j=0; j<nncs; j++)  {d[j]=Y[k][j];}//+
		propagationavant(poidbcc,pcc,biaiscc,pcs, biaiscs,poidbcs,yc,x,ys ,nnce,nncc,nncs);//Fonction de propagation avant 	
		erreurdesortie(es,ys,d,nncs);//erreur de sortie 
		
                for(int l=0;l<nncs;l++)//+
                em+=0.5*es[l]*es[l]; //erreur quadratique correspondante à l'ensemble  des sorties     
		
                deltacs(dk, es ,ys ,nncs);//gradient local de la couche de sortie 

		actpoidcs(pcs,dk,yc,nncc,nncs);//actualisation de poid de couche de sortie 

                actbpcs(poidbcs,dk,nncs);//actualisation de poid de biais  couche de sortie
		
                deltacc(pcs,dk,dj,yc,nncc,nncs);//gradient local de la couche cachée 
		
		actpoidcc(pcc,dj,x ,nncc,nnce);//actualisation de poid de couche de cachée 

                actbpcc(poidbcc,dj,nncc);//actualisation de poids de biai de couche de cachée 

           
   }
  

g=em/nbf;

cout<<"itération N° "<<t<<" Erreur Quadratique Moyenne "<<g<<endl;
t2=clock();
s+=(double)(t2-t1)/CLOCKS_PER_SEC;
printf("%3f secondes \n",(double)(t2-t1)/CLOCKS_PER_SEC);

}while((t<tmax) && (g>eps));

printf("%3f Temps total \n",s);

cout<<" fin de l'apprentissage dans >"<<t<<" itération"<<endl;

          


//enregistrement des poids synaptiques 


               cout<<"Nom du fichier pour enregistrer les poids synaptiques :"; cin>>wcc;
                          f2=fopen(wcc,"w");


		for(int i=0;i<nncc;i++){for(int j=0;j<nnce;j++){ fprintf(f2,"%f\t",pcc[i][j]);}fprintf(f2,"\n");}// enregistrement des pcc 
		for(int i=0;i<nncs;i++){for(int j=0;j<nncc;j++){ fprintf(f2,"%f\t",pcs[i][j]);}fprintf(f2,"\n");}// enregistrement des pcs 
		for(int i=0;i<nncc;i++){{fprintf(f2,"%f\t",poidbcc[i]);fprintf(f2,"%f\t",biaiscc[i]);fprintf(f2,"\n");}}
		for(int i=0;i<nncs;i++){{fprintf(f2,"%f\t",poidbcs[i]);fprintf(f2,"%f\t",biaiscs[i]);fprintf(f2,"\n");}}
		fclose(f2);}




//étape de test



cout<< " pour faire un test "<<endl;
cout<<"Nom du fichier contenant  les poids synaptiques : "; cin>>wcc;
f2=fopen(wcc,"rt");

		for(int i=0;i<nncc;i++){for(int j=0;j<nnce;j++){ fscanf(f2,"%f",&pcc[i][j]);}}//  
		for(int i=0;i<nncs;i++){for(int j=0;j<nncc;j++){ fscanf(f2,"%f",&pcs[i][j]);}}// 
		for(int i=0;i<nncc;i++){{fscanf(f2,"%f",&poidbcc[i]);fscanf(f2,"%f",&biaiscc[i]);}}
		for(int i=0;i<nncs;i++){{fscanf(f2,"%f",&poidbcs[i]);fscanf(f2,"%f",&biaiscs[i]);}}
		fclose(f2);



		cout<<"entrer le fichier contenant la base de test "<<endl;
		cin>>test;
		cout<<"le nombre d'exemples  de test ";cin>>nbft;
		f3=fopen(test,"rt");


for(int i=0; i<nbft; i++){ for(int j=0; j<nnce; j++){ fscanf(f3,"%f",&X[i][j]);} for(int k=0;k<nncs;k++)fscanf(f3,"%f",&Y[i][k]);
    }


for(int i=0;i<nbft;i++){for(int j=0;j<nnce;j++){ x[j]=X[i][j];}for(int k=0;k<nncs;k++){d[k]=Y[i][k];if(d[k]==1)  cout<<"Appliance N°"<<k<<endl;}

          
 
		propagationavant(poidbcc,pcc,biaiscc,pcs, biaiscs,poidbcs,yc,x,ys ,nnce,nncc,nncs);
                 
		//cout<<"appliance   "<<i<<endl; 
		for(int i=0;i<nncs;i++)
		cout<<ys[i]<<setw(3)<<"  ";
		 cout<<endl;


		for(int i=0;i<nncs;i++)

                            if(fabs(d[i]-ys[i])>0.3) comp++;
                                  int a=wta(ys,nncs);
           
                            ys[a]=1;  if(ys[a]!=d[a]) compt++;
                 }

 cout<<endl;
 cout<<endl;
 cout<<"Le taux de reconnaissance  par la méthode de seuil ------------------->"<<100-((comp*100)/nbft)<<"  % "<<endl;
 cout<<endl;
 cout<<endl;
 cout<<"Le taux de reconnaissance  par la methode the Winner Takes All (WTA)-->"<<100-((compt*100)/nbft)<<"  % "<<endl;

 cout<<endl;
 cout<<endl;

		
return 0;}

