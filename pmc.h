//declaration des prototypes de fonctions 
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <math.h>
#include <fstream>
#include<algorithm>
using namespace std;





void poidbiaiscc(float*,float*,int);

void poidbiaiscs(float*,float*,int);

void poidcc(float**,int,int);

void poidcs(float**,int,int);

float sigmoide(float );

void propagationavant(float*,float**,float*,float**,float*,float*,float*,float*,float*,int,int,int);

void erreurdesortie(float*,float*,float*,int);

void deltacs(float*,float*,float*,int);

void actpoidcs(float**,float*,float*,int,int);

void deltacc(float **,float *,float *,float *,int,int);

void actpoidcc(float**,float*,float* ,int,int);

void actbpcc(float*,float*,int);

void actbpcs(float* ,float*,int);

void permut(float ** , int ,int );
int wta(float *,int);

