#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os 
import time

def sigmoid(z):
    ''' fonction d'activation tanh '''
    return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
    ''' la dérivée de tanh '''
    return sigmoid(z)*(1-sigmoid(z))

class pmc:
    ''' perceptron multicouches . '''

    def __init__(self, *taille):
        ''' initialisation du perceptron selon sa taille .  '''
        numpy_rng = np.random.RandomState(1234)
        self.numpy_rng=numpy_rng

        self.shape =taille
        n = len(taille)
        # Build couches
        self.couches = []
        #build biais 
        self.biais = []
        # Input layer (+1 unit for biais)
        self.couches.append(np.ones(self.shape[0]))
        # couches cachées +la couche de sortie 
        for i in range(1,n):
            self.couches.append(np.ones(self.shape[i]))
        #biais 
            self.biais.append(numpy_rng.uniform(-0.5,0.5,self.shape[i]))
        # # construire la matrice des poids (0.5 -0.5)
        self.poids = []
        for i in range(n-1):
            self.poids.append(np.zeros((self.couches[i].size,
                                         self.couches[i+1].size)))

 

        # initialisations des poids
        #self.init_poids()

    def init_poids(self):
        ''' init_poids poids'''

        for i in range(len(self.poids)):
            #Z = np.random.random((self.couches[i].size,self.couches[i+1].size))
            #self.poids[i][...] = (2*Z-1)*0.5
            self.poids[i][...] =self.numpy_rng.uniform(-0.5,0.5,size=(self.couches[i].size,self.couches[i+1].size))

    def propagation_avant(self,data):
        ''' propagation des données de l'entrée vers la sortie. '''

        # chargement de données 
        self.couches[0] = data
        #self.couches[0] = data

        # propagation de la couche 0(couche d'entrée)  à n-1 (couche de sortie)
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.couches[i][...] =sigmoid(np.dot(self.couches[i-1],self.poids[i-1])+self.biais[i-1])

        # Returner la sortie du réseau
        return self.couches[-1]

    def retro_propagation(self, desiree, t_a=0.0001):
        ''' la retropropagation. '''

        deltas = []

        # calcule de l'erreur de la couche de sortie 
        erreur = desiree - self.couches[-1]
        delta = erreur*dsigmoid(self.couches[-1])
        deltas.append(delta)

        # calcule de l'erreur des couche cahcées
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.poids[i].T)*dsigmoid(self.couches[i])
            deltas.insert(0,delta)
        #mise à jours des biais 

        for i in range(len(self.biais)):
            self.biais[i]=self.biais[i]+t_a*deltas[i]

        # mise à jours des poids 
        for i in range(len(self.poids)):
            layer = np.atleast_2d(self.couches[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.poids[i]= self.poids[i] +t_a*dw


        # retourner l'errur quadratique moyenne des sortie 
        return ((erreur**2).sum())*0.5

    
    def affiche(self):
		print  self.poids
		print "----------------------------------------"
        
def learn(reseau,data,data_t ,it=100, t_a=0.009):
        # entrainement  
        eqm=[]
        s=[]
        for i in range(it):
              eqm=0
              d=time.clock()
              for k in range(data.size):
                    #n = np.random.randint(data.size)
                    reseau.propagation_avant( data['entree'][k] )
                    reseau.retro_propagation( data['sortie'][k], t_a)
                    eqm= reseau.retro_propagation( data['sortie'][k], t_a)
                    eqm+=eqm
              s.append(eqm/data.size)
              f=time.clock()
              print i
              print eqm/data.size
              print (f-d),'s'
        np.savetxt('pop2.txt',s)
            
        # Test
        comp=0
        for i in range(data_t.size):
            a= reseau.propagation_avant( data_t['entree'][i] )
            #print i, data['entree'][i], '%.2f' % a,
            #print 'desiree  ',data_t['sortie'][i],"obsevee",a
            if (np.argmax(data_t['sortie'][i])!=np.argmax(a)):
                comp+=1
                print 'desiree  ',data_t['sortie'][i],"obsevee",a,
        recon_er=(100-(100*comp)/data_t.size)
        print 'taux de recon = ',recon_er
        print 'comteur = ',comp
            #print a
        print 



if __name__ == '__main__':
    os.system("clear")


    #taille du réseau 
    reseau = pmc(784,300,10)
    #la forme des données 

    data = np.zeros(10000, dtype=[('entree',  float,784), ('sortie', float,10)])
    data_t = np.zeros(6000, dtype=[('entree',  float, 784), ('sortie', float, 10)])
    reseau.init_poids()
    data['entree']=np.loadtxt('train-mnist.txt')
    data['sortie']=np.loadtxt('train-label.txt')
    data_t['entree']=np.loadtxt('test-mnist.txt')
    data_t['sortie']=np.loadtxt('test-label.txt')
    d=time.clock()
    learn(reseau,data,data_t)
    f=time.clock()
    print (f-d),'s'
