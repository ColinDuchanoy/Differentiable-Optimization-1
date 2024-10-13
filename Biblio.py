# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:41:19 2022

@author: colin
"""

#%%
"""
                                   3 Méthodes Numériques
"""
#%% 3.1 Sans Contraintes

import numpy as np 
import  copy

def u0(x):
    return (1/2)*np.sin(np.pi*x)

def v0(x):
    return -4*np.sin(np.pi*x)

def MatlapA(M):
   A=2*np.eye(M)-np.diag(np.ones(M-1),1)-np.diag(np.ones(M-1),-1) 
   return A

def GPC(A,b,x0,epsilon) : 
    x=copy.copy(x0)
    d=b-A@x
    compteur=0
    y=x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) >epsilon and compteur <1000 : 
         y=copy.copy(x)
         t=-(d.T @ (A@x-b))/(d.T @A @d)
         x=x+t*d
         beta=(d.T @ A @ (A@x-b)) / (d.T @ A @ d)
         d=-(A@x-b)+beta*d
         compteur+=1
    #print("GPC : La convergence à {} près est obtenue pour {} itérations.".format(epsilon,compteur))
    return x


#%% 3.2 Avec Contraintes

def proj(x) :
    lim = 1/2
    xproj=np.maximum(np.minimum(x,lim*np.ones(np.shape(x))),-lim*np.ones(np.shape(x)))
    return xproj

def Gproj(A,b,rho,x0,epsilon) : 
    x=copy.copy(x0)
    w=b-A@x
    compteur=0
    y=x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) >epsilon and compteur <1000 : 
         y=copy.copy(x)
         x=proj(x+rho*w)
         w=b-A@x
         compteur+=1
    #print("GProj : La convergence à {} près est obtenue pour {} itérations.".format(epsilon,compteur))
    return x