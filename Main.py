# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:41:43 2022

@author: colin
"""
import numpy as np
import Biblio as bbl 
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

#%%
"""
                                   3 Méthodes Numériques
"""
#%% 
"""
3.1 Sans Contraintes
"""
#%% Données intiales
M , N = 200, 300
t0 , tf = 0 , 10
c = 1
theta = 1
L = 1
dt=tf/(N+1)
dx=1/(M+1)
gamma=c*(dt/dx)
x=np.linspace(dx,1-dx,M)


#%% Inititialisation du schéma numérique

#Matrice de Laplace
A=bbl.MatlapA(M)

u0=bbl.u0(x)
u1=u0+dt*bbl.v0(x) +0.5*(dt**2)*(-(c/dx)**2)*A@u0

P2=np.eye(M)+(gamma**2)*(theta/2)*A
P1=(2*np.eye(M)+(gamma**2)*(1-theta)*A)
P0=-P2

b=P1@u1+P0@u0

#%% Application Gradient Conjugué

epsilon=10**(-6)
x0=np.ones((M,1))

# Chaque colonne de Utot représente la position de la corde à un instant 
Utot=np.zeros((M,N+3))
# Les deux premières colonnes sont donc connues :
Utot[:,0]=u0.reshape(M)
Utot[:,1]=u1.reshape(M)

# Calcul de la suite de la matrice Utot à partir des valeurs initiales grâce au gradient projeté
for i in tqdm(range(0,N),desc="Calcul Utot"):
    b=P1@Utot[:,i+1].reshape(M,1)+P0@Utot[:,i].reshape(M,1)
    Utot[:,i+2]=bbl.GPC(P2,b,x0,epsilon).reshape(M)
#On force les extrémités de la corde à être fixées à 0
    Utot[0,i]=0
    Utot[-1,i]=0

#%% Sortie vidéo sans contrainte

#Création de l'animation
fig, ax=plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(-2,2)
ax.set_title("Oscillations d'une corde sans contrainte")

line, = ax.plot(0,0)
def ani(i):
    line.set_xdata(x)
    line.set_ydata(Utot[:,i])   
animation = FuncAnimation(fig, func=ani, frames=np.arange(0,N,1),interval=dt)

#On enregistre la vidéo sous forme de gif
animation.save('video_sans_contraintes.gif', writer = 'pillow', fps = 40)
print("L'animation sans contraintes est générée")
#%% Comparaison avec solution approchée

#Définition de la solution approchée
def u(x,t):
    return (0.5*(bbl.u0(x-c*t)+bbl.u0(x+c*t))+(2/(np.pi*c))*(np.cos(np.pi*(x+c*t))-np.cos(np.pi*(x-c*t))))

#On crée une matrice comme précédement une matrice contenant la position de la corde à chaque instant
Utotapprox=np.zeros((M,N+2))
time=0
for i in tqdm(range(N+2),desc="Calcul Utotapprox"):
    time+=dt
    for index ,l in enumerate(x):
            Utotapprox[index,i]=u(l,time)
            
#%% Calcul de l'erreur maximale entre les 2 méthodes

erreur=np.zeros((N+2))
for i in tqdm(range(N+2),desc="Calcul erreur"):
    erreur[i]=round((np.linalg.norm(Utotapprox[:,i-1]-Utot[:,i]))/(np.linalg.norm(Utot[:,i])),2)
    
maxerror=np.max(erreur)

#%% Sortie vidéo  :

#Création de l'animation
fig, ax=plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(-2,2)
line1, = ax.plot(0,0,"r--",label="U total")
line2, = ax.plot(0,0,"b--" ,label="U approximé")
legend = ax.legend(loc='upper right', shadow=True)
def ani(i):     
    line1.set_xdata(x)
    line1.set_ydata(Utot[:,i])  
    line2.set_xdata(x)
    line2.set_ydata(Utotapprox[:,i-1]) 
    erreur=round((np.linalg.norm(Utotapprox[:,i-1]-Utot[:,i]))/(np.linalg.norm(Utot[:,i])),2)
    ax.set_title("Oscillations d'une corde sans contraintes\n Erreur absolue d'approximation : {} \n Erreur maximale : {}".format(erreur,maxerror))
animation = FuncAnimation(fig, func=ani, frames=np.arange(1,N,1),interval=dt)
#On enregistre la vidéo sous forme de gif
animation.save('video_comparaison_sans_contraintes.gif', writer = 'pillow', fps = 60)
print("L'animation de comparaison sans contraintes est générée")
    
#%% 
"""
3.2 Avec Contraintes
"""
#%% Données intiales
M , N = 200, 300
t0 , tf = 0 , 10
c = 1
theta = 1
L = 1
dt=tf/(N+1)
dx=1/(M+1)
gamma=c*(dt/dx)
x=np.linspace(dx,1-dx,M)

#%% Inititialisation du schéma numérique 
A=bbl.MatlapA(M)
u0=bbl.u0(x)
u1=u0+dt*bbl.v0(x) +0.5*(dt**2)*(-(c/dx)**2)*A@u0
P2=np.eye(M)+(gamma**2)*(theta/2)*A
P1=(2*np.eye(M)+(gamma**2)*(1-theta)*A)
P0=-P2
b=P1@u1+P0@u0

#%% Application Gradient Projeté

alpha=np.min(np.abs(np.linalg.eigvals(P2)))
m= (np.max(np.abs(np.linalg.eigvals(P2))))
rhomax=2*alpha/m
rho=rhomax/2  # On divise par deux pour être sur d'avoir un rho qui fonctionne, même si ce n'est pas forcément le meilleur.
epsilon=10**(-10)
x0=np.ones((M,1))
#Création de Utotsc, relativement selon le même principe que pour la méthode sans contraintes.
Utotsc=np.zeros((M,N+2))
Utotsc[:,0]=u0.reshape(M)
Utotsc[:,1]=u1.reshape(M)
Utotsc[0,:]=0
Utotsc[-1,:]=0
for i in tqdm(range(0,N),desc="Calcul Utotsc"):
    b=P1@Utotsc[:,i+1].reshape(M,1)+P0@Utotsc[:,i].reshape(M,1)
    Utotsc[:,i+2]=bbl.Gproj(P2,b,rho,x0,epsilon).reshape(M)

#%% Sortie vidéo 

lim=0.5*np.ones((M))

#Création de l'animation
fig, ax=plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(-2,2)

ax.plot(x,lim,color='r')
ax.plot(x,-lim,color='r')
line, = ax.plot(0,0)
def ani(i):
    line.set_xdata(x)
    line.set_ydata(Utotsc[:,i])  
    ax.set_title("Oscillations d'une corde avec une contrainte entre {} et {}".format(lim[0],-lim[0]))
animation = FuncAnimation(fig, func=ani, frames=np.arange(0,N,1),interval=dt)
#On enregistre la vidéo sous forme de gif
animation.save('video_contrainte.gif', writer = 'pillow', fps = 40)
print("L'animation avec contraintes est générée")