#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:02:28 2020

@author: nvthaomy
"""
import numpy as np
from scipy.integrate import simps
from numpy.linalg import det

#import matrix

class RPA():
    def __init__(self, nSpecies,nMolecule):
        '''RPA class
        Implement for DGC of homopolymer and copolymer
                      CGC of homopolymer
        TO DO: implement copolymer model for CGC'''
        
        self.M = nMolecule # molecule species
        self.Cm = np.ones(self.M) # vector of BEAD concentration for each molecule species, Cm = DOP_m * nm / V
        self.DOP = np.ones(self.M)
        self.struc = [] # list of chain structures made up of bead indices
        self.molCharge = np.zeros(self.M) # vector of charge/segment of molecues
        self.beadFrac = [] # matrix of bead fraction in each molecue
        self.n = np.ones(self.M) # number of chains for each molecue species  
        
        self.S = nSpecies # bead species                
        self.charge = np.zeros(self.S) # charges of bead types       
        self.C  = np.ones(self.S) # vector of bead species concentrations        
        self.abead = np.ones(self.S) # smearing length of beads
        self.u0 = np.ones((self.S,self.S)) # excluded vol matrix        
        self.b = np.ones(self.S) # RMS length of bond, can be anything for small molecules

        self.lB = 1.0 # Bjerrum length in kT
        self.V = 10.   
        self.kmin = 0.001 
        self.kmax = 1000 
        self.nk =  1000 # number of mesh point
        self.chain = 'DGC' # DGC or CGC
        
        self.USI = np.ones((self.S,self.S))
        self.a = np.ones((self.S,self.S)) # smearing length of interaction matrix
        
        
        self.k = [] 
        self.dk = 0.1  
        self.U = [] # k-space matrices of excluded vol int.
        self.Ue = []              
        self.Gpp = []
        self.Gee = []
        self.gD = None # for CGC, list of gD for each molecule, if provided, dont have to recalculate
        self.gm_list = None # for DGC, list of structure factor like matrices (at all k numbers) for each molecule, independent of concentration
        self.gme_list = None
        self.IncludeEvRPA = True # include perturbation term for the excluded volume interaction        
    
    def Setstruc(self,struc):
        self.struc = struc
        self.SetDOP()
        beadFrac = np.zeros([self.S,self.M])
        for m,struc in enumerate(self.struc):
            beads, counts = np.unique(struc, return_counts=True)
            f = np.array(counts,dtype=float)/float(np.sum(counts))
            for i, bead in enumerate(beads):
                beadFrac[int(bead),m] = f[i]
        self.beadFrac = beadFrac
        
    def SetDOP(self):
        DOP = np.zeros(self.M)
        for i,struc in enumerate(self.struc):
            DOP[i] = float(len(struc))
        self.DOP = DOP
    def Setcharge(self,charge):
        self.charge = np.array(charge, dtype=float)
    def SetCm(self,Cm):
        self.Cm = np.array(Cm, dtype=float)
        self.n = self.Cm * self.V / self.DOP
        if len(self.struc) == 0:
            raise Exception('Need to set the chain structure with Setstruc(struc)')
        else:
            self.C =  self.beadFrac@self.Cm
    def Setabead(self,abead):
        self.abead = np.array(abead, dtype=float)
    def Setu0(self,u0):
        self.u0 = np.array(u0, dtype=float)
    def SetlB(self,lB):
        self.lB = float(lB)
    def Setb(self,b):
        self.b = np.array(b, dtype=float)
    def SetV(self,V):
        self.V = float(V)
    def Setkmin(self,kmin):
        self.kmin = float(kmin)
    def Setkmax(self,kmax):
        self.kmax = float(kmax)
    def Setnk(self,nk):
        self.nk = int(nk)
    def Setchain(self,chain):
        self.chain = chain
    def gD_DGC(self,i):
        '''Debye function for discreate gaussian chain of species i at all wavenumbers'''
        N = self.DOP[i]
        b = self.b[i]  
        k = self.k
        gD = np.zeros(len(self.k))
        for m in range(int(N)):
            for j in range(m,int(N)): #calculating the upper half
                if m==j:
                    gD += 0.5 * np.exp(-b**2 * k**2/6. * np.abs(m-j)) #multiply by 0.5 to account for over counting when multiply by 2 at the end
                else:
                    gD += np.exp(-b**2 * k**2/6. * np.abs(m-j))
        return 1./N**2 *2 * gD

    def gD_CGC(self,i):
        '''Debye function for continuous gaussian chain of species i at all wavenumbers'''
        N = self.DOP[i]
        b = self.b[i]  
        k = self.k
        gD = np.zeros(len(self.k))
        x = N * k**2 * b**2 /6.
        gD = 2. * x**-2 * (x + np.exp(-x) - 1.)
        return gD
       
    
    def GetG(self, V=None, i=None, n=None):
        '''Density and charge correlation matrices at all wavenumbers'''
        C = self.C  
        Gpp = np.zeros((len(self.k),self.S,self.S))
        Gee = np.zeros((len(self.k),self.S,self.S))

        if self.chain == 'DGC':
            # list of structure factor like matrices (at all k numbers) for each molecule, independent of concentration
            if (isinstance(self.gm_list,list) or isinstance(self.gm_list,np.ndarray)) and (isinstance(self.gme_list,list) or isinstance(self.gme_list,np.ndarray)):
                gm_list = self.gm_list
                gme_list = self.gme_list
            else:
                gm_list = [] 
                gme_list = []
                for m, struc in enumerate(self.struc):
                    gm = np.zeros([len(self.k), self.S, self.S])
                    gme = np.zeros([len(self.k), self.S, self.S])
                    for l in range(len(struc)):
                        bead1 = struc[l]
                        # using the Kuhn length of bead l for now
                        b = self.b[bead1]
                        for j in range(len(struc)):
                            bead2 = struc[j]
                            val = np.exp(-b**2 * self.k**2 / 6. * np.abs(l-j))
                            gm[:,bead1,bead2] += val
                            gme[:,bead1,bead2] += val * self.charge[bead1] * self.charge[bead2]
                    gm_list.append(gm)
                    gme_list.append(gme)
                self.gm_list = gm_list
                self.gme_list = gme_list
            for m in range(self.M):
                Gpp[:,:,:] += self.Cm[m]/self.DOP[m] * gm_list[m]
                Gee[:,:,:] += self.Cm[m]/self.DOP[m] * gme_list[m]

        elif self.chain == 'CGC':    
            gD_list = []   
            for i in range(self.S):
                if isinstance(self.gD,list) or isinstance(self.gD,np.ndarray):
                    gD = self.gD[i]
                elif self.gD == None:
                    gD = self.gD_CGC(i)
                gD_list.append(gD)
                g = C[i] * self.DOP[i] * gD
                ge = C[i] * self.DOP[i] * self.charge[i]**2 * gD
                Gpp[:,i,i] = g
                Gee[:,i,i] = ge
                self.gD = gD_list
        return Gpp, Gee
        
    def GetU(self):
        '''Excluded volume interaction matrix at all wavenumbers'''
        U = np.zeros((len(self.k),self.S,self.S))
        k = self.k
        for i in range(self.S):
            for j in range(self.S):
                U[:,i,j] = self.u0[i,j] * np.exp(-self.a[i,j]**2 * k**2)
        return U
    
    def GetUe(self):
        '''Electrostatic interaction matrix at all wavenumbers'''
        Ue = np.zeros((len(self.k),self.S,self.S))
        k = self.k
        for i in range(self.S):
            for j in range(self.S):
                Ue[:,i,j] = 4. * np.pi * self.lB / k**2 * np.exp(-self.a[i,j]**2 * k**2)
        return Ue
    
    def DotUG(self):
        '''Matrix multiplication of U * Gpp, Ue *Gee'''    
        U = self.U
        Ue = self.Ue
        Gee = self.Gee
        Gpp = self.Gpp 
        UGpp = np.zeros([len(self.k),self.S, self.S])
        UeGee = np.zeros([len(self.k),self.S, self.S])
        
        for k in range(len(self.k)):
            UGpp[k] = np.matmul(U[k],Gpp[k])
            UeGee[k] = np.matmul(Ue[k],Gee[k])
        return UGpp, UeGee

    def Initialize(self):
        '''Smearing matrix'''
        a = np.ones((self.S,self.S))
        for i in range(self.S):
            for j in range(self.S):
                a[i,j] = np.sqrt((self.abead[i]**2 + self.abead[j]**2 )/2.)
        self.a = a
                            
        '''Self interaction energy in kB'''
        aii = self.a.diagonal()    
        u0ii = self.u0.diagonal()
        U = 0.5 * self.C * self.V *( u0ii/(4. * np.pi * aii**2) + self.lB)
        self.USI = np.sum(U) 

        '''wave number'''
        self.k, self.dk = np.linspace(self.kmin, self.kmax, self.nk, endpoint = True, retstep = True)
        
        '''Struture factor related var'''
        self.U = self.GetU()
        self.Gpp, self.Gee = self.GetG()
        self.Ue = self.GetUe()
        self.UGpp, self.UeGee = self.DotUG()
        I = np.array([np.identity(self.S)]*len(self.k))
        self.Xs = I + self.UGpp
        self.Ys = I + self.UeGee
        self.invXs = None #inverse of Xs[k,:,:] for all k values
        self.invYs = None #inverse of Ys[k,:,:] for all k values

    def dAdn(self, a):
        '''Jacobian matrix of [I + betaU*G] and [I + betaU_e*G_e] with respect to number of molecule a'''
        J = np.zeros([len(self.k),self.S, self.S])
        Je = np.zeros([len(self.k),self.S, self.S])  
        
        for k in range(len(self.k)):    
            J[k,:,:] = 1/self.DOP[a] * np.dot(self.U[k,:,:],self.gm_list[a][k,:,:])
            Je[k,:,:] = 1/self.DOP[a] * np.dot(self.Ue[k,:,:],self.gme_list[a][k,:,:])   
        J *= self.DOP[a]/self.V 
        Je *= self.DOP[a]/self.V 
        return J, Je
        
    
    def dAdV(self):
        '''Jacobian matrix of [I + betaU*G] and [I + betaU_e*G_e] with respect to V'''
        J = -1/self.V * self.UGpp
        Je = -1/self.V * self.UeGee
        return J, Je   
        
    def F(self):
        ''' Free energy'''
        FMF = np.sum(self.n * np.log(self.n/self.V) - self.n) + self.V/2. * np.dot(np.dot(self.C,self.u0),self.C)
                
        y = self.k**2 * np.log(det(self.Ys))                
        Fee = self.V/(4.*np.pi**2) * simps(y,self.k)
        F = FMF + Fee
 
        if self.IncludeEvRPA:
            y = self.k**2 * np.log(det(self.Xs)) 
            Fpp = self.V/(4.*np.pi**2) * simps(y,self.k)
            F += Fpp        

        return F

    def mu(self, i):
        ''' Chemical potential'''
        
        muMF = np.log(self.Cm[i]) - np.log(self.DOP[i]) 
        
        y = 0
        struc = self.struc[i]
        for a in range(self.S):
            for b in range(self.S):
                if a in struc and not b in struc:
                    y += 2. * self.beadFrac[a,i] * self.C[b] * self.u0[a,b]
                elif a in struc and b in struc:
                    y += (self.beadFrac[a,i] * self.C[b] + self.beadFrac[b,i] * self.C[a]) * self.u0[a,b]
        y *= self.DOP[i]/2.
        muMF += y
        
        dXdns, dYdns = self.dAdn(i)
        y = []
        for j,k in enumerate(self.k):                       
            X = self.Xs[j]
            Y = self.Ys[j]
            dXdn = dXdns[j]
            dYdn = dYdns[j]

            T = np.trace(np.matmul(np.linalg.inv(Y),dYdn))
            if self.IncludeEvRPA:
                W = np.trace(np.matmul(np.linalg.inv(X),dXdn))
                y.append(k**2 * (W+T))
            else:
                y.append(k**2 * T)            
        muRPA = simps(y,self.k)
        muRPA *= self.V/(4.*np.pi**2)
        return muMF + muRPA

    def P(self):
        '''Pressure'''
        
        P_MF = np.sum(self.Cm/self.DOP) + 1./2. * np.dot(np.dot(self.C,self.u0),self.C)        
        dXdVs, dYdVs = self.dAdV()
        y = []
        for j,k in enumerate(self.k):                       
            X = self.Xs[j]
            Y = self.Ys[j]
            dXdV = dXdVs[j]
            dYdV = dYdVs[j]
            
            T = np.trace(np.matmul(np.linalg.inv(Y),dYdV))
            if self.IncludeEvRPA:
                W = np.trace(np.matmul(np.linalg.inv(X),dXdV))
                y.append(k**2 * (self.V*(W+T) + np.log(det(X)) + np.log(det(Y))))
            else:
                y.append(k**2 * (self.V*(T) + np.log(det(Y))))                
        PRPA = -1/(4.*np.pi**2) * simps(y,self.k)

        return P_MF + PRPA
