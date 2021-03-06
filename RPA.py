#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:02:28 2020

@author: nvthaomy
"""
import numpy as np
from scipy.integrate import simps
from numpy.linalg import det
from scipy.misc import derivative
#import matrix

class RPA():
    def __init__(self, nSpecies,nChain):
        self.S = nSpecies # bead species
        self.nChain = nChain # chain species
        self.DOP = np.ones(self.S)
        self.charge = np.zeros(self.S) # vector of charge/segment
        self.C  = np.ones(self.S) # vector of species concentrations 
        self.abead = np.ones(self.S) # smearing length of beads
        self.u0 = np.ones((self.S,self.S)) # excluded vol matrix
        self.lB = 1.0 # Bjerrum length in kT
        self.b = np.ones(self.S) # RMS length of bond, can be anything for small molecules
        self.V = 10.   
        self.kmin = 0.001 
        self.kmax = 1000 
        self.nk =  1000 # number of mesh point
        self.chain = 'DGC' # DGC or CGC
        
        self.USI = np.ones((self.S,self.S))
        self.a = np.ones((self.S,self.S)) # smearing length of interaction matrix
        self.n = np.ones(self.S) # number of molecules for each species  
        self.k = [] 
        self.dk = 0.1  
        self.U = [] # k-space matrices of excluded vol int.
        self.Ue = []              
        self.Gpp = []
        self.Gee = []
        self.gD = None # list of gD for each molecule, if provided, dont have to recalculate
        self.IncludeEvRPA = True # include perturbation term for the excluded volume interaction        
        
    def SetDOP(self,DOP):
        self.DOP = np.array(DOP, dtype=float)
    def Setcharge(self,charge):
        self.charge = np.array(charge, dtype=float)
    def SetC(self,C):
        self.C = np.array(C, dtype=float)
        self.n = self.C * self.V / self.DOP
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
       
    
    def GetGpp(self, V=None, i=None, n=None):
        '''Density correlation matrix at all wavenumbers'''
        C = self.C  
        # Perturb C[i] and V to calculate the derivative
        if not i == None and not n == None:
            C[i] = n * self.DOP[i]/self.V
        elif not V == None:
            C *= self.V/V
        gD_list = []   
        Gpp = np.zeros((len(self.k),self.S,self.S))
        for i in range(self.S):
            if isinstance(self.gD,list) or isinstance(self.gD,np.ndarray):
                gD = self.gD[i]
            elif self.gD == None:
                if self.chain == 'DGC':
                    gD = self.gD_DGC(i)
                elif self.chain == 'CGC':
                    gD = self.gD_CGC(i)
            gD_list.append(gD)
            g = C[i] * self.DOP[i] * gD
            Gpp[:,i,i] = g
        return Gpp, gD_list
    
    def GetGee(self, V=None, i=None, n=None):
        '''Charge correlation matrix'''
        C = self.C        
        # Perturb C[i] and V to calculate the derivative
        if not i == None and not n == None:
            C[i] = n * self.DOP[i]/self.V
        elif not V == None:
            C *= self.V/V  
            
        Gee = np.zeros((len(self.k),self.S,self.S))
        for i in range(self.S):
            if isinstance(self.gD,list) or isinstance(self.gD,np.ndarray):
                gD = self.gD[i]
            elif self.gD == None:
                if self.chain == 'DGC':
                    gD = self.gD_DGC(i)
                elif self.chain == 'CGC':
                    gD = self.gD_CGC(i)
            g = C[i] * self.DOP[i] * self.charge[i]**2 * gD
            Gee[:,i,i] = g
        return Gee
    
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
        U = 0.5 * ( self.n * self.DOP * u0ii/(4. * np.pi * aii**2) + self.n * self.DOP * self.lB)
        self.USI = np.sum(U) 

        '''Number of molecules'''
        self.n = self.C * self.V / self.DOP
        
        '''wave number'''
        self.k, self.dk = np.linspace(self.kmin, self.kmax, self.nk, endpoint = True, retstep = True)
        
        '''Struture factor related var'''
        self.U = self.GetU()
        self.Gpp, self.gD = self.GetGpp()
        self.Ue = self.GetUe()
        self.Gee =  self.GetGee() 
        self.UGpp, self.UeGee = self.DotUG()
        I = np.array([np.identity(self.S)]*len(self.k))
        self.Xs = I + self.UGpp
        self.Ys = I + self.UeGee
        self.invXs = None #inverse of Xs[k,:,:] for all k values
        self.invYs = None #inverse of Ys[k,:,:] for all k values

    def dAdn(self, a):
        '''Jacobian matrix of [I + betaU*G] and [I + betaU_e*G_e] with respect to number of species a'''
        J = np.zeros([len(self.k),self.S, self.S])
        Je = np.zeros([len(self.k),self.S, self.S])    
        #for i in range(self.S):
        #     #for j in range(self.S):
        #     j=a
        #     for k in range(len(self.k)):
        #         J[k,i,j] = np.dot(self.U[k,i,:],self.Gpp[k,:,j]) / self.C[j]
        #         Je(k,i,j) = np.dot(self.Ue[k,i,:],self.Gee[k,:,j]) / self.C[j]

        J[:,:,a] = self.DOP[a]/self.V * self.UGpp[:,:,a]/ self.C[a]
        Je[:,:,a] = self.DOP[a]/self.V * self.UeGee[:,:,a]/ self.C[a]
        return J, Je
        
    
    def dAdV(self):
        '''Jacobian matrix of [I + betaU*G] and [I + betaU_e*G_e] with respect to V'''
        J = -1/self.V * self.UGpp
        Je = -1/self.V * self.UeGee
        return J, Je   
        
    def F(self):
        ''' Free energy'''
        FMF = np.sum(self.n * np.log(self.n/self.V) - self.n) + self.V/2. * np.dot(np.dot(self.C,self.u0),self.C)
        
        U = self.U
        Gpp = self.Gpp
        Ue = self.Ue
        Gee =  self.Gee
        y=[]
        for i,k in enumerate(self.k):                    
            EE =  k**2 * np.log(det(np.identity(self.S)+ np.dot(Ue[i],Gee[i])))
            y.append(EE)
        Fee = self.V/(4.*np.pi**2) * simps(y,self.k)
        F = FMF + Fee
 
        if self.IncludeEvRPA:
            y = []
            for i,k in enumerate(self.k):
                PP = k**2 * np.log(det(np.identity(self.S)+ np.dot(U[i],Gpp[i])))
                y.append(PP)
            Fpp = self.V/(4.*np.pi**2) * simps(y,self.k)
            F += Fpp        

        return F

    def mu(self, i):
        ''' Chemical potential'''
        
        muMF = np.log(self.C[i]) - np.log(self.DOP[i]) + self.DOP[i] * np.dot(self.C,self.u0[i,:])
                
        dXdns, dYdns = self.dAdn(i)
        y = []
        for j,k in enumerate(self.k):                       
            X = self.Xs[j]
            Y = self.Ys[j]
            dXdn = dXdns[j]
            dYdn = dYdns[j]
            # calculate W = d/dni ln(det(X)) and T = d/dni ln(det(Y) using Jacobi's formula
#            try:
            T = np.trace(np.matmul(np.linalg.inv(Y),dYdn))
#            except:
#                m = matrix.Matrix(Y.tolist())
#                adjY = m.adjoint()
#                adjY = np.array([adjY[t] for t in range(Y.shape[0])])
#                T = 1/det(Y) * np.trace(np.matmul(adjY,dYdn))
            if self.IncludeEvRPA:
#                try:
                W = np.trace(np.matmul(np.linalg.inv(X),dXdn))
#                except:
#                    m = matrix.Matrix(X.tolist())
#                    adjX = m.adjoint()
#                    adjX = np.array([adjX[t] for t in range(X.shape[0])])
#                    W = 1/det(X) * np.trace(np.matmul(adjX,dXdn))
                y.append(k**2 * (W+T))
            else:
                y.append(k**2 * T)            
        muRPA = simps(y,self.k)
        muRPA *= self.V/(4.*np.pi**2)
        return muMF + muRPA

    def P(self):
        '''Pressure'''
        
        P_MF = np.sum(self.C/self.DOP) + 1./2. * np.dot(np.dot(self.C,self.u0),self.C)        
        dXdVs, dYdVs = self.dAdV()
        y = []
        for j,k in enumerate(self.k):                       
            X = self.Xs[j]
            Y = self.Ys[j]
            dXdV = dXdVs[j]
            dYdV = dYdVs[j]
            # calculate W = d/dV ln(det(X)) and T = d/dV ln(det(Y) using Jacobi's formula
#            try:
            T = np.trace(np.matmul(np.linalg.inv(Y),dYdV))
#            except:
#                m = matrix.Matrix(Y.tolist())
#                adjY = m.adjoint()
#                adjY = np.array([adjY[t] for t in range(Y.shape[0])])
#                T = 1/det(Y) * np.trace(np.matmul(adjY,dYdV))
            
            if self.IncludeEvRPA:
#                try:
                W = np.trace(np.matmul(np.linalg.inv(X),dXdV))
#                except:
#                    m = matrix.Matrix(X.tolist())
#                    adjX = m.adjoint()
#                    adjX = np.array([adjX[t] for t in range(X.shape[0])])
#                    W = 1/det(X) * np.trace(np.matmul(adjX,dXdV))
                y.append(k**2 * (self.V*(W+T) + np.log(det(X)) + np.log(det(Y))))
            else:
                y.append(k**2 * (self.V*(T) + np.log(det(Y))))                
        PRPA = -1/(4.*np.pi**2) * simps(y,self.k)

        return P_MF + PRPA
