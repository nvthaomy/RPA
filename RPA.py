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
        gD = 2. * N**2 * x**-2 * (x + np.exp(-x) - 1.)
        return gD
       
    
    def GetGpp(self, V=None, i=None, n=None):
        '''Density correlation matrix at all wavenumbers'''
        C = self.C  
        # Perturb C[i] and V to calculate the derivative
        if not i == None and not n == None:
            C[i] = n * self.DOP[i]/self.V
        elif not V == None:
            C *= self.V/V
            
        Gpp = np.zeros((len(self.k),self.S,self.S))
        for i in range(self.S):
            if self.chain == 'DGC':
                gD = self.gD_DGC(i)
            elif self.chain == 'CGC':
                gD = self.gD_CGC(i)
            g = C[i] * self.DOP[i] * gD
            Gpp[:,i,i] = g
        return Gpp
    
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
        
        self.U = self.GetU()
        self.Gpp = self.GetGpp()
        self.Ue = self.GetUe()
        self.Gee =  self.GetGee() 
        
    def F(self):
        ''' Free energy'''
        FMF = np.sum(self.n * np.log(self.C) - self.n) + self.V/2. * np.dot(np.dot(self.C,self.u0),self.C)
        
        U = self.U
        Gpp = self.Gpp
        Ue = self.Ue
        Gee =  self.Gee
        y=[]
        for i,k in enumerate(self.k):                    
            EE =  k**2 * np.log(det(np.identity(self.S)+ np.dot(Ue[i],Gee[i])))
            y.append(EE)
        Fee = self.V/(4.*np.pi**2) * simps(EE,self.k)
        F = FMF + Fee
 
        if self.IncludeEvRPA:
            y = []
            for i,k in enumerate(self.k):
                PP = k**2 * np.log(det(np.identity(self.S)+ np.dot(U[i],Gpp[i])))
                y.append(PP)
            Fpp = self.V/(4.*np.pi**2) * simps(PP,self.k)
            F += Fpp        

        return F

    def mu(self, i, dn = 10e-9):
        ''' Chemical potential'''
        
        muMF = np.log(self.C[i]/self.DOP[i]) + self.DOP[i] * np.dot(self.C,self.u0[i,:])
        
        U = self.U
        Ue = self.Ue
        Gee1 = self.GetGee(i=i,n=self.n[i]-dn)
        Gpp1 = self.GetGpp(i=i,n=self.n[i]-dn)   
        Gee2 = self.GetGee(i=i,n=self.n[i]+dn)
        Gpp2 = self.GetGpp(i=i,n=self.n[i]+dn)  
        
        y = []
        for j,k in enumerate(self.k):
                       
            A1 = np.identity(self.S)+ np.dot(U[j],Gpp1[j])
            B1 = np.identity(self.S)+ np.dot(Ue[j],Gee1[j])
            f1 =  np.log(det(B1))
            if self.IncludeEvRPA:
                f1 += np.log(det(A1))
            
            A2 = np.identity(self.S)+ np.dot(U[j],Gpp2[j])
            B2 = np.identity(self.S)+ np.dot(Ue[j],Gee2[j])
            f2 =  np.log(det(B2))
            if self.IncludeEvRPA:
                f2 += np.log(det(A2))

            der = (f2-f1)/(2*dn)
            y.append(k**2 * der)
            
        muRPA = simps(y,self.k)
        muRPA *= self.V/(4.*np.pi**2)
        return muMF + muRPA
    
    def P(self, dV = 10e-9):
        '''Pressure'''
        P_MF = np.sum(self.C/self.DOP) + 1./2. * np.dot(np.dot(self.C,self.u0),self.C) 

        U = self.U
        Ue = self.Ue
        Gee1 = self.GetGee(V=self.V-dV)
        Gpp1 = self.GetGpp(V=self.V-dV)   
        Gee2 = self.GetGee(V=self.V+dV)
        Gpp2 = self.GetGpp(V=self.V+dV) 
        y = []
        for j,k in enumerate(self.k):                      
            A1 = np.identity(self.S)+ np.dot(U[j],Gpp1[j])
            B1 = np.identity(self.S)+ np.dot(Ue[j],Gee1[j])
            f1 = (self.V-dV) *  np.log(det(B1))
            if self.IncludeEvRPA: 
                f1 += (self.V-dV) *  np.log(det(A1))                

            A2 = np.identity(self.S)+ np.dot(U[j],Gpp2[j])
            B2 = np.identity(self.S)+ np.dot(Ue[j],Gee2[j])
            f2 = (self.V+dV) * np.log(det(B2))
            if self.IncludeEvRPA:             
                f2 += (self.V+dV) *  np.log(det(A2))

            der = (f2-f1)/(2*dV)
            y.append(k**2 * der)
        PRPA = -1/(4.*np.pi**2) * simps(y,self.k)
        return P_MF + PRPA
        
