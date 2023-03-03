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
        self.Ue = [] # bare electrostatic interactions, no charge
        self.Ue_charge = [] # electrostatic interactions with charges of bead i and j
        self.Gpp = []
        self.Gee = []
        self.gD = None # for CGC, list of gD for each molecule, if provided, dont have to recalculate
        self.gm_list = None # for DGC, list of structure factor like matrices (at all k numbers) for each molecule, independent of concentration
        self.gme_list = None
        self.IncludeEvRPA = True # include perturbation term for the excluded volume interaction        
        self.MF = False # do MF approximation 
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
        '''Debye function for discrete gaussian chain of type i at all wavenumbers'''
        b = self.b[i] 
        N = self.DOP[i]
        kRg2 = self.k**2 * N * b**2 /6.
        phi = np.exp(-kRg2/N)
        gD = N*(1-phi**2.) + 2*phi*(phi**N-1)
        gD/= N**2*(1-phi)**2        
        return gD
    
    def gD_FJC(self,i):
        '''Debye function for freely jointed chain of type i at all wavenumbers'''
        b = self.b[i]  
        k = self.k
        N = self.DOP[i]
        phi = np.sin(b*k)/(b*k)
        gD = N*(1-phi**2.) + 2*phi*(phi**N-1)
        gD/= N**2*(1-phi)**2        
        return gD

    def gD_CGC(self,i):
        '''Debye function for continuous gaussian chain of species i at all wavenumbers'''
        N = self.DOP[i]
        b = self.b[i]  
        k = self.k
        gD = np.zeros(len(self.k))
        x = N * k**2 * b**2 /6.
        gD = 2. * x**-2 * (x + np.exp(-x) - 1.)
        return gD

    def gD_rod(self,i):
        '''Alternative for the Debye function, used for semiflexible rod''' 
        N = self.DOP[i]
        b = self.b[i]
        k = self.k
        return 1./(1.+N*b*k/np.pi)

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
                    if len(np.unique(struc)) == 1: # use non-summation version for quicker calculation if the chain only has one bead species
                        bead1=struc[0]
                        if self.DOP[m]>1:
                            gm[:,bead1,bead1] = self.DOP[m]**2 * self.gD_DGC(bead1) 
                            gme[:,bead1,bead1] = self.DOP[m]**2 * self.gD_DGC(bead1) * self.charge[bead1]**2.
                        else: # set to exact solution for small molecule to avoid error at low k
                            gm[:,bead1,bead1] = np.ones(len(self.k)) 
                            gme[:,bead1,bead1] = np.ones(len(self.k)) * self.charge[bead1]**2. 
                    else:
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

        elif self.chain in ['CGC', 'rod']:    
            # Assume homopolymer chains
            gD_list = []
            gm_list = []
            gme_list = []
            for m, struc in enumerate(self.struc):
                gm = np.zeros([len(self.k), self.S, self.S])
                gme = np.zeros([len(self.k), self.S, self.S])
                if self.DOP[m]>1:
                    if len(np.unique(struc)) != 1:
                        raise Exception('Continuous Gaussian and rod-like chains are only supported for homogeneous polymers')
                    if self.chain == 'CGC':
                        gD = self.gD_CGC(struc[0])
                    elif self.chain == 'rod':
                        gD = self.gD_rod(struc[0])
                    gm[:,struc[0],struc[0]] = self.DOP[m]**2. * gD 
                    gme[:,struc[0],struc[0]] = self.DOP[m]**2. * gD * self.charge[struc[0]]**2
                else: # set to exact solution for small molecule to avoid error at low k
                    gm[:,struc[0],struc[0]] = np.ones(len(self.k))
                    gme[:,struc[0],struc[0]] = np.ones(len(self.k)) * self.charge[struc[0]]**2.                
                gD_list.append(gD)
                gm_list.append(gm)
                gme_list.append(gme)               
            self.gD = gD_list
            self.gm_list = gm_list
            self.gme_list = gme_list

        for m in range(self.M):
            Gpp[:,:,:] += self.Cm[m]/self.DOP[m] * gm_list[m]
            Gee[:,:,:] += self.Cm[m]/self.DOP[m] * gme_list[m]

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
        Ue_charge = np.zeros((len(self.k),self.S,self.S))
        k = self.k
        for i in range(self.S):
            for j in range(self.S):
                Ue[:,i,j] = 4. * np.pi * self.lB / k**2 * np.exp(-self.a[i,j]**2 * k**2)
                Ue_charge[:,i,j] = Ue[:,i,j] * self.charge[i] * self.charge[j]
        return Ue, Ue_charge
    
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

    def GetSk(self):
        '''Structure factor matrix'''
        Sk = np.zeros([len(self.k),self.S, self.S])
        Sk_inv = np.zeros([len(self.k),self.S, self.S])

        # loop through k vect and calculate inverse of S_pp, S_ee
        for k in range(len(self.k)):
            Sk_inv[k] = self.U[k] + self.Ue_charge[k] + np.linalg.inv(self.Gpp[k])
            Sk[k] = np.linalg.inv(Sk_inv[k])
        return Sk, Sk_inv

    def GetSe(self):
        '''Electrostatic structure factor 
           Se = [1] Sk [1]'''
        See = np.zeros([len(self.k),self.S, self.S])
        See_inv = np.zeros([len(self.k),self.S, self.S])
        Se = np.zeros(len(self.k))
        # loop through k vect and calculate inverse of S_pp, S_ee
        for k in range(len(self.k)):
            See_inv[k] = self.Ue_charge[k] + np.linalg.inv(self.Gpp[k])
            See[k] = np.linalg.inv(See_inv[k])
            Se[k] = np.dot(np.dot(self.charge,See[k]),self.charge)
#            Se[k] = np.sum(See[k])
        return Se

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
        self.Ue, self.Ue_charge = self.GetUe()
        if not self.MF:        
            self.Gpp, self.Gee = self.GetG()
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
        tmp = self.n * (np.log(self.n) - np.log(self.V)) - self.n
        tmp1 = np.where(tmp==np.nan)[0] #screen out underflow with high concentration and recalculate with the equation below
        tmp2 = np.where(tmp==np.inf)[0]
        infIdx = np.concatenate((tmp1,tmp2))
        if len(infIdx)>0:
            tmp[infIdx] = np.log(self.n[infIdx]**(self.n[infIdx])) - self.n[infIdx] * np.log(self.V) - self.n[infIdx]
        FMF = np.sum(tmp) + self.V/2. * np.dot(np.dot(self.C,self.u0),self.C)
        F = FMF      

        if not self.MF:
            if any(self.charge):
                y = self.k**2 * np.log(det(self.Ys))
                Fee = self.V/(4.*np.pi**2) * simps(y,self.k)
                F += Fee
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
        mu = muMF
        if not self.MF:
            dXdns, dYdns = self.dAdn(i)
            y = []
            for j,k in enumerate(self.k):                       
                X = self.Xs[j]
                Y = self.Ys[j]
                dXdn = dXdns[j]
                dYdn = dYdns[j]

                if any(self.charge):
                    T = np.trace(np.matmul(np.linalg.inv(Y),dYdn))
                else:
                    T = 0.
                if self.IncludeEvRPA:
                    W = np.trace(np.matmul(np.linalg.inv(X),dXdn))
                    y.append(k**2 * (W+T))
                else:
                    y.append(k**2 * T)            
            if any(self.charge) or self.IncludeEvRPA:
                muRPA = simps(y,self.k)
                muRPA *= self.V/(4.*np.pi**2)
            else:
                muRPA = 0
            mu += muRPA
        return mu
    def P(self):
        '''Pressure'''
        P_MF = np.sum(self.Cm/self.DOP) + 1./2. * np.dot(np.dot(self.C,self.u0),self.C)        
        P = P_MF
        if not self.MF:
            dXdVs, dYdVs = self.dAdV()
            y = []
            for j,k in enumerate(self.k):                       
                X = self.Xs[j]
                Y = self.Ys[j]
                dXdV = dXdVs[j]
                dYdV = dYdVs[j]
                
                if any(self.charge):
                    T = np.trace(np.matmul(np.linalg.inv(Y),dYdV))
                else:
                    T = 0
                if self.IncludeEvRPA:
                    W = np.trace(np.matmul(np.linalg.inv(X),dXdV))
                    y.append(k**2 * (self.V*(W+T) + np.log(det(X)) + np.log(det(Y))))
                else:
                    y.append(k**2 * (self.V*(T) + np.log(det(Y))))                
            if any(self.charge) or self.IncludeEvRPA:
                PRPA = -1/(4.*np.pi**2) * simps(y,self.k)
            else:
                PRPA = 0.
            P += PRPA
        return P
