#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:25:52 2020

@author: nvthaomy
"""
import sys
sys.path.append('/home/mnguyen/bin/RPA/')
import GibbsRPA_LSQ as Gibbs
#import GibbsRPA_Ele_LSQ as Gibbs
from scipy.optimize import root
import numpy as np
import time
from scipy.optimize import least_squares
import time
import os

dataFile = 'gibbs1.dat'
logDir = 'log'
nIter0 = 10
nIter = 5
dtV0 = 0.01
dtC0 = 0.02
dtV = 0.00
dtC = 0.0
try:
    os.mkdir(logDir)
except:
    pass

IncludeEvRPA = False

# Composition
xSalts = np.linspace(0.001, 0.1, num=20, endpoint=True)
#xSalts = np.flip(xSalts)
xPE = 2*0.00006
fPAA = 0.3
x1 = xPE*fPAA
x2 = xPE-x1
Ctot = 33.3378644184
CI0 = [0.05*x1*Ctot, 0.05*x2*Ctot, (xSalts[0]+x1)*Ctot, (xSalts[0]+x2)*Ctot, (1-x1-x2-(xSalts[0]+x1)-(xSalts[0]+x2))*Ctot]
#CI0 =  [0.19187351,   0.12935853,   0.51519509,   0.45268012,  32.11688125] 
fI0 = 0.55

ensemble = 'NPT'
Ptarget = 285.9924138 

# Molecules
number_species = 5 
chargedPairs = [[0,2],[1,3],[2,3]]       # only include pairs
PAADOP = 50
PAHDOP = 50
DOP = [PAADOP,PAHDOP,1,1,1]
charges = [-1,1,1,-1,0]

# Forcefield
u0 = [[2.4362767487,2.91300224542,0.025694451934,2.11006226319,1.13846068423],
      [2.91300224542,6.11007508231,1.75498875979,1.35437785696,1.31135889157],
      [0.025694451934,1.75498875979,0.794886183768,0.0132698230775,0.0132691595333],
      [2.11006226319,1.35437785696,0.0132698230775,1.92109327616,0.680809658472],
      [1.13846068423,1.31135889157,0.0132691595333,0.680809658472,0.449843180313]]
abead = [0.45,0.45,0.31,0.31,0.31]
lB = 9.349379737083224
b = [0.142895474876,0.147135133373, 1,1,1]

# Numerical
V = 20.
kmin = 1.e-5
kmax = 20
nk = 500
VolFracBounds=[0.00001,1 - 0.00001]

LSQftol = 1e-8
LSQxtol = 1e-8

'''Set up and Run'''

GM = Gibbs.GibbsSolver(number_species)
GM.SetEnsemble(ensemble)
if ensemble == 'NPT':
    GM.SetPtarget(Ptarget)
GM.SetCharges(charges)
GM.SetChargedPairs(chargedPairs)
GM.SetDOP(DOP)
GM.Setabead(abead)
GM.Setu0(u0)
GM.SetlB(lB)
GM.Setb(b)
GM.SetV(V)
GM.Setkmin(kmin)
GM.Setkmax(kmax)
GM.Setnk(nk)
GM.SetVolFracBounds(VolFracBounds)
GM.SetLSQTol(LSQftol,LSQxtol )
GM.IncludeEvRPA = IncludeEvRPA
GM.nIter = nIter
GM.dtV = dtV
GM.dtC = dtC

log = open(dataFile,'w')
log.write('# xSalt Ctot fI fII CI1 CII1 CI2 CII2 CI3 CII3 CI4 CII4 CI5 CII5 dP dmuPAA_Na dmuPAH_Cl dmuNaCl dmuW PI muI1 muI2 muI3 muI4 muI5 PII muII1 muII2 muII3 muII4 muII5\n')
log.flush()

for i, xSalt in enumerate(xSalts):
    print('==xSalt {}=='.format(xSalt))
    x3 = x1+xSalt
    x4 = x2+xSalt
    x5 = 1.0 - (x1+x2+x3+x4)
    xs = np.array([x1,x2,x3,x4,x5])
    t0 = time.time()
    if i == 0: 
        fI_Init = fI0
        CI_Init = CI0
        GM.nIter = nIter0
        GM.dtV = dtV0
        GM.dtC = dtC0
    else:
        fI_Init = fI
        CI_Init = CIs
        CI_Init[2] = Ctot*x3
        CI_Init[3] = Ctot*x4
        if i % 20 == 0:
            GM.nIter = nIter0 
            GM.dtV = dtV0
            GM.dtC = dtC0
        else:
            GM.nIter = nIter
            GM.dtV = dtV
            GM.dtC = dtC
    GM.SetInitialfI(fI_Init)
    GM.SetInitialCI(CI_Init)
    GM.SetCtot(Ctot)
    GM.SetSpeciesVolFrac(xs)
    GM.LogFileName = '{}/xNaCl{}_log.txt'.format(logDir,xSalt)

    fI,CIs,fII,CIIs,errs, PI, muIs, PII, muIIs, cost = GM.Run()
    Ctot = GM.Ctot
    t1 = time.time()
    s = '{} {} {} {} '.format(xSalt,Ctot,fI,fII)
    for i in range(5):
        s+= str(CIs[i])+' '+str(CIIs[i])+' '
    for err in errs:
        s +='{} '.format(err)
    s += '{} {} {} {} {} {} '.format(PI,*muIs)
    s += '{} {} {} {} {} {} '.format(PII,*muIIs)
    log.write(s+'\n')
    log.flush()
    print('LSQ cost {}'.format(cost))
    print('Run time {:3.3f} min'.format((t1-t0)/60.))
