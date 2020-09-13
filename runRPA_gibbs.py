#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:25:52 2020

@author: nvthaomy
"""
import sys
sys.path.append('/home/mnguyen/bin/scripts')
sys.path.append('/home/mnguyen/bin/RPA/')
import GibbsRPA as Gibbs
#import GibbsRPA_Ele_LSQ as Gibbs
from scipy.optimize import root
import numpy as np
import time
from scipy.optimize import least_squares
import time
import os

dataFile = 'gibbs.dat'
logFile = 'log.txt'
Dt = [0.0005,0.02,0.02,0.02,0.02,0.1]
DtCpair = [0.05,0.05,0.05]
program = 'polyFTS'
jobtype = 'RPA' 
GibbsTolerance = 5e-5

IncludeEvRPA = False

# Composition
xSalts = [0.01] 
xPE = 0.04  
fPAA = 0.5

x1 = xPE*fPAA                                                                             
x2 = xPE-x1
Ctot = 33.2
CI0 = [0.05*x1*Ctot, 0.05*x2*Ctot, (xSalts[0]+x1)*Ctot, (xSalts[0]+x2)*Ctot, (1-x1-x2-(xSalts[0]+x1)-(xSalts[0]+x2))*Ctot]

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
nk = 200
VolFracBounds=[0.00001,1 - 0.00001]

LSQftol = 1e-8
LSQxtol = 1e-8

'''Set up and Run'''

GM = Gibbs.Gibbs_System(program,number_species)
GM.SetJobType(jobtype)
GM.SetEnsemble(ensemble)

GM.SetEnsemble(ensemble)
if ensemble == 'NPT':
    GM.SetPtarget(Ptarget)
GM.SetCharges(charges)
GM.SetChargedPairs(chargedPairs)
GM.SetSpeciesDOP(DOP)
GM.Setabead(abead)
GM.Setu0(u0)
GM.SetlB(lB)
GM.Setb(b)
GM.SetV(V)
GM.Setkmin(kmin)
GM.Setkmax(kmax)
GM.Setnk(nk)
GM.SetDt(Dt)
GM.SetDtCpair(DtCpair)
GM.SetVolFracBounds(VolFracBounds)
GM.IncludeEvRPA = IncludeEvRPA

GM.GibbsLogFileName = 'gibbs.dat'
GM.GibbsErrorFileName = 'error.dat'

data = open(dataFile,'w')
data.write('# xSalt Ctot fI fII CI1 CII1 CI2 CII2 CI3 CII3 CI4 CII4 CI5 CII5 dP dmuPAA_Na dmuPAH_Cl dmuNaCl dmuW PI PII\n')
data.flush()

log = open(logFile,'w')
log.flush()

cwd = os.getcwd()

for i, xSalt in enumerate(xSalts):
    try:
        os.mkdir('xNaCl{}'.format(xSalt))
    except:
        pass
    os.chdir('xNaCl{}'.format(xSalt))
    print('==xSalt {}=='.format(xSalt))
    log.write('\n====xSalt {}====\n'.format(xSalt))
    log.flush()
    #move to new folder
    x3 = x1+xSalt
    x4 = x2+xSalt 
    x5 = 1.0 - (x1+x2+x3+x4)
    xs = np.array([x1,x2,x3,x4,x5])
    
    Cs = xs*Ctot
    GM.SetSpeciesCTotal(Cs)    
    Ctot = GM.RPABaroStat()
    
    # update Cs after barostat
    log.write('Ctot after barostat {}\n'.format(Ctot))
    log.flush()    
    Cs = xs*Ctot
    GM.SetSpeciesCTotal(Cs) 
    
    # Initialize
    if i == 0: 
        fI_Init = fI0
        CI_Init = np.array(CI0)        
    else:
        fI_Init = fI
        CI_Init = np.array(CIs)
        
    fII_Init  = 1.-fI_Init
    CII_Init = (Cs-CI_Init*fI_Init)/fII_Init
    VarInit = [fI_Init,fII_Init]
    for i,CI in enumerate(CI_Init):
        VarInit.extend([CI,CII_Init[i]])
    GM.SetInitialGuess(VarInit)
    GM.ValuesCurrent = VarInit 
    GM.DvalsCurrent = [1.] * (GM.Nspecies+2)
    GM.Iteration = 1
    fracErr = 10
    
    t0 = time.time()
    step = 0
    log.write('\n=step\tFracErr\tdP\tdMus=\n')
    while fracErr > GibbsTolerance:
        step += 1
        dVals = []
        GM.TakeGibbsStep()  
        dVals = [GM.DvalsCurrent[1]]
        Vals =  [GM.OperatorsCurrent[4]]
        dVals.extend(GM.dMuPair)
        Vals.extend(GM.MuPair1)
        dVals.append(GM.DvalsCurrent[4+2])
        Vals.append(GM.OperatorsCurrent[24])            
        #dVals = np.array([GM.DvalsCurrent[1], *GM.dMuPair, GM.DvalsCurrent[4+2]])
        #Vals = np.array([GM.OperatorsCurrent[4],*GM.MuPair1,GM.OperatorsCurrent[24]])
        dVals = np.array(dVals)
        Vals = np.array(Vals)
        fracErr = np.max(np.abs(dVals/Vals))
        s = '{} {} {} '.format(step,fracErr,GM.DvalsCurrent[1])
        for a in GM.dMuPair:
            s+= '{} '.format(a)
        s+= '{} \n'.format(GM.DvalsCurrent[4+2])
        
        log.write(s)
        log.flush() 
        
    t1 = time.time()
    t = t1-t0
    log.write('==Finish after {} minutes==\n'.format(t/60.))
    
    fI = GM.ValuesCurrent[0]
    fII = GM.ValuesCurrent[1]
    CIs = GM.ValuesCurrent[2::2]
    CIIs = GM.ValuesCurrent[3::2]

    s = '{} {}'.format(xSalt,Ctot)
    for a in GM.ValuesCurrent:
        s += '{} '.format(a)
    for a in dVals:
        s += '{} '.format(a)
    s += '{} '.format(GM.OperatorsCurrent[4])
    s += '{} \n'.format(GM.OperatorsCurrent[6])
    data.write(s)
    data.flush()
    os.chdir(cwd)


