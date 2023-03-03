#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:25:52 2020
Define the system and run RPA calculations
@author: My Nguyen
"""
import sys
sys.path.append('/home/mnguyen/bin/scripts')
sys.path.append('/home/mnguyen/bin/RPA/')
sys.path.append('/Users/nvthaomy/Desktop/rpa/RPA/')
from scipy.optimize import root
import numpy as np
import time, copy
import os

dataFile = 'operators.dat'
logFile = 'log.txt'

MF = True
IncludeEvRPA = False
chain = 'DGC'
log = open(logFile,'w')
log.flush()
muWtarget = float(sys.argv[1])
muIDPtarget = float(sys.argv[2])
Ctot = float(sys.argv[3]) #33.49116106142355
tolerance = 1e-10

# Molecules
number_species = 6
charges = np.array([0, 0, 0, 1, -1, 0], dtype = float) # A-, A, B+, B, Na+, Cl-, HOH
number_molecules = 2
struc = [[2, 1, 1, 4, 1, 3, 2, 1, 0, 4, 0, 3, 2, 1, 1, 4, 1, 3, 2, 1, 0, 4, 0, 3, 2, 1, 1, 4, 1, 3, 2, 1, 0, 4, 0, 3, 2, 1, 1, 4, 1, 3, 2, 1, 0, 4, 0, 3, 2, 1, 1, 4, 1, 3, 2, 1, 0, 4, 0, 3, 2, 1, 1, 4, 1, 3, 2, 1, 0, 4, 0, 3, 2, 1, 1, 4, 1, 3, 2, 1, 0, 4, 0, 3, 0, 0, 2, 1, 0, 1, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [5]]
molCharges = []
for s in struc:
    c = 0
    for i in s:
        c += charges[i]
    c /= len(s)
    molCharges.append(c)
molCharges = np.array(molCharges, dtype = float) # charge per segment length of each molecule type

# Forcefield
u0 = [[15.760520808387364, 9.620286929056373, 5.931417975266296, 5.931417975266296, 5.931417975266296, 1.5916018379087862], [9.620286929056373, 0.47265940950298235, 2.757684625446694, 2.757684625446694, 2.757684625446694, 0.8144758600054185], [5.931417975266296, 2.757684625446694, 15.348333517386306, 15.348333517386306, 15.348333517386306, 1.0129079034563466], [5.931417975266296, 2.757684625446694, 15.348333517386306, 15.348333517386306, 15.348333517386306, 1.0129079034563466], [5.931417975266296, 2.757684625446694, 15.348333517386306, 15.348333517386306, 15.348333517386306, 1.0129079034563466], [1.5916018379087862, 0.8144758600054185, 1.0129079034563466, 1.0129079034563466, 1.0129079034563466, 0.13270884748289072]] 
abead = [0.4999999999999999, 0.5, 0.5, 0.5, 0.5, 0.31]
lB = 0.744
b = [0.16787670578191757, 0.16787670578191757, 0.16787670578191757, 0.16787670578191757, 0.16787670578191757, 1.0]

# Numerical
V = 20.
kmin = 1.e-5
kmax = 20
nk = 200

'''Set up RPA object'''
import RPA_v2 as RPAModule
RPA = RPAModule.RPA(number_species,number_molecules)
RPA.Setstruc(struc)
RPA.Setchain(chain)
RPA.Setcharge(charges)
RPA.Setabead(abead)
RPA.Setu0(u0)
RPA.SetlB(lB)
RPA.Setb(b)
RPA.SetV(V)
RPA.Setkmin(kmin)
RPA.Setkmax(kmax)
RPA.Setnk(nk)
RPA.IncludeEvRPA = IncludeEvRPA
RPA.MF = MF
np.random.seed(555) 
gme_list = None
gm_list = None

data = open(dataFile,'w')
data.write('# xIDP Ctot Hamiltonian Pressure ')
for i in range(number_molecules):
    data.write('ChemicalPotential{} '.format(i+1))
data.write('\n')

def func(Ctot,xIDP):
    vol_frac = np.array([xIDP,1-xIDP])
    Cs = Ctot * vol_frac
    # check neutrality
    totcharge = molCharges * Cs
    totcharge = np.sum(totcharge)
    log.write('Total charge: {}\n'.format(totcharge))
    log.flush()
    if np.abs(totcharge) > 1e-4:
        log.write('System is not neutral. Break')
        log.flush()
        raise Exception('System is not neutral. Break')

    RPA.gm_list = gm_list
    RPA.gme_list = gme_list
    RPA.SetCm(Cs)
    RPA.Initialize()
    P = RPA.P()
    muW = RPA.mu(1)
    return np.abs((muW-muWtarget)/muWtarget)

# Run calculation
def solver(xIDPs_, Ctot):
    for i, xIDP in enumerate(xIDPs_):
        if xIDP > 1.0:
            continue
        sol = root(func, Ctot, args=(xIDP),method='broyden1', tol=tolerance)
        Ctot = sol.x
        gm_list = RPA.gm_list
        gme_list = RPA.gme_list

        P = RPA.P()
        F = RPA.F()   
        mu = np.zeros(number_molecules) 
        for j in range(number_molecules):
            mu[j] = RPA.mu(j)
        vals = [F/V,P]
        vals.extend(mu)

        # Write data 
        data.write('{} {} {}\n'.format(xIDP, Ctot, ' '.join(str(v) for v in vals)))
        data.flush()

        # refine xIDP
        if np.abs((mu[0] - muIDPtarget)/muIDPtarget) <= tolerance:
            #print('=== Converged ===\n\txIDP: {}, muIDP: {}'.format(xIDP, mu[0]))
            return xIDP, Ctot
        elif i == 0 and mu[0] > muIDPtarget:
            xIDPs = np.logspace(np.log(xIDP) - 5, 0.3, num=100, endpoint=True)
            print('Reduce xIDP range: [{:.4e} , {:.4e}]'.format(xIDPs[0],xIDPs[-1]))
            return solver(xIDPs, Ctot)
        elif i != 0 and mu[0] > muIDPtarget:
            xIDPs = np.linspace( xIDPs_[i-1], xIDP, num=100, endpoint=True)
            print('Refine xIDP range: [{:.4e} , {:.4e}]'.format(xIDPs_[i-1],xIDP))
            return solver(xIDPs, Ctot)
        
xIDPs = np.logspace(-50, 0.3, num=100, endpoint=True)
xIDPs = xIDPs[np.where(xIDPs <= 1e-2)]
xIDPs_2 = np.linspace(1e-2, 1, num=200, endpoint=True)
xIDPs = np.concatenate((xIDPs,xIDPs_2), axis=0)

print('\nSweeping xIDP range: [{:.4e} , {:.4e}]'.format(xIDPs[0],xIDPs[-1]))
xIDP, Ctot = solver(xIDPs,Ctot)
# check
vol_frac = np.array([xIDP,1-xIDP])
Cs = Ctot * vol_frac
# check neutrality
totcharge = molCharges * Cs
totcharge = np.sum(totcharge)
log.write('Total charge: {}\n'.format(totcharge))
log.flush()
if np.abs(totcharge) > 1e-4:
    log.write('System is not neutral. Break')
    log.flush()
    raise Exception('System is not neutral. Break')

RPA.gm_list = gm_list
RPA.gme_list = gme_list
RPA.SetCm(Cs)
RPA.Initialize()
P = RPA.P()
F = RPA.F()/V  
mu = np.zeros(number_molecules) 
for j in range(number_molecules):
    mu[j] = RPA.mu(j)
print('=== Converged ===\n\txIDP: {}, Ctot: {}\n\tmuIDP: {}, P: {}\n\tH: {}'.format(xIDP, Ctot, mu[0], P, F))
