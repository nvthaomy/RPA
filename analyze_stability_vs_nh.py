"""
Calculate the stability matrix and map out the unstable window in the composition-chain length plane
"""
import numpy as np
import os
import compute_DIS_G

import matplotlib
matplotlib.rc('font', size=7)
matplotlib.rc('axes', titlesize=7)
showPlots = True
try:
  os.environ["DISPLAY"] #Detects if display is available
except KeyError:
  showPlots = False
  matplotlib.use('Agg') #Need to set this so doesn't try (and fail) to open interactive graphics window
import matplotlib.pyplot as plt

xIDPs = np.logspace(-10, 0.3, num=4000, endpoint=True)
xIDPs = xIDPs[np.where(xIDPs <= 1e-2)]
xIDPs_2 = np.linspace(1e-2, 0.8, num=2000, endpoint=True)
xIDPs = np.concatenate((xIDPs,xIDPs_2), axis=0)
n0 = 0
n1 = 16
nhs = np.linspace(n0,n1,num=int((n1-n0)/0.5) + 1,endpoint=True)

# ../6IDP-2Yx2A_L10nm_xIDP5e-3/
G_mic = -1.0907491349e+02 #intensive
muIDPtarget = 3.3417262478e+03
muWtarget = 7.9902913827e+00
Ptarget = 1.0907650812e+02
barostat = True

def BaroStat(xs, Ptarget, Ctot, dtC=0.2, maxIter = 1000 ,tol = 1.e-6):
        import math
        '''
        RPA class with defined interactionsn
        dtC: step size
        maxIter: max iteration step'''

        C1 = Ctot
        err = 10.
        step = 1
        while err>tol:
            P1 = analytical.Pressure_from_Composition(xs, C1)
            err = np.abs(P1-Ptarget)/np.abs(Ptarget)
            #self.Write2Log('{} {} {} {}\n'.format(step,C1,P1,err))
            C1 = C1 * ( 1. + dtC*(math.sqrt(Ptarget/P1) - 1.) )
            if err <= tol:
                #print('error is below tolerance {}\n'.format(tol))
                break
            else:
                step += 1
        return C1

vIDP = 0.13 # nm^3
vW = 0.03
G_dis = []
muIDP = []
muW = []
P = []
H = []

stable = []
Ctots = []
NIDPs = []
for nh in nhs:
    analytical = compute_DIS_G.MF(nh)
    _stable = []
    _Ctots = []
    NIDPs.append(analytical.Nofmolecule[0])
    for xIDP in xIDPs:
        xW = 1-xIDP
        vol_frac = [xIDP,xW] # bead basis
        Ctot = 1/(vIDP*xIDP + vW*xW)  

        # Barostat
        if barostat:
            Ctot = BaroStat(vol_frac, Ptarget, Ctot)
        _Ctots.append(Ctot)

        G = analytical.GrandFreeEnergy_from_Composition( vol_frac, Ctot)
        noverV = analytical. MoleculeDensitiesFromVolFracs( vol_frac, Ctot )
        #mu_p = analytical.ChemicalPotentialsFromComposition( noverV )
        #P_ = analytical.Pressure_from_Composition(vol_frac, Ctot)
        #H_ = analytical.HelmholtzFreeEnergy_Method2( noverV )
        eigval = analytical.compute_stability( noverV)
        if G:
            if np.isfinite(G): #not np.isnan(G) and not np.isinf(G):
                if np.prod(eigval) < 0:
                    _stable.append(0) # unstable
                else:
                    _stable.append(1) # stable
    stable.append(_stable)
    Ctots.append(_Ctots)

Ctots = np.array(Ctots)
stable = np.array(stable)
CIDPs = xIDPs[np.newaxis, :] * Ctots
x_spinodal = []   # xIDP value
x_spinodal_uM = [] # CIDP in uM
y_spinodal = [] # nh value
# find the spinodal line
for j, xIDP in enumerate(xIDPs):
    for i, nh in enumerate(nhs):
        if i > 0:
            current = stable[i,j]
            if current == 1 and prev == 0: # transition from unstable (0) to stable (1)
                x_spinodal.append(xIDP) 
                x_spinodal_uM.append(xIDP * Ctots[i,j] *10**24/(6.022e23)*10**6 )
                y_spinodal.append(np.mean([nh, nhs[i-1]])) # take the of nh value
        prev = stable[i,j]
x_spinodal = np.array(x_spinodal)

print('\n--- Upper limit for stability is nh = {}---'.format(max(y_spinodal)))

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
plt.semilogx(x_spinodal, y_spinodal, marker='None', ms=4, c='k', lw=1)
plt.fill_between(x_spinodal, y_spinodal, 0, alpha = 0.3, label='unstable')
plt.xlabel('$x_{IDP}$')
plt.ylabel('$n_h$')
plt.xlim(min(xIDPs),max(xIDPs))
plt.ylim(0,max(nhs))
ylim = ax.get_ylim()
xlim = ax.get_xlim()
plt.legend(loc='upper right',prop={'size':5})
plt.savefig('spinodal.png', dpi=500,transparent=False,bbox_inches="tight")

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
plt.semilogx(x_spinodal_uM, y_spinodal, marker='None', ms=4, c='k', lw=1)
plt.fill_between(x_spinodal_uM, y_spinodal, 0, alpha = 0.3, label='unstable')
plt.xlabel('$\\rho_{IDP}$ $(\mu M$ bead)')
plt.ylabel('$n_h$')
plt.ylim(0,max(nhs))
plt.xlim(min(xIDPs) * 30 *10**24/(6.022e23)*10**6,max(xIDPs)* 30 *10**24/(6.022e23)*10**6)
ylim = ax.get_ylim()
xlim = ax.get_xlim()
plt.legend(loc='upper right',prop={'size':5})
plt.savefig('spinodal_uM.png', dpi=500,transparent=False,bbox_inches="tight")
