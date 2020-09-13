from scipy.interpolate import interp1d
import os, sys, re
import numpy as np
import mdtraj as md
import matplotlib
sys.path.append('/home/mnguyen/bin/scripts/')
import stats

showPlots = True
try:
  os.environ["DISPLAY"] #Detects if display is available
except KeyError:
  showPlots = False
  matplotlib.use('Agg') #Need to set this so doesn't try (and fail) to open interactive graphics window
import matplotlib.pyplot as plt
#plt.style.use('seaborn-dark')
matplotlib.rc('font', size=7)
matplotlib.rc('axes', titlesize=7)
colors = ['#6495ED','r','#6da81b','#483D8B','#FF8C00','#2E8B57','#800080','#008B8B','#949c2d', '#a34a17','#c43b99','#949c2d','#1E90FF']

######
dirs = [
'xSalt0.01/xNaCl0.01',
'xSalt0.02/xNaCl0.02',
'xSalt0.03/xNaCl0.03',
'xSalt0.032/xNaCl0.032',
#'xSalt0.035/xNaCl0.035',
'xSalt0.037/xNaCl0.037',
#'xSalt0.04/xNaCl0.04',
#'xSalt0.042/xNaCl0.042',
'xSalt0.045/xNaCl0.045',
'xSalt0.047/xNaCl0.047',
'xSalt0.05/xNaCl0.05',
'xSalt0.06/xNaCl0.06',
'xSalt0.07/xNaCl0.07',
'xSalt0.08/xNaCl0.08',
'xSalt0.09/xNaCl0.09',
'xSalt0.1/xNaCl0.1']

legends = ['0', '1']

#plotName = 'C PAH'
fPAA = 0.5
xPE = 0.0002
#xlabel = 'CPAH'
ylabel = 'CSalt'
CIPAAcol = 3 #C PAA
CIPAHcol = 5
CIycol = 7 #C na
CIIycol = 8
CIHOHcol = 11

nSpecies = 5
DOI = 1.
Cs = np.zeros(len(dirs))
xSalts = np.array([0.01,0.02,0.03,0.032,0.037,0.045,0.047,0.05,0.06,0.07,0.08,0.09,0.1])
x_PAAs = fPAA * xPE * np.ones(len(dirs))
x_PAHs = (1-fPAA) * xPE * np.ones(len(dirs))
x_ys = x_PAAs*DOI + xSalts
x_Cls = x_PAHs*DOI + xSalts 
x_HOHs = np.ones(len(dirs)) - x_PAAs - x_PAHs - x_ys - x_Cls
datafile = 'gibbs.dat'
newdatafile = 'gibbs0.dat'
errorfile = 'error.dat'
logfile = 'log.txt'
fIcol = 1 


#=========
cwd = os.getcwd()

fIs = np.zeros(len(dirs))
CIPAAs = np.zeros(len(dirs))
CIIPAAs = np.zeros(len(dirs))
CIPAHs = np.zeros(len(dirs))
CIIPAHs = np.zeros(len(dirs))
CIys = np.zeros(len(dirs))
CIIys = np.zeros(len(dirs))
CIHOHs = np.zeros(len(dirs))
CIIHOHs = np.zeros(len(dirs))
relerrors = np.zeros(len(dirs))
errors = []
for i,dir in enumerate(dirs):
    #remove first line of data
    file = open(os.path.join(cwd,dir,datafile), 'r')
    lines = file.readlines()
    lastline = lines[-1] 
#    lines = [lines[0],*lines[2:]]
#    lines = ''.join(lines)
#    file = open(os.path.join(cwd,dir,newdatafile), 'w')
#    file.write(lines)
#    file.close()
#    file = open(os.path.join(cwd,dir,newdatafile), 'r')
#    filename = os.path.join(cwd,dir,newdatafile) 

    # get total C
    vals = [float(a) for a in lastline.split()] #np.loadtxt(os.path.join(cwd,dir,newdatafile))[0]    
    fI = vals[1]
    CIs = vals[3:3+nSpecies*2:2]
    CIIs = vals[4:3+nSpecies*2:2]
    Cs[i] = fI * np.sum(CIs) + (1-fI) * np.sum(CIIs)
    print('Tot C {}'.format(Cs[i]))
    # get average volume fraction and concentration in boxI
    fIs[i] = vals[fIcol] #np.loadtxt(filename)[-1,fIcol]
    CIPAAs[i] = vals[CIPAAcol] #np.loadtxt(filename)[-1,CIPAAcol]
    CIPAHs[i] = vals[CIPAHcol] #np.loadtxt(filename)[-1,CIPAHcol]
    CIys[i] = vals[CIycol] #np.loadtxt(filename)[-1,CIycol]
    CIHOHs[i] = vals[CIHOHcol] #np.loadtxt(filename)[-1,CIHOHcol]

    #get max relative errors
    file = open(os.path.join(cwd,dir,'..',logfile), 'r')
    lines = file.readlines()
    try:
        lastline = lines[-1]
        vals = [float(a) for a in lastline.split()] 
    except:
        lastline = lines[-2]
        vals = [float(a) for a in lastline.split()]
    relerrors[i] = vals[1]

    #get error of operators
    errorfileN = os.path.join(cwd,dir,errorfile)
    file = open(os.path.join(cwd,dir,errorfile), 'r')
    lines = file.readlines()
    line0 = lines[0]
    line0 = line0.split('#')[-1]
    obsName = line0.split()[2:] # dP, dmu...
    nObs = len(obsName) #number of columns for P, mu errors
    lastline = lines[-1]
    vals = [float(a) for a in lastline.split()]
    error = []
    for j in range(2, 2+nObs):
        error.append(vals[j])
    errors.append(error)
errors = np.abs(np.array(errors))
CPAAs = Cs*x_PAAs #total conc of PAA
CPAHs = Cs*x_PAHs
CHOHs = Cs*x_HOHs
Cys = Cs*x_ys #total conc of Nacl
CIIPAAs = (CPAAs - fIs*CIPAAs)/(1-fIs) 
CIIPAHs = (CPAHs - fIs*CIPAHs)/(1-fIs)
CIIys = (Cys - fIs*CIys)/(1-fIs)
CIIHOHs = (CHOHs - fIs*CIHOHs)/(1-fIs)
#get salt concentration
CIsalts = CIys  - DOI * CIPAAs
CIIsalts = CIIys  - DOI * CIIPAAs

# **** swap last data point
#CIItemp = CIIPAAs[-1]
#CIIPAAs[-1] = CIPAAs[-1]
#CIPAAs[-1] = CIItemp
#CIItemp = CIIys[-1]
#CIIys[-1] = CIys[-1]
#CIys[-1] = CIItemp
        
# write data to file
data = np.stack((Cs,fIs,CIPAAs,CIIPAAs,CIPAHs, CIIPAHs, CIsalts,CIIsalts, CIHOHs, CIIHOHs),axis=1)
np.savetxt('gibbs.dat',data,header='Ctot fI CIPAA CIIPAA CIPAH CIIPAH CISalt CIISalt CIHOH CIIHOH')
#convert to real units
MoAA = 94
MoAH = 93.5
Mw = 18
Mna = 23
Mcl = 35.5

##PAA##
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
ax.set_prop_cycle('color', colors)
ax.plot(CIPAAs, CIsalts, marker='o', ms=5,ls='None',lw=1)
ax.plot(CIIPAAs, CIIsalts, marker='o', ms=5,ls='None',lw=1)
#tie line
for i in range(len(CIys)):
    ax.plot([CIPAAs[i],CIIPAAs[i]], [CIsalts[i],CIIsalts[i]], marker='None', ls=':', lw=1, label = 'xNaCl{}'.format(x_ys[i]))
plt.xlabel('C PAA')
plt.ylabel(ylabel)
plt.legend(loc='best',prop={'size':5})
title = 'C PAA'
plt.title(title, loc = 'center')
plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")

##PAH##
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
ax.set_prop_cycle('color', colors)
ax.plot(CIPAHs, CIsalts, marker='o', ms=5,ls='None',lw=1)
ax.plot(CIIPAHs, CIIsalts, marker='o', ms=5,ls='None',lw=1)
#tie line
for i in range(len(CIys)):
    ax.plot([CIPAHs[i],CIIPAHs[i]], [CIsalts[i],CIIsalts[i]], marker='None', ls=':', lw=1, label = 'xNaCl{}'.format(x_ys[i]))
plt.xlabel('C PAH')
plt.ylabel(ylabel)
plt.legend(loc='best',prop={'size':5})
title = 'C PAH'
plt.title(title, loc = 'center')
plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")

##PE##
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
ax.set_prop_cycle('color', colors)
ax.plot(CIPAAs+CIPAHs, CIsalts, marker='o', ms=5,ls='None',lw=1)
ax.plot(CIIPAAs+CIIPAHs, CIIsalts, marker='o', ms=5,ls='None',lw=1)
#tie line
for i in range(len(CIys)):
    ax.plot([CIPAAs[i]+CIPAHs[i],CIIPAAs[i]+CIIPAHs[i]], [CIsalts[i],CIIsalts[i]], marker='None', ls=':', lw=1, label = 'xNaCl{}'.format(x_ys[i]))
plt.xlabel('C PE')
plt.ylabel(ylabel)
plt.legend(loc='best',prop={'size':5})
title = 'C PE'
plt.title(title, loc = 'center')
plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")

#semilogx
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
ax.set_prop_cycle('color', colors)
ax.semilogx(CIPAAs, CIsalts, marker='o', ms=5,ls='None',lw=1)
ax.semilogx(CIIPAAs, CIIsalts, marker='o', ms=5,ls='None',lw=1)
#tie line
for i in range(len(CIys)):
    ax.semilogx([CIPAAs[i],CIIPAAs[i]], [CIsalts[i],CIIsalts[i]], marker='None', ls=':', lw=1, label = 'xNaCl{}'.format(x_ys[i]))
plt.xlabel('C PAA')
plt.ylabel(ylabel)
plt.legend(loc='best',prop={'size':5})
title = 'CPAA semilog'
plt.title(title, loc = 'center')
plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")


#errors
for i in range(nObs):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
    y = errors[:,i]
    ax.plot(x_ys,y,marker='o', ms=5,ls=':',lw=0.5, c='k')
    plt.xlabel('xSalt')
    plt.ylabel('{}'.format(obsName[i]))
    title = '{}'.format(obsName[i])
    plt.title(title, loc = 'center')
    plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=[3,2])
ax.semilogy(xSalts,relerrors, marker='o', ms=5,ls=':',lw=0.5, c='k')
plt.xlabel('xSalt')
plt.ylabel('{}'.format('Max relative error'))
title = 'max relative error'
plt.title(title, loc = 'center')
plt.savefig('_'.join(re.split(' |=|,',title))+'.png',dpi=500,transparent=True,bbox_inches="tight")
plt.show()
