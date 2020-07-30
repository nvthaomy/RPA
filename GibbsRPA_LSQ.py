import numpy as np
import time
import sys
import math
sys.path.append('/home/mnguyen/bin/scripts')
sys.path.append('/home/mnguyen/bin/RPA')

import RPA as RPAModule 
#import RPA_Ele as RPA
from scipy.optimize import least_squares

class GibbsSolver():    
    def __init__(self, Nspecies):
        self.Nspecies           = Nspecies
        self.xs = []
        self.Ctot = 33.
        self.CtotSpecies = []
        self.CtotNeutralSpecies = []
        self.ensemble = 'NPT' # or NVT
        self.Ptarget = 285.9924138
        
        self.NchargedSpecies   = 0
        self.NeffSpecies       =  Nspecies # number of "effective" species (ChargedPairs + neutral species)
        self.Charges           = np.zeros(Nspecies) # charges of all species, for polymer, this is the average charge/monomer 
        self.Valencies         = np.zeros(Nspecies) 
        self.ChargedPairs = []   
        self.ChargedSpecies = []
        self.NeutralSpecies = []
        
        self.DOP = np.ones(Nspecies)
        self.chain = 'DGC'
        self.Charges = [-1,1,1,-1,0]
        
        self.IncludeEvRPA = True
        self.abead = np.ones(self.Nspecies ) # smearing length of beads
        self.u0 = np.ones((self.Nspecies ,self.Nspecies )) # excluded vol matrix
        self.lB = 1.0 # Bjerrum length in kT
        self.b = np.ones(self.Nspecies ) # RMS length of bond, can be anything for small molecules
        self.V = 20
        self.VolFracBounds=[0.00001,1 - 0.00001]
        self.kmin = 1.e-5
        self.kmax = 20
        self.nk = 500
        self.LSQftol = 1e-5
        self.LSQxtol = 1e-5
        self.nIter = 5 # number of time to run LSQ  
        self.dtV = 0.01
        self.dtC = 0.1
        
        self.CI_Init = []
        self.fI_Init = 0.55
        self.LogFileName = 'log.txt'
        self.SetLogFile(self.LogFileName) # initialize LogFile
        
    def UpdateC(self):
        self.CtotSpecies = self.Ctot  * self.xs

    def Write2Log(self,_text):
        ''' Write out 2 Log file '''
        self.LogFile = open(self.LogFileName,'a')
        self.LogFile.write(str(_text))
        self.LogFile.flush()
        self.LogFile.close()
    
    def SetLogFile(self,_LogFileName):
        ''' Create Log File For the Run '''
        try:    
            self.LogFile = open(str(_LogFileName),'w')
        except:
            pass
            
            self.LogFile.close()
    
    def SetLSQTol(self,ftol,xtol):
        self.LSQftol = ftol
        self.LSQxtol = xtol
        
    
    def SetInitialCI(self,CI):
        self.CI_Init = np.array(CI,dtype=float)
    
    def SetInitialfI(self,fI):
        self.fI_Init = float(fI)
    
    
    def SetVolFracBounds(self, bounds):
        self.VolFracBounds = np.array(bounds)
        
    def SetEnsemble(self,_Ensemble):
        ''' The Ensemble to Use.
                (1) NVT
                (2) NPT        
        '''
        self.ensemble = str(_Ensemble)
    
    def SetPtarget(self,P):
        self.Ptarget = np.float(P)
        
    def SetDOP(self,DOP):
        self.DOP = np.array(DOP,dtype=float)

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

    def SetCtot(self,Ctot):
            self.Ctot = float(Ctot)
            
    def SetSpeciesVolFrac(self,xs):
        self.xs = np.array(xs)
        self.CtotSpecies = self.Ctot  * self.xs
        
    def SetChargedSpecies(self):
        ''' Set indices and number of charged species and indices of neutral species'''
        self.ChargedSpecies = np.where(self.Charges != 0.)[0]
        self.NeutralSpecies = np.where(self.Charges == 0.)[0]
        self.NchargedSpecies = len(self.ChargedSpecies)
        
    def SetCharges(self,_Charges):
        ''' Set charges of all species in the system '''
        if len(_Charges) != self.Nspecies:
            raise Exception('Must provide charges to all species')
        self.Charges = np.array(_Charges)
        self.Valencies = np.abs(self.Charges)
        return self.SetChargedSpecies()
       
    def SetChargedPairs(self,_ChargedPairs):
        ''' Set pairs of species that make up effective neutral species 
            and set the effective number of species'''

        if len(_ChargedPairs) != self.NchargedSpecies - 1:
            raise Exception('Number of charged pairs must be number of charged species - 1')
            
        for i1,i2 in _ChargedPairs:
            if self.Charges[i1] * self.Charges[i2] >= 0.:
                raise Exception('Species in a ChargedPair must be of opposite charges')
        self.ChargedPairs = _ChargedPairs
        self.NeffSpecies = int(len(self.ChargedPairs) + (self.Nspecies - self.NchargedSpecies))

    def GetEffC(self,Cs):
        ''' Get the total concentration of a charged pair provided the total concentration of individual species'''        
        
        Cpairs = np.zeros(len(self.ChargedPairs))        
        PairIds = [] # indices of pairs that involve species i
        zjs = [] # number of species i in pairs that it involves
        for i in self.ChargedSpecies:
            PairId = []
            zj = []
            for indx, Pair in enumerate(self.ChargedPairs):
                    if i in Pair:
                        PairId.append(indx)
                        j = int(np.array(Pair)[np.where(np.array(Pair) != i)[0]])
                        zj.append(self.Valencies[j])
            PairIds.append(PairId)
            zjs.append(zj)
            if len(PairId) == 1: # this species only involves in one pair, so its concentration directly gives the pair concentration
                Cpair = Cs[i]/zj[0]
                if Cpairs[PairId[0]] != 0.: # This pair concentration has alredy been calculated, check if consistent with prev. value
                    relErr = np.abs((Cpair-Cpairs[PairId[0]])/Cpairs[PairId[0]])
                    if relErr >= 5.0e-3:
                        self.Write2Log('\nInconsistent values for concentration of pair {} '.format(PairId[0]))
                        raise Exception('Inconsistent values for concentration of pair {} '.format(PairId[0]))
                else:
                    Cpairs[PairId[0]] = Cpair        
        # calculate concentration of pairs consisting of 2 species in more than one pair, assume no species involve in more than 2 pairs
        for i,ii in enumerate(self.ChargedSpecies):
            for k,indx in enumerate(PairIds[i]):
                if Cpairs[indx] == 0.: 
                    otherPairIds = np.array(PairIds[i])[np.where(np.array(PairIds[i]) != indx)[0]]
                    otherCpairs = Cpairs[otherPairIds]
                    otherPairZjs = np.array(zjs[i])[np.where(np.array(PairIds[i]) != indx)[0]]
                    Ci_in_other = np.multiply(otherCpairs,otherPairZjs)
                    Ci = Cs[ii]
                    Cpair = (Ci - np.sum(Ci_in_other))/zjs[i][k]
                    Cpairs[indx] = Cpair
        # Include neutral species  
        Cneutral = Cpairs.tolist()
        for i in self.NeutralSpecies:
            Cneutral.append(Cs[i])
        return np.array(Cneutral)

    def GetChargedC(self, i, Cpairs):
        ''' Get the concentration of an individual charged species 
            Cpairs: array of neutral species conc.
            i: index of charged species i'''                        
        C = 0
        for indx, Pair in enumerate(self.ChargedPairs):
            if i in Pair:
                j = int(np.array(Pair)[np.where(np.array(Pair) != i)[0]])
                zj = self.Valencies[j] # valency of the j species in pair
                Cpair = Cpairs[indx]
                C += zj * Cpair
        return C
    
    def BaroStat(self,RPA,dtC=0.2, maxIter = 1000 ,tol = 1.e-6):
        import math 
        '''
        RPA class with defined interactionsn
        dtC: step size
        maxIter: max iteration step'''
        
        C1 = self.Ctot
        self.Write2Log('==Barostat at P {}==\n'.format(self.Ptarget))
        self.Write2Log('# step C P Err\n')
        err = 10.
        step = 1
        while err>tol:        
            P1 = RPA.P()
            err = np.abs(P1-self.Ptarget)/np.abs(self.Ptarget)
            self.Write2Log('{} {} {} {}\n'.format(step,C1,P1,err))
            C1 = C1 * ( 1. + dtC*(math.sqrt(self.Ptarget/P1) - 1.) )
            RPA.SetC(self.xs*C1)
            RPA.Initialize()
            if err <= tol:
                self.Write2Log('error is below tolerance {}\n'.format(tol))
                break
            else:
                step += 1
        return C1
    
    def GetEffMu(self, Pair, Mu1, Mu2):
        ''' Get the combined chemical potential of a charged pair'''
        i1,i2 = Pair
        z1, z2 = [self.Valencies[i1],self.Valencies[i2]]
        N1, N2 = [self.DOP[i1],self.DOP[i2]]
        Mu = z2 * Mu1/N1 + z1 * Mu2/N2
        return Mu
    
    def Obj(self,x,RPA,Cs):
        '''x: volume fraction and concentration of EFFECTIVE species
        x = [fI,CIpair1,CIpair2,CIpair3,...CIn]
        RPA class with defined interactions'''
    
        fI = x[0]
        CIeffs = x[1:]
        #Cs = self.CtotSpecies   
        
        CIs = np.zeros(self.Nspecies)
        for i in self.ChargedSpecies:
            CIs[i] = self.GetChargedC(i,CIeffs )
        for ii,i in enumerate(self.NeutralSpecies):
            CIs[i] = CIeffs[len(self.ChargedPairs)+ii]
        
        fII = 1 - fI
        CIIs = (Cs - fI*CIs)/fII
        
        #box1
        RPA.SetC(CIs)
        RPA.Initialize()
        PI = RPA.P()
        muIs = np.zeros(self.Nspecies)
        for i in range(self.Nspecies):
            muIs[i] = RPA.mu(i)
        muIpairs = np.zeros(self.NeffSpecies)
        for i,[i1,i2] in enumerate(self.ChargedPairs):
            muIpairs[i] = self.GetEffMu([i1,i2],muIs[i1],muIs[i2])
        for ii,i in enumerate(self.NeutralSpecies):
            muIpairs[ii+len(self.ChargedPairs)] = muIs[i]

        #box2
        RPA.SetC(CIIs)
        RPA.Initialize()
        PII = RPA.P()
        muIIs = np.zeros(self.Nspecies)
        for i in range(self.Nspecies):
            muIIs[i] = RPA.mu(i)
        muIIpairs = np.zeros(self.NeffSpecies)
        for i,[i1,i2] in enumerate(self.ChargedPairs):
            muIIpairs[i] = self.GetEffMu([i1,i2],muIIs[i1],muIIs[i2])
        for ii,i in enumerate(self.NeutralSpecies):
            muIIpairs[ii+len(self.ChargedPairs)] = muIIs[i]
                
        f = [PII-PI]
        f.extend(muIIpairs-muIpairs)
        self.Write2Log('\n')
        self.Write2Log('fI {}\n'.format(fI))
        self.Write2Log('CIs {}\n'.format(CIs))
        self.Write2Log('err {}\n'.format(f))
        self.Write2Log('LSQ {}\n'.format(np.sum(np.array(f)**2)))
        return f

    
    def Run(self):    
        
        '''Make RPA Class'''
        RPA = RPAModule.RPA(self.Nspecies,self.Nspecies)
        RPA.Setchain(self.chain)
        RPA.SetDOP(self.DOP)
        RPA.Setcharge(self.Charges)
        RPA.Setabead(self.abead)
        RPA.Setu0(self.u0)
        RPA.SetlB(self.lB)
        RPA.Setb(self.b)
        RPA.SetV(self.V)
        RPA.Setkmin(self.kmin)
        RPA.Setkmax(self.kmax)
        RPA.Setnk(self.nk)
        RPA.IncludeEvRPA = self.IncludeEvRPA
        RPA.SetC(self.xs*self.Ctot)
        RPA.Initialize()
        
        if self.ensemble == 'NPT':
            self.Ctot = self.BaroStat(RPA)
            self.UpdateC()
         
        # calculate total neutral species C
        self.CtotNeutralSpecies = self.GetEffC(self.CtotSpecies)

        # LSQ
        CINeutral_init = self.GetEffC(self.CI_Init)
        x0 = [self.fI_Init]
        x0.extend(CINeutral_init)
        x0_org = x0
        
        # Set bounds
        bounds = np.array([len(x0)*[1.0e-200],len(x0)*[np.inf]])
        bounds[0,0] = self.VolFracBounds[0]
        bounds[1,0] = self.VolFracBounds[1]           

        self.Write2Log('\n=== LSQ solutions ===\n')
        cost = 1e5 
        LSQ0 = 1e5
        dtV = self.dtV
        dtC = self.dtC
        costs = []
        LSQs = []
        for k in range(self.nIter):
           self.Write2Log('==Iter {}==\n'.format(k+1))
           print('==Iter {}==\n'.format(k+1))
           sol = least_squares(self.Obj, x0, args = (RPA,self.CtotSpecies), bounds=bounds, ftol=self.LSQftol, xtol=self.LSQxtol) 

           #update solution           
           try:
               errs = self.Obj(sol.x,RPA,self.CtotSpecies)
               LSQ = np.sum(np.array(errs)**2)
               self.Write2Log('success {}\n'.format(sol.success))
               self.Write2Log('x {}\n'.format(sol.x))
               self.Write2Log('LSQ {}\n'.format(LSQ))
               self.Write2Log('cost {}\n'.format(sol.cost))
               print('success {}'.format(sol.success))
               print('x {}'.format(sol.x))
               print('LSQ {}'.format(LSQ))
               print('cost {}\n'.format(sol.cost))
               if LSQ < LSQ0:
                   cost = sol.cost
                   LSQ0 = LSQ
                   fI = sol.x[0]
                   CIpairs = sol.x[1:]
                   x_final = sol.x
                   costs.append(cost)
                   LSQs.append(LSQ)
           except:
               pass

           #update and perturb initial guess
           if not any(sol.x) == np.nan and not any(sol.x) == np.inf:
               x0 = sol.x.copy()
           elif self.nIter > 10 and k % 10 == 0: # go back to the original guess 
               x0 = x0_org.copy()
               
           for i, val in enumerate(x0):
                if i == 0:
                    newval = np.random.uniform(val-dtV*val, val+dtV*val)
                    dtV_adjust = dtV
                    while newval < 0. or newval > 1.:
                        dtV_adjust *= 0.9
                        newval = np.random.uniform(val-dtV_adjust*val, val+dtV_adjust*val)
                    x0[i] = newval
                else:
                    newval = np.random.uniform(val-dtC*val, val+dtC*val)
                    dtC_adjust = dtC
                    while newval < 0. :#or newval > self.CtotNeutralSpecies[i-1]/x0[0]:
                        dtC_adjust *= 0.9
                        newval = np.random.uniform(val-dtC_adjust*val, val+dtC_adjust*val)
                    x0[i] = newval

        self.Write2Log('accepted costs {}\n'.format(costs))
        self.Write2Log('accepted LSQs {}\n'.format(LSQs))

        # final run on the lowest cost results
        self.Write2Log('\n==Final run on the lowest cost results==\n')
        sol = least_squares(self.Obj, x_final, args = (RPA,self.CtotSpecies), bounds=bounds, ftol=self.LSQftol, xtol=self.LSQxtol)
        cost = sol.cost
        fI = sol.x[0]
        CIpairs = sol.x[1:]
        x_final = sol.x
        errs = self.Obj(sol.x,RPA,self.CtotSpecies)
        LSQ = np.sum(np.array(errs)**2)
        self.Write2Log('x {}\n'.format(sol.x))
        self.Write2Log('LSQ {}\n'.format(LSQ))

        CIs = np.zeros(self.Nspecies)
        for i in self.ChargedSpecies:
            CIs[i] = self.GetChargedC(i,CIpairs)                
        for ii,i in enumerate(self.NeutralSpecies):
            CIs[i] = CIpairs[len(self.ChargedPairs)+ii]
                       
        fII = 1-fI    
        CIIs = (self.CtotSpecies - fI*CIs)/(fII)
        
        self.Write2Log('fI {}\n'.format(fI))
        self.Write2Log('CIs {}\n'.format(CIs))
        self.Write2Log('fII {}\n'.format(fII))
        self.Write2Log('CIIs {}\n'.format(CIIs))
        print('fI {}\n'.format(fI))
        print('CIs {}\n'.format(CIs))
        print('fII {}\n'.format(fII))
        print('CIIs {}\n'.format(CIIs))
        
        gibbsErr = self.Obj(x_final,RPA,self.CtotSpecies)
        self.Write2Log('\n===errors===\n')
        s = '# dP '
        for i in range(len(self.ChargedPairs)):
            s += 'dmuPair_{} '.format(i)
        for i in self.NeutralSpecies:
            s  += 'dmu_{} '.format(i)
        self.Write2Log(s+'\n')
        for err in gibbsErr:
            self.Write2Log(err)
            self.Write2Log(' ')

        self.Write2Log('\n===final observables===\n')
        header = ''
        obs = ''
        RPA.SetC(CIs)
        RPA.Initialize()
        PI = RPA.P()
        header +='# PI '
        obs += '{} '.format(PI)
        muIs = np.zeros(self.Nspecies)
        for i in range(self.Nspecies):
            header +='muI_{} '.format(i)
            muIs[i] = RPA.mu(i)
            obs += '{} '.format(muIs[i])

        RPA.SetC(CIIs)
        RPA.Initialize()
        PII = RPA.P()
        header +='# PII '
        obs += '{} '.format(PII)
        muIIs = np.zeros(self.Nspecies)
        for i in range(self.Nspecies):
            muIIs[i] = RPA.mu(i)
            header +='muII_{} '.format(i)
            obs += '{} '.format(muIIs[i])
        self.Write2Log(header+'\n')
        self.Write2Log(obs+'\n')
        return fI,CIs,fII,CIIs,gibbsErr, PI, muIs, PII, muIIs, cost

    
