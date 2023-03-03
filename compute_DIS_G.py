#!/usr/bin/env python3
"""
Compute the stability matrix (second derivative of the Helmholtz free energy wrt composition) the using mean-field approximation
and other thermodynamic quantities
"""
import numpy as np

BExclVol = np.array([[15.760520808387364, 9.620286929056373, 5.931417975266296, 5.931417975266296, 5.931417975266296, 1.5916018379087862], [9.620286929056373, 0.47265940950298235, 2.757684625446694, 2.757684625446694, 2.757684625446694, 0.8144758600054185], [5.931417975266296, 2.757684625446694, 15.348333517386306, 15.348333517386306, 15.348333517386306, 1.0129079034563466], [5.931417975266296, 2.757684625446694, 15.348333517386306, 15.348333517386306, 15.348333517386306, 1.0129079034563466], [5.931417975266296, 2.757684625446694, 15.348333517386306, 15.348333517386306, 15.348333517386306, 1.0129079034563466], [1.5916018379087862, 0.8144758600054185, 1.0129079034563466, 1.0129079034563466, 1.0129079034563466, 0.13270884748289072]])


class MF():
  def __init__(self, nh, nspecies = 6):
    # Model Parameters
    self.nmoleculetypes = 2
    self.nspecies = int(nspecies)
    self.nh =  float(nh)
    bead_map = {('W','Y','I','L','V'): 'A1', ('A','G','P'): 'A2', ('S','Q'): 'A3', ('K'): 'A3+', ('E'): 'A3-'}
    #4IDP-2Yx2A
    if nh % 1. == 0.:
        head = int(nh) * ['S', 'P', 'A', 'E', 'A', 'K', 'S', 'P', 'V', 'E', 'V', 'K']
    elif nh % 1. == 0.5:
        head = int(nh - (nh % 1.)) * ['S', 'P', 'A', 'E', 'A', 'K', 'S', 'P', 'V', 'E', 'V', 'K'] + ['S', 'P', 'A', 'E', 'A', 'K']
    else:
        raise Exception('float rather than x.5 is not implemented')
    tail = ['Y', 'W', 'S', 'A', 'Y', 'G', 'A', 'Y', 'A', 'Q', 'Y', 'V', 'Y', 'I', 'Y', 'A', 'Y', 'W', 'Y', 'L']
    idp_seq = head + tail

    idp_seq_ = []
    # parse sequence
    for a in idp_seq:
        for key, bead in bead_map.items():
            if a in key:
                idp_seq_.append(bead)
                continue
            
    self.Nofmolecule = [len(idp_seq_), 1]
    self.speciesVolumeFractionsPerMolecule = np.array([[ float(idp_seq_.count('A1'))/float(len(idp_seq_)), float(idp_seq_.count('A2'))/float(len(idp_seq_)), float(idp_seq_.count('A3'))/float(len(idp_seq_)), float(idp_seq_.count('A3+'))/float(len(idp_seq_)), float(idp_seq_.count('A3-'))/float(len(idp_seq_)), 0],
                                                    [ 0., 0., 0., 0., 0., 1.] ] )

    print('speciesVolumeFractionsPerMolecule\n',self.speciesVolumeFractionsPerMolecule)
    # For a homogeneous state, can transform interactions between bead species to
    # effective interactions between MOLECULE types using the number of beads of each type in a molecule
    self.BExclVolBetweenMolecules = np.array( [ np.zeros( self.nmoleculetypes ), np.zeros( self.nmoleculetypes ) ] )
    for i in range(nspecies):
      for j in range(nspecies):
        for p in range(self.nmoleculetypes):
          for pp in range(self.nmoleculetypes):
            self.BExclVolBetweenMolecules[p][pp] = self.BExclVolBetweenMolecules[p][pp] + BExclVol[i][j] * self.speciesVolumeFractionsPerMolecule[p][i] * self.speciesVolumeFractionsPerMolecule[pp][j] *self.Nofmolecule[p] * self.Nofmolecule[pp]

    # DIAGNOSTIC for model stability. Check eigenvalues of molecule-molecule excluded-volume matrix.
    # If all eigenvalues are positive, the mixture is unconditionally stable across all compositions.
    # If there's a negative eigenvalue, spinodals and binodals can be computed by including entropic terms in stability analysis.
    print("MIXTURE STABILITY ANALYSIS")
    print("- Excluded volume matrix BETWEEN MOLECULES:\n{}".format(self.BExclVolBetweenMolecules))
    # Eigensystem
    w, v = np.linalg.eig(self.BExclVolBetweenMolecules)
    print("\n- Eigenvalues = {}\n\n".format(w))


  #Stability matrix d^2 H/V / d(ni/V) / d(nj/V) 
  def compute_stability(self, _noverV):
    "compute the stability matrix of the entropic and enthalpic terms"
    stability_matrix = np.array([[self.BExclVolBetweenMolecules[0,0] +  1/_noverV[0], self.BExclVolBetweenMolecules[0,1]],
                                  [self.BExclVolBetweenMolecules[0,1],  self.BExclVolBetweenMolecules[1,1] +  1/_noverV[1]]])
    w, v = np.linalg.eig(stability_matrix)

    return w
  # 
  # Method to compute n/V for all molecule types from volume fractions and total bead density
  #
  def MoleculeDensitiesFromVolFracs(self, _moleculevolfracs, _C ):
    noverV = np.zeros( len(_moleculevolfracs) )
    for p in range( len(_moleculevolfracs) ):
      noverV[p] = _C * _moleculevolfracs[p] / self.Nofmolecule[p]
    return noverV

  #
  # Compute the Helmholtz free energy for a given composition.
  # Arguments:
  #   moleculevolfracs = array [ n_p N_p / (\sum_q n_q N_q) ]  =>  volume fraction of each molecule type (assuming equal bead volumes)
  #   C = \sum_p (n_p N_p) / V is the total bead density.
  #
  def HelmholtzFreeEnergy(self,_moleculevolfracs, _C ):
    # Overall volume fractions of each species
    globalVolFracsOfSpecies = np.zeros( self.nspecies )
    for p in range(self.nmoleculetypes):
      globalVolFracsOfSpecies = globalVolFracsOfSpecies + _moleculevolfracs[p] * self.speciesVolumeFractionsPerMolecule[p]

  #  print(" Volume fractions by species = {}; check sum = {}".format(globalVolFracsOfSpecies, np.sum(globalVolFracsOfSpecies)))

    # Compute the enthalpic part of the intensive Helmholtz free energy
    EnergyOverV = 0.
    for i in range(nspecies):
      for j in range(nspecies):
        EnergyOverV = EnergyOverV + globalVolFracsOfSpecies[i] * globalVolFracsOfSpecies[j] * BExclVol[i][j]
    EnergyOverV = EnergyOverV * _C * _C * 0.5

    # Entropic terms
    nOverV = self.MoleculeDensitiesFromVolFracs( _moleculevolfracs, _C )
    EntropyOverV = 0.
    for p in range(self.nmoleculetypes):
      EntropyOverV = EntropyOverV + nOverV[p] * ( np.log( nOverV[p] ) - 1. )

    # Full Helmholtz free energy (intensive)
    HoverV = EnergyOverV + EntropyOverV

  #  print("Helmholtz free energy, method 1:")
  #  print("E/V = {}, -TS/V = {}, H/V = {}\n".format(EnergyOverV, EntropyOverV, HoverV))

    return HoverV

  #
  # Alternative Helmholtz free energy method for passing n/V for each molecule type.
  #
  def HelmholtzFreeEnergy_Method2(self, _noverV ):
    # Compute the enthalpic part of the intensive Helmholtz free energy
    UoverV = 0.
    for p in range(self.nmoleculetypes):
      for pp in range(self.nmoleculetypes):
        UoverV = UoverV + 0.5 * _noverV[p] * _noverV[pp] * self.BExclVolBetweenMolecules[p][pp]

    # Entropic terms
    EntropyOverV = 0.
    for p in range(self.nmoleculetypes):
      EntropyOverV = EntropyOverV + _noverV[p] * ( np.log( _noverV[p] ) - 1. )

    # Full Helmholtz free energy (intensive)
    HoverV = UoverV + EntropyOverV

    return HoverV

  #
  # Compute the chemical potential from input n/V for all molecules.
  #   Evaluated using the n derivative of H at fixed V,T.
  #
  def ChemicalPotentialsFromComposition(self, _noverV ):
    mu = np.zeros( len(_noverV) )

    for p in range( len(_noverV) ):
      mu[p] = 0.
      # Enthalpic contribution
      for pp in range( len(_noverV) ):
        mu[p] = mu[p] +  _noverV[pp] * self.BExclVolBetweenMolecules[p][pp]
      # Ideal gas entropy
      mu[p] = mu[p] + np.log(_noverV[p])

    return mu

  #
  # Compute the grand potential G/V = H/V - \sum_p n_p \mu_p / V
  #
  def GrandFreeEnergy_from_Composition(self, _moleculevolfracs, _C):
    print("COMPUTE G FROM COMPOSITION")
    # Get particle numbers
    noverV = self.MoleculeDensitiesFromVolFracs( _moleculevolfracs, _C )
    print("  Input composition: C = {}, volfrac = {}".format(_C, _moleculevolfracs))
    print("  Converted to n/V = {}".format(noverV))

    # Get chemical potentials
    mu_p = self.ChemicalPotentialsFromComposition( noverV )
    print("Chemical potentials:")
    for p in range( self.nmoleculetypes ):
      print("  mu[{}] = {}".format(p, mu_p[p]))

    # Get the intensive Helmholtz free energy
    HoverV = self.HelmholtzFreeEnergy_Method2( noverV )
    print("Helmholtz free energy:")
    print("  H/V = {}".format(HoverV))

    # Compute the grand free energy
    GoverV = HoverV
    for p in range( self.nmoleculetypes ):
      GoverV = GoverV - mu_p[p] * noverV[p]

    print("Grand free energy:")
    print("  G/V = {}\n".format(GoverV))

    return GoverV

  #
  # Compute the grand potential G/V = H/V - \sum_p n_p \mu_p / V
  #
  def Pressure_from_Composition(self, _moleculevolfracs, _C):
  #  print("COMPUTE P FROM COMPOSITION")
    # Get particle numbers
    noverV = self.MoleculeDensitiesFromVolFracs( _moleculevolfracs, _C )
    #print("  Input composition: C = {}, volfrac = {}".format(_C, _moleculevolfracs))
    #print("  Converted to n/V = {}".format(noverV))


    P = 0
    # Compute the enthalpic part of the intensive Helmholtz free energy
    for p in range(self.nmoleculetypes):
      for pp in range(self.nmoleculetypes):
        P = P + 0.5 * noverV[p] * noverV[pp] * self.BExclVolBetweenMolecules[p][pp]

    # Entropic terms
    for p in range(self.nmoleculetypes):
      P = P + noverV[p]

  #  print("Pressure:")
  #  print("  P = {}\n".format(P))

    return P

  #
  # Compute the composition from input chemical potentials using optimization of a harmonic cost function
  # Arguments:
  #   mu: array of chemical potentials for each molecule type
  #   noverV_guess: initial estimate of the n/V values.
  #
  # Note that if the mixture is stable, [n/V] <-> [mu] is one-to-one and optimization should converge to a unique
  # value regardless of initial guess of n/V. However, if there is an unstable mode in the enthalpic contribution to H,
  # there will be multiple possible [n/V] for each collection of [mu]. Then the resulting [mu] set will be sensitive to
  # initial composition guess.
  def CompositionFromChemicalPotentials_Optimization(self, _mu, _noverV_guess):
    import scipy.optimize as so

    def penalty( _noverVcurrent, _mutarget):
      # What chemical potentials do we have now?
      _mucurrent = self.ChemicalPotentialsFromComposition( _noverVcurrent )
      # Harmonic penalty from target mu
      cost = np.sum(np.square(_mucurrent - _mutarget))
      # Diagnostic
  #    print(" _noverVcurrent = {}, _mucurrent = {}, _mucurrent - _mutarget = {}, cost = {}".format(_noverVcurrent, _mucurrent, _mucurrent-_mutarget, cost))
      return cost

    def grad_penalty( _noverVcurrent, _mutarget):
      # Gradient of the penalty function's "cost" w.r.t. n_p/V
      grad_cost = np.zeros( len(_noverVcurrent) )
      _mucurrent = self.ChemicalPotentialsFromComposition( _noverVcurrent )
      for p in range( len(_noverVcurrent) ):
        for pp in range( len(_noverVcurrent) ):
          Hhessian = self.BExclVolBetweenMolecules[p][pp]
          if p == pp:
            Hhessian = Hhessian + 1./_noverVcurrent[p]
          grad_cost[p] = grad_cost[p] + 2. * (_mucurrent[pp] - _mutarget[pp]) * Hhessian
      return(grad_cost)

    # Check the gradient manually
  #  print("Checking gradient")
  #  graddelta  = 1e-6
  #  print("  Analytic  = {}".format(grad_penalty(_noverV_guess, _mu)))
  #  print("  Numerical, 1 = {}".format( 0.5 * ( penalty(_noverV_guess + np.array([graddelta, 0.]), _mu) - penalty(_noverV_guess - np.array([graddelta, 0]), _mu))/graddelta ))
  #  print("  Numerical, 2 = {}".format( 0.5 * ( penalty(_noverV_guess + np.array([0., graddelta]), _mu) - penalty(_noverV_guess - np.array([0., graddelta]), _mu))/graddelta ))

    # With diagnostic printing...
  #  res = so.minimize( penalty, _noverV_guess, args=_mu, method='L-BFGS-B', jac=grad_penalty, bounds=[ (1e-10, 1e6), (1e-10, 1e6) ], tol=1e-20, options = {'maxiter': 15000, 'disp': True} )
  #  print(res)
    # Without diagnostic printing
    res = so.minimize( penalty, _noverV_guess, args=_mu, method='L-BFGS-B', jac=grad_penalty, bounds=[ (1e-10, 1e6), (1e-10, 1e6) ], tol=1e-20, options = {'maxiter': 15000} )

    return(res.x)

  # As above but using root finding.
  def CompositionFromChemicalPotentials_Root(self, _mu, _noverV_guess):
    import scipy.optimize as so

    def penalty( _noverVcurrent, _mutarget):
      # What chemical potentials do we have now?
      _mucurrent = self.ChemicalPotentialsFromComposition( _noverVcurrent )
      # Diagnostic
  #    print(" _noverVcurrent = {}, _mucurrent = {}, _mucurrent - _mutarget = {}".format(_noverVcurrent, _mucurrent, _mucurrent-_mutarget))
      return (_mucurrent - _mutarget)

    def grad_penalty( _noverVcurrent, _mutarget):
      # Gradient of the penalty function w.r.t. n_p/V
      # Start with enthalpic part of Hessian
      grad_pen = np.copy(self.BExclVolBetweenMolecules)
      # Add entropic contribution to diagonals
      for p in range( len(_noverVcurrent) ):
        grad_pen[p][p] = grad_pen[p][p] + 1./_noverVcurrent[p]
      return(grad_pen)

    # Check the gradient manually
  #  print("Checking gradient at composition coordinate = {}".format(_noverV_guess))
  #  graddelta  = 1e-8
  #  print("\n  Analytic  = {}".format(grad_penalty(_noverV_guess, _mu)))
  #  print("\n")
  #  print("  Numerical, 1 = {}".format( 0.5 * ( penalty(_noverV_guess + np.array([graddelta, 0.]), _mu) - penalty(_noverV_guess - np.array([graddelta, 0]), _mu))/graddelta ))
  #  print("  Numerical, 2 = {}".format( 0.5 * ( penalty(_noverV_guess + np.array([0., graddelta]), _mu) - penalty(_noverV_guess - np.array([0., graddelta]), _mu))/graddelta ))

    # With diagnostic printing...
  #  res = so.root( penalty, _noverV_guess, args=_mu, method='hybr', jac=grad_penalty, tol=1e-10, options = {'disp': True})
  #  print(res)
    # Without diagnostic printing
    res = so.root( penalty, _noverV_guess, args=_mu, method='hybr', jac=grad_penalty, tol=1e-20)

    return(res.x)


  # Compute the grand potential G from chemical potentials.
  # This method first obtains n/V, then calculates G from H, n, mu
  def GrandFreeEnergy_from_ChemPots(self, _mu, _noverV_guess):
    print("COMPUTE G FROM CHEMICAL POTENTIALS")
  #  print("  Input mu = {}, guess for n/V = {}".format(_mu, _noverV_guess))
    _noverV = self.CompositionFromChemicalPotentials_Optimization( _mu, _noverV_guess )
  #  _noverV = CompositionFromChemicalPotentials_Root( _mu, _noverV_guess )
  #  print("  Output n/V = {}".format(_noverV))

    # Compute the C parameter and volume fractions
    _C = 0.
    for p in range( len(_noverV) ):
      _C = _C + _noverV[p] * self.Nofmolecule[p]

    _moleculevolfracs = np.zeros( len(_mu) )
  #  print(_moleculevolfracs, _noverV, Nofmolecule)
    for p in range( len(_noverV) ):
      _moleculevolfracs[p] = _noverV[p] * self.Nofmolecule[p] / _C
  #  print("  C = {}, volfrac = {}, sum_volfrac = {}".format(_C, _moleculevolfracs, np.sum(_moleculevolfracs)))

    # Compute G/V
    GoverV = self.GrandFreeEnergy_from_Composition(_moleculevolfracs, _C)

    return(GoverV)

if __name__ == "__main__":
  # For command-line runs, build the relevant parser
  import argparse as ap
  parser = ap.ArgumentParser(description='Compute SCFT Grand Potential for Water + IDP CG model')
#  parser.add_argument('-t','--test',default=False,dest='test',action='store_true',help='Test thermodynamic functions')
  parser.add_argument('-g','--grand',default=False,dest='grand',action='store_true',help='Grand canonical ensemble?')
  parser.add_argument('-mu1', '--mu_IDP',default=1e-6,type=float,help='GCE: Chemical potential of the IDP')
  parser.add_argument('-mu2', '--mu_W',default=10.,type=float,help='GCE: Chemical potential of the water')
  parser.add_argument('-c1', '--cguess_IDP',default=-1.,type=float,help='GCE: Guess of n/V for IDP')
  parser.add_argument('-c2', '--cguess_W',default=-1.,type=float,help='GCE: Guess of n/V for water')
  parser.add_argument('-C', '--ctot', default=10., type=float, help='CE: Total C = sum_i(n_i * N_i)/V')
  parser.add_argument('-x', '--xIDP', default=0.1, type=float, help='CE: Volume fraction of IDP')
  parser.add_argument('-nh', default=10, type=int, help='number of repeating 6-mer unit in the head domain')
  # Parse the command-line arguments
  args=parser.parse_args()

#  if args.test == True:
#    print("Grand potential from specified composition")
#    GrandFreeEnergy_from_Composition(moleculevolfracs, C)
#
#    print("Grand potential from chemical potentials")
#    GrandFreeEnergy_from_ChemPots( [2371.398357867112, 10.909680348329285], [0.1, 50])

  if args.grand == True:
    # Get the initial guesses of the IDP and W concentrations
    cguess_IDP = args.cguess_IDP
    cguess_W   = args.cguess_W
    # If these guesses < 0 (default), try to guess using analytic means.
    # In a low-density system, n ~ z, where z = exp(mu)
    # For dense systems, the enthalpic term dominates. Guess using a diagonal approximation
    # Choose whichever is lower
    analytical = MF(args.nh)
    BExclVolBetweenMolecules = MF.BExclVolBetweenMolecules
    if cguess_IDP < 0.:
      cguess_IDP1 = np.exp(np.float128(args.mu_IDP))
      cguess_IDP2 = args.mu_IDP / BExclVolBetweenMolecules[0][0]
      cguess_IDP  = np.float64(np.minimum(cguess_IDP1, cguess_IDP2))
    if cguess_W < 0.:
      cguess_W1 = np.exp(np.float128(args.mu_W))
      cguess_W2 = args.mu_W / BExclVolBetweenMolecules[1][1]
      cguess_W  = np.float64(np.minimum(cguess_W1, cguess_W2))
    analytical.GrandFreeEnergy_from_ChemPots( [args.mu_IDP, args.mu_W], [cguess_IDP, cguess_W])
  else:
    analytical = MF(args.nh)
    analytical.GrandFreeEnergy_from_Composition( [ args.xIDP, 1.-args.xIDP ], args.ctot )


