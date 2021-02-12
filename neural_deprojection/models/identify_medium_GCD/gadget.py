"""This code was provided by Folkert Nobels and is used to extract data from the BAHAMAS simulation"""

import numpy as np
import h5py
import fnmatch
import pprint

# import logging

import pdb

file_type_options = ['snap', 'subh', 'fof', 'particles']

sims = ['OWLS',
        'BAHAMAS']

metals = ['Carbon',
          'Helium',
          'Hydrogen',
          'Iron',
          'Magnesium',
          'Neon',
          'Nitrogen',
          'Oxygen',
          'Silicon']

# list dimensionality for all variables to be reshaped
# after reading in
# n3_snap = ['Coordinates',
#            'Velocity']

n3 = ['Coordinates',
      'Velocity',
      'GroupCentreOfPotential',
      'CentreOfMass',
      'CentreOfMassVelocity',
      'CentreOfPotential',
      'GasSpin',
      'NSFSpin',
      'SFSpin',
      'StarSpin',
      'SubSpin',
      'Position']

n6 = ['MassType',
      'HalfMassProjRad',
      'HalfMassRad',
      'SubHalfMassProj',
      'SubHalfMass'
      'Mass_001kpc',
      'Mass_003kpc',
      'Mass_005kpc',
      'Mass_010kpc',
      'Mass_020kpc',
      'Mass_030kpc',
      'Mass_040kpc',
      'Mass_050kpc',
      'Mass_070kpc',
      'Mass_100kpc']

n9 = ['InertiaTensor']

class Gadget(object):
    """
    An object containing all relevant information for a gadget hdf5 file

    Parameters
    ----------
    model_dir : str

        directory of model to read in

    file_type : [default="snap", "fof", "subh", "particles"]

        Kind of file to read:
            "snap" - standard HDF5 Gadget3 snapshot
            "fof" - friends-of-friends group file
            "subh" - subfind group file
            "particles" - particles file

    snapnum : int

        number of snapshot to look at

    verbose : bool (default=False)
        print(messages

    gadgetunits : bool (default=False)
        keeps quantities in Gadget code units

    Examples
    --------
    #>>> import gadget as g

    Read in the halo masses from the subhalos file
    #>>> model_dir = "/disks/galform11/BAHAMAS/AGN_TUNED_nu0_L100N256_WMAP9/"
    #>>> group_info = g.Gadget(model_dir, "subh", 32)
    #>>> m200m = group_info.read_var("FOF/Group_M_Mean200")
    
    Read in gas temperatures from snapshot
    #>>> part_info = g.Gadget(model_dir, "particles", 32)
    #>>> temp = part_info.read_var("PartType0/Temperature")

    List contents of HDF5 file structure
    #>>> group_info.list_items("/")
    Items:
    [(u'Constants', <HDF5 group "/Constants" (0 members)>),
     (u'FOF', <HDF5 group "/FOF" (22 members)>),
     (u'Header', <HDF5 group "/Header" (0 members)>),
     (u'IDs', <HDF5 group "/IDs" (2 members)>),
     (u'Parameters', <HDF5 group "/Parameters" (1 members)>),
     (u'Subhalo', <HDF5 group "/Subhalo" (38 members)>),
     (u'Units', <HDF5 group "/Units" (0 members)>)]
    ============================================================
    Attrs:
    []
    ============================================================    
    """
    def __init__(self, model_dir, file_type, snapnum, sim='BAHAMAS',
                 smooth=False,verbose=False, gadgetunits=False, **kwargs):
        '''
        Initializes some parameters
        '''
        self.model_dir = model_dir
        if (file_type not in file_type_options or
            (sim == "BAHAMAS" and file_type == "fof")):
            raise ValueError('file_type %s not in options %s'%(file_type,
                                                               file_type_options))
        else:
            self.file_type = file_type
        if sim not in sims:
            raise ValueError('sim %s not in options %s'%(sim, sims))
        else:
            self.sim = sim
        self.filename = self.get_full_dir(model_dir, file_type, snapnum, sim)
        self.snapnum = snapnum
        self.smooth = smooth
        self.verbose = verbose
        self.gadgetunits = gadgetunits

    # ==========================================================================
    def get_full_dir(self, model_dir, file_type, snapnum, sim):
        '''
        Get filename, including full path and load extra info about number of
        particles in file
        '''
        dirpath = model_dir.rstrip('/') + '/data/'
        if sim == 'OWLS':
            if file_type == 'snap':
                dirname = 'snapshot_%.3i/' % snapnum
                fname = 'snap_%.3i.' % snapnum

            elif file_type == 'fof':
                dirname = 'groups_%.3i/' % snapnum
                fname = 'group%.3i.' % snapnum

            elif file_type == 'subh':
                dirname = 'subhalos_%.3i/' % snapnum
                fname = 'subhalo_%.3i.' % snapnum

        elif sim == 'BAHAMAS':
            if file_type == 'snap':
                dirname = 'snapshot_%.3i/' % snapnum
                fname = 'snap_%.3i.' % snapnum
            elif file_type == 'snip':
                dirname = 'snipshot_%.3i/' % snapnum
                fname = 'snip_%.3i.' % snapnum
            elif file_type == 'particles':
                dirname = 'particledata_%.3i/' % snapnum
                fname = 'eagle_subfind_particles_%.3i.' % snapnum
            else:
                dirname = 'groups_%.3i/' % snapnum
                if file_type == 'group':
                    fname = 'group_tab_%.3i.'%snapnum
                elif file_type == 'subh':
                    fname = 'eagle_subfind_tab_%.3i.'%snapnum

        # load actual file
        filename = dirpath + dirname + fname
        print(filename)
        try:
            try:
                # open first file
                f = h5py.File(filename + '0.hdf5', 'r')
            except:
                f = h5py.File(filename + 'hdf5', 'r')
        except:
            raise IOError('file %s does not exist/cannot be opened'%fname)

        if sim == 'OWLS':
            # Read in file and particle info
            self.num_files     = f['Header'].attrs['NumFilesPerSnapshot']
            self.num_part_tot  = f['Header'].attrs['NumPart_Total']
            self.num_part_file = f['Header'].attrs['NumPart_ThisFile']
            if file_type == 'fof':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['FOF'].attrs['Total_Number_of_groups']
                self.num_groups_file = f['FOF'].attrs['Number_of_groups']
            elif file_type == 'subh':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['SUBFIND'].attrs['Total_Number_of_groups']
                self.num_groups_file = f['SUBFIND'].attrs['Number_of_groups']
                self.num_sub_groups_tot  = f['SUBFIND'].attrs['Total_Number_of_subgroups']
                self.num_sub_groups_file = f['SUBFIND'].attrs['Number_of_subgroups']

        elif sim == 'BAHAMAS':
            # Read in file and particle info
            self.num_files     = f['Header'].attrs['NumFilesPerSnapshot']
            self.num_part_tot  = f['Header'].attrs['NumPart_Total']
            self.num_part_file = f['Header'].attrs['NumPart_ThisFile']
            if file_type == 'fof':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['FOF'].attrs['TotNgroups']
                self.num_groups_file = f['FOF'].attrs['Ngroups']
            elif file_type == 'subh':
                self.num_files = f['FOF'].attrs['NTask']
                self.num_groups_tot  = f['FOF'].attrs['TotNgroups']
                self.num_groups_file = f['FOF'].attrs['Ngroups']
                self.num_sub_groups_tot  = f['Subhalo'].attrs['TotNgroups']
                self.num_sub_groups_file = f['Subhalo'].attrs['Ngroups']

        self.read_file_attributes(f)

        f.close()

        return filename

    # --------------------------------------------------------------------------
    def read_file_attributes(self, f):
        '''
        Read in different physical parameters from file, should be simulation
        independent
        '''
        # Read info
        z = f['Header'].attrs['Redshift']
        self.z = round(z, 2)
        try:
            self.a = f['Header'].attrs['ExpansionFactor']
        except:
            self.a = f['Header'].attrs['Time']

        self.masses = f['Header'].attrs['MassTable']
        self.h = f['Header'].attrs['HubbleParam']
        self.boxsize = f['Header'].attrs['BoxSize'] # [h^-1 Mpc]

        # Read conversion units
        self.pi           = f['Constants'].attrs['PI']
        self.gamma        = f['Constants'].attrs['GAMMA']
        self.gravity      = f['Constants'].attrs['GRAVITY']
        self.solar_mass   = f['Constants'].attrs['SOLAR_MASS'] # [g]
        self.solar_lum    = f['Constants'].attrs['SOLAR_LUM'] # [erg/s]
        self.rad_const    = f['Constants'].attrs['RAD_CONST'] # [g]
        self.avogadro     = f['Constants'].attrs['AVOGADRO']
        self.boltzmann    = f['Constants'].attrs['BOLTZMANN']   # [erg/K]
        self.gas_const    = f['Constants'].attrs['GAS_CONST']
        self.c            = f['Constants'].attrs['C']
        self.planck       = f['Constants'].attrs['PLANCK']
        self.cm_per_mpc   = f['Constants'].attrs['CM_PER_MPC']
        self.protonmass   = f['Constants'].attrs['PROTONMASS'] # [g]
        self.electronmass = f['Constants'].attrs['ELECTRONMASS'] # [g]
        self.hubble       = f['Constants'].attrs['HUBBLE'] # [g]
        self.T_cmb        = f['Constants'].attrs['T_CMB0'] # [g]
        self.sec_per_Myr  = f['Constants'].attrs['SEC_PER_MEGAYEAR']
        self.sec_per_year = f['Constants'].attrs['SEC_PER_YEAR']
        self.stefan       = f['Constants'].attrs['STEFAN']
        self.thompson     = f['Constants'].attrs['THOMPSON']
        self.ev_to_erg    = f['Constants'].attrs['EV_TO_ERG']
        self.z_solar      = f['Constants'].attrs['Z_Solar']

        try:
            self.rho_unit = f['Units'].attrs['UnitDensity_in_cgs']
            self.omega_b = f['Header'].attrs['OmegaBaryon']
        except: # if no baryons in file
            self.rho_unit = 0
            self.omega_b = 0

        try:
            self.bh_seed = f['RuntimePars'].attrs['SeedBlackHoleMass_Msun']
            chem = f['Parameters/ChemicalElements'].ref
            self.solarabundance        = f[chem].attrs['SolarAbundance']
            self.solarabundance_oxygen = f[chem].attrs['SolarAbundance_Oxygen']
            self.solarabundance_iron   = f[chem].attrs['SolarAbundance_Iron']
        except:
            pass

    # --------------------------------------------------------------------------
    def list_items(self, var, j=0):
        '''
        List var items and attributes
        '''
        f = h5py.File(self.filename + str(j) + '.hdf5', 'r')
        try:
            items = list(f[var].items())
            print('Items:')
            pprint.pprint(items)
            print('============================================================')
        except AttributeError:
            print('%s is not an HDF5 group'%var)

        attrs = list(f[var].attrs.items())
        print('Attrs:')
        pprint.pprint(attrs)
        print('============================================================')

    # --------------------------------------------------------------------------
    def convert_cgs(self, var, j, verbose=True):
        '''
        Return conversion factor for var
        '''
        f = h5py.File(self.filename + str(j) + '.hdf5', 'r')
        # read in conversion factors
        string = var.rsplit('/')

        try:
            if not 'ElementAbundance' in string[-1]:
                self.a_scaling = f[var].attrs['aexp-scale-exponent']
                self.h_scaling = f[var].attrs['h-scale-exponent']
                self.CGSconversion = f[var].attrs['CGSConversionFactor']
            else:
                metal = f[var+'/'+metals[0]].ref
                self.a_scaling = f[metal].attrs['aexp-scale-exponent']
                self.h_scaling = f[metal].attrs['h-scale-exponent']
                self.CGSconversion = f[metal].attrs['CGSConversionFactor']

        except:
            print('Warning: no conversion factors found in file 0 for %s!'%var)
            self.a_scaling = 0
            self.h_scaling = 0
            self.CGSconversion = 1

        f.close()
        conversion = (self.a**self.a_scaling * self.h**self.h_scaling *
                      self.CGSconversion)

        if verbose:
            print('Converting to physical quantities in CGS units: ')
            print('a-exp-scale-exponent = ', self.a_scaling)
            print('h-scale-exponent     = ', self.h_scaling)
            print('CGSConversionFactor  = ', self.CGSconversion)

        return conversion

    # --------------------------------------------------------------------------
    def read_attr(self, path, j=0):
        '''
        Function to readily read out group attributes
        '''
        f = h5py.File(self.filename + str(j) + '.hdf5', 'r')

        string = path.split('/')
        group = '/'.join(string[:-1])
        attr = string[-1]

        val = f[group].attrs[attr]
        f.close()
        return val

    # --------------------------------------------------------------------------
    def read_var(self, var, gadgetunits=False, verbose=True):
        '''

        '''
        if self.sim == 'OWLS':
            self.data = self.read_owls_array(var,
                                             gadgetunits=gadgetunits,
                                             verbose=verbose)
        elif self.sim == 'BAHAMAS':
            self.data = self.read_bahamas_array(var,
                                                gadgetunits=gadgetunits,
                                                verbose=verbose)

        if verbose: print('Finished reading snapshot')
        return self.data

    # --------------------------------------------------------------------------
    def read_owls_array(self, var, gadgetunits=False, verbose=True):
        '''
        Reading routine that does not use hash table
        '''
        # open file
        try:
            f = h5py.File(self.filename + '0.hdf5', 'r')
        except:
            f = h5py.File(self.filename + 'hdf5', 'r')

        #Set up array depending on what we're reading in
        string = var.rsplit('/')

        self.data = np.empty([0])

        #read data from first file
        if verbose: print('Reading variable ', var)

        # loop over files to find the first one with data in it
        # j tracks file number
        # Ndata contains dimension of data
        read = True
        withdata = -1
        j = 0
        Ndata = 0
        while read:
            f = h5py.File(self.filename + str(j) + ".hdf5", 'r')
            if self.file_type == 'snap':
                num_part_file = f['Header'].attrs['NumPart_ThisFile']
                # parttype is always first part of var
                if 'PartType' in string[0]:
                    parttype = np.int(string[0][-1])
                    Ndata = num_part_file[parttype]

            else:
                if 'FOF' in var:
                    if 'PartType' in var:
                        parttype = string[1][-1]
                        Ndata = f['FOF'].attrs['Number_per_Type'][parttype]
                    else:
                        Ndata = f['FOF'].attrs['Number_of_groups']
                elif 'SUBFIND' in var:
                    if 'PartType' in var:
                        parttype = string[1][-1]
                        Ndata = f['SUBFIND'].attrs['Number_per_Type'][parttype]
                    else:
                        Ndata = f['SUBFIND'].attrs['Number_of_subgroups']

            if Ndata == 0:
                if verbose: print('Npart in file %i is %i continuing...'%(j, Ndata))
                f.close()
                j += 1
                if j >= self.num_files :
                    print('No particles found in any file!')
                    return self.data
            else:
                if withdata < 0: withdata = j
                if verbose: print('Npart in file %i is %i continuing...'%(j, Ndata))
                # read data
                self.append_result(f, var, j,  verbose)
                f.close()
                j += 1
                if j >= self.num_files:
                    read = False

        # convert to CGS units
        if not gadgetunits:
            conversion = self.convert_cgs(var, j-1, verbose=verbose)
            self.data = self.data * conversion
            if verbose: print('Returning data in CGS units')

        if verbose: print('Finished reading data')

        # still need to reshape output
        if string[-1] in n3:
            self.data = self.data.reshape(-1,3)
        elif string[-1] in n6:
            self.data = self.data.reshape(-1,6)
        elif string[-1] in n9:
            self.data = self.data.reshape(-1,9)

        return self.data

    # --------------------------------------------------------------------------
    def read_bahamas_array(self, var, region=None, gadgetunits=False,
                           verbose=True):
        '''
        BAHAMAS is partway between eagle and owls, can use EagleSnapshot
        '''
        # open file
        try:
            f = h5py.File(self.filename + '0.hdf5', 'r')
        except:
            f = h5py.File(self.filename + 'hdf5', 'r')

        #Set up array depending on what we're reading in
        string = var.rsplit('/')
        self.data = np.empty([0])

        #read data from first file
        if verbose: print('Reading variable ', var)

        # loop over files to find the first one with data in it
        # j tracks file number
        # Ndata contains dimension of data
        read = True
        withdata = -1
        j = 0
        Ndata = 0
        while read:
            f = h5py.File(self.filename + str(j) + ".hdf5", 'r')
            if self.file_type == 'snap' or self.file_type == 'particles':
                num_part_file = f['Header'].attrs['NumPart_ThisFile']
                # parttype is always first part of var
                if 'PartType' in string[0]:
                    parttype = np.int(string[0][-1])
                    Ndata = num_part_file[parttype]
                elif 'ParticleID' in var:
                    Ndata = f['Header'].attrs['Nids']

            else:
                if 'FOF' in var:
                    Ndata = f['FOF'].attrs['Ngroups']
                elif 'Subhalo' in var:
                    Ndata = f['Subhalo'].attrs['Ngroups']
                elif 'ParticleID' in var:
                    Ndata = f['IDs'].attrs['Nids']
            if Ndata == 0:
                if verbose: print('Npart in file %i is %i continuing...'%(j, Ndata))
                f.close()
                j += 1
                if j >= self.num_files :
                    print('No particles found in any file!')
                    return self.data
            else:
                if withdata < 0: withdata = j
                if verbose: print('Npart in file %i is %i continuing...'%(j, Ndata))
                # read data
                self.append_result(f, var, j,  verbose)
                f.close()
                j += 1
                if j >= self.num_files:
                    read = False

        # convert to CGS units
        if not gadgetunits:
            conversion = self.convert_cgs(var, j-1, verbose=verbose)
            self.data = self.data * conversion
            if verbose: print('Returning data in CGS units')

        if verbose: print('Finished reading data')

        # still need to reshape output
        if string[-1] in n3:
            self.data = self.data.reshape(-1,3)
        elif string[-1] in n6:
            self.data = self.data.reshape(-1,6)
        elif string[-1] in n9:
            self.data = self.data.reshape(-1,9)

        return self.data

    # --------------------------------------------------------------------------
    def append_result(self, f, var, j, verbose):
        '''
        Append var data from file j to self.data
        '''
        try:
            self.data = np.append(self.data, f[var][:].flatten(), axis=0)
            return
        except KeyError:
            print('KeyError: Variable ' + var + ' not found in file ', j)
            print('Returning value of False')
            return False
# ==============================================================================
