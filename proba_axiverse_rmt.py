##### This script calculates the probability of the axiverse given
##### i) Cosmological constraints
##### ii) A probability distribution for mu, alpha, theta_i
##### It is based on a simple MC algorithm, and a binary check:
##### A model is allowed if it does not exceed the 95%CL on Om_zc.
##### credit: Vivian Poulin, Tristan Smith, 09.20.2018


import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from sympy import Eq, Symbol, solve, nsolve
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce 
import numpy.random as rd


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)







#### Define cosmological parameters #####
Omega_m = 0.3
Omega_r = 8e-5
Omega_Lambda = 1-Omega_m

#### Define model parameters #####
n=1
F=7./8
p=1./2 ##will be checked later

#### Read and interpolate constraints #####
Omega_at_zc = np.loadtxt("Omega_at_zc.dat")
interp_constraints_log_extrap = InterpolatedUnivariateSpline(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]),k=1)
interp_constraints_log_interp = interp1d(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]))


#### Define distribution properties #####
###parameters of a log flat distribution (currently centered on 0 +- 2)
# log_mu_min_min = -2 ##minimal value
# log_mu_min_max = 10 ##minimal value
# log_mu_size_min = 1 ##min size of the interval.
# log_mu_size_max = 20 ##max size of the interval.
# log_alpha_min = -2
# log_alpha_size = 2

# ##FOR PLOTTING ONLY##
log_mu_min_min = 4 ##minimal value
log_mu_min_max = 10 ##minimal value##NOTUSED
log_mu_size_min = 4 ##min size of the interval.
log_mu_size_max = 20 ##max size of the interval.##NOTUSED
log_alpha_min = -2
log_alpha_size = 2

##for a future gaussian distribution
##var_log_mu =
## mean_log_mu = 3
## log_mu = var_log_mu*np.abs(np.random.rand())*np.pi+mean_log_mu





######################### RMT PARAMETERS #####################################################

kinetic_shaping_param=0.95    #Distribution shaping parameter for the kinetic matrix 
mass_shaping_param=0.15		  #Distribution shaping parameter for the mass matrix 
decay_first_raw_moment=1	  #Fixes the mean scale for the decacy constants 
mass_first_raw_moment=1		  #Fixes the mean mass scale of the theory
number_of_samples=200		  #Dimensionality of the matrices, should be sufficiently large for universality i.e>50
number_of_iterations=30	      #Accuracy parameter for the three-dimensional sampling space
k_cov = False                 #Fixes wether we model a population covariance matrix for the Kinetic matrix, if False the population covariance matrix is the identity
m_cov = False				  #Fixes wether we model a population covariance matrix for the Mass matrix, if False the population covariance matrix is the identity
model_K = True 				  #Fixes if we consider priors on the kinetic matrix, if false we assume we begin in the kinetically aligned basis

###############################################################################################
mass_samples=np.array([])		  
decay_samples=np.array([])
phi_samples=np.array([])

L1=int(number_of_samples/kinetic_shaping_param) 
L2=int(number_of_samples/mass_shaping_param) 
for j in range(0,number_of_iterations):
    print(j)
    
    if model_K == True:
        if k_cov==True:
            kinetic_sub  = decay_first_raw_moment*(np.random.randn(number_of_samples, L1))
            kinetic_isometric = np.dot(kinetic_sub,(kinetic_sub.T))/L1
            kinetic_covariance = numpy.identity(number_of_samples)
            kinetic_matrix = np.dot(kinetic_isometric,kinetic_covariance)
        else:
            kinetic_sub  = decay_first_raw_moment*(np.random.randn(number_of_samples, L1))
            kinetic_matrix = np.dot(kinetic_sub,(kinetic_sub.T))/L1
    else: 
        kinetic_matrix  = numpy.identity(number_of_samples)	
        
    ev,p = np.linalg.eig(kinetic_matrix) 
    decay_constants = np.sqrt(np.abs(2*ev))
    log_deacy_constants = np.log10(decay_constants)
    decay_samples=np.append(decay_samples,log_deacy_constants)	 
    kD = reduce(np.dot, [p.T, kinetic_matrix, p]) 
    kD[kD < 1*10**-13] = 0 
    perturbative_matrix = np.zeros((number_of_samples, number_of_samples)) 
    np.fill_diagonal(perturbative_matrix, 1/(decay_constants))
    
    if m_cov==True:
        mass_sub = mass_first_raw_moment*(np.random.randn(number_of_samples, L2)) 
        mass_isometric = np.dot(mass_sub,(mass_sub.T))/L2
        mass_covariance = numpy.identity(number_of_samples)
        mass_matrix = np.dot(kinetic_isometric,kinetic_covariance)
    else:
        mass_sub = mass_first_raw_moment*(np.random.randn(number_of_samples, L2)) 
        mass_marix = np.dot(mass_sub,(mass_sub.T))/L2		
    
    mass_eigenstate_matrix = 2.*reduce(np.dot, [perturbative_matrix,p,mass_marix,p.T,perturbative_matrix.T]) 
    mass_eigenstates,mv = np.linalg.eig(mass_eigenstate_matrix) 
    #ma_array = np.sqrt(ma_array)
    log_mass_eigenstates = np.log10(mass_eigenstates)
    mass_samples=np.append(mass_samples,log_mass_eigenstates)
    
    phiin_array = rd.uniform(-np.pi,np.pi,number_of_samples)

    for i in range (0,number_of_samples):
        phiin_array[i] = phiin_array[i]*decay_constants[i]
    phiin_array=np.dot(mv,phiin_array)
    phiin_array = np.log10(np.abs(phiin_array))
    phi_samples=np.append(phi_samples,phiin_array)

##########################################################################

#### some additional checks ####
print_in_file = False
print_in_screen = True
plot_results = True

#### How many iterations? #####
max_iterations = 300
j = 0
fraction_allowed = []
f2 = open('fraction_models_allowed_axiverse.dat', 'w')
f2.write('# log_mu_min \t\t log_mu_size \t\t frac \n') # column titles
fraction_table = []
log_mu_min_table = []
log_mu_size_table = []

log_mu_min = log_mu_min_min
while log_mu_min < log_mu_min_max:
    log_mu_size = log_mu_size_min
    while log_mu_size < log_mu_size_max:
        print("log_mu_min:", log_mu_min, "log_mu_size:", log_mu_size)
        #### Define some tables for writing #####
        if plot_results is True:
            mu_excluded = []
            alpha_excluded = []
            zc_excluded = []
            frac_zc_excluded = []
            Theta_i_excluded = []
            mu_allowed = []
            alpha_allowed = []
            zc_allowed = []
            frac_zc_allowed = []
            Theta_i_allowed = []
            all_mu = []
            all_alpha = []
            all_Theta = []

        #### The code really starts here! #####
        total = 0
        allowed = 0
        excluded = 0

        while total < max_iterations:
            total +=1
            ##Draw from log flat distribution##
            log_mu = np.random.rand()*log_mu_size+log_mu_min #np.random.rand() draws in [0,1) ##log_mu_min is negative!
            mu = 10**log_mu
            log_alpha = np.random.rand()*log_alpha_size+log_alpha_min
            alpha = 10**log_alpha
            ##Draw from flat distribution
            Theta_i = np.random.rand()*np.pi ##draws from [0;pi)
            if plot_results is True:
                all_mu.append(log_mu)
                all_alpha.append(log_alpha)
                all_Theta.append(Theta_i)
            ##Calculate Omega_at_zc and zc
            Omega_zc = 1./6*alpha*alpha*mu*mu*(1-np.cos(Theta_i))**n
            xc = (1-np.cos(Theta_i))**((1-n)/2)/mu*np.sqrt((1-F)*(6*p+2)*Theta_i/(n*np.sin(Theta_i)))
            zc = Symbol('zc')
            Results = solve(Omega_m*(1+zc)**3.0+Omega_r*(1+zc)**4+Omega_Lambda-(p/xc)**2,zc) ##nb: this assumes negligible Omega_zc. What if not??
            zc_found = 0
            ##Results is a table with 4 entries: the 2 firsts are real, one of which is positive: this is our zc.
            if Results[0] > 0:
                 zc_found = Results[0]
            if Results[1] > 0:
                zc_found = Results[1]

            ##Check that our hypothesis of zc>z_eq was correct. Otherwise we recalculate.
            if zc_found < Omega_m/Omega_r:
                p=2./3
                xc = (1-np.cos(Theta_i))**((1-n)/2)/mu*np.sqrt((1-F)*(6*p+2)*Theta_i/(n*np.sin(Theta_i)))
                zc = Symbol('zc')
                Results = solve(Omega_m*(1+zc)**3.0+Omega_r*(1+zc)**4+Omega_Lambda-(p/xc)**2,zc)
                if Results[0] > 0:
                     zc_found = Results[0]
                if Results[1] > 0:
                    zc_found = Results[1]

            if zc_found == 0:
                zc_found = Results[1]
                if print_in_screen is True:
                    print("Weird! zc is negative: it might mean that the field is a dark energy candidate.")
                    print("mu:", mu, "alpha:",  alpha, "Theta_i:",  Theta_i, "Omega_zc:", Omega_zc, "xc:", xc, "zc:", Results[1])
                    #print(Results)
                ## Currently compares to Lambda: if = or less, assume it is viable. TO BE IMPROVED.
                if Omega_zc <= Omega_Lambda:
                    if plot_results is True:
                        mu_allowed.append(log_mu)
                        alpha_allowed.append(log_alpha)
                        zc_allowed.append(-np.log10(float((zc_found*zc_found)**0.5)))
                        frac_zc_allowed.append(0)
                        Theta_i_allowed.append(Theta_i)
                    allowed+=1
                else:
                    if plot_results is True:
                        mu_excluded.append(log_mu)
                        alpha_excluded.append(log_alpha)
                        zc_excluded.append(np.log10(float(zc_found)))
                        frac_zc_excluded.append(np.log10(Omega_zc/(Omega_m*(1+float(zc_found))**3.0+Omega_r*(1+float(zc_found))**4+Omega_Lambda)))
                        Theta_i_excluded.append(Theta_i)
                    excluded+=1
            else:
                if print_in_screen is True:
                    print("mu:", mu, "alpha:",  alpha, "Theta_i:",  Theta_i, "Omega_zc:", Omega_zc, "xc:", xc, "log10_ac:", np.log10(float(1/zc_found)))
                ##interp1d is more stable but cannot extrapolate.
                if 0 > np.log10(float(1/zc_found)) > -6:
                    # print "interpolation",np.log10(float(1/zc_found))
                    if interp_constraints_log_interp(np.log10(float(1/zc_found))) < np.log10(Omega_zc) :
                        if print_in_screen is True:
                            print("EXCLUDED",interp_constraints_log_interp(np.log10(float(1/zc_found))), "<",np.log10(Omega_zc), "at ac=",1/zc_found)
                        if plot_results is True:
                            mu_excluded.append(log_mu)
                            alpha_excluded.append(log_alpha)
                            zc_excluded.append(np.log10(float(zc_found)))
                            frac_zc_excluded.append(np.log10(Omega_zc/(Omega_m*(1+float(zc_found))**3.0+Omega_r*(1+float(zc_found))**4+Omega_Lambda)))
                            Theta_i_excluded.append(Theta_i)
                        excluded+=1 ##increment the number of points excluded
                    else:
                        if print_in_screen is True:
                            print("allowed",interp_constraints_log_interp(np.log10(float(1/zc_found))), ">", np.log10(Omega_zc), "at ac=",1/zc_found)
                        if plot_results is True:
                            mu_allowed.append(log_mu)
                            alpha_allowed.append(log_alpha)
                            zc_allowed.append(np.log10(float(zc_found)))
                            frac_zc_allowed.append(np.log10(Omega_zc/(Omega_m*(1+float(zc_found))**3.0+Omega_r*(1+float(zc_found))**4+Omega_Lambda)))
                            Theta_i_allowed.append(Theta_i)
                        allowed+=1 ##increment the number of points allowed
                else:
                    # print "extrapolation"
                    if interp_constraints_log_extrap(np.log10(float(1/zc_found))) < np.log10(Omega_zc) :
                        if print_in_screen is True:
                            print("EXCLUDED",interp_constraints_log_extrap(np.log10(float(1/zc_found))), "<",np.log10(Omega_zc), "at ac=",1/zc_found)
                        if plot_results is True:
                            mu_excluded.append(log_mu)
                            alpha_excluded.append(log_alpha)
                            Theta_i_excluded.append(Theta_i)
                        excluded+=1 ##increment the number of points excluded
                    else:
                        if print_in_screen is True:
                            print("allowed",interp_constraints_log_extrap(np.log10(float(1/zc_found))), ">", np.log10(Omega_zc), "at ac=",1/zc_found)
                        if plot_results is True:
                            mu_allowed.append(log_mu)
                            alpha_allowed.append(log_alpha)
                            zc_allowed.append(np.log10(float(zc_found)))
                            frac_zc_allowed.append(np.log10(Omega_zc/(Omega_m*(1+float(zc_found))**3.0+Omega_r*(1+float(zc_found))**4+Omega_Lambda)))
                            Theta_i_allowed.append(Theta_i)
                        allowed+=1 ##increment the number of points allowed
        fraction = allowed*1.0/max_iterations
        print("fraction of models allowed = ", fraction)
        fraction_table.append(fraction)
        log_mu_min_table.append(log_mu_min)
        log_mu_size_table.append(log_mu_size)
        f2.write(str(log_mu_min) + '\t\t' + str(log_mu_size) + '\t\t' + str(fraction) +'\n') # writes the array to file


        #### if needed, write to output file ####
        if print_in_file is True:
            f = open('allowed_models_axiverse.dat', 'w')
            f.write('# mu \t\t alpha \t\t Theta_i \n') # column titles
            for i in range(allowed):
            	f.write(str(mu_allowed[i]) + '\t\t' + str(alpha_allowed[i]) + '\t\t' + str(Theta_i_allowed[i]) +'\n') # writes the array to file
            f.close()
        #### if needed, plot the allowed parameters ####
        if plot_results is True:
            fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax2 = fig.add_subplot(211, projection='3d')
            # # ax.set_xlim(log_mu_min,log_mu_min+log_mu_size)
            # # ax.set_ylim(log_alpha_min,log_alpha_min+log_alpha_size)
            # ax.set_zlim(0,np.pi)
            # ax.set_xlabel(r"$z_c$", fontsize=16)
            # ax.set_ylabel(r"$f(z_c)$", fontsize=16)
            # ax.set_zlabel(r"$\Theta_i$", fontsize=16)
            # print(len(zc_allowed),len(frac_zc_allowed),len(Theta_i_allowed))
            # ax.scatter(zc_allowed,frac_zc_allowed,Theta_i_allowed,c='b')
            ax2 = fig.add_subplot(111, projection='3d')
            ax2.set_xlim(log_mu_min,log_mu_min+log_mu_size)
            ax2.set_ylim(log_alpha_min,log_alpha_min+log_alpha_size)
            ax2.set_zlim(0,np.pi)
            ax2.set_xlabel(r"$\mu$", fontsize=16)
            ax2.set_ylabel(r"$\alpha$", fontsize=16)
            ax2.set_zlabel(r"$\Theta_i$", fontsize=16)
            ax2.scatter(mu_allowed,alpha_allowed,Theta_i_allowed,c='b')
            ax2.scatter(mu_excluded,alpha_excluded,Theta_i_excluded,c='r')
            # print("total=",len(all_mu),len(mu_allowed)+len(mu_excluded),mu_allowed,mu_excluded)
            plt.show()
        #### calculate the fraction of allowed models ####
        log_mu_size+=1


    log_mu_min+=1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r"$\mu_{\rm min}$", fontsize=16)
ax.set_ylabel(r"$\Delta \mu$", fontsize=16)
ax.set_zlabel(r"$p_{\rm allowed}$", fontsize=16)
print(len(log_mu_min_table),len(log_mu_size_table),len(fraction_table))
ax.scatter(log_mu_min_table,log_mu_size_table,fraction_table,c='b')
plt.show()
f2.close()
