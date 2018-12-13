import numpy as np
import math


""" This function is used to calculate the trajectories, given the beam
    properties and beamline geometry. """
#%%
def trajectories_function(zone_of_freezing, beamline_geometry,
                          beam, molecule, N_molecules,store_trajectories):
    #%% Set up parameters for the simulation
    #Set gravitational acceleration
    g = 9.81 # m/s**2
    
    #%%Set up counters for the number of molecules that hit each element and
    # a structure to store trajcetories
    counts = {}
    trajectories = {}
    velocities = {}
    
    for i in beamline_geometry:
        name = beamline_geometry[i]['name']
        counts[name] = 0;
        trajectories[name] = []
    
    #Also set up counters for the total numbers of molecules that are detected
    #or hit the apparatus
    counts['detected'] = 0
    counts['dead'] = 0
    
    #Also initialize storage for the trajectories of detected molecules
    trajectories['detected'] = []
    velocities['detected'] = []
    
    #Initialize arrays for storing positions and velocities
    #First calculate how large the arrays need to be:
    number_of_points = (int(np.round(2*len(beamline_geometry))))
    #Then initialize
    x = np.zeros((3,number_of_points))
    v = np.zeros((3,number_of_points))
    #%% Start loop to propagate through the beamline
    #Use an extra loop here to decrease memoery requirements due to storing
    #initial positions and velocities which are generated randomly
    N_loops = 100 #Number of loops
    #Calculate the size of arrays needed for each iteration of the loop (i.e.
    #number of trajectories simulated each iteration)
    size = int(np.rint(N_molecules/N_loops))
    for n_loop in range(0,N_loops):
        
        #Generate initial positions
        theta = np.random.rand(1,size)*2*np.pi
        r = (np.sqrt(np.random.rand(1,size)) 
            * zone_of_freezing['d']/2)
        
        x_ini = np.zeros((size,3))
        #Store the initial positions in an array. Each new row of the array
        #corresponds to a new molecule. Starting z-position is set later.
        x_ini[:,0] = r*np.cos(theta)
        x_ini[:,1] = r*np.sin(theta)
    
        #Next generate initial velocities from normal distributions:
        v_ini = np.zeros((size,3))
        v_ini[:,0] = (beam['sigma_v_x']
                      *np.random.randn(size,)
                      + beam['v_t'])
        v_ini[:,1] = (beam['sigma_v_y']
                      *np.random.randn(size,)
                      + beam['v_t'])
        v_ini[:,2] = (beam['sigma_v_z']
                      *np.random.randn(size,)
                      + beam['v_z'])
        
        #%%Now start loop over molecules to get the trajectory for each
        for i in range(0,size):
            
            #Then initialize
            x = np.zeros((3,number_of_points))
            v = np.zeros((3,number_of_points))
            
            #Set up initial position and velocity
            #Get initial position from array generated earlier
            x[:,0] = x_ini[i,:]
            #Set z position to z coordinate of 'zone of freezing'
            x[2,0] = zone_of_freezing['z']
            #Get velocity from array generated earlier
            v[:,0] = v_ini[i,:]
                        
            #Molecule starts 'alive' so set that to true
            alive = True
            #Also initialize a counter to keep track of indexing in arrays
            n = 0
            
            ### Now start propagating through the beamline ###
            #Start loop over beamline elements
            for j in beamline_geometry:
                #Get beamline element from beamline geometry storage
                beamline_element = beamline_geometry[j]
                if beamline_element['name'] == 'lens':
                    #If element is lens, use propagate function for lens
                    x, v, alive, n = (propagate_through_lens(x, v, a_values, r_values, beamline_element, molecule, n, alive, 'circle', g))
                else:
                    #If it's not the lens, use the regular function
                    x, v, alive, n = (propagate_through(
                                              x, v, beamline_element,
                                              n, alive,
                                              beamline_element['type'], g))
                #Test if molecule is still alive after passing through the
                #current beamline element
                if not alive:
                    #Increase the relevant counts
                    name = beamline_element['name']
                    counts[name] = counts[name] + 1
                    counts['dead'] = counts['dead'] + 1
                    
                    #Check if the molecule hit one of the specified elements
                    #and save trajectory if it did
                    if name == 'lens' or name == 'field_plate' or name =='Fluorescence_region':
                        trajectories[name].append(x[:,0:n+1])
                    #If molecule is dead, break out of for loop through
                    #beamline geometry and start with next molecule
                    break
            
            #If molecule made it through the beamline element, store its
            #trajectory
            if alive:
                counts['detected'] = counts['detected'] + 1
                
                if store_trajectories:
                    trajectories['detected'].append(x[:,0:n+1])
                    velocities['detected'].append(v[:,0:n+1])
                                      
                    
                
    return trajectories, velocities, counts
#%% 
""" This function calculates the Stark shifts in the TlF electronic gorundstate
    for given electric field values. Based on MATLAB code by Adam West""" 
def stark_potentials(E_values,molecule_J):
    Evals = E_values                       # E-field (V/m)
    D=4.23                                 # Molecule frame dipole moment (Debye)
    D=D*3.336e-30                          # Molecule frame dipole moment (C.m)
    Br=0.22315                             # Rotational constant (cm^-1)
    Br=Br*100*3e8*6.63e-34                 # Rotational constant (J)
    kb=1.38e-23                            # Boltzmann's constant (J/K)
    Jmax=10                                # Maximum J value to consider
    msize=(Jmax +1)**2                       # Size of the resulting Hamiltonian matrix (msize x msize)
    mass=224*1.67e-27                      # Mass (kg)
    
    Js = np.array([]) #Define list of J values
    Ms = np.array([]) #Define list of M values
    
    for J in range(0,Jmax+1):
        for m in range(-J,J+1):
            Js = np.append(Js,J)
            Ms = np.append(Ms,m)
        
    #Preallocate array for the energy spectrum
    V = np.zeros((np.size(Evals),msize))
    
    #Preallocate Hamiltonian
    H = np.zeros((msize,msize))
    H_rot = np.zeros((msize,msize))
    H_stark = np.zeros((msize,msize))
    
    
    for i in range(0,msize):
        for j in range(0, msize):
            #Retrieve values for quantum numbers
            J1 = Js[i]
            M1 = Ms[i]
            J2 = Js[j]
            M2 = Ms[j]
            
            #Rotational part of Hamiltonian
            if J1 == J2 and M1 == M2:
                H_rot[i,j] = Br*J1*(J1+1)
                H_stark[i,j] = 0
            
            #Stark part (multiply by value of electric field later)
            elif J1 + 1 == J2 and M1 == M2:
                H_stark[i,j] = - math.sqrt(((J1-M1+1)*(J1+M1+1))
                            /((2*J1+1)*(2*J1+3))) * D
                H_rot[i,j] = 0
                
            elif J1 - 1 == J2 and M1 == M2:
                H_stark[i,j] = -math.sqrt(((J1 - M1)*(J1 + M1))
                            /((2*J1-1)*(2*J1+1))) * D
                H_rot[i,j] = 0
            else:
                H_rot[i,j] = 0
                H_stark[i,j] = 0
    
    #Set value of E-field
    n = 0                
    for E in Evals:
        #Generate full Hamiltonian from rotational and stark Hamiltonian
        H = H_rot + E*H_stark
        #Get eigenvalues, sort them and store in V matrix
        energies = np.sort(np.linalg.eigvalsh(H))
        V[n,:] = energies
        n = n + 1
        
    V_array = np.array
        
    if molecule_J == 2:
        V_array = V[:,8] - V[0,8]
            
    elif molecule_J == 3:
        V_array = V[:,15] - V[0,15]
    
    return V_array
    
    
#%% Function to build a table for an interpolating function for electrostatic 
#lens by calculating the Stark shift at various electric field values and
#then taking gradient

def acceleration_table(lens_V,lens_R,dr,J,m):
    r_values = np.array
    a_values = np.array
    
    #Make an array of radius values for which to calculate the Stark shift
    r_values = np.linspace(0,lens_R,int(np.round(lens_R/dr)))
    #Convert radius values into electric field values (assuming E = 2*V/R^2 * r within the lens radius)
    E_values = 2*lens_V/(lens_R**2) * r_values
    #Find the Stark shifts for the given electric field values by using a different function
    V_stark = stark_potentials(E_values,J)
    #Calculate radial acceleration at each radius based on dV_stark/dr
    a_values = -np.gradient(V_stark,dr)/m
    
    return a_values, r_values



#%%
    """ 
    propagate_through is a function which takes a molecule and propagates 
    it through a given beamline element (e.g. aperture or field plates)
    """
    
def propagate_through(x, v, beamline_element, n, alive,
                      option_1, g):
    
    #First check if molecule is alive or dead:
    if not alive:
        #If molecule is dead, keep position and velocity constant
        #Position first
        x[:,n+1] = x[:,n]
        x[:,n+2] = x[:,n]
        #Then veclocity
        v[:,n+1] = v[:,n]
        v[:,n+2] = v[:,n]
        
        #Also update index (up by two)
        n = n + 2
        
    #If molecule is alive, propagate it through the beamline element
    else:
        #First move to start of element
        #Calculate the change in z from input position to start of element
        delta_z = beamline_element['z_1'] - x[2,n]
        
        #Calculate the time taken to reach start of element from input position
        t_1 = delta_z/v[2,n]
        
        #Increase index by 1 before updating positions and velocities
        n = n + 1
        
        #Calculate position at start of element
        x[0,n] = x[0,n-1] + v[0,n-1]*t_1
        x[1,n] = x[1,n-1] + v[1,n-1]*t_1 - (g*t_1**2) / 2
        x[2,n] = x[2,n-1] + delta_z
        
        #Then calculate velocities
        v[0,n] = v[0,n-1]
        v[1,n] = v[1,n-1] - g * t_1
        v[2,n] = v[2,n-1]
        
        #Cut the molecules outside the allowed region. First check the type of
        #cross section the beamline element has
        if option_1 == 'circle':
            #Calculate the radial position
            rho = np.sqrt(x[0,n]**2 + x[1,n]**2)
            #Test if molecule is inside the allowed region
            alive = alive and rho < beamline_element['d_1']/2
        
        #Next case is in case the cross section is rectangular
        elif option_1 == 'square':
            alive = (alive and x[0,n] > beamline_element['x_1'] and
                     x[0,n] < beamline_element['x_2'] and 
                     x[1,n] > beamline_element['y_1'] and
                     x[1,n] < beamline_element['y_2'])
            
        #If element is the field plate, there's a special case
        elif option_1 == 'fplate':
            alive = (alive and x[0,n] >  beamline_element['x_1'] and
                     x[0,n] < beamline_element['x_2'])
        else:
            #If none of the options is valid print an error message
            print("Error: invalid option in propagate through")
            
        #Next propagate through the element itself. Only do this if element
        #has non-negligible thickness
        if beamline_element['z_2'] - beamline_element['z_1'] > 0 and alive:
            
            #Calculate change in z-postion within the element
            delta_z = beamline_element['z_2'] - beamline_element['z_1']
            
            #Calculate time taken to propagate through the element
            t_2 = delta_z/v[2,n]
            
            #Increase index by one before calculating positions
            n = n + 1
            
            #Calculate position at end of element
            x[0,n] = x[0,n-1] + v[0,n-1]*t_2
            x[1,n] = x[1,n-1] + v[1,n-1]*t_2 - g*t_2**2 / 2
            x[2,n] = x[2,n-1] + delta_z
            
            #Then calculate velocities
            v[0,n] = v[0,n-1]
            v[1,n] = v[1,n-1] - g * t_2
            v[2,n] = v[2,n-1]
            
            #Cut the molecules that are outside the allowed region
            if option_1 == 'circle':
                
                #Calculate radial position of molecule
                rho = np.sqrt(x[0,n]**2 + x[1,n]**2)
                
                #Molecules stay alive if they are within the allowed region
                #determined by the radius of element
                alive = alive and rho < beamline_element['d_2']/2
                
                #If molecule hits the element, calculate where
                if not alive:
                    #code in this if statement is not complete since I won't
                    #need to know where the molecules hit the round apertures.
                    #See the MATLAB version if you wan't to complete this part
                    pass
                    
#                    #Calculate how the radius of the element changes with z
#                    #(for cones)
#                    if beamline_element['d_1'] - beamline_element['d_2'] > 0:
#                        diff_rho = ((beamline_element['d_2'] - 
#                                    beamline_element['d_1'])
#                                    /(2*(beamline_element['z_2'] 
#                                    - beamline_element['z_1'])))
#                    else:
#                        diff_rho = 0
#                    
#                    #Now some messy maths to calculate where the molecule hits
#                    #the beamline element (sorry for not documenting the
#                    #calculation)
#                    t_vec = np.zeros((1,2))
#                    t_vec[1] = (diff_rho*v[2,n]*beamline_element['d_1'] - 2*v[0,n]*x[0,n] - 2*v[1,n]*x[1,n] 
#                        + sqrt((diff_rho*v[2,n]*beamline_element['d_1'] - 2*v[1,n]*x[1,n] - 2*v[2,n]*x[2,n])^2 
#                        -(4*rho^2-beamline_element.d_1^2)*(v(1,n)^2 + v(2,n)^2 - diff_rho^2*v(3,n)^2))) ...
#                        /(v(1,n)^2 + v(2,n)^2 - diff_rho^2*v(3,n)^2);
            #Test if molecule is alive if cross section is square        
            elif option_1 == 'square':
                
                #Molecule stays alive if it is within the square cross section
                alive = (alive and x[0,n] > beamline_element['x_1'] and  
                         x[0,n] < beamline_element['x_2'] and
                         x[1,n] > beamline_element['y_1'] and
                         x[1,n] < beamline_element['y_2']) 
                if not alive:
                    #code in this if statement is not complete since I won't
                    #need to know where the molecules hit the round apertures.
                    #See the MATLAB version if you wan't to complete this part
                    pass 
                 
            #Finally we have a special case for the field plates
            elif option_1 == 'fplate':
                #Alive if the molecule is still between the plates
                alive = (alive and x[0,n] > beamline_element['x_1'] and 
                         x[0,n] < beamline_element['x_2'])
                
                #If molecule hits field plates, calculate where
                if not alive:
                    #Calculate time taken to hit field plate if
                    #molecule is moving in -ve x-direction
                    if v[0,n] < 0:
                        delta_t = ((beamline_element['x_1']-x[0,n-1])
                                    /v[0,n-1])
                    #Calculate time taken to hit field plate if
                    #molecule is moving in +ve x-direction
                    elif v[0,n] > 0:
                        delta_t = ((beamline_element['x_2']-x[0,n-1])
                                    /v[0,n-1])
                    
                    
                    #Then calculate the position where the particle ends up
                    x[0,n] = x[0,n-1] + v[0,n-1] * delta_t
                    x[1,n] = x[1,n-1] + v[1,n-1] * delta_t - (g*delta_t**2)/2
                    x[2,n] = x[2,n-1] +  delta_t*v[2,n-1]
            else:
                #If none of the options is valid print out an error message
                print("Error: invalid option in propagate through")
                
    #Return the position and velocity arrays
    return x, v, alive, n
                    


#%% propagate_through_lens is a function that propagates the input molecule
#   through the lens

def propagate_through_lens(x,v,a_values,r_values,lens,molecule,n,alive,option_1,g):
    #First check that molecule is alive
    if not alive:
        #If molecule is dead, keep position and velocity constant
        #Position first
        x[:,n+1] = x[:,n]
        x[:,n+2] = x[:,n]
        #Then veclocity
        v[:,n+1] = v[:,n]
        v[:,n+2] = v[:,n]
        
        #Also update index (up by two)
        n = n + 2
            
    #Next check that lens has finite length
    if lens['z_2'] - lens['z_1'] <= 0:

        #If lens has 0 length, keep position and velocity constant
        #Position first
        x[:,n+1] = x[:,n]
        x[:,n+2] = x[:,n]
        #Then veclocity
        v[:,n+1] = v[:,n]
        v[:,n+2] = v[:,n]
        
        #Also update index (up by two)
        n = n + 2
        
    else:
       #%%Now move to start of the lens
        #First move to start of lens
        #Calculate the change in z from input position to start of element
        delta_z = lens['z_1'] - x[2,n]

        #Calculate the time taken to reach start of element from input position
        t_1 = delta_z/v[2,n]
        
        #Increase index by 1 before updating positions and velocities
        n = n + 1
        
        #Calculate position at start of element
        x[0,n] = x[0,n-1] + v[0,n-1]*t_1
        x[1,n] = x[1,n-1] + v[1,n-1]*t_1 - (g*t_1**2) / 2
        x[2,n] = x[2,n-1] + delta_z
        
        #Then calculate velocities
        v[0,n] = v[0,n-1]
        v[1,n] = v[1,n-1] - g * t_1
        v[2,n] = v[2,n-1]
        
        #Cut the molecules outside the allowed region. First check the type of
        #cross section the lens has
        if option_1 == 'circle':
            #Calculate the radial position
            rho = np.sqrt(x[0,n]**2 + x[1,n]**2)
            #Test if molecule is inside the allowed region
            alive = alive and rho < lens['d_1']/2
        
        #Next case is in case the cross section is rectangular
        elif option_1 == 'square':
            alive = (alive and x[0,n] > lens['x_1'] and
                     x[0,n] < lens['x_2'] and 
                     x[1,n] > lens['y_1'] and
                     x[1,n] < lens['y_2'])
        
        #%%Next propagate through the lens itself by using RK4 integration
        #For RK4 have:
        # x(t+dt) = x(t) + dt/6*(k_1+2*k_2+2k_3+k_4)
        # v(t+dt) = v(t) + dt/6*(l_1+2*l_2+2l_3+l_4)
        #Only do this part if lens has finite thickness and molecule didn't hit
        #lens earlier
        if lens['L'] > 0 and alive:
            N_steps = int(np.rint((lens['L']/lens['dz'])))
            dt = lens['dz']/v[2,n]
            #Start loop over steps within lens
            for i in range(n, n + N_steps):
                #Only keep doing this while molecule is alive
                if alive:
                    #Calculate parameters for RK4
                    k_1 = v[:,n]
                    l_1 = acceleration_interpolation(x[:,n],a_values,
                                                     r_values,g)  
                    k_2 = v[:,n]+dt*l_1/2
                    l_2 = acceleration_interpolation(x[:,n]+dt*k_1/2,a_values,
                                                     r_values,g)
                    k_3 = v[:,n]+dt*l_2/2
                    l_3 = acceleration_interpolation(x[:,n]+dt*k_2/2,a_values,
                                                     r_values,g)
                    k_4 = v[:,n]+dt*l_3
                    l_4 = acceleration_interpolation(x[:,n]+dt*k_3,a_values,
                                                     r_values,g)
                    n = i
                    #Calculate x(t+dt) and v(t+dt)
                    x[:,n+1] = x[:,n] + dt*(k_1+2*k_2+2*k_3+k_4)/6
                    v[:,n+1] = v[:,n] + dt*(l_1+2*l_2+2*l_3+l_4)/6
                    
                    #Cut molecules outside the allowed region
                    rho = np.sqrt(x[0,n]**2 + x[1,n]**2)
                    alive = rho < lens['d_2']/2 and alive
                    
                else:
                    #If molecule is dead keep position and velocity constant
                    x[:,n+1] = x[:,n]
                    v[:,n+1] = v[:,n]
    #Return position, velocity and n-index        
    return x, v, alive, n


#%% acceleration_interpolation calculates acceleration within the lens based
#on a given table of acceleration values at given radial position
def acceleration_interpolation(x,a_values,r_values,g):
    
    #print("size of x: ", np.size(x), np.shape(x))
    #Initialize acceleration array
    a = np.zeros((3,))
    
    #Calculate radial position of molecule
    rho = np.sqrt(x[0]**2 + x[1]**2)
    #print(np.size(rho))
    
    #Calculate radial acceleration based on interpolation table
    a_r = interpolate_custom(r_values,a_values,rho)
    
    if rho != 0:
        #Resolve acceleration into components
        a[0] = a_r*x[0]/rho
        a[1] = a_r*x[1]/rho - g
        a[2] = 0
    else:
        a[:] = 0
        
    return a


#%% interpolate_custom is used to get acceleration values by interpolating
def interpolate_custom(r_values,a_values,r):
    
    #Find the index to which the current radial position of the molecule
    #corresponds
    
    r_max = r_values[-1]
    index = int(r/r_max * np.size(r_values))
    
    #If the index exceeds the size of the interpolation tabels, the particle
    #must be very close to the edge of the lens so set the  index so that the
    #acceleration outputted is the acceleration at r = r_max
    if index > np.size(r_values)-1:
        index = np.size(r_values)-1
        
    #Calculate acceleration by interpolating linearly
    if index < np.size(r_values)-1:
        a_r = (a_values[index] + (a_values[index+1]-a_values[index])/
               (r_values[index+1] - r_values[index]) * (r - r_values[index]))
    elif index == np.size(r_values)-1:
        a_r = a_values[index]
    else:
        print("Error in interpolate_custom")
    
    return a_r
    
    