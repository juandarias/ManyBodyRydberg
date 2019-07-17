"""
* TODO:
* - separate code for calculation of operators from time evolution
* - clean-up kwargs. See https://medium.com/@Jernfrost/function-arguments-in-julia-and-python-9865fb88c697
* - move observables to another module
* --------------------------------------------------------------
* NOTES:
* - The variable step Leap-Frog algorithm is based on 'kick-drift-kick' variant form. https://en.wikipedia.org/wiki/Leapfrog_integration and https://arxiv.org/pdf/astro-ph/9710043.pdf or https://www.unige.ch/~hairer/poly_geoint/week2.pdf or http://www.mcmchandbook.net/HandbookChapter5.pdf
* - Resetting the time-step to the initial value will add some unnecessary calculations.
* - When doing the time propagation in the electronic space, the initial state of the propagation corresponds to the final state of the previous propagation
* - list of kwargs accepted by integrator: withvdW, withPT, mcwfm , iters, laseroff
"""


module algorithms

export LeapFrogIntegrator, LeapFrogIntegratorVariable, LeapFrogIntegratorDark

using JLD2, Distributed, Dates
using hamiltonian, methods_cloud, electrostatics, time_evolution


"=================================================================================="
# Leap frog algorithms
"=================================================================================="


    function LeapFrogIntegrator(cloud, topology, atomsGroups, tfinal, time_step, cutOff, refresh_rate, test_location, withvdW = true, withPT = true)
        global step = 0
        numberSteps = tfinal/time_step
        pairs_atoms = PairGenerator(cloud, cutOff)
        for step in 1:numberSteps
            ElectronicPropagation(topology, atomsGroups, time_step, step,test_location, withvdW, mcwfm, iters)
            SpacePropagation(cloud, pairs_atoms, topology, time_step, withPT)
            mod(step, refresh_rate) == 0 && return pairs_atoms = PairGenerator(cloud, cutOff)
            PostUpdate()
        end
    
    end

    
    function LeapFrogIntegratorVariable(cloud, topology, atomsGroups::Array{Any,1}, tfinal::Float64, time_step::Float64, cutOff::Float64, test_location::AbstractString; kwargs...)
        ### Initialization
        global step = 0
        global collisions = 0
        global time_simulation = 0.0

        refresh_rate = 10
        pairs_atoms = PairGenerator(cloud, cutOff)
        
        ### Step 0
        ElectronicPropagation(topology, atomsGroups, time_simulation, time_step, step, test_location; kwargs...) #I am introducing an error here by integrating for a time_step
        AccelerationAtoms(cloud, pairs_atoms, topology, time_simulation)
        status = open("simulation_status.out", "a+")
        println(status, "System time, Simulation time (mus), Collisions, Kinetic energy, Potential energy")
        close(status)

        ### Step n>1
        while time_simulation < tfinal
            step += 1
            @save test_location*"step"*string(step)*"_cloud.jld2" cloud
            LeapFrogStep(cloud, topology, atomsGroups, pairs_atoms, time_simulation, time_step, step, test_location; kwargs...)
            if mod(step, refresh_rate) == 0
                pairs_atoms = PairGenerator(cloud, cutOff)
            end
            collisions = CollisionCounter(cloud, time_simulation)
            CloudState(atomsGroups, cloud, step, test_location)
            energy = PotentialEnergy(cloud, topology, pairs_atoms, time_simulation; kwargs...)
            time_simulation += time_step
            ### Status output and conclusion of simulation
            status = open("simulation_status.out", "a+")
            println(status, Dates.Time(Dates.now()),",",time_simulation*1e6,",",collisions,",",energy[1],",",energy[2])
            close(status)
            collisions != 0 && break
        end
        return collisions, time_simulation
    end


    
    function LeapFrogIntegratorDark(cloud, topology, atomsGroups::Array{Any,1}, tfinal::Float64, time_step::Float64, cutOff::Float64, test_location::AbstractString; kwargs...)
        ### Initialization
        global step = 0
        global collisions = 0
        global time_simulation = 0.0

        refresh_rate = 10
        pairs_atoms = PairGenerator(cloud, cutOff)
        
        ### Step 0 ###
        ElectronicPropagation(topology, atomsGroups, time_simulation, time_step, step, test_location; kwargs...) #I am introducing an error here by integrating for a time_step
        AccelerationAtoms(cloud, pairs_atoms, topology, time_simulation)
        status = open("simulation_status.out", "a+")
        println(status, "System time, Simulation time (mus), Collisions, Kinetic energy, Potential energy")
        close(status)

        ### Step n>1 ###
        tpulse = topology.laserTypes["laserRyd"].pulse_length

        ### Evolution during coupling with laser field, calculated with TDSE. If decoherence/dephasing is important, use mcwfm = true
        while time_simulation < tpulse
            step += 1
            @save test_location*"step"*string(step)*"_cloud.jld2" cloud
            if mod(step, refresh_rate) == 0
                pairs_atoms = PairGenerator(cloud, cutOff)
            end
            LeapFrogStep(cloud, topology, atomsGroups, pairs_atoms, time_simulation, time_step, step, test_location; mcwfm = false, kwargs...)
            CloudState(atomsGroups, cloud, step, test_location)
            collisions = CollisionCounter(cloud, time_simulation)
            energy = PotentialEnergy(cloud, topology, pairs_atoms, time_simulation; kwargs...)
            time_simulation += time_step
            ### Status output and conclusion of simulation
            ExportStatus(time_simulation, collisions, energy)
            collisions != 0 && break
        end

        ### Dark evolution. No laser and time evolution calculated with MCWFM
        while time_simulation < tfinal
            step += 1
            @save test_location*"step"*string(step)*"_cloud.jld2" cloud
            if mod(step, refresh_rate) == 0
                pairs_atoms = PairGenerator(cloud, cutOff)
            end
            LeapFrogStep(cloud, topology, atomsGroups, pairs_atoms, time_simulation, time_step, step, test_location; mcwfm = true, laseroff = true, kwargs...)
            CloudState(atomsGroups, cloud, step, test_location)
            collisions = CollisionCounter(cloud, time_simulation)
            energy = PotentialEnergy(cloud, topology, pairs_atoms, time_simulation; kwargs...)
            time_simulation += time_step
            ### Status output and conclusion of simulation
            ExportStatus(time_simulation, collisions, energy)
            collisions != 0 && break
        end

        return collisions, time_simulation
    end



    function LeapFrogStep(cloud, topology, atomsGroups::Array{Any,1}, pairs_atoms, time_simulation::Float64, time_step::Float64, step::Int64, test_location::AbstractString; max_displacement = 100e-9, kwargs...)
        i = 1
        ### First and second substep of LF
        while i < length(cloud) #Loop through atoms
            new_velocity = cloud[i].velocity + cloud[i].acceleration*time_step/2 #1,v1/2
            new_position = cloud[i].position + cloud[i].velocity*time_step #2, x1
            displacement = sqrt(sum((new_position - cloud[i].position).*(new_position - cloud[i].position)));
            if displacement > max_displacement
                time_step /= 2; #halves time_step
                i = 1; #resets loops on atom, to recalculate all displacements
            else
                #updates velocities and positions to new values
                cloud[i].velocity = new_velocity
                cloud[i].position = new_position
                #time_step = copy(time_step_0) #resets time-step to original value
                i += 1; #continue to next atom
            end
        end
        
        ### Update accelerations
        ElectronicPropagation(topology, atomsGroups, time_simulation, time_step, step, test_location; kwargs...) #u1
        AccelerationAtoms(cloud, pairs_atoms, topology, time_simulation)
        
        ### Third substep of Leap-Frog algorithm
        for atom in cloud
            atom.velocity = atom.velocity + atom.acceleration*time_step/2 #3, v1
        end


    end


"========================================================"
# Algorithms for propagation in real and electronic space
"========================================================"
    #
    function ElectronicPropagation(topology, atomsGroups::Array{Any,1}, time_simulation::Float64, tpropagation::Float64, step::Int64, test_location::AbstractString; withvdW = true, withPT = true, mcwfm = false, iters = 1e6, laseroff = false)
        @sync @distributed for i in 1:length(atomsGroups)
            
            save_location = test_location*"step"*string(step)*"_g"*string(i)*"_"          
            H_total = Htotal(atomsGroups[i], topology, time_simulation, save_location, withvdW, withPT, laseroff)
            
            if step > 0
                previous_step_location = test_location*"step"*string(step-1)*"_g"*string(i)*"_"
                psi_f = last(jldopen(previous_step_location*"Psi_t.jld2", "r", mmaparrays=true)["psi_t"])
                initial_state = psi_f.data
                mcwfm == false && TDSE(H_total, tpropagation, initial_state, save_location; maxiters= iters, steps =2)
                mcwfm == true && MCWFM(H_total, topology, tpropagation, initial_state, save_location; maxiters= iters, steps =2)
            else
                mcwfm == false && TDSE(H_total, tpropagation, save_location; maxiters= iters, steps =2)
                mcwfm == true && MCWFM(H_total, topology, tpropagation, save_location; maxiters= iters, steps =2)
            end
        end
    end

    
    function SpacePropagation(cloud, pairs_atoms, topology, time_step, withPT = true) ###Obsolete
        ResetAcceleration(cloud)
        CalculateGradvdW(pairs_atoms, cloud, topology)
        if withPT == true
            CalculateGradIonPT(cloud, topology, step, time_step)
        else
            CalculateGradIon(cloud, topology)
        end
        NewtonPropagation(cloud, time_step)
    end
  

"========================================"
# Algorithms for observables of system
"========================================"
    
    function PotentialEnergy(cloud, topology, pairs, time_simulation; withvdW = true, withPT = true, mcwfm = false, iters = 1e6, laseroff = false)
        
        if withvdW == true
            @sync for pair in pairs
                @async CalculatevdW(pair, cloud, topology)
            end
        end
        
        @sync for atom in cloud
            withPT == false && @async CalculateIonRydberg(atom, topology)
            withPT == true && @async CalculatePTIon(atom, topology, time_simulation)
        end
        
        mass = topology.atomTypes["atomRyd"].mass;
        energy= zeros(2) 
        energy =  @distributed (+) for atom in cloud
            kin_energy = 1/2*mass*(atom.velocity[1]^2 + atom.velocity[2]^2 + atom.velocity[3]^2);
            [atom.stark_shift, kin_energy]
        end
        return energy
    end

    function CollisionCounter(cloud, time::Float64)
        Collisions = 0
        Collisions = @distributed (+) for atom in cloud
            distance = sqrt(sum(atom.position.*atom.position));
            distance < 200e-9 && 1
        end
        return Collisions
    end

    function ExportStatus(time_simulation, collisions, energy)
        status = open("simulation_status.out", "a+")
        println(status, Dates.Time(Dates.now()),",",time_simulation*1e6,",",collisions,",",energy[1],",",energy[2])
        close(status)
    end
    
end


