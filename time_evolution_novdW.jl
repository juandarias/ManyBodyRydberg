"""
# TODO
# -----------------------------------------------------------
# -update definition of Jump operators
# -add arguments to Hamiltonian from topology properties
# -add function for space propagation
# -add some mean field hamiltonian to facilitation to take into account interaction with other Rydberg atoms in gas
"""
module propagation


### Julia modules
using Distributed
using QuantumOptics
using Random; Random.seed!(0)
using LinearAlgebra
using JLD2, FileIO
using SparseArrays

### Own modules
include("./topology.jl")
include("./basis_operators.jl")
include("./methods.jl")
include("./hamiltonian.jl")

#using .topology,.hamiltonian,.basis_operators,.methods_cloud


function JumpOperators(topology, proj_ops)
    Jmp, Γ = [], []
    laserRyd = topology.laserTypes["laserRyd"]
    n_atoms = length(proj_ops["Excitation_Operators"])
    
    ### Decay rates
    Γ_weak = 2 * π * 50*10^3 # See 10.1103/PhysRevA.89.033421
    Γeff = 40*10^3 #Spontaneous decay + BBR decay for Li 24S. Obtained from ARC
    Γ_1 = Γeff

    ### Dephasing rates
    γint = 9e4 #Scattering from intermediate state
    γFT = 1/laserRyd.pulse_length #Fourier broadening due to laser pulse length
    γ_1 = laserRyd.bandwidth + γFT + γint #Dephasing: convolution of two Lorentzians, whose width is equal to the sum of individual widths
             
    
    for i in 1:n_atoms
        push!(Jmp, proj_ops["Decay_Operators"][i])
        append!(Γ, +(Γ_1)^0.5)
    end

    for i in 1:n_atoms
        push!(Jmp, proj_ops["Projectors"][i])
        append!(Γ, +(γ_1/2)^0.5)
    end
    Γ = convert(Array{Float64,1}, Γ)
    return Jmp, Γ
end

function MCWFM(atoms_positions, topology, trajectories, save_location)
    #positions, states = Array{Array{Float64,1},1}(), Array{Array{Float64,1},1}()
    #for atom in atoms
    #    push!(positions, atom.position)
    #    push!(states, atom.state)
    #end
    
    ### Read topology
    laserRyd = topology.laserTypes["laserRyd"]
    atomRyd = topology.atomTypes["atomRyd"]
        
    ### Prepare basis and operators
    b_mb, posopx, posopy, posopz = basis_operators.ManyBodyBasis(atoms_positions, save_location);
    basis_operators.PositionOperators(b_mb, posopx, posopy, posopz, save_location);
    basis_operators.ProjectionOperators(b_mb, save_location);
    
    sleep(5)
    pos_ops = jldopen(save_location*"pos_ops.jld2", "r", mmaparrays=true)
    proj_ops = jldopen(save_location*"proj_ops.jld2", "r", mmaparrays=true)
    n_atoms = length(proj_ops["Excitation_Operators"])


    ### Prepare Hamiltonian
    
    hamiltonian.Hatom(laserRyd.Δ , proj_ops, save_location);
    hamiltonian.Hrabi(laserRyd.Ω, proj_ops, save_location);
    hamiltonian.Hion(atomRyd.C4, pos_ops, proj_ops, save_location);
    #hamiltonian.HvdW(atomRyd.C6, n_atoms, save_location);


    sleep(5)
    h_total_array = hamiltonian.HtotalnovdW(save_location);
    H_total = SparseOperator(b_mb,b_mb, h_total_array)

    ### Decay and decoherence operators
    Jmp, Γ = JumpOperators(topology, proj_ops) #Update function definition to use topology data

    ### Prepare MCWFM
    iters = 1e9
    tfinal, tstep = laserRyd.pulse_length, 1e-6
    T = [0:tstep:tfinal;]
    n_trajectories = trajectories

    ### Run MCWFM
    psi0 = basisstate(b_mb,1)
    psi_t = @distributed (+) for i in 1:n_trajectories
        timeevolution.mcwf(T, psi0, H_total, Jmp; rates=Γ, seed=UInt(i))[2]
    end

    @save save_location*"Psi_t.jld2" psi_t
    cp(save_location*"Psi_t.jld2", save_location*"Psi_t.out.jld2", force=true) # 
    #Solution for issue https://github.com/JuliaIO/JLD2.jl/issues/55    
end


end
