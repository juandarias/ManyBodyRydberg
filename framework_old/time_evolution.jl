"""
# TODO
# -----------------------------------------------------------
# -add some mean field hamiltonian to facilitation to take into account interaction with other Rydberg atoms in gas
# -think of smart gridding of space
# -review gaussian wavepackage definition in Trotter propagation
# -define types for arguments of functions
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
    n_atoms = length(proj_ops["RXG"])
    
    ### Decay rates
    Γ_weak = 2 * π * 50*10^3 # See 10.1103/PhysRevA.89.033421
    Γeff = 40*10^3 #Spontaneous decay + BBR decay for Li 24S. Obtained from ARC
    Γ_1 = Γeff

    ### Dephasing rates
    γint = 9e4 #Scattering from intermediate state
    γFT = 1/laserRyd.pulse_length #Fourier broadening due to laser pulse length
    γ_1 = laserRyd.bandwidth + γFT + γint #Dephasing: convolution of two Lorentzians, whose width is equal to the sum of individual widths
             
    
    for i in 1:n_atoms
        push!(Jmp, proj_ops["GXR"][i])
        append!(Γ, +(Γ_1)^0.5)
    end

    for i in 1:n_atoms
        push!(Jmp, proj_ops["RXR"][i])
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
    n_atoms = length(proj_ops["RXG"])


    ### Prepare Hamiltonian
    
    hamiltonian.Hatom(laserRyd.Δ , proj_ops, save_location);
    hamiltonian.Hrabi(laserRyd.Ω, proj_ops, save_location);
    hamiltonian.Hion(atomRyd.C4, pos_ops, proj_ops, save_location);
    hamiltonian.HvdW(atomRyd.C6, n_atoms, save_location);


    sleep(5)
    h_total_array = hamiltonian.Htotal(save_location);
    H_total = SparseOperator(b_mb,b_mb, h_total_array)

    ### Decay and decoherence operators
    Jmp, Γ = JumpOperators(topology, proj_ops) #Update function definition to use topology data

    ### Prepare MCWFM
    iters = 1e9
    tfinal, tstep = laserRyd.pulse_length, laserRyd.pulse_length/10
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


function TDSE(atoms_positions, topology, save_location)
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
    n_atoms = length(proj_ops["RXG"])


    ### Prepare Hamiltonian
    
    hamiltonian.Hatom(laserRyd.Δ , proj_ops, save_location);
    hamiltonian.Hrabi(laserRyd.Ω, proj_ops, save_location);
    hamiltonian.Hion(atomRyd.C4, pos_ops, proj_ops, save_location);
    hamiltonian.HvdW(atomRyd.C6, n_atoms, save_location);


    sleep(5)
    h_total_array = hamiltonian.Htotal(save_location);
    H_total = SparseOperator(b_mb,b_mb, h_total_array)

    ### Decay and decoherence operators
    Jmp, Γ = JumpOperators(topology, proj_ops) #Update function definition to use topology data

    ### Prepare TDSE
    iters = 1e6
    tfinal, tstep = laserRyd.pulse_length, laserRyd.pulse_length/10
    T = [0:tstep:tfinal;]
    #n_trajectories = trajectories

    ### Solve TDSE
    psi0 = basisstate(b_mb,1)
    psi_t = timeevolution.schroedinger(T, psi0, H_total, maxiters=iters)[2]

    @save save_location*"Psi_t.jld2" psi_t
    cp(save_location*"Psi_t.jld2", save_location*"Psi_t.out.jld2", force=true) # 
    #Solution for issue https://github.com/JuliaIO/JLD2.jl/issues/55    
end


function TrotterSpace(atom_position, cloud_size, resolution, time_step)

    ### Constants
    ħ = 1; a = 1; m = 1;
    
    ### Limits of real space
    atom_position[1] > 0 ? xmax = cloud_size : xmax = -cloud_size
    atom_position[2] > 0 ? ymax = cloud_size : ymax = -cloud_size
    atom_position[3] > 0 ? zmax = cloud_size : zmax = -cloud_size

    ### Position and momentum basis
    Npoints = cloud_size/resolution;
    b_position_x = PositionBasis(xmin, xmax, Npoints); b_momentum_x = MomentumBasis(b_position_x);
    b_position_y = PositionBasis(ymin, ymax, Npoints); b_momentum_y = MomentumBasis(b_position_y);
    b_position_z = PositionBasis(zmin, zmax, Npoints); b_momentum_z = MomentumBasis(b_position_z);
    b_comp_xyz = b_position_x ⊗ b_position_y ⊗ b_position_z;
    b_comp_kxkykz = b_momentum_x ⊗ b_momentum_y ⊗ b_momentum_z;


    ### Operators for base transformation
    Txp = transform(b_comp_xyz, b_comp_kxkykz); #This transforms are done with a DFT
    Tpx = transform(b_comp_kxkykz, b_comp_xyz);

    ### Kinetic and potential operators
    nmax = Npoints;

    Tn = [(π^2*ħ^2)/(2*m*a)*(nx^2 + ny^2 + nz^2) for nx in 1:nmax, ny in 1:nmax, nz in 1:nmax];
    Tn_sparse = sparse(diagm(0 => vec(Tn)));
    Tn_op = SparseOperator(b_comp_kxkykz, b_comp_kxkykz, Tn_sparse);

    UIon(x,y,z) = C4*(x^2 + y^2 + z^2)^(-2);
    Ur = [UIon(x,y,z) for x in 1:nmax, y in 1:nmax, z in 1:nmax];
    Ur_sparse = sparse(diagm(0 => vec(Ur)));
    Ur_op = SparseOperator(b_comp_xyz, b_comp_xyz, Ur_sparse);


    ### Propagators and Trotter step
    M=1000; Δt = time_step;
    λ = -sqrt(im)*Δt/(M*ħ);
    #propT = dense(Txp)*exp.(λ*Tn_op)*dense(Tpx);
    propT = LazyProduct(Txp,exp.(λ*Tn_op),Tpx)
    propU = exp.(λ/2*Ur_op)
    
    # Trotter step
    stepTrotter = LazyProduct(propU,propT,propU)

    ### Atomic wavefunction

    ψx = gaussianstate(b_position_x, atom_position[1], p0_x, σ)
    ψy = gaussianstate(b_position_y, atom_position[2], p0_y, σ)
    ψz = gaussianstate(b_position_z, atom_position[3], p0_z, σ)
    ψ = ψx ⊗ ψy ⊗ ψz

    ### Propagate wavefunction
    ψt = stepTrotter* ψ
    return ψt
end


end