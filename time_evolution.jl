"""
# TODO
# -----------------------------------------------------------
# -add some mean field hamiltonian to facilitation to take into account interaction with other Rydberg atoms in gas
# -think of smart gridding of space
# -review gaussian wavepackage definition in Trotter propagation
# -define types for arguments of functions
# -distribute Newtonian propagation
"""
module time_evolution

export TDSE, MCWFM, NewtonPropagation, testfun

### Julia modules
using Distributed
using QuantumOptics
using JLD2, FileIO


#using .topology,.hamiltonian,.basis_operators,.methods_cloud

function testfun(number)
    println(number)
end

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

function MCWFM(H_total, tfinal, save_location::AbstractString; n_trajectories=1000)
    
    ### Decay and decoherence operators
    proj_ops = jldopen(save_location*"proj_ops.jld2", "r", mmaparrays=true)
    Jmp, Γ = JumpOperators(topology, proj_ops) #Update function definition to use topology data

    ### Prepare MCWFM
    tstep = tfinal/10
    T = [0:tstep:tfinal;]

    ### Run MCWFM
    b_mb = basis(H_total)
    psi0 = basisstate(b_mb,1)
    psi_t = @distributed (+) for i in 1:n_trajectories
        timeevolution.mcwf(T, psi0, H_total, Jmp; rates=Γ, seed=UInt(i))[2]
    end

    psi_t /= n_trajectories

    @save save_location*"Psi_t.jld2" psi_t
    cp(save_location*"Psi_t.jld2", save_location*"Psi_t.out.jld2", force=true) # 
    #Solution for issue https://github.com/JuliaIO/JLD2.jl/issues/55    
end



function MCWFM(H_total, tfinal, initial_state::Array{Complex{Float64},1}, save_location::AbstractString; n_trajectories=1000)
    
    ### Decay and decoherence operators
    proj_ops = jldopen(save_location*"proj_ops.jld2", "r", mmaparrays=true)
    Jmp, Γ = JumpOperators(topology, proj_ops) #Update function definition to use topology data

    ### Prepare MCWFM
    tstep = tfinal/10
    T = [0:tstep:tfinal;]

    ### Run MCWFM
    b_mb = basis(H_total)
    psi0 = Ket(b_mb, initial_state)
    psi_t = @distributed (+) for i in 1:n_trajectories
        timeevolution.mcwf(T, psi0, H_total, Jmp; rates=Γ, seed=UInt(i))[2]
    end

    psi_t /= n_trajectories

    @save save_location*"Psi_t.jld2" psi_t
    cp(save_location*"Psi_t.jld2", save_location*"Psi_t.out.jld2", force=true) # 
    #Solution for issue https://github.com/JuliaIO/JLD2.jl/issues/55    
end



function TDSE(H_total, tfinal::Float64, save_location::AbstractString, iters=1e6)
    

    ### Prepare TDSE
    #tfinal, tstep = laserRyd.pulse_length, laserRyd.pulse_length/10
    tstep = tfinal/10
    T = [0:tstep:tfinal;]
    #n_trajectories = trajectories

    ### Solve TDSE
    b_mb = basis(H_total)
    psi0 = basisstate(b_mb,1)
    psi_t = timeevolution.schroedinger(T, psi0, H_total, maxiters=iters)[2]

    @save save_location*"Psi_t.jld2" psi_t
    cp(save_location*"Psi_t.jld2", save_location*"Psi_t.out.jld2", force=true)
    sleep(5) # 
    #Solution for issue https://github.com/JuliaIO/JLD2.jl/issues/55    
end


function TDSE(H_total, tfinal::Float64, initial_state::Array{Complex{Float64},1}, save_location::AbstractString, iters=1e6)
    

    ### Prepare TDSE
    #tfinal, tstep = laserRyd.pulse_length, laserRyd.pulse_length/10
    tstep = tfinal/10
    T = [0:tstep:tfinal;]
    #n_trajectories = trajectories

    ### Solve TDSE
    b_mb = basis(H_total)
    psi0 = Ket(b_mb, initial_state)
    psi_t = timeevolution.schroedinger(T, psi0, H_total, maxiters=iters)[2]

    @save save_location*"Psi_t.jld2" psi_t
    cp(save_location*"Psi_t.jld2", save_location*"Psi_t.out.jld2", force=true)
    sleep(5) # 
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
    
    ### Trotter step
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


function NewtonPropagation(cloud, time_step)
    for i in 1:length(cloud)
        a_ti = cloud[i].gradient/mass
        cloud[i].velocity = cloud[i].velocity + a_ti*time_step
        cloud[i].position = cloud[i].position + cloud[i].velocityt*time_step
    end
end

function NewtonPropagationVariable(cloud, time_step)
    for i in 1:length(cloud)
        a_ti = cloud[i].gradient/mass #0
        cloud[i].velocity = cloud[i].velocity + a_ti*time_step/2 #1
        cloud[i].position = cloud[i].position + cloud[i].velocityt*time_step #2
        
        cloud[i].velocity = cloud[i].velocity + a_ti*time_step/2 #1
    end
end


end