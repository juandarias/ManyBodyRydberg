"""
# TODO
# -----------------------------------------------------------
# -add argument to function for savefolder location and update functions accordingly: Done
# -add mean field potential hamiltonian
# -import HvdW everywhere in main!
# -Hatom and Hrabi only need to be calculated once, not for every atom group. Rewrite code could save some simulation time.
# -calculate matrix elements only for atoms within a maximum distance
# -----------------------------------------------------------
# ISSUES
# -----------------------------------------------------------
# HvdW cannot be executed in external program. See https://stackoverflow.com/questions/53344955/julia-parallel-function-inside-module
# https://discourse.julialang.org/t/distributed-parallelism-within-packages-applications/14320/5
"""
module hamiltonian

export Htotal, Hdark

using Distributed
using QuantumOptics, LinearAlgebra
using JLD2, FileIO 
using SparseArrays

using basis_operators, constants


"""
Constants
"""
hbar = constants.hbar
hplanck = constants.hplanck
ε_0 = constants.ε_0
C_e = constants.e



### Laser detuning
function HatomOld(Δ::Float64, proj_ops, save_location::AbstractString)
    n_atoms = length(proj_ops["RXG"])
    dim_b = 2^n_atoms
    H_atom = spzeros(dim_b,dim_b)
    for i in 1:n_atoms
        h_ai = 0.5 * Δ * proj_ops["RXR"][i].data
        H_atom += h_ai #TODO: is global needed
    end
    save(save_location*"hatom.jld2", "H_atom", H_atom)
end
### Laser detuning

### Laser detuning
function Hatom(Δ::Float64, proj_ops, save_location::AbstractString)
    n_atoms = length(proj_ops["RXG"])
    dim_b = 2^n_atoms
    H_atom = spzeros(dim_b,dim_b)
    for i in 1:n_atoms
        h_ai = 0.5 * Δ * (proj_ops["RXR"][i].data - proj_ops["GXG"][i].data)
        H_atom += h_ai #TODO: is global needed
    end
    save(save_location*"hatom.jld2", "H_atom", H_atom)
end
### Laser detuning


### Population transfer
function Hrabi(Ω::Float64, proj_ops, save_location::AbstractString)
    n_atoms = length(proj_ops["RXG"])
    dim_b = 2^n_atoms
    H_rabi = spzeros(dim_b,dim_b)
    for i in 1:n_atoms
        h_r = Ω * (proj_ops["RXG"][i].data + proj_ops["GXR"][i].data)
        H_rabi += h_r
    end
    save(save_location*"hrabi.jld2", "H_rabi", H_rabi)
end
### Population transfer




### Ion-Rydberg
function Hion(C4::Float64, pos_ops, proj_ops, save_location::AbstractString)
    n_atoms = length(proj_ops["RXG"])
    dim_b = 2^n_atoms
    H_ion = spzeros(dim_b,dim_b)
    for i in 1:n_atoms
        #di = pos_ops["Position_x"][i]^2 + pos_ops["Position_y"][i]^2 + pos_ops["Position_z"][i]^2
        di_sq = (pos_ops["Position_x"][i].data).^2 + (pos_ops["Position_y"][i].data).^2 + (pos_ops["Position_z"][i].data).^2
        d4 = abs.(diag(di_sq)).^-2
        d4_matrix = sparse(diagm(0 => d4))
        #d4_operator = SparseOperator(b_mb, b_mb, d4_matrix)
        h_i = C4 * proj_ops["RXR"][i].data * d4_matrix
        H_ion += h_i
    end
    save(save_location*"hion.jld2", "H_ion", H_ion)
end
### Ion-Rydberg


### Paul trap + ion
function HPTIon(topology, pos_ops, proj_ops, time_simulation::Float64, save_location::AbstractString)
    α = topology.atomTypes["atomRyd"].α;
    YbTrap = topology.trapTypes["YbTrap"];
    a1, a2, a3, b1, b2, b3 = [-0.5, -0.5, 1, 1, -1, 0];
    Ωrf = YbTrap.Ωrf
    ωz = YbTrap.ωz
    mYb = YbTrap.mIon
    q = YbTrap.q
    ϕ = YbTrap.ϕ
    t = time_simulation
    C = -ωz^2 + q*Ωrf^2*cos(Ωrf*t+ϕ)
    D = -ωz^2 - q*Ωrf^2*cos(Ωrf*t+ϕ)
    E = 2*ωz^2
    
    n_atoms = length(proj_ops["RXG"])
    dim_b = 2^n_atoms
    H_PTIon = spzeros(dim_b,dim_b)
    for i in 1:n_atoms
        #di = pos_ops["Position_x"][i]^2 + pos_ops["Position_y"][i]^2 + pos_ops["Position_z"][i]^2
        d_sq = (pos_ops["Position_x"][i].data).^2 + (pos_ops["Position_y"][i].data).^2 + (pos_ops["Position_z"][i].data).^2
        H_ipt =  -(α/4)*mYb/(4*pi*ε_0)*d_sq^(-3/2)*(C*pos_ops["Position_x"][i].data^2 + D*pos_ops["Position_y"][i].data^2 + E*pos_ops["Position_z"][i].data^2)*proj_ops["RXR"][i].data;
        H_PTIon += H_ipt
    end
    save(save_location*"hption.jld2", "H_PTIon", H_PTIon)
end
### Paul trap +alphaon




### vdW Interaction
function h_vdW_ij(i::Int8, j::Int8, C6::Float64, save_location::AbstractString)
    pos_ops = jldopen(save_location*"pos_ops.jld2", "r", mmaparrays=true)
    proj_ops = jldopen(save_location*"proj_ops.jld2", "r", mmaparrays=true)
    

    dx_sq= (pos_ops["Position_x"][i].data - pos_ops["Position_x"][j].data).^2

    dy_sq= (pos_ops["Position_y"][i].data - pos_ops["Position_y"][j].data).^2
    dz_sq= (pos_ops["Position_z"][i].data - pos_ops["Position_z"][j].data).^2
    dij_sq = dx_sq + dy_sq + dz_sq
    #Short distance cut-off for vdW interaction
    if abs(dij_sq[1]) <= 50e-9^2
        return dij_sq=one(dij_sq)*100e-9^2
    end
    d6 = abs.(diag(dij_sq)).^-3
    #d6[d6.==Inf] .=1 #d6 must be a vector
    d6_matrix = spdiagm(0 => d6)
    h_vdw_ij = C6 * sparse(proj_ops["RXR"][j].data*proj_ops["RXR"][i].data*d6_matrix) #Conversion to sparse matrix to save bytes
    return h_vdw_ij
end

function HvdW(C6::Float64, n_atoms::Int64, save_location::AbstractString)
    iter_vdW = Tuple{Int8,Int8}[]
    #n_atoms = length(proj_ops["RXG"])
    for i in 1:n_atoms, j in i+1:n_atoms
        push!(iter_vdW, (i,j))
    end

    H_vdW = []
    H_vdW = @distributed (+) for (i,j) in iter_vdW
        hamiltonian.h_vdW_ij(i,j, C6, save_location)
    end
    save(save_location*"hvdW.jld2", "H_vdW", H_vdW)
end

### vdW Interaction


function Htotal(atoms_positions::Array{Any,1}, topology, time_simulation::Float64, save_location::AbstractString, withvdW = true, withPT = false)

    ### Read topology
    n_atoms = length(atoms_positions)
    laserRyd = topology.laserTypes["laserRyd"]
    atomRyd = topology.atomTypes["atomRyd"]
        
    ### Prepare basis and operators
    b_mb, posopx, posopy, posopz = basis_operators.ManyBodyBasis(atoms_positions, save_location);
    basis_operators.PositionOperators(b_mb, posopx, posopy, posopz, save_location);
    basis_operators.ProjectionOperators(b_mb, save_location);
    
    sleep(5)
    pos_ops = jldopen(save_location*"pos_ops.jld2", "r", mmaparrays=true)
    proj_ops = jldopen(save_location*"proj_ops.jld2", "r", mmaparrays=true)
    

    ### Prepare Hamiltonian
    hamiltonian.Hatom(laserRyd.Δ , proj_ops, save_location);
    hamiltonian.Hrabi(laserRyd.Ω, proj_ops, save_location);

    if withPT == true
        hamiltonian.HPTIon(topology, pos_ops, proj_ops, time_simulation, save_location);
    else
        hamiltonian.Hion(atomRyd.C4, pos_ops, proj_ops, save_location);
    end
    
    if withvdW == true 
        hamiltonian.HvdW(atomRyd.C6, n_atoms, save_location);
    end


    #Open JLD files with output of Hamiltonians
    fatom = jldopen(save_location*"hatom.jld2", mmaparrays=true)
    frabi = jldopen(save_location*"hrabi.jld2", mmaparrays=true)
    
    if withPT == true
        fption = jldopen(save_location*"hption.jld2", mmaparrays=true)
    else
        fion = jldopen(save_location*"hion.jld2", mmaparrays=true)
    end

    if withvdW == true
        fvdW = jldopen(save_location*"hvdW.jld2", mmaparrays=true)
    end
    
    
    #Generate empty sparse array for Hamiltonian, save to disk and empty allocation
    dim_b = length(b_mb)
    H_total_0 = spzeros(dim_b,dim_b)
    save(save_location*"htotal_0.jld2", "H_total", H_total_0)
    

    #Create Hamiltonian
    ftotal = jldopen(save_location*"htotal_0.jld2", mmaparrays=true)
    ftotal["H_total"] .+= fatom["H_atom"]
    ftotal["H_total"] .+= frabi["H_rabi"]
    
    if withPT == true
        ftotal["H_total"] .+= fption["H_PTIon"] 
    else
        ftotal["H_total"] .+= fion["H_ion"] 
    end


    if withvdW == true 
        ftotal["H_total"] .+= fvdW["H_vdW"]
    end
    
    H_total = SparseOperator(b_mb,b_mb, ftotal["H_total"])
    
    #Save Hamiltonian and close files
    save(save_location*"htotal.jld2", "H_total", ftotal["H_total"])
    
    if withPT == true
        close(fption)
    else
        close(fion)
    end
    
    if withvdW == true
        close(fvdW)
    end
    close(fatom), close(frabi), close(ftotal)
    return H_total
end


function Hdark(atoms_positions::Array{Any,1}, topology, time_simulation::Float64, save_location::AbstractString, withvdW = true, withPT = false)

    ### Read topology
    n_atoms = length(atoms_positions)
    laserRyd = topology.laserTypes["laserRyd"]
    atomRyd = topology.atomTypes["atomRyd"]
        
    ### Prepare basis and operators
    b_mb, posopx, posopy, posopz = basis_operators.ManyBodyBasis(atoms_positions, save_location);
    basis_operators.PositionOperators(b_mb, posopx, posopy, posopz, save_location);
    basis_operators.ProjectionOperators(b_mb, save_location);
    
    sleep(5)
    pos_ops = jldopen(save_location*"pos_ops.jld2", "r", mmaparrays=true)
    proj_ops = jldopen(save_location*"proj_ops.jld2", "r", mmaparrays=true)
    

    ### Prepare Hamiltonian
    if withPT == true
        hamiltonian.HPTIon(topology, pos_ops, proj_ops, time_simulation, save_location);
    else
        hamiltonian.Hion(atomRyd.C4, pos_ops, proj_ops, save_location);
    end
    
    if withvdW == true 
        hamiltonian.HvdW(atomRyd.C6, n_atoms, save_location);
    end


    #Open JLD files with output of Hamiltonians
    if withPT == true
        fption = jldopen(save_location*"hption.jld2", mmaparrays=true)
    else
        fion = jldopen(save_location*"hion.jld2", mmaparrays=true)
    end

    if withvdW == true
        fvdW = jldopen(save_location*"hvdW.jld2", mmaparrays=true)
    end
    
    
    #Generate empty sparse array for Hamiltonian, save to disk and empty allocation
    dim_b = length(b_mb)
    H_dark_0 = spzeros(dim_b,dim_b)
    save(save_location*"hdark_0.jld2", "H_dark", H_dark_0)
    

    #Create Hamiltonian
    fdark = jldopen(save_location*"hdark_0.jld2", mmaparrays=true)
    
    if withPT == true
        fdark["H_dark"] .+= fption["H_PTIon"] 
    else
        fdark["H_dark"] .+= fion["H_ion"] 
    end

    if withvdW == true 
        fdark["H_dark"] .+= fvdW["H_vdW"]
    end

    H_dark = SparseOperator(b_mb,b_mb, fdark["H_dark"])
    
    #Save Hamiltonian and close files
    save(save_location*"hdark.jld2", "H_dark", fdark["H_dark"])
    
    if withPT == true
        close(fption)
    else
        close(fion)
    end
    if withvdW == true
        close(fvdW)
    end
    close(fdark)
    return H_dark
end




### Full Hamiltonian
function HtotalOld(save_location)
    
    #Open JLD files
    fatom = jldopen(save_location*"hatom.jld2", mmaparrays=true)
    fion = jldopen(save_location*"hion.jld2", mmaparrays=true)
    frabi = jldopen(save_location*"hrabi.jld2", mmaparrays=true)
    fvdW = jldopen(save_location*"hvdW.jld2", mmaparrays=true)
    
    
    #Generate empty sparse array for Hamiltonian, save to disk and empty allocation
    dim_b = size(fatom["H_atom"])[1]
    H_total_0 = spzeros(dim_b,dim_b)
    save(save_location*"htotal_0.jld2", "H_total", H_total_0)
    H_total_0 = nothing


    #Create Hamiltonian
    ftotal = jldopen(save_location*"htotal_0.jld2", mmaparrays=true)
    ftotal["H_total"] .+= fatom["H_atom"]
    ftotal["H_total"] .+= fion["H_ion"]
    ftotal["H_total"] .+= frabi["H_rabi"]
    ftotal["H_total"] .+= fvdW["H_vdW"]
    #Save Hamiltonian
    save(save_location*"htotal.jld2", "H_total", ftotal["H_total"])
    close(fatom), close(frabi), close(fion), close(fvdW)
    return ftotal["H_total"]
end
### Full Hamiltonian


### Hamiltonian no vdW
function HtotalnovdWOld(save_location)
    
    #Open JLD files
    fatom = jldopen(save_location*"hatom.jld2", mmaparrays=true)
    fion = jldopen(save_location*"hion.jld2", mmaparrays=true)
    frabi = jldopen(save_location*"hrabi.jld2", mmaparrays=true)
    #fvdW = jldopen(save_location*"hvdW.jld2", mmaparrays=true)
    
    
    #Generate empty sparse array for Hamiltonian, save to disk and empty allocation
    dim_b = size(fatom["H_atom"])[1]
    H_total_0 = spzeros(dim_b,dim_b)
    save(save_location*"htotal_0.jld2", "H_total", H_total_0)
    H_total_0 = nothing


    #Create Hamiltomnian
    ftotal = jldopen(save_location*"htotal_0.jld2", mmaparrays=true)
    ftotal["H_total"] .+= fatom["H_atom"]
    ftotal["H_total"] .+= fion["H_ion"]
    ftotal["H_total"] .+= frabi["H_rabi"]
    #ftotal["H_total"] .+= fvdW["H_vdW"]
    #Save Hamiltonian
    save(save_location*"htotal.jld2", "H_total", ftotal["H_total"])
    close(fatom), close(frabi), close(fion)# , close(fvdW)
    return ftotal["H_total"]
end
### Hamiltonian no vdW

end