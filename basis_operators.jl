"""
# TODO
# -----------------------------------------------------------
# -update definition of Jump operators
# -write appropiate arguments for Jump function
"""
module basis_operators

using QuantumOptics
using JLD2, FileIO




function ManyBodyBasis(positions, save_location)
    Position_Operators_x, Position_Operators_y, Position_Operators_z = [], [], []
    Basis_Positions_x, Basis_Positions_y, Basis_Positions_z = [], [], []
    global b_mb = []
    n_atoms = length(positions)
    
    #Single particle operator and basis
    for i in 1:n_atoms
        bx = PositionBasis(positions[i][1], positions[i][1], 1)
        by = PositionBasis(positions[i][2], positions[i][2], 1)
        bz = PositionBasis(positions[i][3], positions[i][3], 1)
        #Ii = identityoperator(bx)
        #Basis_Positions["bxp$(i)"] = bx #the double quotation mark is important
        #Position_Operators["x$(i)"] = xi
        push!(Basis_Positions_x, bx)
        push!(Basis_Positions_y, by)
        push!(Basis_Positions_z, bz)
        #push!(Identities, Ii)
    end




    
    ### Many-body basis
    bs = SpinBasis(1//2)
    for i in 1:n_atoms
        global b_mb
        if i == 1
            global b_mb
            b_mb = Basis_Positions_x[i] ⊗ Basis_Positions_y[i] ⊗ Basis_Positions_z[i] ⊗ bs
        elseif i >1
            b_mb = b_mb ⊗ Basis_Positions_x[i] ⊗ Basis_Positions_y[i] ⊗ Basis_Positions_z[i] ⊗ bs
        end
    end
    @save save_location*"many_body_basis.jld2" b_mb
    cp(save_location*"many_body_basis.jld2", save_location*"many_body_basis.out.jld2", force=true)
    return b_mb, Basis_Positions_x, Basis_Positions_y, Basis_Positions_z
end

function PositionOperators(many_body_basis, Basis_Positions_x, Basis_Positions_y, Basis_Positions_z, save_location)
    Position_x, Position_y, Position_z = [], [], []
    n_atoms = count(i->(i==2),many_body_basis.shape)
    for i in 1:n_atoms
        xi = embed(many_body_basis, (4i-3), position(Basis_Positions_x[i]))
        yi = embed(many_body_basis, (4i-2), position(Basis_Positions_y[i]))
        zi = embed(many_body_basis, (4i-1), position(Basis_Positions_z[i]))
        push!(Position_x, xi)
        push!(Position_y, yi)
        push!(Position_z, zi)
    end
    @save save_location*"pos_ops.jld2" Position_x Position_y Position_z
    cp(save_location*"pos_ops.jld2", save_location*"pos_ops.out.jld2", force=true) #Solution for issue https://github.com/JuliaIO/JLD2.jl/issues/55
    #pos_ops = jldopen(folder*testname*"pos_ops.jld2", "r", mmaparrays=true) 
    #return pos_ops
end

function ProjectionOperators(many_body_basis, save_location)
    RXR, GXG, RXG, GXR = [], [], [], []
    bs = SpinBasis(1//2)
    sup = basisstate(bs,2)
    sdown = basisstate(bs,1)
    eXe = sparse(sup ⊗ dagger(sup))
    gXg = sparse(sdown ⊗ dagger(sdown))
    eXg = sparse(sup ⊗ dagger(sdown))
    gXe = sparse(sdown ⊗ dagger(sup))
    
    n_atoms = count(i->(i==2),many_body_basis.shape)
    for i in 1:n_atoms
        eiXei = embed(many_body_basis,(4i), eXe)
        giXgi = embed(many_body_basis,(4i), gXg)
        giXei = embed(many_body_basis, (4i), gXe)
        eiXgi = embed(many_body_basis, (4i), eXg)
        push!(RXR, sparse(eiXei))
        push!(GXG, sparse(giXgi))
        push!(RXG, sparse(eiXgi))
        push!(GXR, sparse(giXei))
    end
    @save save_location*"proj_ops.jld2" RXR GXG RXG GXR
    cp(save_location*"proj_ops.jld2", save_location*"proj_ops.out.jld2", force=true) #Solution for issue https://github.com/JuliaIO/JLD2.jl/issues/55
    #proj_ops = jldopen(folder*testname*"proj_ops.jld2", "r", mmaparrays=true) 
    #return proj_ops
end

end