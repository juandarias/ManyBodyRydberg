"""
# TODO
# -----------------------------------------------------------
# -test cloud generation: Done
"""
module topology



mutable struct Atom
    name
    index::Int64
    state::Array{Float64,1}
    position::Array{Float64,1} #the dimension of the position basis
    stark_shift::Float64
    function Atom(name, index::Int64, state::Array{Float64,1}, position::Array{Float64,1}, shift::Float64)
        new(name, index, state, position, shift)
    end
end

struct AtomType
    α::Float64
    C6::Float64
    C4::Float64
    τ::Float64
    charge::Int8
    Rydberg_state
end


struct LaserType
    Δ::Float64
    ω::Float64
    Ω::Float64
    bandwidth::Float64;
    field_itensity::Float64;
    pulse_length::Float64;
    rep_rate::Float64;
end



struct AtomCloud
    density::Float64
    size::Float64
    temperature::Float64
    cloud::Array{Atom}
    function AtomCloud(size::Float64, density::Float64)
        size = size*1e-6
        L = density^(-1/3) #Spacing between atoms
        nAtoms = trunc(Int, size^3 * density)
        nAtomscubic = trunc(Int, nAtoms^(1/3))       
        ### Generate coordinates of atoms
        atomPos = [[x*L + L*rand(1)[1], y*L + L*rand(1)[1], z*L + L*rand(1)[1]] for x in -floor(nAtomscubic/2):floor(nAtomscubic/2), y in -floor(nAtomscubic/2):floor(nAtomscubic/2), z in -floor(nAtomscubic/2):floor(nAtomscubic/2)]
        ### Generating cloud of ground state atoms
        ground_state = [0.0, 1.0]
        name = "Li" 
        cloud = [Atom(name, i, ground_state, atomPos[i], 0.0) for i in 1:length(atomPos)]
        return cloud, atomPos
    end
end

function AtomCloud2D(size::Float64, density::Float64) #density in atoms/m^3 and size in mum
    size = size*1e-6
    L = density^(-1/2) #Spacing between atoms
    nAtoms = trunc(Int, size^2 * density)
    nAtomsroot = trunc(Int, nAtoms^(1/2))       
    ### Generate coordinates of atoms
    atomPos = [[x*L + L*rand(1)[1], y*L + L*rand(1)[1], 0] for x in 1:floor(nAtomsroot), y in 1:floor(nAtomsroot)]
    ### Generating cloud of ground state atoms
    ground_state = [0.0, 1.0]
    name = "Li" 
    cloud = [Atom(name, i, ground_state, atomPos[i], 0.0) for i in 1:length(atomPos)]
    return cloud, atomPos
end


function AtomCloud1D(size::Float64, density::Float64) #density in atoms/m^3 and size in mum
    size = size*1e-6
    L = 1/density #Spacing between atoms
    nAtoms = trunc(Int, size * density)
    ### Generate coordinates of atoms
    atomPos = [[x*L + L*rand(1)[1], 0, 0] for x in 1:floor(nAtoms)]
    ### Generating cloud of ground state atoms
    ground_state = [0.0, 1.0]
    name = "Li" 
    cloud = [Atom(name, i, ground_state, atomPos[i], 0.0) for i in 1:length(atomPos)]
    return cloud, atomPos
end





#mutable struct Topologies
    #DictAtom = Dict{Any, AtomType}
    #DictLaser = Dict{Any, LaserType}
    #atomTypes::DictAtom
    #laserTypes::DictLaser
    #atomTypes::Dict{Any, AtomType}
    #laserTypes::Dict{Any, LaserType}
    #atomCloud::AtomCloud
    #function Topologies(atomTypes::DictAtom=DictAtom(), atomTypes::DictLaser=DictLaser())
    #    new(atomTypes, laserTypes)
    #end
#end



mutable struct Topologies
    atomTypes::Dict{Any, AtomType}
    laserTypes::Dict{Any, LaserType}
    #atomCloud::AtomCloud
    function Topologies(atomTypes::AT=Dict{Any, AtomType}(),
        laserTypes::LT=Dict{Any, LaserType}()) where {AT<:Dict{Any, AtomType}, LT<:Dict{Any, LaserType}, DIM}
        new(atomTypes, laserTypes)
    end
end


end
