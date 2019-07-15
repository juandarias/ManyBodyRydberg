"""
# TODO
# -----------------------------------------------------------
# -test cloud generation: Done
"""
module topology

using constants
using Distributions: MvNormal, rand!

export AtomType, LaserType, TrapType, AtomCloudSphere, AtomCloud2DSquare, AtomCloud, AtomCloud2D, AtomCloud1D, Topologies

mutable struct Atom
    name
    index::Int64
    state::Array{Float64,1}
    position::Array{Float64,1} #the dimension of the position basis
    velocity::Array{Float64,1}
    acceleration::Array{Float64,1}
    stark_shift::Float64
    function Atom(name, index::Int64, state::Array{Float64,1}, position::Array{Float64,1}, velocity::Array{Float64,1}, acceleration::Array{Float64,1}, shift::Float64)
        new(name, index, state, position, velocity, acceleration, shift)
    end
end

struct AtomType
    α::Float64
    C6::Float64
    C4::Float64
    τ::Float64
    charge::Int8
    mass::Float64
    Rydberg_state
end


mutable struct LaserType
    Δ::Float64
    ω::Float64
    Ω::Float64
    bandwidth::Float64;
    field_itensity::Float64;
    pulse_length::Float64;
    rep_rate::Float64;
end

mutable struct TrapType
    Ωrf::Float64
    ωz::Float64
    ϕ::Float64
    mIon::Float64
    q::Float64
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
        nAtomscubic = nAtoms^(1/3)
        ### Generate coordinates of atoms
        atomPos = [[x*L + L*rand(1)[1], y*L + L*rand(1)[1], z*L + L*rand(1)[1]] for x in -floor(nAtomscubic/2):floor(nAtomscubic/2), y in -floor(nAtomscubic/2):floor(nAtomscubic/2), z in -floor(nAtomscubic/2):floor(nAtomscubic/2)]
        ### Generating cloud of ground state atoms
        initial_state = [1.0, 0.0] #Ground state
        initial_velocity = [0.0, 0.0, 0.0]
        initial_acceleration = [0.0, 0.0, 0.0]
        initial_shift = 0.0
        name = "Li" 
        cloud = [Atom(name, i, initial_state, atomPos[i], initial_velocity, initial_acceleration, initial_shift) for i in 1:length(atomPos)]
        return cloud, atomPos
    end
end


struct AtomCloudSphere
    density::Float64
    radius::Float64
    temperature::Float64
    cloud::Array{Atom}
    function AtomCloudSphere(radius::Float64, density::Float64, temperature::Float64)
        size = 2*radius*1e-6
        L = density^(-1/3) #Spacing between atoms
        nAtoms = size^3 * density
        nAtomscubic = nAtoms^(1/3)
        ### Generate coordinates of atoms in cube
        atomsPosCube = [[x*L + L*rand(1)[1], y*L + L*rand(1)[1], z*L + L*rand(1)[1]] for x in -ceil(nAtomscubic/2):ceil(nAtomscubic/2), y in -ceil(nAtomscubic/2):ceil(nAtomscubic/2), z in -ceil(nAtomscubic/2):ceil(nAtomscubic/2)]
        ### Intersect sphere
        atomsPosSphere = []
        for atomPos in atomsPosCube
            distance = sqrt(sum(atomPos.*atomPos))
            if distance < radius*1e-6
                push!(atomsPosSphere, atomPos)
            end
        end
        ### Generate initial velocities of atoms, as a multivariate normal distribution, with μ_v = 0
        mLi = constants.Li6.mass;
        σ_v = sqrt( constants.kB * temperature / mLi );
        vel_distribution = MvNormal(3, σ_v);
        initial_velocity = zeros(3,length(atomsPosSphere))
        initial_velocity = rand!(vel_distribution, initial_velocity)
        ### Generating cloud of ground state atoms
        initial_state = [1.0, 0.0] #Ground state
        initial_acceleration = [0.0, 0.0, 0.0]
        initial_shift = 0.0
        name = "Li" 
        cloud = [Atom(name, i, initial_state, atomsPosSphere[i], initial_velocity[:,i], initial_acceleration, initial_shift) for i in 1:length(atomsPosSphere)]
        return cloud, atomsPosSphere
    end
end


struct AtomCloud2DSquare
    density::Float64
    size::Float64
    cloud::Array{Atom}
    function AtomCloud2DSquare(size::Float64, density::Float64)
        size = size*1e-6
        L = density^(-1/2) #Spacing between atoms
        nAtoms = size^2 * density
        nAtomsroot = trunc(Int, nAtoms^(1/2)) 
        ### Generate coordinates of atoms in cube
        atomsPosSquare = [[x*L + L*rand(1)[1], y*L + L*rand(1)[1], 0] for x in -ceil(nAtomsroot/2):ceil(nAtomsroot/2), y in -ceil(nAtomsroot/2):ceil(nAtomsroot/2)]
        ### Generating cloud of ground state atoms
        initial_state = [1.0, 0.0] #Ground state
        initial_velocity = [0.0, 0.0, 0.0]
        initial_acceleration = [0.0, 0.0, 0.0]
        initial_shift = 0.0
        name = "Li" 
        cloud = [Atom(name, i, initial_state, atomsPosSquare[i], initial_velocity, initial_acceleration, initial_shift) for i in 1:length(atomsPosSquare)]
        return cloud, atomsPosSquare
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
    initial_state = [1.0, 0.0] #Ground state
    initial_velocity = [0.0, 0.0, 0.0]
    initial_acceleration = [0.0, 0.0, 0.0]
    initial_shift = 0.0
    name = "Li" 
    cloud = [Atom(name, i, initial_state, atomPos[i], initial_velocity, initial_acceleration, initial_shift) for i in 1:length(atomPos)]
    return cloud, atomPos
end


function AtomCloud1D(size::Float64, density::Float64) #density in atoms/m^3 and size in mum
    size = size*1e-6
    L = 1/density #Spacing between atoms
    nAtoms = trunc(Int, size * density)
    ### Generate coordinates of atoms
    atomPos = [[x*L + L*rand(1)[1], 0, 0] for x in -floor(nAtoms)/2:floor(nAtoms)/2]
    ### Generating cloud of ground state atoms
    initial_state = [1.0, 0.0] #Ground state
    initial_velocity = [0.0, 0.0, 0.0]
    initial_acceleration = [0.0, 0.0, 0.0]
    initial_shift = 0.0
    name = "Li" 
    cloud = [Atom(name, i, initial_state, atomPos[i], initial_velocity, initial_acceleration, initial_shift) for i in 1:length(atomPos)]
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
    trapTypes::Dict{Any, TrapType}
    #atomCloud::AtomCloud
    function Topologies(atomTypes::AT=Dict{Any, AtomType}(),
        laserTypes::LT=Dict{Any, LaserType}(), trapTypes::TT=Dict{Any, TrapType}()) where {AT<:Dict{Any, AtomType}, LT<:Dict{Any, LaserType}, TT<:Dict{Any, TrapType}, DIM}
        new(atomTypes, laserTypes, trapTypes)
    end
end


end
