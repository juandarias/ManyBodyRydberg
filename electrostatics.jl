module electrostatics

export AccelerationAtoms, CalculateIonRydberg, CalculatePTIon, CalculatevdW, ResetAcceleration

using constants

"==========="
# Constants #
"==========="
hbar = constants.hbar
hplanck = constants.hplanck
ε_0 = constants.ε_0
C_e = constants.e


"=========================="
# Electrostatic potentials #
"=========================="


function CalculatevdW(pair, cloud, topology)
    C6 = topology.atomTypes["atomRyd"].C6;
    α = topology.atomTypes["atomRyd"].α;

    atom_i = cloud[pair[1]];
    atom_j = cloud[pair[2]];
    
    pos_i = atom_i.position;
    pos_j = atom_j.position;
    d_sq = (pos_i[1] - pos_j[1])^2 + (pos_i[2] - pos_j[2])^2 + (pos_i[3] - pos_j[3])^2;
    abs(d_sq ) < 50e-9^2 && (d_sq =100e-9^2)

    ampRyd_i = atom_i.state[2];
    ampRyd_j = atom_j.state[2];


    #####
    #Assignment of potential energy and forces. Potential energy is added to both atoms only if both are Rydberg, otherwise it will be added only to GS atom. Only if both atoms are in Rydberg state then the forces will be calculated

    if ampRyd_i!=0 && ampRyd_j!=0
        pot_energy_vdW = hbar * C6 * d_sq^-3 * ampRyd_i * ampRyd_j;
        atom_i.stark_shift += pot_energy_vdW;
        atom_j.stark_shift += pot_energy_vdW;
    else
        atom_j.stark_shift += 0
        atom_i.stark_shift += 0
    end
end
   
function CalculateIonRydberg(atom, topology)
    α = topology.atomTypes["atomRyd"].α;
    pos = atom.position;
    d_sq = pos[1]^2 + pos[2]^2 + pos[3]^2
    prefactor = C_e/(4*pi*ε_0);
    U_ion = -1/2 * hplanck * α  * prefactor^2 * d_sq^-2;
    atom.stark_shift += U_ion
end


function CalculatePTIon(atom, topology, time_simulation::Float64)
    α = topology.atomTypes["atomRyd"].α; 
    YbTrap = topology.trapTypes["YbTrap"];
    mass = topology.atomTypes["atomRyd"].mass;
    Ωrf = YbTrap.Ωrf;
    ωz = YbTrap.ωz;
    mYb = YbTrap.mIon;
    q = YbTrap.q;
    ϕ = YbTrap.ϕ;
    t = time_simulation;
    C = -ωz^2 + q*Ωrf^2*cos(Ωrf*t+ϕ);
    D = -ωz^2 - q*Ωrf^2*cos(Ωrf*t+ϕ);
    E = 2*ωz^2;
    prefactor = C_e/(4*pi*ε_0);;

    pos = atom.position[1:3];
    d_sq = pos[1]^2 + pos[2]^2 + pos[3]^2;
                
    Etot = prefactor*d_sq^(-3/2)*pos + mYb/(2*C_e)*[pos[1]*C, pos[2]*D, pos[3]*E]; #Eion + Ept
    U_PT_ion = -1/2*hplanck*α*(Etot[1]^2 + Etot[2]^2 + Etot[3]^2)
    atom.stark_shift += U_PT_ion
    

end


"======================"
# Electrostatic forces #
"======================"




function AccelerationvdW(pairs, cloud, topology)
    C6 = topology.atomTypes["atomRyd"].C6;
    mass = topology.atomTypes["atomRyd"].mass;
    for pair in pairs
        atom_i = cloud[pair[1]];
        atom_j = cloud[pair[2]];
        
        pos_i = atom_i.position;
        pos_j = atom_j.position;
        rij = pos_i - pos_j;
        rij_sq = sum(rij.*rij);

        ∇UvdW = -6*hbar*C6*atom_i.state[2]*atom_j.state[2]*rij_sq^(-4)*rij;
        atom_i.acceleration += -∇UvdW/mass;
        atom_j.acceleration += -1*-∇UvdW/mass; #Taking into account direction of rij vector
    end
end


   
function AccelerationIon(cloud, topology)
    C4 = topology.atomTypes["atomRyd"].C4;
    mass = topology.atomTypes["atomRyd"].mass;
    for atom in cloud
        pos = atom.position;
        d_sq = pos[1]^2 + pos[2]^2 + pos[3]^2;
        ∇UIon = -4*hplanck*C4*atom.state[2]*d_sq^(-3)*pos;
        atom.acceleration += -∇UIon/mass; #Direction of force determined by sign of C4
    end
end



function AccelerationIonPT(cloud, topology, time_simulation::Float64)
    ### Parameters Paul Trap and atom
    α = topology.atomTypes["atomRyd"].α; 
    YbTrap = topology.trapTypes["YbTrap"];
    mass = topology.atomTypes["atomRyd"].mass;
    a1, a2, a3, b1, b2, b3 = [-0.5, -0.5, 1, 1, -1, 0]
    Ωrf = YbTrap.Ωrf
    ωz = YbTrap.ωz
    mYb = YbTrap.mIon
    q = YbTrap.q
    ϕ = YbTrap.ϕ
    t = time_simulation
    C = -ωz^2 + q*Ωrf^2*cos(Ωrf*t+ϕ)
    D = -ωz^2 - q*Ωrf^2*cos(Ωrf*t+ϕ)
    E = 2*ωz^2
    
    postfactor = C_e/(4*pi*ε_0);
    #atomfactor = hbar * α / mLi;
    for atom in cloud
        ### See derivation in Limiting_equations v2.0, under gradients header
        pos = atom.position[1:3];
        d_sq = pos[1]^2 + pos[2]^2 + pos[3]^2;
        
        M = 3*(C*pos[1]^2 + D*pos[2]^2 + E*pos[3]^2);
        
        ∇Eion_sq = -4*postfactor^2*d_sq^(-3)*pos
        ∇Ept_sq =  2*(mYb/(2*C_e))^2*([C,D,E].^2).*pos
        ∇Eion_pt = mYb/(4*pi*ε_0)*d_sq^(-5/2)*
        [pos[1]*(2*C*d_sq-M),
        pos[2]*(2*D*d_sq-M),
        pos[3]*(2*E*d_sq-M)]


        
        ∇Utot = -1/2*hplanck*α*(∇Eion_sq + ∇Ept_sq + 2*∇Eion_pt)*atom.state[2];
        atom.acceleration += -∇Utot/mass;
    end
        
    #accel_tot = (nabla_E_PT_ion + nabla_E_PT_sq + nabla_E_ion_sq) * -atomfactor/2

end

function AccelerationAtoms(cloud, pairs_atoms, topology, time_simulation::Float64; withPT = true)
    ResetAcceleration(cloud)
    AccelerationvdW(pairs_atoms, cloud, topology)
    if withPT == true
        AccelerationIonPT(cloud, topology, time_simulation)
    else
        AccelerationIon(cloud, topology)
    end
end

function ResetAcceleration(cloud)
    for atom in cloud
        atom.acceleration = [0.0, 0.0, 0.0]
    end
end




"====="
# Old #
"====="

function CalculatePTIonOld(cloud, topology, time_simulation::Float64)
    α = topology.atomTypes["atomRyd"].α; 
    YbTrap = topology.trapTypes["YbTrap"];
    mass = topology.atomTypes["atomRyd"].mass;
    a1, a2, a3, b1, b2, b3 = [-0.5, -0.5, 1, 1, -1, 0]
    Ωrf = YbTrap.Ωrf
    ωz = YbTrap.ωz
    mYb = YbTrap.mIon
    q = YbTrap.q
    ϕ = YbTrap.ϕ
    t = time
    C = -ωz^2 + q*Ωrf^2*cos(Ωrf*t+ϕ)
    D = -ωz^2 - q*Ωrf^2*cos(Ωrf*t+ϕ)
    E = 2*ωz^2


    for atom in cloud
        pos = atom.position[1:3];
        Uion_pt =  -(α/4)*mYb/(4*pi*ε_0)*d_sq^(-3/2)*(C*pos[1]^2 + D*pos[2]^2 + E*pos[3]^2);
        atom.stark_shift += Uion_pt
    end
end


end