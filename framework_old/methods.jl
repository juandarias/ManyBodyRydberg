"""
# TODO
# -----------------------------------------------------------
# -check if return functions are needed
# -use a mean field approach to calculate interaction energy with other Rydberg atoms
"""
module methods_cloud

function PairGenerator(cloud, cutOff)
    nAtoms = length(cloud)
    pairs = []
    for i in 1:nAtoms
        for j in i+1:nAtoms
            pos_i = cloud[i].position
            pos_j = cloud[j].position
            distance_sq = (pos_i[1] - pos_j[1])^2 + (pos_i[2] - pos_j[2])^2 + (pos_i[3] - pos_j[3])^2
            if distance_sq < cutOff^2
                push!(pairs, (i,j))
            end
        end
    end
    return pairs   
end

function RandomSelectorOld(position_atoms, size_group)
    atomGroups = []
    global position = []
    position = copy(position_atoms[:])
    numberGroups = trunc(Int, length(position)/size_group)
    subgroups = zeros(Float64, numberGroups, size_group, 4)
    for i in 1:numberGroups
        for j in 1:size_group
            pos_atom = rand(position) #chooses random position
            index = indexin([pos_atom], position)[1] #finds index of position in list
            atom_number = indexin([pos_atom], position_atoms)[1]
            subgroups[i,j,1] = atom_number
            subgroups[i,j,2] = pos_atom[1]
            subgroups[i,j,3] = pos_atom[2]
            global position = deleteat!(position,index)
        end
    end
    return subgroups
end


function RandomSelector(position_atoms, size_group)
    subgroups = []
    global position = copy(position_atoms[:])
    numberGroups = trunc(Int, length(position)/size_group)
    for i in 1:numberGroups
        subgroup = []
        for j in 1:size_group
            pos_atom = rand(position) #chooses random position
            idx = indexin([pos_atom], position)[1] #finds index of position in list
            atom_number = indexin([pos_atom], position_atoms[:])[1]
            #subgroups[i,j,1] = atom_number
            #subgroups[i,j,2] = pos_atom[1]
            #subgroups[i,j,3] = pos_atom[2]
            append!(subgroup, [pos_atom])
            global position = deleteat!(position,idx)
        end
        push!(subgroups, subgroup)
    end
    return subgroups
end


function RandomSelectorIndex(position_atoms, size_group)
    subgroups = []
    global position = copy(position_atoms[:])
    numberGroups = trunc(Int, length(position)/size_group)
    for i in 1:numberGroups
        subgroup = []
        for j in 1:size_group
            pos_atom = rand(position) #chooses random position
            idx = indexin([pos_atom], position)[1] #finds index of position in list
            atom_index = indexin([pos_atom], position_atoms[:])[1]
            #subgroups[i,j,1] = atom_index
            #subgroups[i,j,2] = pos_atom[1]
            #subgroups[i,j,3] = pos_atom[2]
            pos_index = append!([pos_atom], atom_index)
            append!(subgroup, pos_index)
            global position = deleteat!(position,idx)
        end
        push!(subgroups, subgroup)
    end
    return subgroups
end




function CalculatevdW(pair, cloud, topology)
    C6 = topology.atomTypes["Li"].C6;
    alpha = topology.atomTypes["Li"].alpha;

    atom_i = cloud[pair[0]];
    atom_j = cloud[pair[1]];
    
    pos_i = atom_i.position;
    pos_j = atom_j.position;
    distance_sq = (pos_i[1] - pos_j[1])^2 + (pos_i[2] - pos_j[2])^2 + (pos_i[3] - pos_j[3])^2;

    probRyd_i = atom_i.state[1]^2;
    probRyd_j = atom_j.state[1]^2;


    #####
    #Assignment of potential energy and forces. Potential energy is added to both atoms only if both are Rydberg, otherwise it will be added only to GS atom. Only if both atoms are in Rydberg state then the forces will be calculated

    if probRyd_i!=0 && probRyd_j!=0
        pot_energy_vdW = C6 * distance_sq^-3 * probRyd_i * probRyd_j;
        atom_i.stark_shift += pot_energy_vdW;
        atom_j.stark_shift += pot_energy_vdW;
    elseif probRyd_i!=0
        pot_energy_vdW = C6 * distance_sq^-3 * probRyd_i;
        atom_j.stark_shift += pot_energy_vdW;
    else
        pot_energy_vdW = C6 * distance_sq^-3 * probRyd_j;
        atom_i.stark_shift += pot_energy_vdW;
    end
end
   
function CalculateIonRydberg(atom, topology)
    alpha = topology.atomTypes["Li"].alpha;
    pos = atom.position
    distance_sq = pos[1]^2 + pos[2]^2 + pos[3]^2
    
    prefactor = C_e/(4*pi*epsilon_0);
    pot_energy_ion = -0.5 * alpha * prefactor^2 * distance_sq^-2;

    atom.stark_shift += pot_energy_ion
end


end