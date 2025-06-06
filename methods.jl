"""
# TODO
# -----------------------------------------------------------
# -check if return functions are needed
# -use a mean field approach to calculate interaction energy with other Rydberg atoms
"""
module methods_cloud

export PairGenerator, RandomSelector, RandomSelectorIndex, CloudState, ResetGradient

function PairGeneratorOld(cloud, cutOff)
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

function PairGenerator(cloud, cutOff)
    nAtoms = length(cloud)
    pairs = []
    for i in 1:nAtoms
        for j in i+1:nAtoms
            pos_i = cloud[i].position
            pos_j = cloud[j].position
            distance_sq = (pos_i[1] - pos_j[1])^2 + (pos_i[2] - pos_j[2])^2 + (pos_i[3] - pos_j[3])^2
            if distance_sq < cutOff^2
                push!(pairs, (cloud[i].index, cloud[j].index))
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
            append!(pos_atom, atom_index)
            append!(subgroup, [pos_atom])
            global position = deleteat!(position,idx)
        end
        push!(subgroups, subgroup)
    end
    return subgroups
end

function CloudState(atomGroups, test_location)
    atomState, cloudState = [], []
    numberGroups = length(atomGroups)
    for i in 1:numberGroups #loop through groups
        save_location = test_location*"_g"*string(i)*"_"
        proj_ops = jldopen(save_location*"proj_ops.jld2", "r", mmaparrays=true)
        psi_t = jldopen(save_location*"Psi_t.jld2", "r", mmaparrays=true)["psi_t"]
        psi_f = last(psi_t)
        subgroup = atomsGroups[i]
        for j in 1:length(subgroup) #loop through atoms in group
            #Rydberg probability amplitude
            rjXrj = proj_ops["RXR"][j]
            cr = expect(rjXrj, psi_f)
            cg = (1-cr^2)^(0.5)
            #Update state of atom in cloud
            atomPos_Index = deepcopy(subgroup[j])
            atomIndex = atomPos_Index[4]
            cloud[atomIndex].state = [cg,cr]
        end
    end

end

function ResetGradient(cloud)
    for atom in 1:length(cloud)
        cloud[atom].gradient = [0.0, 0.0, 0.0]
    end
end

end