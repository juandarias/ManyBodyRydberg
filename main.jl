"""
# TODO
# -----------------------------------------------------------
# -update arguments related to laser
# -
# -
"""


function CalculatePotential(cloud, topology, pairs)
    for atom in cloud
        CalculateIonRydberg(atom, topology)
    end

    for pair in pairs
        CalculatevdW(pair, cloud, topology)
    end
    
end


"""
# Facilitation of atoms
# ----------------------------------------------------
# -> Notes:
#	-MCWFM is done only for a group of atoms with certain detuning bandwidth
#
"""



function ExciteAtoms(atoms, topology)
    resAtoms = []
    for atom in atoms
        deltaE = abs(atom.stark_shift - laserRyd.detuning)
        if deltaE < laserRyd.bandwidth
            push!(resAtoms, atom)
        end
	end
	MCWFM(resAtoms, time)
end