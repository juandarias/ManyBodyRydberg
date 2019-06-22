using PyPlot
using QuantumOptics
using DelimitedFiles
using JLD2, FileIO
using SparseArrays
using Interpolations


"""
# Correlation function for a quantum state:
# ----------------------------------------
# Unless there is a complete transfer to a excited states, atoms will be in a superpositon of ground and excited states. 
# Therefore the projection of the electronic state of atom i in the system state Ψ, <Ψ|e_iXe_i|Ψ> < 1
# ----------------------------------------
# Notes:
# -Due to the limited amount of atoms, the graininess of the correlation function is too big. An input with more atoms are needed, e.g. after propagation of all the atomic cloud
# -To verify that simulations indeed show expected behavior, different C6 coefficients can be used in the simulations
"""

folder = "/home/jdiego/Project_Juan_Diego/Code/Julia/MBRyd/testa1/"
testname = "run2_"

function paircorrelation(folder, testname, binSize, maxDistance, trajectories)
    distancePair, g2_r, numRyd, numPairs = [], [], [], []
    #Load wavefunction and operators
    pos_ops = jldopen(folder*testname*"_pos_ops.out.jld2", "r", mmaparrays=true)
    proj_ops = jldopen(folder*testname*"_proj_ops.out.jld2", "r", mmaparrays=true)
    proj_ops = jldopen(test_location*"g1_proj_ops.out.jld2", "r", mmaparrays=true)
    psi_t = jldopen(folder*testname*"_Psi_t.out.jld2", "r", mmaparrays=true)
    psi_t = jldopen(test_location*"g1_Psi_t.out.jld2", "r", mmaparrays=true)
    frame_final = length(psi_t["psi_t"])
    Ψ = psi_t["psi_t"][frame_final]/trajectories
    n_atoms = Int8(log(length(Ψ))/log(2))
    #Calculate g^2(r) using binning and according the definition of Gärtner et. al. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.88.043410
    numBins = maxDistance/binSize
    binPos = [(2*n-1)*binSize/2 for n in 1:numBins] #Midpoint of bins
    totalRyd = sum([abs(dagger(Ψ) * proj_ops["RXR"][i] * Ψ) for i in 1:n_atoms])
    for n in 1:numBins
        global atom_pairs = 0
        global ryd_pairs = 0
        binLow = (n-1)*binSize
        binMax = n*binSize
        for i in 1:n_atoms
            for j in i:n_atoms
                #Distance between atoms
                #dij = ((pos_ops["Position_x"][i] - pos_ops["Position_x"][j])^2 + (pos_ops["Position_y"][i] - pos_ops["Position_y"][j])^2 + (pos_ops["Position_z"][i] - pos_ops["Position_z"][j])^2)
                dij = abs((((pos_ops["Position_x"][i] - pos_ops["Position_x"][j])^2 + (pos_ops["Position_y"][i] - pos_ops["Position_y"][j])^2).data[1]))^0.5
                #Projection operator 
                pair_operator = proj_ops["RXR"][j]*proj_ops["RXR"][i]
                state_pair = abs(dagger(Ψ) * pair_operator * Ψ)
                if state_pair == 1 #Creates a list of atoms pairs in a pure Rydberg state
                    push!(distancePair, dij)
                end
                if binMax > dij && dij > binLow
                    ryd_pairs += state_pair
                    atom_pairs += 1 #Counts the number of pairs within a bin
                end
            end
        end
        #pairs *= (n_atoms*totalRyd)^2 #Total number of pairs times total Rydbergs squared
        push!(numPairs, atom_pairs)
        push!(numRyd, ryd_pairs)
        push!(g2_r, ryd_pairs/atom_pairs)
    end
    return g2_r, binPos, numRyd, numPairs
end



function radialcorrelation2D(test_location, binSize, maxDistance, trajectories, frame, numGroups, atomsxgroup, int_order=1)
    #Calculate g^2(r) using binning and according the definition of Gärtner et. al. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.88.043410
    numBins = Int8(maxDistance ÷ binSize)
    g2_r = zeros(numBins)
    norm = zeros(numBins)
    binPos = [(2*n-1)*binSize/2 for n in 1:numBins] #Midpoint of bins
    for g in 1:numGroups
        #Load wavefunction and operators
        pos_ops = jldopen(test_location*"_g"*string(g)*"_pos_ops.out.jld2", "r", mmaparrays=true)
        pos_x = pos_ops["Position_x"]
        pos_y = pos_ops["Position_y"]
        pos_z = pos_ops["Position_z"]
        proj_ops = jldopen(test_location*"_g"*string(g)*"_proj_ops.out.jld2", "r", mmaparrays=true)
        psi_t = jldopen(test_location*"_g"*string(g)*"_Psi_t.out.jld2", "r", mmaparrays=true)["psi_t"]/trajectories
        Ψ = psi_t[frame]
        totalRyd = sum([abs(dagger(Ψ) * proj_ops["RXR"][i] * Ψ) for i in 1:atomsxgroup])
        for a in 1:atomsxgroup
            probRyd = expect(proj_ops["RXR"][a],Ψ)
            posAtom = abs(((pos_ops["Position_x"][a]^2 + pos_ops["Position_y"][a]^2).data[1])^0.5)*1e6
            binNum = Int8(posAtom ÷ binSize)+2
            if binNum < numBins
                g2_r[binNum] += probRyd
                norm[binNum] += 1
            end
        end
    end
    for b in 1:numBins
        areaBin = π*(b * binSize)^2
        #g2_r[b] = g2_r[b]/areaBin
        norm[b]!= 0 && (g2_r[b] = g2_r[b]/norm[b])
    end
    ###Interpolation of data
    int_grid = [x for x in 1:1/int_order:numBins];
    g2_r_int = interpolate(g2_r, BSpline(Quadratic(Line(OnCell()))) )
    g2_r_quad = [g2_r_int(x) for x in int_grid];
    binSize_int = binSize/int_order;
    numBins_int = Int8(maxDistance ÷ binSize_int);
    binPos_int = [(2*n+4)*binSize_int/2 for n in 1:numBins_int]; #Midpoint of bins
    return binPos, g2_r, binPos_int, g2_r_quad
end
            


function radialcorrelation2DOld(test_location, binSize, maxDistance, trajectories, frame, numGroups, atomsxgroup, int_order=1)
    #Calculate g^2(r) using binning and according the definition of Gärtner et. al. https://journals.aps.org/pra/pdf/10.1103/PhysRevA.88.043410
    numBins = Int8(maxDistance ÷ binSize)
    g2_r = zeros(numBins)
    norm = zeros(numBins)
    binPos = [(2*n-1)*binSize/2 for n in 1:numBins] #Midpoint of bins
    for g in 1:numGroups
        #Load wavefunction and operators
        pos_ops = jldopen(test_location*"_g"*string(g)*"_pos_ops.out.jld2", "r", mmaparrays=true)
        pos_x = pos_ops["Position_x"]
        pos_y = pos_ops["Position_y"]
        pos_z = pos_ops["Position_z"]
        proj_ops = jldopen(test_location*"_g"*string(g)*"_proj_ops.out.jld2", "r", mmaparrays=true)
        psi_t = jldopen(test_location*"_g"*string(g)*"_Psi_t.out.jld2", "r", mmaparrays=true)["psi_t"]/trajectories
        Ψ = psi_t[frame]
        totalRyd = sum([abs(dagger(Ψ) * proj_ops["Projectors"][i] * Ψ) for i in 1:atomsxgroup])
        for a in 1:atomsxgroup
            probRyd = expect(proj_ops["Projectors"][a],Ψ)
            posAtom = abs(((pos_ops["Position_x"][a]^2 + pos_ops["Position_y"][a]^2).data[1])^0.5)*1e6
            binNum = Int8(posAtom ÷ binSize)+2
            if binNum < numBins
                g2_r[binNum] += probRyd
                norm[binNum] += 1
            end
        end
    end
    for b in 1:numBins
        areaBin = π*(b * binSize)^2
        #g2_r[b] = g2_r[b]/areaBin
        norm[b]!= 0 && (g2_r[b] = g2_r[b]/norm[b])
    end
    ###Interpolation of data
    int_grid = [x for x in 1:1/int_order:numBins];
    g2_r_int = interpolate(g2_r, BSpline(Quadratic(Line(OnCell()))) )
    g2_r_quad = [g2_r_int(x) for x in int_grid];
    binSize_int = binSize/int_order;
    numBins_int = Int8(maxDistance ÷ binSize_int);
    binPos_int = [(2*n+4)*binSize_int/2 for n in 1:numBins_int]; #Midpoint of bins
    return binPos, g2_r, binPos_int, g2_r_quad
end


######
### Plots

PyPlot.bar(binPos, numPairs_testa4, width=0.3)
PyPlot.bar(binPos, numRyd_testa4, bottom=numPairs_testa4, width=0.3)


######
### Adaptive binning

function binning(positions)
    sort(positions)
    for pos in positions
        global k=0
        while pos > binPos[k]
            global k+=1
        end
    end    
end


### Adaptive binning
######
