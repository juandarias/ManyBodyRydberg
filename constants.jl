module constants
"""
# All constants in SI units, except energies which are expressed in Hz
# --------------------------------------------------------------------
# Sources of constants:
# --------------------------------------------------------------------
# - Strontium:
# DOI: 10.1103/PhysRevX.7.021038 (polarizability of ion)
# DOI: 10.1088/0953-4075/44/18/184010
"""

hbar = 2.109143635292313e-34
hplanck = 1.32521403e-33
ε_0 = 8.8541878128e-12
e = 1.602176634e-19
kB = 1.380649e-23

mYb = 2.87375415282392e-25

module Li6
mass = 1.1528239202657808e-26
C624S = 2*π*407e3*10^-36
C650S = 2*π*1.893e9*10^-36
α24S = 313 #Hz/(V/m)^2
α50S = 49.3*10^3
τ24S = 11e-6
τ50S = 103e-6
C424S = -3.2*10^-16
C450S = -5.11*10^-14
end

module Rb
mass = 1.4196013289036546e-25
C643S = 2*π*2.43e9*10^-36
end


module Sr
C642S = -2*π*935e6*10^-36
α42S = 437
C442S = -4.53749*10^-16
end

end