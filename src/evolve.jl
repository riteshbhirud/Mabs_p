"""
    tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gate::ITensors.ITensor; kwargs...)
    tebd(psi::BMPS{<:ITensorMPS.MPS,Truncated}, gates::Vector{ITensors.ITensor}; kwargs...)

Perform time evolution using TEBD algorithm.
"""
function tebd(
    psi::BMPS{<:ITensorMPS.MPS,Truncated}, 
    gate::ITensors.ITensor; 
    kwargs...
)
    evolved_mps = ITensors.apply(gate, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end
function tebd(
    psi::BMPS{<:ITensorMPS.MPS,Truncated}, 
    gates::Vector{ITensors.ITensor}; 
    kwargs...
)
    evolved_mps = ITensors.apply(gates, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    tdvp(psi::BMPS{<:ITensorMPS.MPS,Truncated}, H::BMPO{<:ITensorMPS.MPO,Truncated}, dt::Number; kwargs...)

Perform time evolution using Time Dependent Variational Principle (TDVP) algorithm.

Important Note: This performs imaginary-time evolution, useful for:
- Finding ground states by iterative evolution
- Preparing low-energy states  
- Pre-optimization before DMRG

For real-time quantum dynamics, use `tebd` with explicit time evolution operators instead.

# Example: Ground state preparation
```julia
H = harmonic_chain(sites; ω=1.0, J=0.5)
psi = random_bmps(sites, Truncated())

for i in 1:100
    psi = tdvp(psi, H, 0.1)
end

energy, psi_gs = dmrg(H, psi; nsweeps=10)

Arguments:
- psi::BMPS: Input bosonic MPS
- H::BMPO: Hamiltonian as bosonic MPO
- dt::Number: Time step

Keyword Arguments:
- nsweeps::Int: Number of TDVP sweeps (default: 1)
- cutoff::Float64: Truncation cutoff (default: 1e-12)
- maxdim::Int: Maximum bond dimension (default: 10000)
- normalize::Bool: Normalize after evolution (default: true)

Returns:
- BMPS: Evolved bosonic MPS (closer to ground state)
"""
function tdvp(
    psi::BMPS{<:ITensorMPS.MPS,Truncated}, 
    H::BMPO{<:ITensorMPS.MPO,Truncated}, 
    dt::Number; 
    nsweeps::Int=1,
    cutoff::Float64=1e-12,
    maxdim::Int=10000,
    kwargs...
)
    evolved_mps = ITensorMPS.tdvp(
        H.mpo, 
        -1im * dt,  
        psi.mps; 
        nsweeps=nsweeps,
        cutoff=cutoff,
        maxdim=maxdim,
        normalize=true,
        kwargs...
    )
    return BMPS(evolved_mps, psi.alg)
end

"""
    tebd(psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, gate::ITensors.ITensor; kwargs...)
    tebd(psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, gates::Vector{ITensors.ITensor}; kwargs...)

Perform TEBD evolution on <:PseudoSite representation.
"""
function tebd(
    psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    gate::ITensors.ITensor; 
    kwargs...
)
    evolved_mps = ITensors.apply(gate, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

function tebd(
    psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    gates::Vector{ITensors.ITensor}; 
    kwargs...
)
    evolved_mps = ITensors.apply(gates, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    tdvp(psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, H::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, dt::Number; kwargs...)

Perform TDVP evolution on <:PseudoSite representation.
"""
function tdvp(
    psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    H::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, 
    dt::Number; 
    nsweeps::Int=1,
    cutoff::Float64=1e-12,
    maxdim::Int=10000,
    kwargs...
)
    psi.alg == H.alg || throw(ArgumentError("Algorithms must match"))
    
    evolved_mps = ITensorMPS.tdvp(
        H.mpo, 
        -1im * dt,  
        psi.mps; 
        nsweeps=nsweeps,
        cutoff=cutoff,
        maxdim=maxdim,
        normalize=true,
        kwargs...
    )
    return BMPS(evolved_mps, psi.alg)
end