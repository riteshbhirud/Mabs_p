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

Arguments:
- psi::BMPS: Input bosonic MPS
- H::BMPO: Hamiltonian as bosonic MPO
- dt::Number: Time step

Keyword Arguments:
- kwargs...: Additional parameters passed to `ITensorMPS.tdvp`

Returns:
- BMPS: Time-evolved bosonic MPS
"""
function tdvp(
    psi::BMPS{<:ITensorMPS.MPS,Truncated}, 
    H::BMPO{<:ITensorMPS.MPO,Truncated}, 
    dt::Number; 
    kwargs...
)
    evolved_mps = ITensorMPS.tdvp(H.mpo, dt, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end