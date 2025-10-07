"""
Bosonic Matrix Product Operator wrapper that supports different algorithms.
Contains an underlying `ITensorMPS.MPO` object and algorithm specification.

Fields:
- mpo::M: The underlying ITensorMPS.MPO object
- alg::A: Algorithm specification (`Truncated`, `PseudoSite`, or `LocalBasis`)
"""
struct BMPO{M<:ITensorMPS.MPO,A<:MabsAlg}
    mpo::M
    alg::A
end

"""
    BMPO(mpo::ITensorMPS.MPO, alg::Truncated)

Create a `BMPO`` from an existing `MPO` using the Truncated algorithm.

Arguments:
- mpo::ITensorMPS.MPO: Input matrix product operator
- alg::Truncated: Algorithm specification

Returns:
- BMPO: Wrapped bosonic MPO
"""
function BMPO(mpo::ITensorMPS.MPO, alg::Truncated)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

"""
    BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::Truncated)

Create a `BMPO` directly from an OpSum and sites using the Truncated algorithm.

Arguments:
- opsum::ITensors.OpSum: Operator sum specification
- sites::Vector{<:ITensors.Index}: Vector of site indices  
- alg::Truncated: Algorithm specification

Returns:
- BMPO: Bosonic MPO constructed from OpSum
"""
function BMPO(
    opsum::ITensors.OpSum, 
    sites::Vector{<:ITensors.Index}, 
    alg::Truncated
)
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

ITensorMPS.siteinds(bmpo::BMPO) = ITensorMPS.siteinds(bmpo.mpo)
ITensorMPS.maxlinkdim(bmpo::BMPO) = ITensorMPS.maxlinkdim(bmpo.mpo)
Base.eltype(bmpo::BMPO) = eltype(bmpo.mpo)
Base.length(bmpo::BMPO) = length(bmpo.mpo)
Base.copy(bmpo::BMPO) = BMPO(copy(bmpo.mpo), bmpo.alg)
Base.deepcopy(bmpo::BMPO) = BMPO(deepcopy(bmpo.mpo), bmpo.alg)
ITensorMPS.linkind(bmpo::BMPO, i::Int) = ITensorMPS.linkind(bmpo.mpo, i)
ITensorMPS.siteind(bmpo::BMPO, i::Int) = ITensorMPS.siteind(bmpo.mpo, i)
Base.iterate(bmpo::BMPO) = Base.iterate(bmpo.mpo)
Base.iterate(bmpo::BMPO, state) = Base.iterate(bmpo.mpo, state)
Base.eachindex(bmpo::BMPO) = Base.eachindex(bmpo.mpo)
Base.getindex(bmpo::BMPO, i) = bmpo.mpo[i]
Base.setindex!(bmpo::BMPO, val, i) = (bmpo.mpo[i] = val)
Base.firstindex(bmpo::BMPO) = Base.firstindex(bmpo.mpo)
Base.lastindex(bmpo::BMPO) = Base.lastindex(bmpo.mpo)

for f in [
    :(ITensorMPS.findsite),
    :(ITensorMPS.findsites),
    :(ITensorMPS.firstsiteinds),
    :(ITensorMPS.expect),
    :(ITensorMPS.inner),
    :(LinearAlgebra.dot),
    :(ITensorMPS.loginner),
    :(ITensorMPS.logdot),
    :(LinearAlgebra.norm),
    :(ITensorMPS.lognorm),
    :(Base.collect),
    :(Base.length),
    :(Base.size)
]
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}) = ($f)(bmpo.mpo)
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, args...; kwargs...) = ($f)(bmpo.mpo, args...; kwargs...)
end
for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag),
]
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}) = BMPO(($f)(bmpo.mpo), bmpo.alg)
end

"""
    truncate(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Create a truncated copy of the `BMPO`.

Arguments:
- bmpo::BMPO: Input bosonic MPO

Keyword Arguments:
- kwargs...: Truncation parameters passed to `ITensorMPS.truncate`
  (e.g., `maxdim`, `cutoff`, `alg`)

Returns:
- BMPO: Truncated bosonic MPO
"""
function ITensorMPS.truncate(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    truncated_mpo = ITensorMPS.truncate(bmpo.mpo; kwargs...)
    return BMPO(truncated_mpo, bmpo.alg)
end

"""
    truncate!(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Truncate the `BMPO` in place.

Arguments:
- bmpo::BMPO: Bosonic MPO to truncate

Keyword Arguments:
- kwargs...: Truncation parameters passed to ITensorMPS.truncate!
  (e.g., `maxdim`, `cutoff`, `alg`)

Returns:
- BMPO: The truncated BMPO (same object, modified in place)
"""
function ITensorMPS.truncate!(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)
    ITensorMPS.truncate!(bmpo.mpo; kwargs...)
    return bmpo
end

"""
    +(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Add two `BMPO` objects with optional truncation. Also can be called with `add`.

Arguments:
- bmpo1::BMPO: First bosonic MPO
- bmpo2::BMPO: Second bosonic MPO

Keyword Arguments:
- kwargs...: Truncation parameters passed to ITensorMPS.truncate
  (e.g., `maxdim`, `cutoff`, `alg`)

Returns:
- BMPO: Sum of the two bosonic MPO
"""
function Base.:(+)(
    bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; 
    kwargs...
)
    return ITensorMPS.add(bmpo1, bmpo2; kwargs...)
end
function ITensorMPS.add(
    bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; 
    kwargs...
)
    result_mpo = ITensorMPS.add(bmpo1.mpo, bmpo2.mpo; kwargs...)
    return BMPO(result_mpo, bmpo1.alg)
end

"""
    contract(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Contract a `BMPO` with a `BMPS` with optional truncation control.

Arguments:
- bmpo::BMPO: Bosonic MPO
- bmps::BMPS: Bosonic MPS

Keyword Arguments:
- kwargs...: Contraction and truncation parameters passed to `ITensors.contract`
  (e.g., `maxdim`, `cutoff`, `alg`)

Returns:
- BMPS: Result of MPO-MPS contraction
"""
function ITensors.contract(
    bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, 
    bmps::BMPS{<:ITensorMPS.MPS,Truncated}; 
    kwargs...
)
    result_mps = ITensors.contract(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

"""
    apply(bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Apply a `BMPO` to a `BMPS` with optional truncation control.

Arguments:
- bmpo::BMPO: Bosonic MPO to apply
- bmps::BMPS: Bosonic MPS to apply to

Keyword Arguments:
- kwargs...: Parameters passed to `ITensors.apply`
  (e.g., `maxdim`, `cutoff`, `alg`)

Returns:
- BMPS: Result of applying MPO to MPS
"""
function ITensors.apply(
    bmpo::BMPO{<:ITensorMPS.MPO,Truncated}, 
    bmps::BMPS{<:ITensorMPS.MPS,Truncated}; 
    kwargs...
)
    result_mps = ITensors.apply(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

"""
    outer(bmpo1::BMPS{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPS{<:ITensorMPS.MPO,Truncated}; kwargs...)

Compute outer product of two `BMPO` objects.

Arguments:
- bmpo1::BMPO: First bosonic MPO
- bmpo2::BMPO: Second bosonic MPO

Keyword Arguments:
- kwargs...: Additional parameters passed to `ITensorMPS.outer`

Returns:
- BMPO: Outer product result as a bosonic MPO
"""
function ITensorMPS.outer(
    bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; 
    kwargs...
)
    outer_result = ITensorMPS.outer(bmpo1.mpo, bmpo2.mpo; kwargs...)
    return BMPO(outer_result, bmpo1.alg)
end

"""
    dot(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Compute dot product of two `BMPO` objects.

Arguments:
- bmpo1::BMPO: First bosonic MPO
- bmpo2::BMPO: Second bosonic MPO

Keyword Arguments:
- kwargs...: Additional parameters passed to `LinearAlgebra.dot`

Returns:
- Scalar
"""
function LinearAlgebra.dot(
    bmpo1::BMPS{<:ITensorMPS.MPO,Truncated}, 
    bmpo2::BMPS{<:ITensorMPS.MPO,Truncated}; 
    kwargs...
)
    return LinearAlgebra.dot(bmpo1.mpo, bmpo2.mpo; kwargs...)
end

"""
    inner(bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; kwargs...)

Compute inner product of two `BMPO` objects.

Arguments:
- bmpo1::BMPO: First bosonic MPO
- bmpo2::BMPO: Second bosonic MPO

Keyword Arguments:
- kwargs...: Additional parameters passed to `ITensorMPS.inner`

Returns:
- Scalar
"""
function ITensorMPS.inner(
    bmpo1::BMPO{<:ITensorMPS.MPO,Truncated}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,Truncated}; 
    kwargs...
)
    return ITensorMPS.inner(bmpo1.mpo, bmpo2.mpo; kwargs...)
end

"""
    BMPO(mpo::ITensorMPS.MPO, alg::PseudoSite)

Create BMPO from existing MPO using PseudoSite algorithm.
"""
function BMPO(mpo::ITensorMPS.MPO, alg::PseudoSite)
    length(mpo) == length(alg.sites) || 
        throw(ArgumentError("MPO length doesn't match PseudoSite sites"))
    
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

"""
    BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::PseudoSite)

Create BMPO from OpSum using PseudoSite algorithm.

NOTE: Only supports simple single-mode operators (N, Id).
Multi-site operators (Adag, A) are not yet supported via OpSum.
Use explicit operator construction instead.
"""
function BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::PseudoSite)
    sites == alg.sites || 
        throw(ArgumentError("Sites must match algorithm specification"))
    quantics_opsum = _convert_opsum_to_quantics(opsum, alg)
    mpo = ITensorMPS.MPO(quantics_opsum, sites)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

"""
    _convert_opsum_to_quantics(opsum::ITensors.OpSum, alg::PseudoSite)

Convert bosonic OpSum to quantics OpSum.
Currently supports: N (number), Id (identity)
"""
function _convert_opsum_to_quantics(opsum::ITensors.OpSum, alg::PseudoSite)
    quantics_opsum = ITensors.OpSum()
    opsum_terms = opsum.data
    for term in opsum_terms
        coeff = term.coef
        ops = term.ops
        sites_in_term = term.sites
        for (op_name, site_idx) in zip(ops, sites_in_term)
            if op_name == "N"
                cluster_sites = get_mode_cluster(alg, site_idx)
                for (i, _) in enumerate(cluster_sites)
                    weight = coeff * 2^(i-1)
                    global_qubit_idx = (site_idx - 1) * alg.n_qubits_per_mode + i
                    quantics_opsum += weight, "N", global_qubit_idx
                end
            elseif op_name == "Id"
                continue
            else
                error("Operator '$op_name' not supported in PseudoSite OpSum. " *
                      "Supported: N, Id. " *
                      "For other operators, construct them explicitly using quantics functions.")
            end
        end
    end
    
    return quantics_opsum
end

for f in [
    :(ITensorMPS.findsite),
    :(ITensorMPS.findsites),
    :(ITensorMPS.firstsiteinds),
    :(ITensorMPS.expect),
    :(ITensorMPS.inner),
    :(LinearAlgebra.dot),
    :(ITensorMPS.loginner),
    :(ITensorMPS.logdot),
    :(LinearAlgebra.norm),
    :(ITensorMPS.lognorm),
    :(Base.collect),
    :(Base.length),
    :(Base.size)
]
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,<:PseudoSite}) = ($f)(bmpo.mpo)
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, args...; kwargs...) = 
        ($f)(bmpo.mpo, args...; kwargs...)
end

for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag)
]
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,<:PseudoSite}) = BMPO(($f)(bmpo.mpo), bmpo.alg)
end

function ITensorMPS.truncate(bmpo::BMPO{<:ITensorMPS.MPO,<:PseudoSite}; kwargs...)
    truncated_mpo = ITensorMPS.truncate(bmpo.mpo; kwargs...)
    return BMPO(truncated_mpo, bmpo.alg)
end

function ITensorMPS.truncate!(bmpo::BMPO{<:ITensorMPS.MPO,<:PseudoSite}; kwargs...)
    ITensorMPS.truncate!(bmpo.mpo; kwargs...)
    return bmpo
end

function Base.:(+)(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:PseudoSite}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    return ITensorMPS.add(bmpo1, bmpo2; kwargs...)
end

function ITensorMPS.add(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:PseudoSite}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    result_mpo = ITensorMPS.add(bmpo1.mpo, bmpo2.mpo; kwargs...)
    return BMPO(result_mpo, bmpo1.alg)
end

function ITensors.contract(
    bmpo::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, 
    bmps::BMPS{<:ITensorMPS.MPS,PseudoSite}; 
    kwargs...
)
    bmpo.alg == bmps.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    result_mps = ITensors.contract(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

function ITensors.apply(
    bmpo::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, 
    bmps::BMPS{<:ITensorMPS.MPS,PseudoSite}; 
    kwargs...
)
    bmpo.alg == bmps.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    result_mps = ITensors.apply(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

function ITensorMPS.outer(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:PseudoSite}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:PseudoSite}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    outer_result = ITensorMPS.outer(bmpo1.mpo, bmpo2.mpo; kwargs...)
    return BMPO(outer_result, bmpo1.alg)
end