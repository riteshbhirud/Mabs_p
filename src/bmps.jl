"""
Bosonic Matrix Product State wrapper that supports different algorithms.
Contains an underlying `ITensorMPS.MPS` object and algorithm specification.

Fields:
- mps::M: The underlying `ITensorMPS.MPS` object
- alg::A: Algorithm specification (`Truncated`, `PseudoSite`, or `LocalBasis`)
"""
mutable struct BMPS{M<:ITensorMPS.MPS,A<:MabsAlg}
    mps::M
    alg::A
end

"""
    BMPS(mps::ITensorMPS.MPS, alg::Truncated)

Create a `BMPS` from an existing `MPS` using the Truncated algorithm.

Arguments:
- mps::ITensorMPS.MPS: Input matrix product state
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Wrapped bosonic MPS
"""
function BMPS(mps::ITensorMPS.MPS, alg::Truncated)
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

function BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::Truncated)
    mps = ITensorMPS.productMPS(sites, states)
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

ITensorMPS.siteinds(bmps::BMPS) = ITensorMPS.siteinds(bmps.mps)
ITensorMPS.maxlinkdim(bmps::BMPS) = ITensorMPS.maxlinkdim(bmps.mps)
ITensorMPS.linkind(bmps::BMPS, i::Int) = ITensorMPS.linkind(bmps.mps, i)
ITensorMPS.siteind(bmps::BMPS, i::Int) = ITensorMPS.siteind(bmps.mps, i)
Base.eltype(bmps::BMPS) = eltype(bmps.mps[1])  
Base.length(bmps::BMPS) = length(bmps.mps)

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
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}) = ($f)(bmps.mps)
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}, args...; kwargs...) = ($f)(bmps.mps, args...; kwargs...)
end
for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}) = BMPS(($f)(bmps.mps), bmps.alg)
end


Base.copy(bmps::BMPS) = BMPS(copy(bmps.mps), bmps.alg)
Base.deepcopy(bmps::BMPS) = BMPS(deepcopy(bmps.mps), bmps.alg)
Base.iterate(bmps::BMPS) = Base.iterate(bmps.mps)
Base.iterate(bmps::BMPS, state) = Base.iterate(bmps.mps, state)
Base.eachindex(bmps::BMPS) = Base.eachindex(bmps.mps)
Base.getindex(bmps::BMPS, i) = bmps.mps[i]
Base.setindex!(bmps::BMPS, val, i) = (bmps.mps[i] = val)
Base.firstindex(bmps::BMPS) = Base.firstindex(bmps.mps)
Base.lastindex(bmps::BMPS) = Base.lastindex(bmps.mps)

"""
    normalize[!](bmps::BMPS{<:ITensorMPS.MPS,Truncated}; (lognorm!)=[])

Normalize the `BMPS` in place such that `norm(bmps) ≈ 1`.

Arguments:
- bmps::BMPS: Bosonic MPS to normalize

Keyword Arguments:
- lognorm!=[]: Mutable vector to store the log norm. Pass an empty vector 
  that will be filled with the log norm value.

Returns:
- BMPS: The normalized BMPS (same object, modified in place)
```
"""
function LinearAlgebra.normalize!(bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    LinearAlgebra.normalize!(bmps.mps; kwargs...)
    return bmps
end
function LinearAlgebra.normalize(bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    normalized_mps = LinearAlgebra.normalize(bmps.mps; kwargs...)
    return BMPS(normalized_mps, bmps.alg)
end

"""
    orthogonalize[!](bmps::BMPS{<:ITensorMPS.MPS,Truncated}, j::Int)

Orthogonalize the `BMPS`` to site j.

Arguments:
- bmps::BMPS: Bosonic MPS to orthogonalize
- j::Int: Site to orthogonalize to

Returns:
- BMPS: The orthogonalized BMPS (same object, modified in place)
"""
function ITensorMPS.orthogonalize!(
    bmps::BMPS{<:ITensorMPS.MPS,Truncated}, 
    j::Int; 
    kwargs...
)
    ITensorMPS.orthogonalize!(bmps.mps, j; kwargs...)
    return bmps
end
function ITensorMPS.orthogonalize(
    bmps::BMPS{<:ITensorMPS.MPS,Truncated}, 
    j::Int; 
    kwargs...
)
    orthog_mps = ITensorMPS.orthogonalize(bmps.mps, j; kwargs...)
    return BMPS(orthog_mps, bmps.alg)
end

"""
    truncate[!](bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Create a truncated copy of the `BMPS`.

Arguments:
- bmps::BMPS: Input bosonic MPS

Keyword Arguments:
- kwargs...: Truncation parameters passed to `ITensorMPS.truncate` 
  (e.g., `maxdim`, `cutoff`, `alg`)

Returns:
- BMPS: Truncated bosonic MPS
"""
function ITensorMPS.truncate(bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    truncated_mps = ITensorMPS.truncate(bmps.mps; kwargs...)
    return BMPS(truncated_mps, bmps.alg)
end
function ITensorMPS.truncate!(bmps::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)
    ITensorMPS.truncate!(bmps.mps; kwargs...)
    return bmps
end

"""
    +(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Add two BMPS objects with optional truncation. Can also use `add`.

Arguments:
- bmps1::BMPS: First bosonic MPS
- bmps2::BMPS: Second bosonic MPS

Keyword Arguments:
- kwargs...: Truncation parameters passed to ITensorMPS.truncate (e.g., `maxdim`, `cutoff`)

Returns:
- BMPS: Sum of the two bosonic MPS
"""
function Base.:(+)(
    bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, 
    bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; 
    kwargs...
)
    result_mps = Base.:(+)(bmps1.mps, bmps2.mps; kwargs...)
    return BMPS(result_mps, bmps1.alg)
end
function ITensorMPS.add(
    bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, 
    bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; 
    kwargs...
)
    return Base.:(+)(bmps1, bmps2; kwargs...)
end

"""
    outer(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Compute outer product of two `BMPS` objects.

Arguments:
- bmps1::BMPS: First bosonic MPS
- bmps2::BMPS: Second bosonic MPS

Keyword Arguments:
- kwargs...: Additional parameters passed to `ITensorMPS.outer`

Returns:
- BMPO: Outer product result as a bosonic MPO
"""
function ITensorMPS.outer(
    bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, 
    bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; 
    kwargs...
)
    outer_result = ITensorMPS.outer(bmps1.mps, bmps2.mps; kwargs...)
    return BMPO(outer_result, bmps1.alg)
end

"""
    dot(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Compute dot product of two `BMPS` objects.

Arguments:
- bmps1::BMPS: First bosonic MPS
- bmps2::BMPS: Second bosonic MPS

Keyword Arguments:
- kwargs...: Additional parameters passed to LinearAlgebra.dot

Returns:
- Scalar: Dot product ⟨bmps1|bmps2⟩
"""
function LinearAlgebra.dot(
    bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, 
    bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; 
    kwargs...
)
    return LinearAlgebra.dot(bmps1.mps, bmps2.mps; kwargs...)
end

"""
    inner(bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; kwargs...)

Compute inner product of two `BMPS` objects.

Arguments:
- bmps1::BMPS: First bosonic MPS
- bmps2::BMPS: Second bosonic MPS

Keyword Arguments:
- kwargs...: Additional parameters passed to `ITensorMPS.inner`

Returns:
- Scalar: Inner product ⟨bmps1|bmps2⟩
"""
function ITensorMPS.inner(
    bmps1::BMPS{<:ITensorMPS.MPS,Truncated}, 
    bmps2::BMPS{<:ITensorMPS.MPS,Truncated}; 
    kwargs...
)
    return ITensorMPS.inner(bmps1.mps, bmps2.mps; kwargs...)
end

"""
    BMPS(mps::ITensorMPS.MPS, alg::PseudoSite)

Create BMPS from existing MPS using PseudoSite algorithm.
"""
function BMPS(mps::ITensorMPS.MPS, alg::PseudoSite)
    length(mps) == length(alg.sites) || 
        throw(ArgumentError("MPS length $(length(mps)) doesn't match PseudoSite sites $(length(alg.sites))"))
    
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

"""
    BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::PseudoSite)

Create product state BMPS in PseudoSite representation.

Arguments:
- sites::Vector{ITensors.Index}: Should be alg.sites
- states::Vector: Bosonic occupation numbers for each mode
- alg::PseudoSite: Algorithm specification

Returns:
- BMPS: Product state in quantics representation
"""
function BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::PseudoSite)
    sites == alg.sites || 
        throw(ArgumentError("Sites must match algorithm specification"))
    
    length(states) == alg.n_modes || 
        throw(ArgumentError("Number of states $(length(states)) must match modes $(alg.n_modes))"))
    
    binary_states = Int[]
    for (mode_idx, n) in enumerate(states)
        n <= alg.original_max_occ || 
            throw(ArgumentError("State $n exceeds maximum $(alg.original_max_occ)"))
        
        binary_state = decimal_to_binary_state(n, alg.n_qubits_per_mode)
        append!(binary_states, binary_state)
    end
    
    mps = ITensorMPS.productMPS(sites, binary_states)
    
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
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
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}) = ($f)(bmps.mps)
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, args...; kwargs...) = 
        ($f)(bmps.mps, args...; kwargs...)
end

for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}) = BMPS(($f)(bmps.mps), bmps.alg)
end

function LinearAlgebra.normalize!(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; kwargs...)
    LinearAlgebra.normalize!(bmps.mps; kwargs...)
    return bmps
end

function LinearAlgebra.normalize(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; kwargs...)
    normalized_mps = LinearAlgebra.normalize(bmps.mps; kwargs...)
    return BMPS(normalized_mps, bmps.alg)
end

function ITensorMPS.orthogonalize!(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, j::Int; kwargs...)
    ITensorMPS.orthogonalize!(bmps.mps, j; kwargs...)
    return bmps
end

function ITensorMPS.orthogonalize(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, j::Int; kwargs...)
    orthog_mps = ITensorMPS.orthogonalize(bmps.mps, j; kwargs...)
    return BMPS(orthog_mps, bmps.alg)
end

function ITensorMPS.truncate(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; kwargs...)
    truncated_mps = ITensorMPS.truncate(bmps.mps; kwargs...)
    return BMPS(truncated_mps, bmps.alg)
end

function ITensorMPS.truncate!(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; kwargs...)
    ITensorMPS.truncate!(bmps.mps; kwargs...)
    return bmps
end

function Base.:(+)(
    bmps1::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    result_mps = Base.:(+)(bmps1.mps, bmps2.mps; kwargs...)
    return BMPS(result_mps, bmps1.alg)
end

function ITensorMPS.add(
    bmps1::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; 
    kwargs...
)
    return Base.:(+)(bmps1, bmps2; kwargs...)
end

function ITensorMPS.outer(
    bmps1::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    outer_result = ITensorMPS.outer(bmps1.mps, bmps2.mps; kwargs...)
    return BMPO(outer_result, bmps1.alg)
end

function LinearAlgebra.dot(
    bmps1::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    return LinearAlgebra.dot(bmps1.mps, bmps2.mps; kwargs...)
end

function ITensorMPS.inner(
    bmps1::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:PseudoSite}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError("PseudoSite algorithms must match"))
    return ITensorMPS.inner(bmps1.mps, bmps2.mps; kwargs...)
end