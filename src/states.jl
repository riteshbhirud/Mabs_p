"""
    random_bmps(sites::Vector{<:ITensors.Index}, alg::Truncated; kwargs...)

Create a random bosonic MPS using the `Truncated` algorithm.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Random bosonic MPS
"""
function random_bmps(sites::Vector{<:ITensors.Index}, alg::Truncated; linkdims = 1)
    mps = ITensorMPS.random_mps(sites; linkdims)
    return BMPS(mps, alg)
end

"""
    vacuumstate(sites::Vector{<:ITensors.Index}, alg::Truncated)

Create a vacuum state |0,0,...,0⟩ BMPS.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Vacuum state bosonic MPS
"""
function vacuumstate(sites::Vector{<:ITensors.Index}, alg::Truncated)
    states = fill(1, length(sites))
    return BMPS(sites, states, alg)  
end

"""
    coherentstate(sites::Vector{<:ITensors.Index}, α::Number, alg::Truncated)
    coherentstate(sites::Vector{<:ITensors.Index}, αs::Vector{<:Number}, alg::Truncated)

Create an approximate coherent state `BMPS` using truncated expansion.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- α::Number: Single coherent state amplitude (applied to all modes)
- αs::Vector{<:Number}: Vector of coherent state amplitudes (one per mode)
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Coherent state bosonic MPS (approximated by truncation)
"""
function coherentstate(sites::Vector{<:ITensors.Index}, α::Number, alg::Truncated)
    αs = fill(α, length(sites))
    return coherentstate(sites, αs, alg)
end

function coherentstate(sites::Vector{<:ITensors.Index}, αs::Vector{<:Number}, alg::Truncated)
    N = length(sites)
    length(αs) == N || error("Number of amplitudes ($(length(αs))) must match number of sites ($N)")
    tensors = Vector{ITensors.ITensor}(undef, N)
    @inbounds for (i, site) in enumerate(sites)
        α = αs[i]
        max_occ = ITensors.dim(site) - 1
        coeffs = Vector{ComplexF64}(undef, max_occ+1)
        normalization = exp(-abs2(α)/2)
        @inbounds for n in 0:max_occ
            coeff = normalization * (α^n) / sqrt(_safe_factorial(n))
            coeffs[n+1] = convert(ComplexF64, coeff)
        end
        norm_factor = sqrt(sum(abs2, coeffs))
        coeffs ./= norm_factor
        if i == 1
            if N == 1
                tensor = ITensors.ITensor(ComplexF64, site)
                @inbounds for n in 0:max_occ
                    tensor[n+1] = coeffs[n+1]
                end
            else
                right_link = ITensors.Index(1, "Link,l=$i")
                tensor = ITensors.ITensor(ComplexF64, site, right_link)
                @inbounds for n in 0:max_occ
                    tensor[n+1, 1] = coeffs[n+1]
                end
            end
        elseif i == N
            left_link = ITensors.Index(1, "Link,l=$(i-1)")
            tensor = ITensors.ITensor(ComplexF64, left_link, site)
            @inbounds for n in 0:max_occ
                tensor[1, n+1] = coeffs[n+1]
            end
        else
            left_link = ITensors.Index(1, "Link,l=$(i-1)")
            right_link = ITensors.Index(1, "Link,l=$i")
            tensor = ITensors.ITensor(ComplexF64, left_link, site, right_link)
            @inbounds for n in 0:max_occ
                tensor[1, n+1, 1] = coeffs[n+1]
            end
        end
        tensors[i] = tensor
    end
    mps = ITensorMPS.MPS(tensors)
    return BMPS(mps, alg)
end

"""
    random_bmps(sites::Vector{<:ITensors.Index}, alg::PseudoSite; linkdims=1)

Create random BMPS using PseudoSite algorithm.
"""
function random_bmps(sites::Vector{<:ITensors.Index}, alg::PseudoSite; linkdims=1)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    mps = ITensorMPS.random_mps(sites; linkdims=linkdims)
    return BMPS(mps, alg)
end

"""
    vacuumstate(sites::Vector{<:ITensors.Index}, alg::PseudoSite)

Create vacuum state |0,0,...,0⟩ in PseudoSite representation.
Each mode is represented by all qubits in |0⟩ state.
"""
function vacuumstate(sites::Vector{<:ITensors.Index}, alg::PseudoSite)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    states = fill(1, length(sites))
    mps = ITensorMPS.productMPS(sites, states)
    
    return BMPS(mps, alg)
end

"""
    coherentstate(sites::Vector{<:ITensors.Index}, α::Number, alg::PseudoSite)

Create coherent state in PseudoSite representation (single amplitude for all modes).
"""
function coherentstate(sites::Vector{<:ITensors.Index}, α::Number, alg::PseudoSite)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    αs = fill(α, alg.n_modes)
    return coherentstate(sites, αs, alg)
end

"""
    coherentstate(sites::Vector{<:ITensors.Index}, αs::Vector{<:Number}, alg::PseudoSite)

Create multi-mode coherent state in PseudoSite representation.
Uses optimized MPS construction with proper gauge fixing.
"""
function coherentstate(sites::Vector{<:ITensors.Index}, αs::Vector{<:Number}, alg::PseudoSite)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    length(αs) == alg.n_modes || 
        throw(ArgumentError("Number of amplitudes must match modes"))
    
    if alg.fock_cutoff <= 15
        return _coherent_state_direct_sum(sites, αs, alg)
    else
        return _coherent_state_via_displacement(sites, αs, alg)
    end
end

"""
Direct summation approach for coherent states - OPTIMIZED VERSION
"""
function _coherent_state_direct_sum(sites::Vector{<:ITensors.Index},
                                     αs::Vector{<:Number},
                                     alg::PseudoSite)
    n_modes = alg.n_modes
    n_qubits = n_qubits_per_mode(alg)
    max_occ = alg.fock_cutoff
    
    all_fock_coeffs = [_coherent_fock_coefficients_raw(α, max_occ) for α in αs]
    max_coeff = maximum(maximum(abs.(coeffs)) for coeffs in all_fock_coeffs)
    cutoff = 1e-12 * max_coeff
    
    fock_states_list = Vector{Vector{Int}}()
    coefficients_list = Vector{ComplexF64}()
    
    function enumerate_states!(current_state::Vector{Int}, mode_idx::Int)
        if mode_idx > n_modes
            coeff = ComplexF64(1.0)
            for m in 1:n_modes
                coeff *= all_fock_coeffs[m][current_state[m] + 1]
            end
            
            if abs(coeff) >= cutoff
                push!(fock_states_list, copy(current_state))
                push!(coefficients_list, coeff)
            end
            return
        end
        
        for n in 0:max_occ
            if abs(all_fock_coeffs[mode_idx][n + 1]) >= cutoff
                current_state[mode_idx] = n
                enumerate_states!(current_state, mode_idx + 1)
            end
        end
    end
    
    current_state = zeros(Int, n_modes)
    enumerate_states!(current_state, 1)
    
    if isempty(fock_states_list)
        error("No significant Fock states found")
    end
    
    norm_factor = sqrt(sum(abs2, coefficients_list))
    coefficients_list ./= norm_factor
    
    result_mps = nothing
    for (idx, (fock_state, coeff)) in enumerate(zip(fock_states_list, coefficients_list))
        binary_states = Int[]
        for mode in 1:n_modes
            append!(binary_states, decimal_to_binary_state(fock_state[mode], n_qubits))
        end
        
        term_mps = ITensorMPS.productMPS(sites, binary_states)
        term_norm_before = ITensorMPS.norm(term_mps)
        term_mps[1] *= coeff
        term_norm_after = ITensorMPS.norm(term_mps)
        
        if result_mps === nothing
            result_mps = term_mps
        else
            prev_norm = ITensorMPS.norm(result_mps)
            result_mps = ITensorMPS.add(result_mps, term_mps; cutoff=1e-15)
            new_norm = ITensorMPS.norm(result_mps)
        end
    end
    
    final_norm_before = ITensorMPS.norm(result_mps)
    inner_val = ITensorMPS.inner(result_mps, result_mps)
    ITensorMPS.normalize!(result_mps)
    norm_after_normalize = ITensorMPS.norm(result_mps)
    final_norm = ITensorMPS.norm(result_mps)
    inner_final = ITensorMPS.inner(result_mps, result_mps)
    
    bmps = BMPS(result_mps, alg)
    LinearAlgebra.normalize!(bmps)
    
    return bmps
end

"""
    _coherent_fock_coefficients_raw(α::Number, max_occ::Int)

Compute RAW (unnormalized) Fock state expansion coefficients for coherent state |α⟩.
Returns: cₙ = exp(-|α|²/2) × α^n / √(n!)

These are NOT individually normalized - the normalization happens after 
summing all product state contributions.
"""
function _coherent_fock_coefficients_raw(α::Number, max_occ::Int)
    coeffs = zeros(ComplexF64, max_occ + 1)
    exp_factor = exp(-abs2(α) / 2)
    
    for n in 0:max_occ
        coeffs[n+1] = exp_factor * (α^n) / sqrt(_safe_factorial(n))
    end    
    return coeffs
end

"""
Displacement operator approach for coherent states.
More efficient for large Hilbert spaces.
"""
function _coherent_state_via_displacement(sites::Vector{<:ITensors.Index},
                                          αs::Vector{<:Number},
                                          alg::PseudoSite)
    psi = vacuumstate(sites, alg)
    for mode_idx in 1:alg.n_modes
        α = αs[mode_idx]
        if abs(α) > 1e-10  
            cluster_sites = get_mode_cluster(sites, alg, mode_idx)
            D = displace_op_quantics(cluster_sites, α)
            psi.mps = ITensors.apply(D, psi.mps; cutoff=1e-12, maxdim=256)
        end
    end    
    LinearAlgebra.normalize!(psi)
    return psi
end

"""
    _state_vector_to_mps(state_vector::Vector{ComplexF64}, 
                        sites::Vector{ITensors.Index})

Convert full state vector to MPS via tensor train decomposition.
Properly normalized by construction through SVD.
"""
function _state_vector_to_mps(state_vector::Vector{ComplexF64}, 
                              sites::Vector{ITensors.Index})
    n_sites = length(sites)
    total_dim = 2^n_sites
    
    if length(state_vector) != total_dim
        error("State vector size mismatch: got $(length(state_vector)), expected $total_dim")
    end
    norm_val = sqrt(sum(abs2, state_vector))
    if abs(norm_val - 1.0) > 1e-10
        @warn "State vector not normalized: norm = $norm_val"
        state_vector = state_vector / norm_val
    end
    if n_sites == 1
        tensor = ITensors.ITensor(ComplexF64, sites[1])
        tensor[sites[1] => 1] = state_vector[1]
        tensor[sites[1] => 2] = state_vector[2]
        return ITensorMPS.MPS([tensor])
    end
    tensors = Vector{ITensors.ITensor}(undef, n_sites)
    current_data = copy(state_vector)
    left_dim = 1
    prev_right_link::Union{Nothing, ITensors.Index} = nothing
    for site_idx in 1:n_sites
        site = sites[site_idx]
        if site_idx == n_sites
            tensor = ITensors.ITensor(ComplexF64, prev_right_link, site)
            for l in 1:left_dim
                tensor[prev_right_link => l, site => 1] = current_data[2*(l-1) + 1]
                tensor[prev_right_link => l, site => 2] = current_data[2*(l-1) + 2]
            end
            tensors[site_idx] = tensor
        else
            n_remaining_sites = n_sites - site_idx
            right_dim = 2^n_remaining_sites
            row_dim = left_dim * 2
            matrix = reshape(current_data, row_dim, right_dim)
            F = LinearAlgebra.svd(matrix)
            cutoff = 1e-14
            max_bond = 512
            n_keep = count(s -> s > cutoff * F.S[1], F.S)
            n_keep = max(1, min(n_keep, max_bond, length(F.S)))
            U_trunc = F.U[:, 1:n_keep]
            S_trunc = F.S[1:n_keep]
            V_trunc = F.Vt[1:n_keep, :]
            right_link = ITensors.Index(n_keep, "Link,l=$site_idx")
            if site_idx == 1
                tensor = ITensors.ITensor(ComplexF64, site, right_link)
                for s in 1:2, r in 1:n_keep
                    tensor[site => s, right_link => r] = U_trunc[s, r]
                end
            else
                tensor = ITensors.ITensor(ComplexF64, prev_right_link, site, right_link)
                row_idx = 1
                for l in 1:left_dim, s in 1:2
                    for r in 1:n_keep
                        tensor[prev_right_link => l, site => s, right_link => r] = U_trunc[row_idx, r]
                    end
                    row_idx += 1
                end
            end
            tensors[site_idx] = tensor
            
            prev_right_link = right_link
            current_data = vec(LinearAlgebra.Diagonal(S_trunc) * V_trunc)
            left_dim = n_keep
        end
    end
    mps = ITensorMPS.MPS(tensors)
    mps_norm = ITensorMPS.norm(mps)
    if abs(mps_norm - 1.0) > 1e-10
        ITensorMPS.normalize!(mps)
    end
    return mps
end

"""
    _coherent_fock_coefficients(α::Number, max_occ::Int)

Compute Fock state expansion coefficients for coherent state |α⟩.
Returns normalized coefficients for direct use.
"""
function _coherent_fock_coefficients(α::Number, max_occ::Int)
    coeffs = zeros(ComplexF64, max_occ + 1)
    exp_factor = exp(-abs2(α) / 2)
    for n in 0:max_occ
        coeffs[n+1] = exp_factor * (α^n) / sqrt(_safe_factorial(n))
    end
    norm_factor = sqrt(sum(abs2, coeffs))
    coeffs ./= norm_factor
    return coeffs
end

"""
    _fill_product_state!(state_vector::Vector{ComplexF64}, 
                        all_fock_coeffs::Vector{Vector{ComplexF64}},
                        alg::PseudoSite)

Fill product state vector with Fock state superposition.
Only fills non-negligible coefficients for efficiency.
"""
function _fill_product_state!(state_vector::Vector{ComplexF64},
                               all_fock_coeffs::Vector{Vector{ComplexF64}},
                              alg::PseudoSite)
    n_modes = alg.n_modes
    max_occ = alg.fock_cutoff
    n_qubits = n_qubits_per_mode(alg)
    
    max_coeff = maximum(maximum(abs.(coeffs)) for coeffs in all_fock_coeffs)
    cutoff = 1e-15 * max_coeff
    
    fock_states = zeros(Int, n_modes)
    _recursive_fill!(state_vector, all_fock_coeffs, alg, fock_states, 1, cutoff)
end

"""
    _recursive_fill! with cutoff

Recursively fill state vector, skipping negligible terms.
"""
function _recursive_fill!(state_vector::Vector{ComplexF64},
                         all_fock_coeffs::Vector{Vector{ComplexF64}},
                         alg::PseudoSite,
                         fock_states::Vector{Int},
                         mode_idx::Int,
                         cutoff::Float64)
    n_modes = alg.n_modes
    max_occ = alg.fock_cutoff
    n_qubits_per_mode = n_qubits_per_mode(alg)
    
    if mode_idx > n_modes
        coeff = ComplexF64(1.0)
        for m in 1:n_modes
            coeff *= all_fock_coeffs[m][fock_states[m] + 1]
        end
        if abs(coeff) < cutoff
            return
        end
        binary_state = Int[]
        for m in 1:n_modes
            append!(binary_state, decimal_to_binary_state(fock_states[m], n_qubits_per_mode))
        end
        linear_idx = 1
        for i in 1:length(binary_state)
            bit_val = binary_state[i] - 1  
            linear_idx += bit_val * (1 << (i-1))  
        end
        state_vector[linear_idx] = coeff
        return
    end
    
    for n in 0:max_occ
        if abs(all_fock_coeffs[mode_idx][n+1]) < cutoff
            continue
        end
        fock_states[mode_idx] = n
        _recursive_fill!(state_vector, all_fock_coeffs, alg, fock_states, mode_idx + 1, cutoff)
    end
end