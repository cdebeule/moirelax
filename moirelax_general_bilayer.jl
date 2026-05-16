using DelimitedFiles
using ZChop
using LinearAlgebra
using StaticArrays
using FastGaussQuadrature
using UnPack
using Printf
using TimerOutputs
using Polyester

@views function makechunks(X::AbstractVector, n::Integer)
    c = length(X) ÷ n
    return [X[(1+c*k):(k == n - 1 ? end : c * k + c)] for k in 0:(n-1)]
end

threadschunks(X::AbstractVector) = makechunks(X, Threads.nthreads())

const Timer = TimerOutput()

mix_simple!(x, x_next, β::Real) = @. x = β * x_next + (1 - β) * x

struct MixerDISS{T}
    len::Int
    maxdim::Int
    OVEC::Matrix{T}             # Solution Matrix
    EVEC::Matrix{T}             # Error Matrix
    A::Matrix{T}
    B::Vector{T}
    err::Vector{T}

    function MixerDISS(X::AbstractArray{T}; maxdim::Integer=8, kwargs...) where {T<:Real}
        len = length(X)
        OVEC = Array{T}(undef, len, maxdim)
        EVEC = similar(OVEC)
        A = Array{T}(undef, maxdim + 1, maxdim + 1)
        B = Array{T}(undef, maxdim + 1)
        err = Array{T}(undef, maxdim)
        return new{T}(len, maxdim, OVEC, EVEC, A, B, err)
    end
    """
        MixerDISS: direct inversion of the iterative subspace (DIIS) method
        
    https://www.chem.fsu.edu/~deprince/programming_projects/diis/
    """
    function MixerDISS(X::AbstractArray{Complex{T}}; maxdim::Integer=8, kwargs...) where {T<:Real}
        len = 2length(X)
        OVEC = Array{T}(undef, len, maxdim)
        EVEC = similar(OVEC)
        A = Array{T}(undef, maxdim + 1, maxdim + 1)
        B = Array{T}(undef, maxdim + 1)
        err = Array{T}(undef, maxdim)
        return new{T}(len, maxdim, OVEC, EVEC, A, B, err)
    end
end

@views function mix!(iter::Integer, vec_in::AbstractArray{T}, vec_out::AbstractArray{T}, mixer::MixerDISS{T}; β::Real=0.01, kwargs...) where {T<:Real}

    vvec_in = Base.ReshapedArray(vec_in, (length(vec_in),), ())
    vvec_out = Base.ReshapedArray(vec_out, (length(vec_out),), ())

    OVEC, EVEC = mixer.OVEC, mixer.EVEC

    replace = iter <= mixer.maxdim ? iter : findmax(mixer.err)[2]

    @. OVEC[:, replace] = vvec_out
    @. EVEC[:, replace] = vvec_out - vvec_in
    mixer.err[replace] = norm(EVEC[:, replace])

    lda = min(iter, mixer.maxdim)
    A = mixer.A[1:(lda+1), 1:(lda+1)]
    B = mixer.B[1:(lda+1)]

    if lda > 1
        for i in 1:lda, j in 1:i
            s = zero(T)
            for k in 1:mixer.len
                s += EVEC[k, j] * EVEC[k, i]
            end
            A[j, i] = s
            A[i, j] = s
        end
        @. A[lda+1, 1:lda] = one(T)
        @. A[1:lda, lda+1] = one(T)
        A[lda+1, lda+1] = zero(T)

        @. B[1:lda, 1] = zero(T)
        B[lda+1, 1] = one(T)

        simple = false
        invV = try
            convert.(T, inv(big.(A)))
        catch
            simple = true
        end

        if simple
            mix_simple!(vvec_in, vvec_out, β)
        else
            err = invV * B
            fill!(vvec_in, zero(T))
            for i in 1:lda
                @. vvec_in += err[i] * OVEC[:, i]
            end
        end
    else
        mix_simple!(vvec_in, vvec_out, β)
    end

    return nothing
end

function mix!(iter::Integer, vec_in::AbstractArray{Complex{T}}, vec_out::AbstractArray{Complex{T}}, mixer::MixerDISS{T}; kwargs...) where {T}
    vec_in_reitp = reinterpret(T, vec_in)
    vec_out_reitp = reinterpret(T, vec_out)
    mix!(iter, vec_in_reitp, vec_out_reitp, mixer; kwargs...)
    return nothing
end

function get_error(A::AbstractArray{T}, B::AbstractArray{T}) where {T<:AbstractFloat}
    @assert size(A) == size(B)
    max_atol = zero(T)
    max_rtol = zero(T)
    @inbounds for i in eachindex(A)
        s = abs(A[i] - B[i])
        if s > max_atol
            max_atol = s
        end
        s = abs(s / B[i])
        if isfinite(s) && s > max_rtol
            max_rtol = s
        end
    end
    return max_atol, max_rtol
end

get_error(A::AbstractArray{Complex{T}}, B::AbstractArray{Complex{T}}) where {T} = get_error(reinterpret(T, A), reinterpret(T, B))

# monolayer lattice vectors: zigzag direction along x
const a1 = SA[-1/2, sqrt(3)/2]
const a2 = SA[-1, 0]

# monolayer reciprocal vectors
const b1 = SA[0, 4π/sqrt(3)]
const b2 = SA[-2π, -2π/sqrt(3)]

const b1pb2 = b1 + b2
const b1p2b2 = b1 + 2b2
const b2p2b1 = b2 + 2b1
const b1mb2 = b1 - b2

# we label the components with the monolayer vectors
struct ElasticitySolver{T}
    M::Matrix{T}
    N_shell::Int
    N_star::Int
    bvecs::Matrix{Int}
    gx::Vector{T}
    gy::Vector{T}
    invdotgg::Vector{T}
end

function ElasticitySolver(M::Matrix{Float64}, N_shell::Integer=4)
    N_star = N_shell * (N_shell + 1) ÷ 2 # number of stars

    bvecs = Int[]
    # representative b vectors (no symmetry)
    for n1 in 1:N_shell, n2 in 0:n1-1
        append!(bvecs, n1, n2)
        append!(bvecs, -n2, n1 - n2) # rotated by 2π/3
        append!(bvecs, n2 - n1, -n1) # rotated by 4π/3
    end
    bvecs = reshape(bvecs, 2, :)

    gx = Float64[]
    gy = Float64[]
    invdotgg = Float64[]
    # representative g vectors including N_shell shells
    for m in axes(bvecs, 2)
        b_1 = bvecs[1, m] * b1[1] + bvecs[2, m] * b2[1]
        b_2 = bvecs[1, m] * b1[2] + bvecs[2, m] * b2[2]

        # moiré reciprocal vectors (units of 1/a): g = M^T b
        gx_ = M[1, 1] * b_1 + M[2, 1] * b_2
        gy_ = M[1, 2] * b_1 + M[2, 2] * b_2
        dotgg = abs2(gx_) + abs2(gy_)

        append!(gx, gx_)
        append!(gy, gy_)
        append!(invdotgg, 1 / dotgg)
    end

    return ElasticitySolver{Float64}(M, N_shell, N_star, bvecs, gx, gy, invdotgg)
end

# adhesion potential (for D3 stacking symmetry)
# c1 = Re(V1)
# c2 = V2
# c3 = Re(V3)
# p1 = Im(V1)
# p2 = Im(V3)

# gradient of the adhesion potential (units 1/a)
# cn = Re(Vn), pn = Im(Vn)
function dV(ϕ::AbstractVector{Complex{T}}, c1::Real, c2::Real, c3::Real, p1::Real, p3::Real) where {T<:Real}
    return -2c1 * (b1 * sin(dot(b1, ϕ)) + b2 * sin(dot(b2, ϕ)) + b1pb2 * sin(dot(b1pb2, ϕ))) +
           -2p1 * (b1 * cos(dot(b1, ϕ)) + b2 * cos(dot(b2, ϕ)) - b1pb2 * cos(dot(b1pb2, ϕ))) +
           -2c2 * (b1p2b2 * sin(dot(b1p2b2, ϕ)) + b2p2b1 * sin(dot(b2p2b1, ϕ)) + b1mb2 * sin(dot(b1mb2, ϕ))) +
           -4c3 * (b1 * sin(2dot(b1, ϕ)) + b2 * sin(2dot(b2, ϕ)) + b1pb2 * sin(2dot(b1pb2, ϕ)))
           -4p3 * (b1 * cos(2dot(b1, ϕ)) + b2 * cos(2dot(b2, ϕ)) - b1pb2 * sin(2dot(b1pb2, ϕ)))
end

# displacement field (units a) for given r = s1 L1 + s2 L2
@inline function u(s1::T, s2::T, U_para::AbstractVector{Complex{T}}, U_perp::AbstractVector{Complex{T}}, bvecs::AbstractMatrix{Int}, gx::AbstractVector{T}, gy::AbstractVector{T}, invdotgg::AbstractVector{T}) where {T<:Real}

    ux = zero(T)
    uy = zero(T)

    @inbounds @views @fastmath @simd for m in eachindex(gx)
        # g . (s1 L1 + s2 L2) = b . (s1 a1 + s2 a2) = 2π (s1 n1 + s2 n2) with b = n1 b1 + n2 b2
        foo = -im * exp(im * 2π * (bvecs[1, m] * s1 + bvecs[2, m] * s2)) * invdotgg[m]

        # (upar gx - uperp gy, upar gy + uperp gx)
        ux += (U_para[m] * gx[m] - U_perp[m] * gy[m]) * foo
        uy += (U_para[m] * gy[m] + U_perp[m] * gx[m]) * foo
    end

    return ux + conj(ux), uy + conj(uy)
end

function update!(solver::ElasticitySolver{T}, U_next::AbstractVector{Complex{T}}, U::AbstractVector{Complex{T}}, s::AbstractVector{T}, w::AbstractVector{T}, λ, μ, Vg) where {T}

    c1, c2, c3, p1, p3 = Vg
    @unpack N_star, bvecs, gx, gy, invdotgg = solver

    U1_para = view(U, 1:3N_star)
    U2_para = view(U, 3N_star+1:6N_star)
    U1_perp = view(U, 6N_star+1:9N_star)
    U2_perp = view(U, 9N_star+1:12N_star)
    U1_para_next = view(U_next, 1:3N_star)
    U2_para_next = view(U_next, 3N_star+1:6N_star)
    U1_perp_next = view(U_next, 6N_star+1:9N_star)
    U2_perp_next = view(U_next, 9N_star+1:12N_star)

    @batch for m in axes(bvecs, 2)

        # b . Mr + b . u(r) = s1 n1 + s2 n2 + b.u(s1 l1 + s2 l2) = b. [ s1 a1 + s2 a2 + U(s1 a1 + s2 a2) ]
        # integral only depends on the rigid moiré implicitly through u

        int1, int2 = zero(T), zero(T)

        @inbounds @fastmath for i in eachindex(w), j in eachindex(w)
            s1, s2 = s[i], s[j]
            U1x, U1y = u(s1, s2, U1_para, U1_perp, bvecs, gx, gy, invdotgg)
            U2x, U2y = u(s1, s2, U2_para, U2_perp, bvecs, gx, gy, invdotgg)
            dv1, dv2 = dV(SA[s1*a1[1]+s2*a2[1]+U1x-U2x, s1*a1[2]+s2*a2[2]+U1y-U2y], c1, c2, c3, p1, p3) * exp(-im * 2π * (bvecs[1, m] * s1 + bvecs[2, m] * s2))
            int1 += dv1 * w[i] * w[j]
            int2 += dv2 * w[i] * w[j]
        end

        int1 *= -im * invdotgg[m]
        int2 *= -im * invdotgg[m]
        dotgint = gx[m] * int1 + gy[m] * int2
        dotcgint = -gy[m] * int1 + gx[m] * int2

        # u_para and u_perp
        U1_para_next[m] = dotgint / (λ[1] + 2μ[1])
        U1_perp_next[m] = dotcgint / μ[1]
        U2_para_next[m] = -dotgint / (λ[2] + 2μ[2])
        U2_perp_next[m] = -dotcgint / μ[2]
    end

    return nothing
end

function get_U(
    M::Matrix{<:Real},
    μ::Vector{<:Real},
    λ::Vector{<:Real},
    Vg::Vector{<:Real},
    N_shell::Integer=4,
    N_nodes::Integer=40;
    max_iter_relax::Int=10,
    max_iter::Integer=300,
    β_simplemix::Real=0.2,
    atol_relax::Real=1e-2,
    rtol_relax::Real=1e-2,
    atol::Real=1e-9,
    rtol::Real=1e-6,
    restart::Integer=20,
    U1_para0::AbstractVecOrMat{<:Real}=Float64[],
    U2_para0::AbstractVecOrMat{<:Real}=Float64[],
    U1_perp0::AbstractVecOrMat{<:Real}=Float64[],
    U2_perp0::AbstractVecOrMat{<:Real}=Float64[],
    kwargs...
)

    solver = ElasticitySolver(M, N_shell)
    @unpack N_star = solver

    # quadrature nodes and weights for [-1,1] interval
    s, w = gausslegendre(N_nodes)
    @. s = (s + 1) / 2    # transform nodes from [-1,1] to [0,1]
    @. w = w / 2          # Jacobian

    θ = round(2asind((M[2, 1] - M[1, 2]) / 4); digits=3) # twist angle
    l1 = inv(M) * a1
    l2 = inv(M) * a2
    l = (norm(l1) + norm(l2)) / 2 # mean length of primitive moiré vectors

    U = zeros(ComplexF64, 12N_star)

    if !(isempty(U1_para0) || isempty(U1_perp0)isempty(U2_para0) || isempty(U2_perp0))
        fill!(U, zero(eltype(U)))
        @assert length(U1_para0) == length(U2_para0) == length(U1_perp0) == length(U2_perp0)
        N0 = min(3N_star, length(U1_para0))
        @. U[1:N0] = U1_para0[1:N0]
        @. U[3N_star+1:3N_star+N0] = U2_para0[1:N0]
        @. U[6N_star+1:6N_star+N0] = U1_perp0[1:N0]
        @. U[9N_star+1:9N_star+N0] = U2_perp0[1:N0]
    end

    U_next = similar(U)

    mixer = MixerDISS(U; kwargs...)

    println("|=================================================|")
    println("|=================Moire Elasticity================|")
    println("|=================================================|")
    println(@sprintf("| theta   :%30s    deg  |", θ))
    println(@sprintf("| lambda1 :%30s         |", λ[1]))
    println(@sprintf("| lambda2 :%30s         |", λ[2]))
    println(@sprintf("| mu1     :%30s         |", μ[1]))
    println(@sprintf("| mu2     :%30s         |", μ[2]))
    println(@sprintf("| c1      :%30s         |", Vg[1]))
    println(@sprintf("| c2      :%30s         |", Vg[2]))
    println(@sprintf("| c3      :%30s         |", Vg[3]))
    println(@sprintf("| p1      :%30s         |", Vg[4]))
    println(@sprintf("| p3      :%30s         |", Vg[5]))
    println(@sprintf("| N_shell :%30s         |", solver.N_shell))
    println(@sprintf("| N_star  :%30s         |", N_star))
    println("|-------------------------------------------------|")
    println("|   iter  |              atol |              rtol |")
    println("|-------------------------------------------------|")

    reset_timer!(Timer)

    # simple mixing
    for iter in 1:max_iter_relax
        @timeit Timer "Iter" update!(solver, U_next, U, s, w, λ, μ, Vg)
        max_atol, max_rtol = get_error(U, U_next)
        conv = max_atol <= atol_relax || max_rtol <= rtol_relax
        println(@sprintf("| %7i | %17.6e | %17.6e |", iter, max_atol, max_rtol))
        flush(stdout)
        conv && break
        mix_simple!(U, U_next, β_simplemix)
    end

    # DISS (Pulay mixing)
    for iter in 1:max_iter
        @timeit Timer "Iter" update!(solver, U_next, U, s, w, λ, μ, Vg)
        max_atol, max_rtol = get_error(U, U_next)
        conv = max_atol <= atol || max_rtol <= rtol
        println(@sprintf("| %7i | %17.6e | %17.6e |", iter, max_atol, max_rtol))
        flush(stdout)
        conv && break
        @timeit Timer "Mix" mix!((iter - 1) % restart + 1, U, U_next, mixer; kwargs...)
    end

    print_timer(Timer; compact=true, linechars=:ascii)

    # return u_para and u_perp (definition has extra factor of l/a)
    return view(l * U, 1:3N_star), view(l * U, 3N_star+1:6N_star), view(l * U, 6N_star+1:9N_star), view(l * U, 9N_star+1:12N_star)
end

### Material parameters ###

# TBG : parameters from PRB 98 224102
# ϵ0 = 0.0
# μ = [47352; 47352]
# λ = [69518 - μ[1]; 69518 - μ[2]]
# Vg = [4.064 / 2; -0.374 / 2; -0.095 / 2; 0; 0]

# hBN / graphene : parameters from Nat. Commun. 6, 6308 (2015)
# ϵ0 = 0.0177276
# μ = [42292.3; 50154.8]
# λ = [18977.3; 17032.7]
# Vg = [1.37222; 0; 0; 1.66469; 0]

# hBN / graphene
# ϵ0 = 0.0187
# μ = [37791.3; 44701.8]
# λ = [22620.5; 26756.3]
# Vg = [0.898297; 0; 0; 1.10151; 0]

# aligned WSe2 / WS2
# ϵ0 = 0.0412
# μ = [30099; 31405]
# λ = [24587; 30226]
# Vg = [11.065; -2.09015; -0.737515; 0.683951; -0.136664]

# aligned MoTe2 / WSe2
ϵ0 = 0.0711
μ = [24161; 30099]
λ = [22130; 24587]
Vg = [13.6019; -2.67482; -0.901028; 0.592146; -0.142573]

# folder = "data/hBN-graphene/"
# folder = "data/WSe2-WS2/"
folder = "data/MoTe2-WSe2/"
mkpath(folder)

### Computational parameters ###

N_shell = 14    # number of reciprocal shells
N_nodes = 30    # number of quadrature nodes
eps = 1e-10     # numerical cutoff

# zero twist angle
# θ = 0.0
# var = [0.01:0.001:0.05]
# var = [0.02:0.002:0.1]

# fixed lattice mismatch versus twist angle
var = [0:0.1:0.6]

### Main ###

var = reverse(reduce(vcat, var))

# for (i, ϵ0) in enumerate(var)
for (i, θ) in enumerate(var)

    ϵ = [ϵ0 0; 0 ϵ0]
    M = ([1 0; 0 1] + ϵ / 2) * [cosd(θ / 2) -sind(θ / 2); sind(θ / 2) cosd(θ / 2)] - ([1 0; 0 1] - ϵ / 2) * [cosd(θ / 2) sind(θ / 2); -sind(θ / 2) cosd(θ / 2)]

    U1_para, U2_para, U1_perp, U2_perp = get_U(M, μ, λ, Vg, N_shell, N_nodes);
    
    u_para = U1_para - U2_para
    u_perp = U1_perp - U2_perp
    U_para = U1_para + U2_para
    U_perp = U1_perp + U2_perp

    zchop!(u_para, eps)
    zchop!(u_perp, eps)
    zchop!(U_para, eps)
    zchop!(U_perp, eps)

    writedlm(folder * "u_para_$(θ)_$(ϵ0)_$(N_shell).dat", u_para)
    writedlm(folder * "u_perp_$(θ)_$(ϵ0)_$(N_shell).dat", u_perp)
    writedlm(folder * "U_para_$(θ)_$(ϵ0)_$(N_shell).dat", U_para)
    writedlm(folder * "U_perp_$(θ)_$(ϵ0)_$(N_shell).dat", U_perp)

    # save individual layers

    # zchop!(U1_para, eps)
    # zchop!(U1_perp, eps)
    # zchop!(U2_para, eps)
    # zchop!(U2_perp, eps)

    # writedlm(folder * "U1_para_$(θ)_$(ϵ0)_$(N_shell).dat", U1_para)
    # writedlm(folder * "U1_perp_$(θ)_$(ϵ0)_$(N_shell).dat", U1_perp)
    # writedlm(folder * "U2_para_$(θ)_$(ϵ0)_$(N_shell).dat", U2_para)
    # writedlm(folder * "U2_perp_$(θ)_$(ϵ0)_$(N_shell).dat", U2_perp)

end