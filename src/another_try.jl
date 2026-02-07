using LinearAlgebra
using StaticArrays

# Types
abstract type AbstractRobotManipulator end
abstract type AbstractLink end

struct Inertia
    mass::Float64
    Icm::SMatrix{3,3,Float64}
    c::SVector{3,Float64}
end

struct Link
    parent::Int
    pitch::Float64
    Xtree::SMatrix{6,6,Float64}
    inertia::Inertia
end

struct Model
    N::Int
    links::Vector{Link}
    gravity::SVector{3,Float64}
end

dof(m::Model) = m.N

# Algebra
function skew(v::SVector{3,Float64})
    @SMatrix [
        0.0   -v[3]  v[2];
        v[3]   0.0  -v[1];
       -v[2]  v[1]   0.0
    ]
end

unskew(S::SMatrix{3,3,Float64}) =
    SVector(S[3,2], S[1,3], S[2,1])

const Z33 = @SMatrix zeros(3,3)
const I33 = SMatrix{3,3,Float64}(I)

# Spatial operators
function crm(v::SVector{6,Float64})
    ω   = SVector{3,Float64}(v[1:3])
    vlin = SVector{3,Float64}(v[4:6])
    top    = hcat(skew(ω),   Z33)
    bottom = hcat(skew(vlin), skew(ω))
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

crf(v::SVector{6,Float64}) = -crm(v)'

function spatial_transform(
    R::SMatrix{3,3,Float64},
    r::SVector{3,Float64}
)
    top = hcat(R, Z33)
    bottom = hcat(skew(r)*R, R)
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

function spatial_inertia(I::Inertia)
    m = I.mass
    c = I.c
    Ic = I.Icm
    C = skew(SVector{3,Float64}(c))    
    top_left = Ic + m * C * C'
    top_right = m * C
    bottom_left = m * C'
    bottom_right = m * I33
    top = hcat(top_left, top_right)
    bottom = hcat(bottom_left, bottom_right)
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

# Joint model
function jcalc(pitch::Float64, q::Float64)
    if pitch == 0.0
        E = @SMatrix [
            cos(q) -sin(q) 0.0
            sin(q)  cos(q) 0.0
            0.0     0.0    1.0
        ]
        XJ = spatial_transform(E, @SVector [0.0,0.0,0.0])
        S  = @SVector [0.0,0.0,1.0, 0.0,0.0,0.0]

    elseif pitch == Inf
        XJ = spatial_transform(I33, @SVector [0.0,0.0,q])
        S  = @SVector [0.0,0.0,0.0, 0.0,0.0,1.0]

    else
        E = @SMatrix [
            cos(q) -sin(q) 0.0
            sin(q)  cos(q) 0.0
            0.0     0.0    1.0
        ]
        XJ = spatial_transform(E, @SVector [0.0,0.0,pitch*q])
        S  = @SVector [0.0,0.0,1.0, 0.0,0.0,pitch]
    end

    return XJ, S
end

# Inverse dynamics (RNEA)
function inverse_dynamics(
    model::Model,
    q::Vector{Float64},
    qd::Vector{Float64},
    qdd::Vector{Float64}
)
    n = model.N
    v = Vector{SVector{6,Float64}}(undef, n)
    a = Vector{SVector{6,Float64}}(undef, n)
    f = Vector{SVector{6,Float64}}(undef, n)
    S = Vector{SVector{6,Float64}}(undef, n)
    Xup = Vector{SMatrix{6,6,Float64}}(undef, n)
    a0 = @SVector [0.0, 0.0, 0.0,
                   -model.gravity[1],
                   -model.gravity[2],
                   -model.gravity[3]]

    τ = zeros(n)
    # ----- forward recursion -----
    for i in 1:n
        XJ, S[i] = jcalc(model.links[i].pitch, q[i])
        Xup[i] = model.links[i].Xtree * XJ

        vJ = S[i] * qd[i]

        if model.links[i].parent == 0
            v[i] = vJ
            a[i] = Xup[i]*a0 + S[i]*qdd[i] + crm(v[i])*vJ
        else
            p = model.links[i].parent
            v[i] = Xup[i]*v[p] + vJ
            a[i] = Xup[i]*a[p] + S[i]*qdd[i] + crm(v[i])*vJ
        end
    end
    # ----- backward recursion -----
    for i in n:-1:1
        I = spatial_inertia(model.links[i].inertia)
        f[i] = I*a[i] + crf(v[i])*(I*v[i])

        if model.links[i].parent != 0
            p = model.links[i].parent
            f[p] += Xup[i]'*f[i]
        end

        τ[i] = S[i]' * f[i]
    end
    return τ
end

# Compute mass matrix and force terms for manipulator dynamics.
function compute_mass_matrix_and_force_terms(
    model::Model,
    q::Vector{Float64},
    qd::Vector{Float64}
)
    n = model.N
    M = zeros(n, n)
    bias = inverse_dynamics(model, q, qd, zeros(n))

    for j = 1:n
        qdd_unit = zeros(n); qdd_unit[j] = 1.0
        τ = inverse_dynamics(model, q, zeros(n), qdd_unit)
        M[:, j] .= τ
    end

    return M, bias
end

# Example: SCARA
Icm1 = @SMatrix [
    0.01 0 0
    0    0.01 0
    0    0    0.02
]

Icm2 = @SMatrix [
    0.008 0 0
    0     0.008 0
    0     0     0.015
]

Icm3 = @SMatrix [
    0.002 0 0
    0     0.002 0
    0     0     0.001
]

link1 = Link(
    0, 0.0,
    spatial_transform(I33, @SVector [0.0,0.0,0.0]),
    Inertia(1.0, Icm1, @SVector [0.15,0.0,0.0])
)

link2 = Link(
    1, 0.0,
    spatial_transform(I33, @SVector [0.3,0.0,0.0]),
    Inertia(0.8, Icm2, @SVector [0.125,0.0,0.0])
)

link3 = Link(
    2, Inf,
    spatial_transform(I33, @SVector [0.25,0.0,0.0]),
    Inertia(0.5, Icm3, @SVector [0.0,0.0,-0.05])
)

scara = Model(
    3,
    [link1, link2, link3],
    @SVector [0.0,0.0,9.81]
)

q   = [0.5, 0.3, 0.1]
qd  = [0.1, 0.2, 0.01]
qdd = [0.01,0.02,0.001]

τ = inverse_dynamics(scara, q, qd, qdd)

println("Joint efforts:")
println("τ1 = ", τ[1])
println("τ2 = ", τ[2])
println("f3 = ", τ[3])

q   = [0.0, 0.0, 0.0]
qd  = [0.0, 0.0, 0.0]
qdd = [0.0, 0.0, 0.0]

τ = inverse_dynamics(scara, q, qd, qdd)

println("Гравитационные моменты в покое:")
println(τ)


q   = [0.0, 0.0, 0.1]
qd  = [0.0, 0.0, 0.0]
qdd = [0.0, 0.0, 0.0]

τ = inverse_dynamics(scara, q, qd, qdd)
println("τ при q₃ = 0.1 м:", τ[3])


q   = [0.0, π/4, 0.0]
qd  = [1.0, 1.0, 0.0]
qdd = [0.0, 0.0, 0.0]

τ = inverse_dynamics(scara, q, qd, qdd)
println("Центробежные/кориолисовы:", round.(τ, digits=4))


q   = [0.0, 0.0, 0.0]
qd  = [0.0, 0.0, 0.0]
qdd = [2.0, 3.0, 4.0]

τ = inverse_dynamics(scara, q, qd, qdd)
println("Инерционные моменты:", round.(τ, digits=4))

q   = [π/6, π/3, 0.15]
qd  = [0.8, 1.2, 0.05]
qdd = [1.5, 2.0, 0.3]

τ = inverse_dynamics(scara, q, qd, qdd)
println("Полный пример:")
println(round.(τ, digits=4))