using StaticArrays
using LinearAlgebra

abstract type AbstractRobotManipulator end
abstract type AbstractLink end

struct Inertia
    mass::Float64
    Icm::SMatrix{3,3,Float64}   # inertia tensor at center of mass
    c::SVector{3,Float64}       # center of mass position in link frame
end

struct Link <: AbstractLink
    parent::Int
    pitch::Float64
    Xtree::SMatrix{6,6,Float64}
    inertia::Inertia
end

struct Model <: AbstractRobotManipulator
    N::Int                      # number of joints = number of moving links
    links::Vector{Link}
    gravity::SVector{3,Float64}
end

const dof(m::Model) = m.N

"""
Algebra
"""

# skew symmetric matrix from a 3D vector
function skew(v::SVector{3,Float64})
    return @SMatrix [
        0.0      -v[3]    v[2];
        v[3]     0.0     -v[1];
       -v[2]     v[1]    0.0
    ]
end

unskew(s::SMatrix{3,3,Float64}) = SVector(s[3,2], s[1,3], s[2,1])

# cross motion product operator
function crm(v::SVector{6,Float64})
    w = SVector{3,Float64}(v[1:3]...)
    v_lin = SVector{3,Float64}(v[4:6]...)
    return @SMatrix [
        skew(w)          zeros(3,3);
        skew(v_lin)     skew(w)
    ]
end

# transform from one frame to another (for motion vectors, for force vectors use Xmotion')
function Xmotion(
    E::SMatrix{3,3,Float64},
    r::SVector{3,Float64}   
)
    P = -skew(r) * E    
    X = zeros(6,6)
    X[1:3, 1:3] = E
    X[4:6, 1:3] = P
    X[4:6, 4:6] = E
    return SMatrix{6,6,Float64}(X)
end

function to_plucker(T::SMatrix{4,4})
    E = T[1:3,1:3]
    p = T[1:3,4]
    @SMatrix [E zeros(3,3); skew(E * p) E]
end

function XtoV(X::SMatrix{6,6,Float64})
    E = X[1:3, 1:3]          # rotation part
    L = X[4:6, 1:3]          # lower-left 3×3 (skew-symmetric part)
    # Small rotation vector from rotation matrix difference
    omega = unskew(E - SMatrix{3,3,Float64}(I))
    # Linear velocity part (from skew-symmetric translation block)
    lin = unskew(L)
    return SVector{6,Float64}(omega[1], omega[2], omega[3], lin[1], lin[2], lin[3])
end

"""
Motion and forces calculation functions
"""
# joint model calculation (returns transfom matrix and screw axis for current joint)
function jcalc(
    pitch::Float64,
    q::Float64
)
    if pitch == 0
        E = @SMatrix[
            cos(q)   -sin(q)    0.0;
            sin(q)    cos(q)    0.0;
            0.0       0.0       1.0
        ]
        XJ = Xmotion(E, @SVector zeros(3))
        S = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    elseif pitch == Inf
        XJ = Xmotion(SMatrix{3,3,Float64}(I), @SVector [0.0, 0.0, q])        
        S = @SVector [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else
        E = @SMatrix[
            cos(q)   -sin(q)    0.0;
            sin(q)    cos(q)    0.0;
            0.0       0.0       1.0
        ]
        XJ = Xmotion(E, @SVector [0.0, 0.0, pitch*q])
        S = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, pitch]
    end
    return XJ, S
end

# calculating inverse dynamics via recursive Newton-Euler algorithm
function inverse_dynamics(
    model::Model,
    q::Vector{Float64},
    qd::Vector{Float64},
    qdd::Vector{Float64}
)
    n = model.N
    v = Vector{SVector{6,Float64}}(undef, n)
    a = Vector{SVector{6,Float64}}(undef, n)
    f = Vector{SVector{6, Float64}}(undef, n)
    S   = Vector{SVector{6,Float64}}(undef, n)
    Xup = Vector{SMatrix{6,6,Float64}}(undef, n)

    a0 = @SVector [0.0; 0.0; 0.0; -model.gravity[1]; -model.gravity[2]; -model.gravity[3]]
    τ = zeros(n)
    for i in 1:n
        XJ, S[i] = jcalc(model.links[i].pitch, q[i])
        Xup[i] = model.links[i].Xtree * XJ
        if model.links[i].parent == 0
            vJ = S[i] * qd[i]
            v[i] = vJ
            a[i] = Xup[i] * a0 + S[i] * qdd[i] + crm(v[i]) * vJ
        else
            j = model.links[i].parent
            vJ = S[i] * qd[i]
            v[i] = Xup[i] * v[j] + vJ
            a[i] = Xup[i] * a[j] + S[i] * qdd[i] + crm(v[i]) * vJ
        end
    end
    for i in n:-1:1
        f[i] = model.links[i].I * a[i] + crm(v[i])' * (model.links[i].I * v[i])
        if model.links[i].parent != 0
            j = model.links[i].parent
            f[j] += Xup[i]' * f[i]
        end
         τ[i] = S[i]' * f[i]
    end
    return τ
end

# body Jacobian calculation
function bodyJac(
    model::Model,
    body::Int,
    q::Vector{Float64}
)
    N = model.N
    chain = falses(N)
    b = body
    # mark joints on kinematic chain
    while b != 0
        chain[b] = true
        b = model.links[b].parent
    end
    Jb = zeros(6, N)
    Xa = Vector{SMatrix{6,6,Float64}}(undef, N)
    # forward propagation
    for i in 1:N
        if !chain[i]
            continue
        end
        link =  model.links[i]
        XJ, S = jcalc(link.pitch, q[i])
        Xup = link.Xtree * XJ
        if link.parent == 0
            Xa[i] = Xup
        else
            Xa[i] = Xup * Xa[link.parent]
        end
        Jb[:, i] = Xa[i] \ S
    end
    return Jb
end

function forward_kinematics(model::Model, q, body::Int=model.N)
    chain = falses(model.N)
    b = body
    while b != 0
        chain[b] = true
        b = model.links[b].parent
    end
    Xa = Vector{SMatrix{6,6,Float64}}(undef, model.N)
    for i in 1:model.N
        if !chain[i]; continue; end
        link = model.links[i]
        XJ, _ = jcalc(link.pitch, q[i])
        Xup = XJ * link.Xtree
        if link.parent == 0
            Xa[i] = Xup
        else
            Xa[i] = Xup * Xa[link.parent]
        end
    end
    Xa[body]
end

function inverse_kinematics(model::Model, target_pose::SMatrix{4,4,Float64}, initial_q::Vector{Float64}; max_iters::Int=100, tol::Float64=1e-6)
    q = copy(initial_q)
    body = model.N
    X_target = to_plucker(target_pose)
    for _ in 1:max_iters
        X_current = forward_kinematics(model, q, body)
        J0 = body_jacobian(model, body, q)
        Jb = X_current * J0
        dXb = X_target * inv(X_current)
        v = XtoV(dXb)
        if norm(v) < tol; break; end
        delta_q = pinv(Jb) * v
        q += delta_q
    end
    q
end

#EXAMPLE USAGE

inertia1 = Inertia(
    1.0,
    diagm(@SVector [0.01, 0.01, 0.02]),
    @SVector [0.15, 0.0, 0.0]
)

inertia2 = Inertia(
    0.8,
    diagm(@SVector [0.008, 0.008, 0.015]),
    @SVector [0.125, 0.0, 0.0]
)

inertia3 = Inertia(
    0.5,
    diagm(@SVector [0.002, 0.002, 0.001]),
    @SVector [0.0, 0.0, -0.05]
)

link1 = Link(
    0,
    0.0,
    Xmotion(SMatrix{3,3,Float64}(I), @SVector [0.0, 0.0, 0.0]),
    inertia1
)

link2 = Link(
    1,
    0.0,
    Xmotion(SMatrix{3,3,Float64}(I), @SVector [0.3, 0.0, 0.0]),
    inertia2
)

link3 = Link(
    2,
    Inf,
    Xmotion(SMatrix{3,3,Float64}(I), @SVector [0.25, 0.0, 0.0]),
    inertia3
)

scara_robot = Model(
    3,
    [link1, link2, link3],
    @SVector [0.0, 0.0, 9.81]
)

println("SCARA робот создан:")
println("Количество суставов: ", dof(scara_robot))
println("Гравитация: ", scara_robot.gravity)

q = [0.5, 0.3, 0.1]
qd = [0.1, 0.2, 0.01]
qdd = [0.01, 0.02, 0.001]

τ = inverse_dynamics(scara_robot, q, qd, qdd)

println("\nУсилия на суставах:")
println("Сустав 1 (вращательный): τ = ", τ[1], " Н·м")
println("Сустав 2 (вращательный): τ = ", τ[2], " Н·м") 
println("Сустав 3 (поступательный): f = ", τ[3], " Н")

X_ee = forward_kinematics(scara_robot, q, 3)
println("\nПоза энд-эффектора (матрица 6x6):")
println(X_ee)

position = X_ee[1:3, 4]
println("Позиция энд-эффектора (x, y, z): ", position)

J = bodyJac(scara_robot, 3, q)
println("\nЯкобиан энд-эффектора (6x3):")
println(J)