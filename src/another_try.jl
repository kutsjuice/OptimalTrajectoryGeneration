abstract type AbstractRobotManipulator end
abstract type AbstractLink end

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

# cross motion product operator
function crm(v::SVector{6,Float64})
    w = v[1:3]
    v_lin = v[4:6]
    return @SMatrix [
        skew(w)          zeros(3,3);
        skew(v_lin)     skew(w)
    ]
end

# transform from one frame to another (for motion vectors, for force vectors use Xmotion')
function Xmotion(
    E::SMatrix{6,6,Float64},
    r::SVector{6,Float64}   
)
    @SMatrix [
    E               zeros(3,3);
    -skew(r*E)          E
    ]
end

# joint model calculation (returns transfom matrix and screw axis for current joint)
function jcalc(
    pitch::Float64,
    q::Float64
)
    if pitch == 0
        E = @SMatrix[
            cos[q]   -sin[q]    0.0;
            sin[q]    cos[q]    0.0;
            0.0       0.0       1.0
        ]
        XJ = Xmotion(E, @SVector zeros(3))
        S = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    elseif pitch == Inf
        XJ = Xmotion(@SMatrix I(3), @SVector [0.0, 0.0, q])
        S = @SVector [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else
        E = @SMatrix[
            cos[q]   -sin[q]    0.0;
            sin[q]    cos[q]    0.0;
            0.0       0.0       1.0
        ]
        XJ = Xmotion(E, @SVector [0.0, 0.0, pitch*q])
        S = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, pitch]
    end
    return XJ, S
end

# calculating inverse dynamics via recursive Newton-Euler algorithm
function inverse_dynamics(
    robot::SerialManipulator,
    q::Vector{Float64},
    qd::Vector{Float64},
    qdd::Vector{Float64}
)
    n = robot.dof
    v = Vector{SVector{6,Float64}}(undef, n)
    a = Vector{SVector{6,Float64}}(undef, n)
    f = Vector{SVector{6, Float64}}(undef, n)
    S   = Vector{SVector{6,Float64}}(undef, n)
    Xup = Vector{SMatrix{6,6,Float64}}(undef, n)

    a0 = @SVector [0.0; 0.0; 0.0; -robot.gravity[1]; -robot.gravity[2]; -robot.gravity[3]]
    for i in 1:n
        XJ, S[i] = jcalc(robot.links[i].pitch, q[i])
        Xup[i] = robot.links[i].Xtree * XJ
        if robot.links[i].parent == 0
            vJ = S[i] * qd[i]
            v[i] = vJ
            a[i] = Xup[i] * a0 + S[i] * qdd[i] + crm(v[i]) * vJ
        else
            j = robot.links[i].parent
            vJ = S[i] * qd[i]
            v[i] = Xup[i] * v[j] + vJ
            a[i] = Xup[i] * a[j] + S[i] * qdd[i] + crm(v[i]) * vJ
        end
    end
    for i in n:-1:1
        f[i] = robot.links[i].I * a[i] + crm(v[i])' * (robot.links[i].I * v[i])
        if robot.links[i].parent != 0
            j = robot.links[i].parent
            f[j] += Xup[i]' * f[i]
        end
    end
end