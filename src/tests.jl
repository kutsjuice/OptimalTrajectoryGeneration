using LinearAlgebra
using StaticArrays
using ForwardDiff

abstract type AbstractRobotManipulator end
abstract type AbstractLink end

function skew(v::AbstractVector)
    skew_of_v = @SMatrix [
        0 -v[3] v[2];
        v[3] 0 -v[1];
        -v[2] v[1] 0
    ]
    return skew_of_v
end

function adjoint(R::SMatrix{3,3}, p::SVector{3})
    @SMatrix[
        R               zeros(3,3);
        skew(p) * R     R
    ]
end

function ad(V)
    omega = V[1:3]
    v = V[4:6]
    @SMatrix [
        skew(ω)      zeros(3,3);
        skew(v)      skew(ω)
    ]
end

function spatial_inertia(
    mass,
    com,
    inertia_com
)
    I3 = I(3)
    S = skew(com)
    @SMatrix [
        inertia_com + mass * S * S'    mass * S;
        mass * S'                      mass * I3
    ]
end

struct RigidBody <: AbstractLink
    screw_axis::SVector{6,Float64}          # Body screw axis
    X_parent::SMatrix{4,4,Float64,16}       # Transform to parent at zero config
    inertia::SMatrix{6,6,Float64,36}        # Spatial inertia
end

struct SerialManipulator <: AbstractRobotManipulator
    links::Vector{RigidBody}
    gravity::SVector{3,Float64}
    dof::Int
end

SerialManipulator(links, gravity) =
    SerialManipulator(links, gravity, length(links))

struct TrajectoryConstraints
    velocity_limits::Vector{Float64}
    acceleration_limits::Vector{Float64}
    torque_limits::Vector{Float64}
    jerk_limits::Vector{Float64}
    position_limits::Tuple{Vector{Float64}, Vector{Float64}}
end

struct JointTrajectory
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    accelerations::Matrix{Float64}
end

struct TrajectoryResult
    trajectory::JointTrajectory
    torques::Matrix{Float64}
    time_vector::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    cartesian_trajectory::Matrix{Float64}
    feasible::Bool
end

function exp_twist(S, theta)
    omega = S[1:3]
    v = S[4:6]

    if norm(omega) < 1e-8
        R = I(3)
        p = v * theta
    else
        omega1 = skew(omega)
        R = I(3) + sin(theta)*omega1 + (1-cos(theta))*(omega1*omega)
        p = (I(3)*theta + (1-cos(theta))*omega1 + (theta - sin(theta))*(omega*omega1))*v
    end

    @SMatrix[
        R p;
        0 1
    ]
end

function forward_kinematics(
    robot,
    joint_positions,
)
    T = Matrix{4,4}(I)
    for i in 1:robot.dof
        T *= exp_twist(robot.links[i].screw_axis, joint_positions[i])
        T *= robot.links[i].X_parent
    end
    return @SVector [T[1,4], T[2,4], T[3,4]]
end

function jacobian(
    robot,
    joint_positions
)
    n = robot.dof
    J = Matrix{Float64}(undef, 6, n)
    Ad_cum = SMatrix{6,6,Float64}(I)
    for i = n:-1:1
        J[:, i] = Vector(Ad_cum * robot.links[i].screw_axis)
        
end

function inverse_kinematics(
    robot,
    target,
    q0,
    tolerance=1e-6,
    max_iterations=100
)
    q = copy(q0)
    for _ in 1:max_iterations
        error = target - forward_kinematics(robot, q)
        if norm(error) < tolerance
            return q
        end
        J = jacobian(robot, q)[4:6, :]
        q += pinv(J)*e
    end
    error("IK not converge")
end