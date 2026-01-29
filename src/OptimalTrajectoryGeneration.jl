module OptimalTrajectoryGeneration

using LinearAlgebra
using StaticArrays
using ForwardDiff

export
    AbstractRobotManipulator,
    AbstractLink,

    RigidBody,
    SerialManipulator,

    TrajectoryConstraints,
    JointTrajectory,
    TrajectoryResult,

    forward_kinematics,
    inverse_kinematics,
    jacobian,
    compute_mass_and_force_terms
    
    evaluate_path,
    compute_limit_path_speed,
    generate_joint_trajectory

"""
Abstract type representing a robot manipulator and robot links.
Users should create concrete subtypes for their specific robots.
"""
abstract type AbstractRobotManipulator end
abstract type AbstractLink end


# Spatial algebra utilities
"""
Skew-symmetric matrix for cross product.
"""
skew(v::SVector{3,Float64}) = @SMatrix [
    0.0 -v[3]  v[2];
    v[3]  0.0 -v[1];
   -v[2]  v[1]  0.0
]

"""
Adjoint transformation matrix for SE(3).
"""
function adjoint(R::SMatrix{3,3}, p::SVector{3})
    @SMatrix [
        R               zeros(3,3);
        skew(p) * R     R
    ]
end

"""
Spatial cross product operator for motion vectors.
"""
function ad(V::SVector{6})
    ω = V[1:3]
    v = V[4:6]
    @SMatrix [
        skew(ω)      zeros(3,3);
        skew(v)      skew(ω)
    ]
end

"""
Spatial inertia matrix.
"""
function spatial_inertia(
    mass::Float64,
    com::SVector{3},
    inertia_com::SMatrix{3,3}
)
    I3 = I(3)
    S = skew(com)
    @SMatrix [
        inertia_com + mass * S * S'    mass * S;
        mass * S'                      mass * I3
    ]
end

# Rigid body & robot definition (URDF-style)
"""
Rigid body with screw axis and spatial inertia.
"""
struct RigidBody <: AbstractLink
    screw_axis::SVector{6,Float64}          # Body screw axis
    X_parent::SMatrix{4,4,Float64,16}       # Transform to parent at zero config
    inertia::SMatrix{6,6,Float64,36}        # Spatial inertia
end

"""
Serial manipulator (chain).
"""
struct SerialManipulator <: AbstractRobotManipulator
    links::Vector{RigidBody}
    gravity::SVector{3,Float64}
    dof::Int
end


SerialManipulator(links, gravity) =
    SerialManipulator(links, gravity, length(links))

"""
Constraints for trajectory optimization.

# Fields
- `velocity_limits::Vector{Float64}`: Maximum joint velocities
- `acceleration_limits::Vector{Float64}`: Maximum joint accelerations  
- `torque_limits::Vector{Float64}`: Maximum joint torques
- `jerk_limits::Vector{Float64}`: Maximum joint jerks (optional)
- `position_limits::Tuple{Vector{Float64}, Vector{Float64}}`: Joint position limits (min, max)
"""
struct TrajectoryConstraints
    velocity_limits::Vector{Float64}
    acceleration_limits::Vector{Float64}
    torque_limits::Vector{Float64}
    jerk_limits::Vector{Float64}
    position_limits::Tuple{Vector{Float64}, Vector{Float64}}
end



"""
Joint space trajectory representation.

# Fields  
- `positions::Matrix{Float64}`: Joint positions [dof × time_steps]
- `velocities::Matrix{Float64}`: Joint velocities [dof × time_steps]
- `accelerations::Matrix{Float64}`: Joint accelerations [dof × time_steps]
"""
struct JointTrajectory
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    accelerations::Matrix{Float64}
end

"""
Result of trajectory computation.

# Fields
- `trajectory::JointTrajectory`: Joint trajectory
- `torques::Matrix{Float64}`: Joint torques [dof × time_steps]
- `time_vector::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}`: Time vector
- `cartesian_trajectory::Matrix{Float64}`: Cartesian trajectory [3 × time_steps]
- `feasible::Bool`: Constraint feasibility
"""
struct TrajectoryResult
    trajectory::JointTrajectory
    torques::Matrix{Float64}
    time_vector::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
    cartesian_trajectory::Matrix{Float64}
    feasible::Bool
end

"""
Compute forward kinematics mapping joint positions to end-effector pose.

# Arguments
- `robot::AbstractRobotManipulator`: Robot instance
- `joint_positions::Vector{Float64}`: Joint positions
- `initial_guess::Vector{Float64}`: Initial pose guess
- `tolerance::Float64=1e-6`: Solution tolerance
- `max_iterations::Int=100`: Maximum iterations

# Returns
- `Vector{Float64}`: Cartesian pose [x, y, z, ...]
"""

"""
Matrix exponential for a screw axis.
"""
function exp_twist(S::SVector{6}, θ::Float64)
    ω = S[1:3]
    v = S[4:6]

    if norm(ω) < 1e-8
        R = I(3)
        p = v * θ
    else
        ω̂ = skew(ω)
        R = I(3) + sin(θ)*ω̂ + (1-cos(θ))*(ω̂*ω̂)
        p = (I(3)*θ + (1-cos(θ))*ω̂ + (θ-sin(θ))*(ω̂*ω̂)) * v
    end

    @SMatrix [
        R  p;
        0  1
    ]
end

function forward_kinematics(
    robot::AbstractRobotManipulator, 
    joint_positions::Vector{Float64}, 
    initial_guess::Vector{Float64}=zeros(length(joint_positions));
    tolerance::Float64=1e-6,
    max_iterations::Int=100)::Vector{Float64}

    error("forward_kinematics not implemented for robot type $(typeof(robot))")

end


"""
Forward kinematics using PoE formulation.
Returns end-effector position.
"""
function forward_kinematics(
    robot::SerialManipulator,
    q::Vector{Float64}
)::SVector{3,Float64}

    T = SMatrix{4,4}(I)
    for i in 1:robot.dof
        T *= exp_twist(robot.links[i].screw_axis, q[i])
        T *= robot.links[i].X_parent
    end

    return @SVector [T[1,4], T[2,4], T[3,4]]
end
"""
Compute inverse kinematics mapping end-effector pose to joint positions.

# Arguments
- `robot::AbstractRobotManipulator`: Robot instance  
- `target_pose::Vector{Float64}`: Desired end-effector pose
- `initial_guess::Vector{Float64}`: Initial joint position guess
- `tolerance::Float64=1e-6`: Solution tolerance
- `max_iterations::Int=100`: Maximum iterations

# Returns
- `Vector{Float64}`: Joint positions
"""
function inverse_kinematics(
    robot::SerialManipulator,
    target::Vector{Float64},
    q0::Vector{Float64}=zeros(robot.dof);
    tolerance=1e-6,
    max_iterations=100
)
    q = copy(q0)
    for _ in 1:max_iterations
        e = target - forward_kinematics(robot, q)
        if norm(e) < tolerance
            return q
        end
        J = jacobian(robot, q)[4:6, :]
        q += pinv(J) * e
    end
    error("IK did not converge")
end

"""
Compute manipulator Jacobian matrix.

# Arguments
- `robot::AbstractRobotManipulator`: Robot instance
- `joint_positions::Vector{Float64}`: Joint positions

# Returns
- `Matrix{Float64}`: Jacobian matrix
"""
function jacobian(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64}
)::Matrix{Float64}
    return ForwardDiff.jacobian(q -> forward_kinematics(robot, q), joint_positions)
end

"""
Recursive Newton-Euler algorithm

# Arguments
- `robot::DHRobot`: Robot instance
- `joint_positions::Vector{Float64}`: Joint positions
- `joint_velocities::Vector{Float64}`: Joint velocities
- `joint_accelerations::Vector{Float64}`: Joint accelerations

# Returns
- `Vector{Float64}`: Generalized forces (torques/forces) at each joint
"""
function newton_euler(
    robot::SerialManipulator,
    q::Vector{Float64},
    qd::Vector{Float64},
    qdd::Vector{Float64}
)
    n = robot.dof

    V = fill(@SVector zeros(6), n)
    Vd = fill(@SVector zeros(6), n)
    F = fill(@SVector zeros(6), n)

    g = @SVector [0.0, 0.0, 0.0; -robot.gravity...]

    # Forward recursion
    for i in 1:n
        S = robot.links[i].screw_axis
        X = exp_twist(S, q[i])

        if i == 1
            V[i]  = S * qd[i]
            Vd[i] = S * qdd[i] - g + ad(V[i]) * (S * qd[i])
        else
            AdX = adjoint(X[1:3,1:3], X[1:3,4])
            V[i]  = AdX * V[i-1] + S * qd[i]
            Vd[i] = AdX * Vd[i-1] + S * qdd[i] + ad(V[i]) * (S * qd[i])
        end
    end

    τ = zeros(n)

    # Backward recursion
    for i in n:-1:1
        I = robot.links[i].inertia
        F[i] = I * Vd[i] + ad(V[i])' * (I * V[i])

        if i < n
            X = exp_twist(robot.links[i+1].screw_axis, q[i+1])
            AdX = adjoint(X[1:3,1:3], X[1:3,4])
            F[i] += AdX' * F[i+1]
        end

        τ[i] = dot(robot.links[i].screw_axis, F[i])
    end

    return τ
end

"""
Compute mass matrix and force terms for manipulator dynamics.

# Arguments
- `robot::AbstractRobotManipulator`: Robot instance
- `joint_positions::Vector{Float64}`: Joint positions  
- `joint_velocities::Vector{Float64}`: Joint velocities

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}}`: (Mass matrix, Force vector)
"""
function compute_mass_and_force_terms(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64},
    joint_velocities::Vector{Float64}
)::Tuple{Matrix{Float64}, Vector{Float64}}
    error("compute_mass_and_force_terms not implemented for robot type $(typeof(robot))")
end

function compute_mass_and_force_terms(
    robot::SerialManipulator,
    q::Vector{Float64},
    qd::Vector{Float64}
)
    n = robot.dof
    M = zeros(n,n)

    for i in 1:n
        qdd = zeros(n); qdd[i] = 1.0
        M[:,i] = newton_euler(robot, q, zeros(n), qdd)
    end

    g = newton_euler(robot, q, zeros(n), zeros(n))
    c = newton_euler(robot, q, qd, zeros(n)) - g

    return M, c
end

"""
AbstractCartesianPath type representing a trajectory path.
"""
abstract type AbstractCartesianPath end

"""
Evaluate point on path at parameter t ∈ [0, 1].

# Arguments
- `path::AbstractPath`: Geometric path
- `t::Float64`: Path parameter in [0, 1]

# Returns
- `Vector{Float64}`: Point on path at parameter t
"""
function evaluate_path(path::AbstractCartesianPath, t::Float64)::Vector{Float64}
    error("evaluate_path not implemented for path type $(typeof(path))")
end

abstract type  AbstractJointPath end
"""
Evaluate joint space path at parameter t ∈ [0, 1].
# Arguments
- `path::AbstractJointPath`: Joint space path
- `t::Float64`: Path parameter in [0, 1]
- `derivative::Int64=0`: Derivative order
# Returns
- `Vector{Float64}`: Joint value or it's derivative at parameter t
"""
function evaluate_path(path::AbstractJointPath, t::Float64, derivative::Int64=0)::Vector{Float64}
end


"""
Compute limit path speed along the joint path given trajectory constraints.
# Arguments
- `joint_path::AbstractJointPath`: Joint space path
- `constraints::TrajectoryConstraints`: Trajectory constraints
# Returns
- `Vector{Float64}`: Limit path speed at discretized points along the path
"""

function compute_limit_path_speed(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath, 
    constraints::TrajectoryConstraints)::Vector{Float64}
    error("compute_limit_phase_velocity not implemented for path type $(typeof(joint_path))")
end

function generate_joint_trajectory(
    robot::AbstractRobotManipulator,
    path_speed::Vector{Vector{Float64}},
    time_step::Float64,
)::TrajectoryResult
    error("generate_joint_trajectory not implemented for robot type $(typeof(robot))")
end

end # module