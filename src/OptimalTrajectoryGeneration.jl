module OptimalTrajectoryGeneration

using LinearAlgebra
using StaticArrays
using ForwardDiff

export
    AbstractRobotManipulator,
    AbstractLink,

    DHLink,
    DHRobotManipulator,

    TrajectoryConstraints,
    JointTrajectory,
    TrajectoryResult,

    forward_kinematics,
    jacobian,
    inverse_kinematics,
    compute_mass_and_force_terms,

    AbstractCartesianPath,
    AbstractJointPath,
    evaluate_path,
    compute_limit_path_speed,
    generate_joint_trajectory

"""
Abstract type representing a robot manipulator.
Users should create concrete subtypes for their specific robots.
"""
abstract type AbstractRobotManipulator end

"""
Abstract type representing a manipulator link.
"""
abstract type AbstractLink end

"""
Struct representing a DH  link.
"""
struct DHLink <: AbstractLink
    a::Float64
    alpha::Float64
    d::Float64
    theta::Float64
end

"""
Struct representing a robot manipulator using DH parameters.
"""
struct DHRobotManipulator <: AbstractRobotManipulator
    dh_params::Vector{DHLink}
    mass::Vector{Float64}
    inertia::Vector{Matrix{Float64}}
    gravity::Vector{Float64}
end

"""
Local joint DH transformation matrix.
"""
function local_transform(joint_variable::Float64, link::DHLink)::Matrix{Float64}
    theta = link.theta + joint_variable
    alpha = link.alpha
    a = link.a
    d = link.d

    T = [
        cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha)  a*cos(theta);
        sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha)  a*sin(theta);
        0       sin(alpha)         cos(alpha)         d;
        0       0              0              1
    ]
    return T
end

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
Compute forward kinematics (end-effector position).

Contract: must be implemented for each concrete manipulator.
"""
function forward_kinematics(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64}
)::Vector{Float64}
    error("forward_kinematics not implemented for $(typeof(robot))")
end

"""
Forward kinematics for serial DH manipulator.
"""
function forward_kinematics(
    robot::DHRobotManipulator,
    joint_positions::Vector{Float64}
)::Vector{Float64}
    T = SMatrix{4,4,Float64,16}(
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    )
    for (q, link) in zip(joint_positions, robot.dh_params)
        T *= local_transform(q, link)
    end
    pos = T[1:3,4]
    r = atan(T[2,3], T[3,3])
    p = -asin(-T[1,3])
    y = atan(T[1,2], T[1,1])
    return [pos; y; p; r]
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
    robot::AbstractRobotManipulator,
    target_pose::Vector{Float64},
    initial_guess::Vector{Float64};
    tolerance::Float64 = 1e-6,
    max_iterations::Int = 100
)::Vector{Float64}

    q = copy(initial_guess)

    for _ in 1:max_iterations
        r = forward_kinematics(robot, q) - target_pose
        norm(r) < tolerance && return q

        J = jacobian(robot, q)
        q -= J \ r
    end

    error("Inverse kinematics did not converge")
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

    f(q) = forward_kinematics(robot, q)
    return ForwardDiff.jacobian(f, joint_positions)
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