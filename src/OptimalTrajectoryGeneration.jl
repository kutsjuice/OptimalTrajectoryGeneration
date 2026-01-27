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
    com::SVector{3, Float64}
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
Abstract constraint-based manipulator
"""
abstract type AbstractConstraintManipulator <: AbstractRobotManipulator end

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
Kinematic constraint equations Φ(q, x)
"""
function kinematic_constraints(
    robot::AbstractConstraintManipulator,
    q::Vector{Float64},
    x::Vector{Float64}
)::Vector{Float64}
    error("kinematic_constraints not implemented for $(typeof(robot))")
end

"""
Dimension of Cartesian state x
"""
function cartesian_dimension(
    robot::AbstractConstraintManipulator
)::Int
    error("cartesian_dimension not implemented")
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
"""
function forward_kinematics(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64};
    initial_guess::Vector{Float64},
    tolerance::Float64 = 1e-6,
    max_iterations::Int = 100
)
    error("forward_kinematics not implemented for $(typeof(robot))")
end

"""
Compute forward kinematics for constraint-based manipulators.
"""
function solve_forward_kinematics(
    robot::AbstractConstraintManipulator,
    joint_positions::Vector{Float64};
    initial_guess::Vector{Float64},
    tolerance::Float64 = 1e-6,
    max_iterations::Int = 100
)::Vector{Float64}

    x = copy(initial_guess)

    for _ in 1:max_iterations
        Φ = kinematic_constraints(robot, q, x)

        if norm(Φ) < tolerance
            return x
        end

        J = ForwardDiff.jacobian(
            ξ -> kinematic_constraints(robot, q, ξ),
            x
        )

        x -= J \ Φ
    end

    error("Forward kinematics did not converge")
end

function forward_kinematics(
    robot::AbstractConstraintManipulator,
    q::Vector{Float64};
    initial_guess::Vector{Float64},
    tolerance::Float64 = 1e-6,
    max_iterations::Int = 100
)::Vector{Float64}

    return solve_forward_kinematics(
        robot,
        q;
        initial_guess = initial_guess,
        tolerance = tolerance,
        max_iterations = max_iterations
    )
end

"""
Compute forward kinematics mapping joint positions to end-effector position.
"""
function forward_kinematics(
    robot::DHRobotManipulator,
    q::Vector{Float64};
    initial_guess::Vector{Float64},
    tolerance::Float64 = 1e-6,
    max_iterations::Int = 100
)::Vector{Float64}

    T = Matrix{Float64}(I, 4, 4)

    for (qi, link) in zip(q, robot.dh_params)
        T *= local_transform(qi, link)
    end

    return T[1:3, 4]
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
    robot::AbstractConstraintManipulator,
    target_pose::Vector{Float64},
    q0::Vector{Float64},
    x0::Vector{Float64};
    tolerance::Float64 = 1e-6,
    max_iterations::Int = 100
)::Vector{Float64}

    q = copy(q0)
    x = copy(x0)

    for _ in 1:max_iterations
        fk = forward_kinematics(
            robot,
            q;
            initial_guess = x,
            tolerance = tolerance,
            max_iterations = max_iterations
        )

        r = fk - target_pose

        if norm(r) < tolerance
            return q
        end

        J = jacobian(robot, q)
        q -= J \ r

        # warm-start для следующей FK
        x = fk
    end

    error("Inverse kinematics did not converge")
end

function inverse_kinematics(
    robot::DHRobotManipulator,
    target_pose::Vector{Float64},
    q0::Vector{Float64};
    tolerance::Float64 = 1e-6,
    max_iterations::Int = 100
)::Vector{Float64}

    q = copy(q0)

    for _ in 1:max_iterations
        r = forward_kinematics(
            robot,
            q;
            initial_guess = zeros(3)
        ) - target_pose

        if norm(r) < tolerance
            return q
        end

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
Return Cartesian position of the center of mass of link i
"""
function link_com_position(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64},
    i::Int
)::Vector{Float64}
    error("link_com_position not implemented for $(typeof(robot))")
end

function link_com_position(
    robot::DHRobotManipulator,
    joint_positions::Vector{Float64},
    i::Int
)::Vector{Float64}

    T = Matrix{Float64}(I, 4, 4)

    for k in 1:i
        T *= local_transform(joint_positions[k], robot.dh_params[k])
    end

    r_local = vcat(robot.dh_params[i].com, 1.0)

    r_world = T * r_local
    return r_world[1:3]
end

function link_com_position(
    robot::AbstractConstraintManipulator,
    joint_positions::Vector{Float64},
    i::Int
)::Vector{Float64}

    x = solve_forward_kinematics(robot, joint_positions; initial_guess=robot.x0)
    return extract_link_com(robot, x, i)
end

"""
Return spatial Jacobian of link i center of mass
"""
function link_com_jacobian(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64},
    i::Int
)::Matrix{Float64}
    error("link_com_jacobian not implemented for $(typeof(robot))")
end

function link_com_jacobian(
    robot::DHRobotManipulator,
    joint_positions::Vector{Float64},
    i::Int
)::Matrix{Float64}

    f(q_) = link_com_position(robot, q_, i)
    return ForwardDiff.jacobian(f, joint_positions)
end

function link_com_jacobian(
    robot::AbstractConstraintManipulator,
    joint_positions::Vector{Float64},
    i::Int
)::Matrix{Float64}

    f(q_) = link_com_position(robot, q_, i)
    return ForwardDiff.jacobian(f, joint_positions)
end

function kinetic_energy(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64},
    joint_velocities::Vector{Float64}
)::Float64

    T = 0.0
    n_links = length(robot.mass)

    for i in 1:n_links
        J = link_com_jacobian(robot, joint_positions, i)
        v = J * joint_velocities
        T += 0.5 * robot.mass[i] * dot(v, v)
    end

    return T
end

function mass_matrix(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64}
)::Matrix{Float64}

    n = length(joint_positions)

    function T_wrapped(joint_velocities)
        kinetic_energy(robot, joint_positions, joint_velocities)
    end

    M = ForwardDiff.hessian(T_wrapped, zeros(n))
    return M
end

function potential_energy(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64}
)::Float64

    P = 0.0
    g = robot.gravity

    for i in 1:length(robot.mass)
        r = link_com_position(robot, joint_positions, i)
        P += robot.mass[i] * dot(g, r)
    end

    return P
end

function gravity_terms(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64}
)::Vector{Float64}

    function P_wrapped(q)
        potential_energy(robot, q)
    end

    g = ForwardDiff.gradient(P_wrapped, joint_positions)
    return g
end

function coriolis_terms(
    robot::AbstractRobotManipulator,
    joint_positions::Vector{Float64},
    joint_velocities::Vector{Float64}
)::Vector{Float64}

    M = mass_matrix(robot, joint_positions)
    n = length(joint_positions)
    C = zeros(n)

    for k in 1:n
        for i in 1:n
            for j in 1:n
                C[k] += 0.5 * (
                    ForwardDiff.derivative(ξ -> mass_matrix(robot, ξ)[k,j], joint_positions)[i] +
                    ForwardDiff.derivative(ξ -> mass_matrix(robot, ξ)[k,i], joint_positions)[j] -
                    ForwardDiff.derivative(ξ -> mass_matrix(robot, ξ)[i,j], joint_positions)[k]
                ) * joint_velocities[i] * joint_velocities[j]
            end
        end
    end

    return C
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

    M = mass_matrix(robot, joint_positions)
    g = gravity_terms(robot, joint_positions)
    c = coriolis_terms(robot, joint_positions, joint_velocities)

    h = c + g
    return M, h
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