module OptimalTrajectoryGeneration

using LinearAlgebra
using StaticArrays

export
    AbstractRobotManipulator,
    AbstractJointPath,
    AbstractCartesianPath,
    
    TrajectoryConstraints,
    JointTrajectory,
    TrajectoryResult,

    forward_kinematics,
    inverse_kinematics,
    jacobian,
    compute_mass_and_force_terms,
    
    evaluate_path,
    compute_limit_path_speed,
    generate_joint_trajectory

"""
Abstract type representing a robot manipulator and robot links.
Users should create concrete subtypes for their specific robots.
"""
abstract type AbstractRobotManipulator end
abstract type AbstractLink end

"""
Type representing a manipulator link with DH parameters.
"""
struct DHLink <: AbstractLink
    a::Float64
    alpha::Float64
    d::Float64
    theta::Float64
    com::SVector{3, Float64}
    inertia::SMatrix{3,3,Float64,9}
    mass::Float64
    is_revolute::Bool
end

"""
Type representing a robot with DH matrix.
"""
struct DHRobot <: AbstractRobotManipulator
    dh_params::Vector{DHLink}
    gravity::SVector{3, Float64}
    dof::Int
end

function DHRobot(links::Vector{DHLink}, gravity::SVector{3, Float64})
    dof = length(links)
    return DHRobot(links, gravity, dof)
end

"""
Compute local transformation matrix for a DH link given joint variable.
"""
function local_transform(joint_variable::Float64, link::DHLink)::SMatrix{4,4,Float64,16}
    theta = link.theta + joint_variable
    alpha = link.alpha
    a = link.a
    d = link.d
    T = @SMatrix [
        cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha)  a*cos(theta);
        sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha)  a*sin(theta);
        0           sin(alpha)             cos(alpha)             d;
        0           0                     0                     1
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
function forward_kinematics(
    robot::AbstractRobotManipulator, 
    joint_positions::Vector{Float64}, 
    initial_guess::Vector{Float64}=zeros(length(joint_positions));
    tolerance::Float64=1e-6,
    max_iterations::Int=100)::Vector{Float64}

    error("forward_kinematics not implemented for robot type $(typeof(robot))")

end

function forward_kinematics(
    robot::DHRobot,
    joint_positions::Vector{Float64},
)::SVector{3, Float64}
    T = SMatrix{4,4,Float64,16}(I)
    for (i, link) in enumerate(robot.dh_params)
        T_link = local_transform(joint_positions[i], link)
        T = T * T_link
    end
    position = @SVector [T[1,4], T[2,4], T[3,4]]
    return position
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
    initial_guess::Vector{Float64}=zeros(length(joint_positions));
    tolerance::Float64=1e-6,
    max_iterations::Int=100
)::Vector{Float64}
    pose = copy(target_pose)
    x = initial_guess
    for iter in 1:max_iterations
        current_pose = forward_kinematics(robot, x)
        error_vec = pose - current_pose
        if norm(error_vec) < tolerance
            return x
        end
        J = jacobian(robot, x)
        Δx = pinv(J) * error_vec
        x += Δx
    end
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
    robot::DHRobot,
    joint_positions::Vector{Float64},
    joint_velocities::Vector{Float64},
    joint_accelerations::Vector{Float64}
)::Vector{Float64}
    num_joints = robot.dof
    gravity_vector = robot.gravity
    robot_links = robot.links

    angular_velocities = Vector{SVector{3, Float64}}(undef, num_joints)
    angular_accelerations = Vector{SVector{3, Float64}}(undef, num_joints)
    linear_accelerations = Vector{SVector{3, Float64}}(undef, num_joints)
    com_accelerations = Vector{SVector{3, Float64}}(undef, num_joints)
    
    forces = Vector{SVector{3, Float64}}(undef, num_joints + 1)
    moments = Vector{SVector{3, Float64}}(undef, num_joints + 1)
    
    generalized_forces = zeros(Float64, num_joints)

    current_angular_velocity = @SVector [0.0, 0.0, 0.0]
    current_angular_acceleration = @SVector [0.0, 0.0, 0.0]
    current_linear_acceleration = gravity_vector

    local_z_axis = @SVector [0.0, 0.0, 1.0]
    
    for link_index in 1:num_joints
        current_link = robot_links[link_index]

        if current_link.is_revolute
            current_joint_angle = current_link.joint_angle + joint_positions[link_index]
            current_link_offset = current_link.link_offset
        else
            current_joint_angle = current_link.joint_angle
            current_link_offset = current_link.link_offset + joint_positions[link_index]
        end

        transformation_matrix = local_transform(
            current_joint_angle,
            current_link
        )

        rotation_matrix = transformation_matrix[1:3, 1:3]
        position_vector = transformation_matrix[1:3, 4]

        angular_velocity = rotation_matrix' * current_angular_velocity
        if current_link.is_revolute
            angular_velocity += local_z_axis * joint_velocities[link_index]
        end

        angular_acceleration = rotation_matrix' * current_angular_acceleration
        if current_link.is_revolute
            angular_acceleration += cross(rotation_matrix' * current_angular_velocity, 
                                          local_z_axis * joint_velocities[link_index])
            angular_acceleration += local_z_axis * joint_accelerations[link_index]
        end

        if link_index == 1
            linear_acceleration = rotation_matrix' * current_linear_acceleration
        else
            previous_position = position_vectors[link_index - 1]
            linear_acceleration = rotation_matrix' * (
                current_linear_acceleration +
                cross(current_angular_acceleration, previous_position) +
                cross(current_angular_velocity, cross(current_angular_velocity, previous_position))
            )
        end

        if !current_link.is_revolute
            linear_acceleration += local_z_axis * joint_accelerations[link_index] +
                                   2 * cross(angular_velocity, local_z_axis * joint_velocities[link_index])
        end

        com_acceleration = linear_acceleration +
                          cross(angular_acceleration, current_link.center_of_mass) +
                          cross(angular_velocity, cross(angular_velocity, current_link.center_of_mass))

        angular_velocities[link_index] = angular_velocity
        angular_accelerations[link_index] = angular_acceleration
        linear_accelerations[link_index] = linear_acceleration
        com_accelerations[link_index] = com_acceleration

        current_angular_velocity = angular_velocity
        current_angular_acceleration = angular_acceleration
        current_linear_acceleration = linear_acceleration

        if link_index == 1
            position_vectors = Vector{SVector{3, Float64}}(undef, num_joints)
        end
        position_vectors[link_index] = position_vector
    end

    # backward pass
    forces[num_joints + 1] = @SVector [0.0, 0.0, 0.0]
    moments[num_joints + 1] = @SVector [0.0, 0.0, 0.0]
    for link_index in num_joints:-1:1
        current_link = robot_links[link_index]

        if current_link.is_revolute
            current_joint_angle = current_link.joint_angle + joint_positions[link_index]
            current_link_offset = current_link.link_offset
        else
            current_joint_angle = current_link.joint_angle
            current_link_offset = current_link.link_offset + joint_positions[link_index]
        end
        transformation_matrix = local_transform(
            current_joint_angle,
            current_link
        )
        rotation_matrix = transformation_matrix[1:3, 1:3]
        position_vector = transformation_matrix[1:3, 4]
        force_at_com = current_link.mass * com_accelerations[link_index]
        moment_at_com = current_link.inertia * angular_accelerations[link_index] +
                        cross(angular_velocities[link_index], current_link.inertia * angular_velocities[link_index])
        
        if link_index == num_joints
            total_force = force_at_com
            total_moment = moment_at_com + cross(current_link.center_of_mass, force_at_com)
        else
            next_link = robot_links[link_index + 1]

            if next_link.is_revolute
                next_joint_angle = next_link.joint_angle + joint_positions[link_index + 1]
            else
                next_joint_angle = next_link.joint_angle
            end

            next_transformation_matrix = local_transform(
                next_joint_angle,
                next_link
            )
            next_rotation_matrix = next_transformation_matrix[1:3, 1:3]
            total_force = next_rotation_matrix * forces[link_index + 1] + force_at_com
            total_moment = moment_at_com +
                           next_rotation_matrix * moments[link_index + 1] +
                           cross(current_link.center_of_mass, force_at_com) +
                           cross(position_vector, next_rotation_matrix * forces[link_index + 1])
        end

        forces[link_index] = total_force
        moments[link_index] = total_moment
        if current_link.is_revolute
            generalized_forces[link_index] = dot(total_moment, local_z_axis)
        else
            generalized_forces[link_index] = dot(total_force, local_z_axis)
        end
    end
    return generalized_forces
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
    robot::DHRobot,
    joint_positions::Vector{Float64},
    joint_velocities::Vector{Float64}
)::Tuple{Matrix{Float64}, Vector{Float64}}
    num_joints = robot.dof
    # Compute mass matrix
    mass_matrix = zeros(Float64, num_joints, num_joints)
    zero_velocities = zeros(Float64, num_joints)
    for i in 1:num_joints
        test_acceleration = zeros(Float64, num_joints)
        test_acceleration[i] = 1.0

        mass_matrix[:, i] = newton_euler(
            robot,
            joint_positions,
            zero_velocities,
            test_acceleration
        )
    end

    symmetric_error = norm(mass_matrix - mass_matrix')
    if symmetric_error > 1e-10
        @warn "Mass matrix is not symmetric"
    end
    # Compute gravity vector
    gravity_vector = newton_euler(
        robot,
        joint_positions,
        zero_velocities,
        zero_velocities
    )

    # Compute Coriolis and centrifugal terms
    total_forces = newton_euler(
        robot,
        joint_positions,
        joint_velocities,
        zero_accelerations
    )

    coriolis_centrifugal = total_forces - gravity_vector

    return (mass_matrix, coriolis_centrifugal)
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