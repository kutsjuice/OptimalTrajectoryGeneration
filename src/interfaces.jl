"""
Abstract type representing a robot manipulator.
Users create concrete subtypes for their specific robots and implement
`forward_kinematics`, `jacobian`, and `compute_mass_and_force_terms` for them.
Everything else in the package (paths, kinematics, TOPP) works for any
robot satisfying this interface.
"""
abstract type AbstractRobotManipulator end

"""
Constraints for trajectory optimization.

# Fields
- `velocity_limits::Vector{Float64}`: Maximum joint velocities |q̇| ≤ v_max
- `acceleration_limits::Vector{Float64}`: Maximum joint accelerations (reserved, not yet used by TOPP)
- `torque_limits::Vector{Float64}`: Maximum joint torques |τ| ≤ τ_max (use `Inf` for an unconstrained DOF)
- `jerk_limits::Vector{Float64}`: Maximum joint jerks (optional, reserved)
- `position_limits::Tuple{Vector{Float64}, Vector{Float64}}`: Joint position limits (min, max)
"""
struct TrajectoryConstraints
    velocity_limits::Vector{Float64}
    acceleration_limits::Vector{Float64}
    torque_limits::Vector{Float64}
    jerk_limits::Vector{Float64}
    position_limits::Tuple{Vector{Float64},Vector{Float64}}
end

"""
Joint space trajectory representation.

# Fields
- `positions::Matrix{Float64}`: Joint positions [time_steps × dof]
- `velocities::Matrix{Float64}`: Joint velocities [time_steps × dof]
- `accelerations::Matrix{Float64}`: Joint accelerations [time_steps × dof]
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
- `torques::Matrix{Float64}`: Joint torques [time_steps × dof]
- `time_vector::Vector{Float64}`: Time stamp for each sample
- `cartesian_trajectory::Matrix{Float64}`: Cartesian trajectory [time_steps × task_dim]
- `path_velocity::Vector{Float64}`: θ̇(θ) — optimized path speed profile
- `feasible::Bool`: Whether torque/velocity constraints were satisfiable everywhere
"""
struct TrajectoryResult
    trajectory::JointTrajectory
    torques::Matrix{Float64}
    time_vector::Vector{Float64}
    cartesian_trajectory::Matrix{Float64}
    path_velocity::Vector{Float64}
    feasible::Bool
end

"""
    forward_kinematics(robot, joint_positions) -> Vector{Float64}

Map joint positions to end-effector pose (task-space coordinates).
Must be implemented for each concrete robot type.
"""
function forward_kinematics(robot::AbstractRobotManipulator, joint_positions::AbstractVector{Float64})::Vector{Float64}
    error("forward_kinematics not implemented for robot type $(typeof(robot))")
end

"""
    jacobian(robot, joint_positions) -> Matrix{Float64}

Task-space Jacobian ∂(forward_kinematics)/∂q evaluated at `joint_positions`.
Must be implemented for each concrete robot type.
"""
function jacobian(robot::AbstractRobotManipulator, joint_positions::AbstractVector{Float64})::Matrix{Float64}
    error("jacobian not implemented for robot type $(typeof(robot))")
end

"""
    compute_mass_and_force_terms(robot, joint_positions, joint_velocities) -> (M, h)

Manipulator dynamics in the form `τ = M(q)q̈ + h(q, q̇)`, where `h` bundles
Coriolis/centrifugal/gravity/friction terms. Must be implemented for each
concrete robot type. If your robot model is purely kinematic (no dynamics
needed), implement it to return an identity-like `M` and zero `h` — but then
torque limits in `TrajectoryConstraints` should be set to `Inf`.
"""
function compute_mass_and_force_terms(
    robot::AbstractRobotManipulator,
    joint_positions::AbstractVector{Float64},
    joint_velocities::AbstractVector{Float64},
)::Tuple{Matrix{Float64},Vector{Float64}}
    error("compute_mass_and_force_terms not implemented for robot type $(typeof(robot))")
end

"""
    dof(robot) -> Int

Number of degrees of freedom. Default implementation looks for a `dof` field;
override for robots that store this information differently.
"""
dof(robot::AbstractRobotManipulator) = robot.dof
