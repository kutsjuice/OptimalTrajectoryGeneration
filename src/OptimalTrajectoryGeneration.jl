module OptimalTrajectoryGeneration

using LinearAlgebra
using StaticArrays
using ForwardDiff
using Statistics

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

function TrajectoryConstraints(dof::Int)
    TrajectoryConstraints(
        fill(1.0, dof),  # velocity_limits
        fill(1.0, dof),  # acceleration_limits
        fill(1.0, dof),  # torque_limits
        fill(1.0, dof),  # jerk_limits
        (fill(-π, dof), fill(π, dof))  # position_limits
    )
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

function inverse_se3(T::SMatrix{4,4,Float64})
    R = T[1:3, 1:3]
    p = T[1:3, 3:4]

    @SMatrix[
        R' (-R' * p);
        0.0 0.0 0.0 1.0
    ]
end

cot(x) = 1 / tan(x)

function log_se3(T::SMatrix{4,4,Float64})
    R = T[1:3,1:3]
    p = T[1:3,4]
    trR = tr(R)
    if abs(trR - 3) < 1e-6
        omega = @SVector zeros(3)
        v = p
    else
        cos_theta = (trR - 1) / 2
        theta = acos(clamp(cos_theta, -1.0, 1.0))
        if theta < 1e-6
            omega = @SVector zeros(3)
            v = p
        else
            omega_skew = (R - R') / (2 * sin(theta))
            omega = @SVector [omega_skew[3,2], omega_skew[1,3], omega_skew[2,1]] * theta
            omega_hat = skew(omega / theta)  # normalized
            if abs(theta) > pi - 1e-2
                theta = theta - 2*pi * sign(theta - pi)
            end
            half_theta = theta / 2
            cot_half = cot(half_theta)
            coef = 1 - half_theta * cot_half
            inv_J = I(3) - 0.5 * omega_hat + (coef / (theta^2)) * (omega_hat * omega_hat)
            v = inv_J * p
        end
    end
    @SVector [omega[1], omega[2], omega[3], v[1], v[2], v[3]]
end

function inv_se3(T::SMatrix{4,4,Float64})
    R = T[1:3, 1:3]
    p = T[1:3, 4]
    return @SMatrix [
        R'  -R' * p;
        0   0   0   1
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

    T
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
        J = jacobian(robot, q)
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

function jacobian(
    robot::SerialManipulator,
    q::Vector{Float64}
)::Matrix{Float64}
    n = robot.dof
    J = Matrix{Float64}(undef, 6, n)
    Ad_cum = SMatrix{6,6,Float64}(I)
    for i = n:-1:1
        J[:, i] = Vector(Ad_cum * robot.links[i].screw_axis)
        X_twist = exp_twist(robot.links[i].screw_axis, q[i])
        X_fixed = robot.links[i].X_parent
        X = X_twist * X_fixed
        Ad = adjoint(X[1:3,1:3], X[1:3,4])
        Ad_cum = Ad * Ad_cum
    end
    J
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
    g = @SVector [0.0, 0.0, 0.0,
              robot.gravity[1],
              robot.gravity[2],
              robot.gravity[3]]

    # Forward recursion
    for i in 1:n
        S = robot.links[i].screw_axis
        X_twist = exp_twist(S, q[i])
        X = X_twist * robot.links[i].X_parent
        X_inv = inv_se3(X)
        R_inv = X_inv[1:3,1:3]
        p_inv = X_inv[1:3,4]
        AdX = adjoint(R_inv, p_inv)

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
            S_next = robot.links[i+1].screw_axis
            X_twist = exp_twist(robot.links[i+1].screw_axis, q[i+1])
            X = X_twist * robot.links[i+1].X_parent
            X_inv = inv_se3(X)
            R_inv = X_inv[1:3,1:3]
            p_inv = X_inv[1:3,4]
            AdX = adjoint(R_inv, p_inv)
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
Abstract Bezier cartesian path based on AbstractCartesianPath
"""
struct BezierCartesianPath <: AbstractCartesianPath
    p0:: SVector{3, Float64}
    p1:: SVector{3, Float64}
    p2:: SVector{3, Float64}
    p3:: SVector{3, Float64}
end

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

function evaluate_path(path::BezierCartesianPath, t::Float64)::Vector{Float64}
    @assert 0.0 ≤ t ≤ 1.0

    u = 1.0 - t
    p = u^3 *path.p0 + 3u^2*t * path.p1 + 3u*t^2 * path.p2 + t^3 * path.p3

    return Vector(p)
end


abstract type  AbstractJointPath end

struct BezierJointPath <: AbstractJointPath
    q0::Vector{Float64}
    q1::Vector{Float64}
    q2::Vector{Float64}
    q3::Vector{Float64}
end

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
    error("evaluate_path not implemented for path type $(typeof(path))")
end

function evaluate_path(
    path::BezierJointPath,
    t::Float64,
    derivative::Int64 = 0
)::Vector{Float64}
    @assert 0.0 ≤ t ≤ 1.0
    @assert derivative ≥ 0
    u = 1.0 - t
    if derivative == 0
        p = u^3 *path.q0 + 3u^2*t * path.q1 + 3u*t^2 * path.q2 + t^3 * path.q3
        return p
    elseif derivative == 1
        p = 3u^2 * (path.q1 - path.q0) + 6u*t * (path.q2 - path.q1) + 3t^2 * (path.q3 - path.q2)
        return p
    elseif derivative == 2
        p = 6u * (path.q2 - 2path.q1 + path.q0) + 6t * (path.q3 - 2path.q2 +path.q1)
        return p
    else
        error("Derivative type not supported")
    end
end

function joint_path_from_cartesian_bezier(
    robot,
    cartesian_path::BezierCartesianPath,
    q_seed::Vector{Float64}
)::BezierJointPath
    q0 = inverse_kinematics(robot, evaluate_path(cartesian_path, 0.0), q_seed)
    q1 = inverse_kinematics(robot, evaluate_path(cartesian_path, 1/3), q0)
    q2 = inverse_kinematics(robot, evaluate_path(cartesian_path, 2/3), q1)
    q3 = inverse_kinematics(robot, evaluate_path(cartesian_path, 1.0), q2)
    return BezierJointPath(q0, q1, q2, q3)
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

function compute_limit_path_speed(
    robot::SerialManipulator,
    joint_path::BezierJointPath, 
    constraints::TrajectoryConstraints
)::Vector{Float64}
    # Discretize path parameter
    N = 100  # Number of discretization points
    theta = range(0.0, 1.0, length=N)
    
    # Initialize velocity profile
    velocity_profile = zeros(Float64, N)
    
    # Extract velocity limits from constraints
    w_max = constraints.velocity_limits
    
    for i in 1:N
        # Get derivative of joint configuration w.r.t. theta
        dq_dtheta = evaluate_path(joint_path, theta[i], 1)
        
        # Compute velocity limits for each joint
        v_limits = zeros(Float64, length(dq_dtheta))
        for j in 1:length(dq_dtheta)
            if abs(dq_dtheta[j]) > 1e-8
                v_limits[j] = abs(w_max[j] / dq_dtheta[j])
            else
                v_limits[j] = Inf
            end
        end
        
        # Take minimum as the limiting factor
        velocity_profile[i] = minimum(v_limits)
    end
    
    return velocity_profile
end

"""
Generate joint trajectory based on path speed profile.
# Arguments
- `robot::AbstractRobotManipulator`: Robot instance
- `joint_path::AbstractJointPath`: Joint space path
- `path_speed::Vector{Float64}`: Path speed profile
- `time_step::Float64`: Time discretization step
# Returns
- `TrajectoryResult`: Complete trajectory with positions, velocities, accelerations, and torques
"""
function generate_joint_trajectory(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath,
    path_speed::Vector{Float64},
    time_step::Float64
)::TrajectoryResult
    error("generate_joint_trajectory not implemented for robot type $(typeof(robot))")
end

function generate_joint_trajectory(
    robot::SerialManipulator,
    joint_path::BezierJointPath,
    path_speed::Vector{Float64},
    time_step::Float64
)::TrajectoryResult
    # Discretize path parameter
    N = length(path_speed)
    theta = range(0.0, 1.0, length=N)
    dtheta = theta[2] - theta[1]
    
    # Compute time from velocity profile (trapezoidal integration)
    time = zeros(Float64, N)
    for i in 2:N-1
        # Average velocity on the segment
        v_avg = (path_speed[i-1] + path_speed[i]) / 2
        if v_avg > 0
            time[i] = time[i-1] + dtheta / v_avg
        else
            time[i] = time[i-1]
        end
    end
    
    # Polynomial extrapolation for last point
    if N >= 4
        A = hcat(ones(3), theta[end-3:end-1], theta[end-3:end-1].^2)
        coeffs = A \ time[end-3:end-1]
        time[end] = coeffs[1] + coeffs[2] * theta[end] + coeffs[3] * theta[end]^2
    else
        time[end] = time[end-1] + dtheta / path_speed[end-1]
    end
    
    # Generate trajectory at specified time steps
    t_final = time[end]
    time_points = 0.0:time_step:t_final
    n_points = length(time_points)
    
    # Initialize result arrays
    dof = robot.dof
    positions = zeros(Float64, dof, n_points)
    velocities = zeros(Float64, dof, n_points)
    accelerations = zeros(Float64, dof, n_points)
    torques = zeros(Float64, dof, n_points)
    
    # Create interpolation function for theta(t)
    # Simple linear interpolation for now
    function theta_of_t(t)
        idx = searchsortedlast(time, t)
        if idx == 0
            return theta[1]
        elseif idx >= N
            return theta[end]
        else
            α = (t - time[idx]) / (time[idx+1] - time[idx])
            return (1 - α) * theta[idx] + α * theta[idx+1]
        end
    end
    
    # Compute trajectory for each time point
    for (idx, t) in enumerate(time_points)
        # Get theta and its derivatives at current time
        θ = theta_of_t(t)
        
        # Find time interval for finite differences
        idx_t = searchsortedlast(time, t)
        if idx_t == 0
            idx_t = 1
        elseif idx_t >= N-1
            idx_t = N-2
        end
        
        # Finite differences for dθ/dt and d²θ/dt²
        dt1 = time[idx_t+1] - time[idx_t]
        dt2 = time[idx_t+2] - time[idx_t+1]
        θ1 = theta[idx_t]
        θ2 = theta[idx_t+1]
        θ3 = theta[idx_t+2]
        
        # First derivative (central difference)
        if idx_t == 1
            dθ_dt = (θ2 - θ1) / dt1
        elseif idx_t >= N-1
            dθ_dt = (θ2 - θ1) / dt1
        else
            dθ_dt = (θ3 - θ1) / (dt1 + dt2)
        end
        
        # Second derivative
        if idx_t == 1 || idx_t >= N-1
            d2θ_dt2 = 0.0
        else
            d2θ_dt2 = 2 * ((θ3 - θ2)/dt2 - (θ2 - θ1)/dt1) / (dt1 + dt2)
        end
        
        # Get joint configuration and its derivatives
        q = evaluate_path(joint_path, θ, 0)
        dq_dθ = evaluate_path(joint_path, θ, 1)
        d2q_dθ2 = evaluate_path(joint_path, θ, 2)
        
        # Compute joint velocities and accelerations using chain rule
        dq_dt = dq_dθ * dθ_dt
        d2q_dt2 = d2q_dθ2 * (dθ_dt^2) + dq_dθ * d2θ_dt2
        
        # Compute torques using robot dynamics
        τ = newton_euler(robot, q, dq_dt, d2q_dt2)
        
        # Store results
        positions[:, idx] = q
        velocities[:, idx] = dq_dt
        accelerations[:, idx] = d2q_dt2
        torques[:, idx] = τ
    end
    
    # Compute cartesian trajectory
    cartesian_trajectory = zeros(Float64, 3, n_points)
    for i in 1:n_points
        T = forward_kinematics(robot, positions[:, i])
        cartesian_trajectory[:, i] = T[1:3, 4]
    end
    
    # Check constraints feasibility (simplified check)
    feasible = true
    for i in 1:dof
        if any(abs.(torques[i, :]) .> constraints.torque_limits[i])
            feasible = false
            break
        end
    end
    
    return TrajectoryResult(
        JointTrajectory(positions, velocities, accelerations),
        torques,
        time_points,
        cartesian_trajectory,
        feasible
    )
end

end