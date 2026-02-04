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
    spatial_inertia,
    SerialManipulator,

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
skew(v) = let w = SVector{3}(v)
    @SMatrix [
         0.0  -w[3]   w[2]
         w[3]  0.0   -w[1]
        -w[2]  w[1]   0.0
    ]
end

"""
Adjoint transformation matrix for SE(3).
"""
function adjoint(R::SMatrix{3,3,Float64}, p::SVector{3,Float64})
    [  R          zero(SMatrix{3,3,Float64})
      skew(p)*R   R                       ]
end

adjoint(R::AbstractMatrix, p::AbstractVector) = 
    adjoint(SMatrix{3,3,Float64}(R), SVector{3,Float64}(p))

"""
Spatial cross product operator for motion vectors.
"""
function ad(V::SVector{6,Float64})
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
    com::SVector{3,Float64},
    inertia_com::SMatrix{3,3,Float64}
)
    I3 = one(SMatrix{3,3,Float64})
    S = skew(com)
    
    A11 = inertia_com + mass * S * S'
    A12 = mass * S
    A21 = mass * S'
    A22 = mass * I3
    
    # Correct SMatrix constructor syntax
    return SMatrix{6,6,Float64}(
        A11[1,1], A11[1,2], A11[1,3], A12[1,1], A12[1,2], A12[1,3],
        A11[2,1], A11[2,2], A11[2,3], A12[2,1], A12[2,2], A12[2,3],
        A11[3,1], A11[3,2], A11[3,3], A12[3,1], A12[3,2], A12[3,3],
        A21[1,1], A21[1,2], A21[1,3], A22[1,1], A22[1,2], A22[1,3],
        A21[2,1], A21[2,2], A21[2,3], A22[2,1], A22[2,2], A22[2,3],
        A21[3,1], A21[3,2], A21[3,3], A22[3,1], A22[3,2], A22[3,3]
    )
end

# Rigid body & robot definition (URDF-style)
"""
Rigid body with screw axis and spatial inertia.
"""
struct RigidBody <: AbstractLink
    screw_axis::SVector{6,Float64}          # Body screw axis
    X_parent::SMatrix{4,4,Float64}          # Transform to parent at zero config
    inertia::SMatrix{6,6,Float64}           # Spatial inertia
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
Matrix exponential for a screw axis.
"""
function exp_twist(S::SVector{6,Float64}, θ::Float64)
    ω = S[1:3]
    v = S[4:6]

    if norm(ω) < 1e-8
        R = one(SMatrix{3,3,Float64})
        p = v * θ
    else
        ω̂ = skew(ω)
        R = one(SMatrix{3,3,Float64}) + sin(θ)*ω̂ + (1-cos(θ))*(ω̂*ω̂)
        p = (one(SMatrix{3,3,Float64})*θ + (1-cos(θ))*ω̂ + (θ-sin(θ))*(ω̂*ω̂)) * v
    end

    @SMatrix [
        R[1,1] R[1,2] R[1,3] p[1];
        R[2,1] R[2,2] R[2,3] p[2];
        R[3,1] R[3,2] R[3,3] p[3];
        0.0    0.0    0.0    1.0
    ]
end

function inverse_se3(T::SMatrix{4,4,Float64})
    R = T[1:3, 1:3]
    p = T[1:3, 4]

    R_inv = R'
    p_inv = -R_inv * p  

    @SMatrix [
        R_inv[1,1] R_inv[1,2] R_inv[1,3] p_inv[1];
        R_inv[2,1] R_inv[2,2] R_inv[2,3] p_inv[2];
        R_inv[3,1] R_inv[3,2] R_inv[3,3] p_inv[3];
        0.0        0.0        0.0        1.0
    ]
end

cot(x) = 1 / tan(x)

function log_se3(T::SMatrix{4,4,Float64})
    R = T[1:3,1:3]
    p = T[1:3,4]
    trR = tr(R)
    
    if abs(trR - 3) < 1e-6
        omega = SVector{3,Float64}(0.0, 0.0, 0.0)
        v = p
    else
        cos_theta = (trR - 1) / 2
        theta = acos(clamp(cos_theta, -1.0, 1.0))
        sin_theta = sin(theta)
        if abs(sin_theta) < 1e-6
            if theta < 0.1
                omega = SVector{3,Float64}(0.0, 0.0, 0.0)
                v = p
            else  # theta ≈ π
                diag_part = [(R[1,1] + 1)/2, (R[2,2] + 1)/2, (R[3,3] + 1)/2]
                diag_part = max.(diag_part, 0.0)
                i = argmax(diag_part)
                n = zeros(SVector{3,Float64})
                n = setindex(n, sqrt(diag_part[i]), i)
                if n[i] > 1e-6
                    for j in 1:3
                        if j != i
                            n = setindex(n, R[i,j] / (2 * n[i]), j)
                        end
                    end
                end
                n = n / norm(n)
                omega = π * n
                theta = π
            end
        else
            omega_skew = (R - R') / (2 * sin(theta))
            omega = SVector{3,Float64}(
                omega_skew[3,2], 
                omega_skew[1,3], 
                omega_skew[2,1]
            ) * theta
            
            omega_hat = skew(omega / theta)
            # if abs(theta) > pi - 1e-2
            #     theta = theta - 2*pi * sign(theta - pi)
            # end
            half_theta = theta / 2
            cot_half = cot(half_theta)
            coef = 1 - half_theta * cot_half
            inv_J = one(SMatrix{3,3,Float64}) - 0.5 * omega_hat + (coef / (theta^2)) * (omega_hat * omega_hat)
            v = inv_J * p
        end
    end

    return SVector{6,Float64}(omega[1], omega[2], omega[3], v[1], v[2], v[3])
end

function inv_se3(T::SMatrix{4,4,Float64})
    R = T[1:3, 1:3]
    p = T[1:3, 4]
    R_inv = R'
    p_inv = -R_inv * p

    return SMatrix{4,4,Float64}(
        R_inv[1,1], R_inv[1,2], R_inv[1,3], p_inv[1],
        R_inv[2,1], R_inv[2,2], R_inv[2,3], p_inv[2],
        R_inv[3,1], R_inv[3,2], R_inv[3,3], p_inv[3],
        0.0,        0.0,        0.0,        1.0
    )
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
)::SMatrix{4,4,Float64}

    T = one(SMatrix{4,4,Float64})
    for i in 1:robot.dof
        T = robot.links[i].X_parent * exp_twist(robot.links[i].screw_axis, q[i]) * T
    end
    return T
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
    target_T::SMatrix{4,4,Float64},    
    q0::Vector{Float64}=zeros(robot.dof);
    tolerance=1e-8,
    max_iterations=500,
    damping_factor=1e-6
)
    q = copy(q0)
    λ = damping_factor
    last_error_norm = Inf
    for iter in 1:max_iterations
        T = forward_kinematics(robot, q)
        T_err = inv_se3(T) * target_T
        e = log_se3(T_err)
        error_norm = norm(e)
        if error_norm < tolerance
            return q
        end
        if iter > 1 && error_norm > last_error_norm * 1.1
            @warn "IK diverging at iteration $iter: error increased from $last_error_norm to $error_norm"
            break
        end
        last_error_norm = error_norm
        J = jacobian(robot, q)
        n = size(J, 2)
        JTJ = J' * J
        DLS_matrix = JTJ + λ^2 * Matrix{Float64}(I, n, n)
        Δq = DLS_matrix \ (J' * e) 
        step_norm = norm(Δq)
        α = min(1.0, 0.5 / max(step_norm, 1e-10))
        q += α * Δq
        
        for i in 1:length(q)
            while q[i] > π
                q[i] -= 2π
            end
            while q[i] < -π
                q[i] += 2π
            end
        end
        if iter % 50 == 0 && λ > 1e-8
            λ *= 0.5
        end
    end
        @warn "IK did not fully converge after $max_iterations iterations. Final error: $last_error_norm"
    return q
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
    ForwardDiff.jacobian(q -> forward_kinematics(robot, q)[1:3,4], joint_positions)
end

function jacobian(
    robot::SerialManipulator,
    q::Vector{Float64}
)::Matrix{Float64}
    n = robot.dof
    J = Matrix{Float64}(undef, 6, n)
    Ad_cum = one(SMatrix{6,6,Float64})
    for i = n:-1:1
        X = robot.links[i].X_parent * exp_twist(robot.links[i].screw_axis, q[i])
        J[:, i] = Ad_cum * robot.links[i].screw_axis
        Ad_cum = Ad_cum * adjoint(X[1:3,1:3], X[1:3,4])
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

    V  = fill(SVector{6,Float64}(zeros(6)), n)
    Vd = fill(SVector{6,Float64}(zeros(6)), n)
    F  = fill(SVector{6,Float64}(zeros(6)), n)
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
            X_twist = exp_twist(S_next, q[i+1])
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

struct BezierQuaternionPath
    q0::SVector{4,Float64}
    q1::SVector{4,Float64}
    q2::SVector{4,Float64}
    q3::SVector{4,Float64}
end

struct BezierSE3Path
    pos_path::BezierCartesianPath
    quat_path::BezierQuaternionPath
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

    return p
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
        p = 3u^2 * (path.q1 - path.q0) + 6u*t * (path.q2 - path.q1) + t^3 * (path.q3 - path.q2)
        return p
    elseif derivative == 2
        p = 6u * (path.q2 - 2path.q1 + path.q0) + 6t * (path.q3 - 2path.q2 +path.q1)
        return p
    else
        error("Derivative type not supported")
    end
end

function quat_to_rot(q::SVector{4,Float64})
    w, x, y, z = q

    @SMatrix [
        1-2y^2-2z^2    2x y - 2 w z    2x z + 2 w y;
        2x y + 2 w z   1-2x^2-2z^2     2y z - 2 w x;
        2x z - 2 w y   2y z + 2 w x    1-2x^2-2y^2
    ]
end

function joint_path_from_cartesian_bezier(
    robot::AbstractRobotManipulator,
    cartesian_path::BezierCartesianPath,
    q_seed::Vector{Float64}
)::BezierJointPath
    
    default_quat = @SVector [1.0, 0.0, 0.0, 0.0]  # w,x,y,z

    function make_full_pose(pos_vec)
        pos = SVector{3,Float64}(pos_vec)
        R = quat_to_rot(default_quat)
        @SMatrix [
            R[1,1] R[1,2] R[1,3] pos[1];
            R[2,1] R[2,2] R[2,3] pos[2];
            R[3,1] R[3,2] R[3,3] pos[3];
            0.0    0.0    0.0    1.0
        ]
    end

    q0 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 0.0)), q_seed)
    q1 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 1/3  )), q0)
    q2 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 2/3  )), q1)
    q3 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 1.0  )), q2)

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
    
    # Simple linear interpolation for theta(t) for now
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
            dθ_dt = (θ2 - θ1) / dt
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