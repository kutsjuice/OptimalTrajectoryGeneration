"""
Time-Optimal Path Parametrization (TOPP) via forward/backward reachability
analysis (Pham/Slotine-style "bang-bang" sweep). This is the generic core
extracted from the original SCARA script: there it was hand-written for a
fixed 4-"joint" (incl. virtual) system with a constant mass matrix. Here it
works for any `robot::AbstractRobotManipulator` and any `dof`, by querying
`compute_mass_and_force_terms` at each path point instead of hardcoding M.

The algorithm finds the fastest θ̇(θ) profile (θ ∈ [0,1] = path parameter)
such that, along the joint path, neither joint velocity limits nor joint
torque limits are violated.
"""

"""
    compute_limit_path_speed(robot, joint_path, constraints, theta) -> Vector{Float64}

Velocity-limit-only bound on θ̇ at each θ in `theta`: the largest θ̇ such
that |q̇_j| = |ψ_j'(θ)| · θ̇ ≤ velocity_limits[j] for every joint j.
This ignores torque limits — it's the first, cheap bound; torque limits are
handled by the forward/backward pass below.
"""
function compute_limit_path_speed(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath,
    constraints::TrajectoryConstraints,
    theta::AbstractVector{Float64},
)::Vector{Float64}
    d = length(constraints.velocity_limits)
    vel_profile = fill(Inf, length(theta))
    for (i, t) in enumerate(theta)
        dpsi = evaluate_path(joint_path, t, 1)
        for j in 1:d
            if abs(dpsi[j]) > 1e-12
                vel_profile[i] = min(vel_profile[i], abs(constraints.velocity_limits[j] / dpsi[j]))
            end
        end
    end
    return max.(vel_profile, 1e-10)
end
function _initial_bound(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath,
    constraints::TrajectoryConstraints,
    theta::AbstractVector,
    ds::Float64,
    side::Symbol,
)::Float64
    t = side == :start ? theta[1] : theta[end]
    dpsi = evaluate_path(joint_path, t, 1)
    ddpsi = evaluate_path(joint_path, t, 2)
    q = evaluate_path(joint_path, t, 0)
    dq = zeros(length(q))  # v=0
    M, _ = compute_mass_and_force_terms(robot, q, dq)
    v_b = M * dpsi
    a_b = M * ddpsi
    Tmax = constraints.torque_limits
    th_buf = Inf
    for j in eachindex(Tmax)
        if abs(v_b[j]) > 1e-12 && isfinite(Tmax[j])
            denom = 0.5 * abs(v_b[j]) / ds + 0.25 * abs(a_b[j])
            if denom > 0
                th_buf = min(th_buf, sqrt(abs(Tmax[j] / denom)))
            end
        end
    end
    return isinf(th_buf) ? 0.0 : th_buf
end

"""
Torque-feasible range [a_min, a_max] for θ̈ at path point `theta_mid`, given
current path speed `v` (≈θ̇). Internally maps to actual joint
position/velocity/acceleration via the joint path's derivatives and the
robot's dynamics (`τ = M·q̈ + h`), then back-projects the per-joint torque
box constraints onto the scalar θ̈ axis.
"""
function _torque_bounds(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath,
    constraints::TrajectoryConstraints,
    theta_mid::Float64,
    v::Float64,
)
    q = evaluate_path(joint_path, theta_mid, 0)
    dpsi = evaluate_path(joint_path, theta_mid, 1)
    ddpsi = evaluate_path(joint_path, theta_mid, 2)
    dq = dpsi .* v

    M, h = compute_mass_and_force_terms(robot, q, dq)
    f1 = M * dpsi                     # coefficient of θ̈ in τ(θ̈)
    f2 = M * (ddpsi .* v^2) .+ h       # remaining (velocity-dependent + bias) term

    Tmax = constraints.torque_limits
    a_min, a_max = -Inf, Inf
    for j in eachindex(Tmax)
        (abs(f1[j]) < 1e-12 || isinf(Tmax[j])) && continue
        if f1[j] > 0
            a_min = max(a_min, (-Tmax[j] - f2[j]) / f1[j])
            a_max = min(a_max, (Tmax[j] - f2[j]) / f1[j])
        else
            a_min = max(a_min, (Tmax[j] - f2[j]) / f1[j])
            a_max = min(a_max, (-Tmax[j] - f2[j]) / f1[j])
        end
    end
    return a_min, a_max
end

function _forward_pass(robot, joint_path, constraints, theta, vel_limit, ds)
    N = length(theta)
    v_f = zeros(N)
    v_f[1] = 0.0
    v_f[2] = _initial_bound(robot, joint_path, constraints, theta, ds, :start)
    feasible = true
    for i in 2:N-1
        dθ = theta[i+1] - theta[i]
        θ_mid = theta[i] + dθ / 2
        v_cur = v_f[i]
        a_min, a_max = _torque_bounds(robot, joint_path, constraints, θ_mid, v_cur)
        a_min > a_max && (feasible = false)
        v_nxt = sqrt(max(0.0, v_cur^2 + 2 * a_max * dθ))
        if v_nxt > vel_limit[i+1]
            a_req = (vel_limit[i+1]^2 - v_cur^2) / (2 * dθ)
            if a_req < a_min
                v_nxt = sqrt(max(0.0, v_cur^2 + 2 * a_min * dθ))
            elseif a_req <= a_max
                v_nxt = vel_limit[i+1]
            end
        end
        v_f[i+1] = max(0.0, v_nxt)
    end
    return v_f, feasible
end

function _backward_pass(robot, joint_path, constraints, theta, vel_limit, ds)
    N = length(theta)
    v_b = zeros(N)
    v_b[N] = 0.0
    v_b[N-1] = _initial_bound(robot, joint_path, constraints, theta, ds, :end)
    feasible = true
    for i in N-2:-1:2
        dθ = theta[i+1] - theta[i]
        θ_mid = theta[i] + dθ / 2
        v_nxt = v_b[i+1]
        a_min, a_max = _torque_bounds(robot, joint_path, constraints, θ_mid, v_nxt)
        a_min > a_max && (feasible = false)
        v_cur = sqrt(max(0.0, v_nxt^2 - 2 * a_min * dθ))
        if v_cur > vel_limit[i]
            a_req = (v_nxt^2 - vel_limit[i]^2) / (2 * dθ)
            if a_req > a_max
                v_cur = sqrt(max(0.0, v_nxt^2 - 2 * a_max * dθ))
            elseif a_req >= a_min
                v_cur = vel_limit[i]
            end
        end
        v_b[i] = max(0.0, v_cur)
    end
    return v_b, feasible
end

"""
    generate_joint_trajectory(robot, joint_path, constraints; n_points=2000) -> TrajectoryResult

Run the full TOPP pipeline on a precomputed `joint_path` (θ ∈ [0,1] →
joint configuration): velocity-limit bound, forward pass, backward pass,
take the pointwise minimum, then convert the resulting θ̇(θ) profile into
a time-parametrized `JointTrajectory` with positions/velocities/accelerations,
joint torques, a time vector, and the Cartesian trajectory.
"""
function generate_joint_trajectory(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath,
    constraints::TrajectoryConstraints;
    n_points::Int=2000,
)::TrajectoryResult
    theta = collect(range(0.0, 1.0, length=n_points))
    N = n_points
    d = length(constraints.velocity_limits)

    vel_limit = compute_limit_path_speed(robot, joint_path, constraints, theta)
    ds = theta[2] - theta[1]  # постоянный шаг
    v_f, feasible_f = _forward_pass(robot, joint_path, constraints, theta, vel_limit, ds)
    v_b, feasible_b = _backward_pass(robot, joint_path, constraints, theta, vel_limit, ds)
    path_velocity = min.(v_f, v_b)
    feasible = feasible_f && feasible_b

    positions = zeros(N, d)
    velocities = zeros(N, d)
    accelerations = zeros(N, d)
    torques = zeros(N, d)
    cart_dim = length(forward_kinematics(robot, evaluate_path(joint_path, theta[1], 0)))
    cartesian_trajectory = zeros(N, cart_dim)
    time_vector = zeros(N)

    for i in 1:N
        q = evaluate_path(joint_path, theta[i], 0)
        dpsi = evaluate_path(joint_path, theta[i], 1)
        positions[i, :] = q
        velocities[i, :] = dpsi .* path_velocity[i]
        cartesian_trajectory[i, :] = forward_kinematics(robot, q)
        if i > 1
            dθ = theta[i] - theta[i-1]
            time_vector[i] = path_velocity[i] > 1e-10 ?
                              time_vector[i-1] + dθ / (0.5 * (path_velocity[i] + path_velocity[i-1])) :
                              time_vector[i-1]
        end
    end

    for i in 2:N-1
        dθ = theta[i+1] - theta[i]
        θ_mid = theta[i] + dθ / 2
        v_mid = 0.5 * (path_velocity[i] + path_velocity[i+1])
        a_th = path_velocity[i] * (path_velocity[i+1] - path_velocity[i-1]) / (2*dθ)  
        
        q_mid = evaluate_path(joint_path, θ_mid, 0)
        dpsi = evaluate_path(joint_path, θ_mid, 1)
        ddpsi = evaluate_path(joint_path, θ_mid, 2)
        dq_mid = dpsi .* v_mid
        ddq_mid = ddpsi .* v_mid^2 .+ dpsi .* a_th

        accelerations[i, :] = ddq_mid
        M, h = compute_mass_and_force_terms(robot, q_mid, dq_mid)
        torques[i, :] = M * ddq_mid .+ h
    end
    accelerations[1, :] = accelerations[2, :]
    accelerations[N, :] = accelerations[N-1, :]
    torques[1, :] = torques[2, :]
    torques[N, :] = torques[N-1, :]

    trajectory = JointTrajectory(positions, velocities, accelerations)
    return TrajectoryResult(trajectory, torques, time_vector, cartesian_trajectory, path_velocity, feasible)
end

"""
    generate_joint_trajectory(robot, cart_path, q_init, constraints; n_points=2000) -> TrajectoryResult

Convenience overload: builds the joint-space path from a Cartesian path via
IK (see `build_joint_path`), then runs the full TOPP pipeline.
"""
function generate_joint_trajectory(
    robot::AbstractRobotManipulator,
    cart_path::AbstractCartesianPath,
    q_init::AbstractVector{Float64},
    constraints::TrajectoryConstraints;
    n_points::Int=2000,
)::TrajectoryResult
    theta = range(0.0, 1.0, length=n_points)
    joint_path, _ = build_joint_path(robot, cart_path, theta, q_init)
    return generate_joint_trajectory(robot, joint_path, constraints; n_points=n_points)
end
