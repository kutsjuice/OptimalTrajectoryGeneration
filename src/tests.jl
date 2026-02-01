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
        X_twist = exp_twist(robot.links[i].screw_axis, q[i])
        X_fixed = robot.links[i].X_parent
        X = X_fixed * X_twist
        Ad = adjoint(X[1:3,1:3], X[1:3,4])
        Ad_cum = Ad * Ad_cum
    end
    J
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
    g = @SVector [0.0, 0.0, 0.0, robot.gravity[1], robot.gravity[2], robot.gravity[3]]

    # Forward recursion
    for i in 1:nameof
        S = robot.links[i].screw_axis
        X_twist = exp_twist(S, q[i])
        X = X_twist * robot.links[i].X_parent
        X_inv = inv_se3(X)
        R_inv = X_inv[1:3, 1:3]
        p_inv = X_inv[1:3, 4]
        AdX = adjoint(R_inv, p_inv)

        if i == 1
            V[i] *= S * qd[i]
            Vd[i] *= S * qdd[i] - g +ad(V[i]) * (S * qd[i])
        else
            AdX = adjoint(X[1:3, 1:3], X[1:3,4])
            V[i] = AdX * V[i-1] + S * qd[i]
            Vd[i] = AdX * Vd[i-1] + S * qdd[i] + ad(V[i]) * (S * qd[i])
        end
    end

    tau = zeros(n)
    
    #Backward recursion
    for i in n:-1:1
        I = robot.links[i].inertia
        F[i] = I * Vd[i] + ad(V[i])' * (I*V[i])

        if i < n
            S_next = rboto.links[i+1].screw_axis
            X_twist = exp.twist(S_next, q[i+1])
            X = X_twist * robot.links[i+1].X_parent
            X_inv = inv_se3(X)
            R_inv = X_inv[1:3, 1:3]
            p_inv = X_inv[1:3, 4]
            AdX = adjoint(R_inv, p_inv)
            F[i] += AdX' * F[i+1]
        end
        tau[i] = dot(robot.links[i].screw_axis, F[i])
    end
    return tau
end

function compute_mass_and_force_terms(
    robot,
    q,
    qd
)
    n = robot.dof
    M = zeros(n,n)
    for i in 1:n
        qdd = zeros(n)
        qdd[i] = 1
        M[:,i] = newton_euler(robot, q, qd, qdd)
    end
    g = newton_euler(robot, q, zeros(n), zeros(n))
    c = newton_euler(robot, q, qd, zeros(n)) - gcd
    return M,c
end

abstract type AbstractCartesianPath end

struct BezierCartesianPath <: AbstractCartesianPath
    p0:: SVector{3, Float64}
    p1:: SVector{3, Float64}
    p2:: SVector{3, Float64}
    p3:: SVector{3, Float64}
end

function evaluate_path(
    path::BezierCartesianPath,
    t
)
    @assert 0.0 <= t <= 1.0
    u = 1.0 - t
    p = u^3 * path.p0 + 3u^2*t * path.p1 + 3u*t^2 * path.p2 + t^3 * path.p3
    return Vector(p)
end


struct BezierJointPath <: AbstractCartesianPath
    q0:: SVector{3, Float64}
    q1:: SVector{3, Float64}
    q2:: SVector{3, Float64}
    q3:: SVector{3, Float64}
end

function evaluate_path(
    path::BezierJointPath,
    t,
    derivative
)
    @assert 0.0 ≤ t ≤ 1.0
    @assert derivative ≥ 0
    u = 1.0 - t
    if derivative == 0
        p = u^3 * path.q0 + 3u^2*t * path.q1 + 3u*t^2 * path.q2
        return p
    elseif derivative == 1
        p = 3u^2 * (path.q1 - path.q0) + 6u*t*(path.q2 - path.q1)
        return p
    elseif derivative == 2
        p = 6u * (path.q2 - 2path.q1 + path.q0) + 6t * (path.q3 - 2path.q2 + path.q1)
        return p
    else
        error("Derivative type not supported")
    end
end

function joint_path_from_cartesian_bezier(
    robot,
    cartesian_path::BezierCartesianPath,
    q_seed
)
    q0 = inverse_kinematics(robot, evaluate_path(cartesian_path, 0.0), q_seed)
    q1 = inverse_kinematics(robot, evaluate_path(cartesian_path, 1/3), q0)
    q2 = inverse_kinematics(robot, evaluate_path(cartesian_path, 2/3), q1)
    q3 = inverse_kinematics(robot, evaluate_path(cartesian_path, 1.0), q2)
    return BezierJointPath(q0, q1, q2, q3)
end

function compute_limit_path_speed(
    robot::SerialManipulator,
    joint_path::BezierJointPath, 
    constraints::TrajectoryConstraints
)::Vector{Float64}
    N = 100
    theta = range(0.0, 1.0, length=N)
    velocity_profile = zeros(Float64, N)
    w_max = constraints.velocity_limits
    for i in 1:N
        dq_dtheta = evaluate_path(joint_path, theta[i], 1)
        v_limits = zeros(Float64, length(dq_dtheta))
        for j in 1:length(dq_dtheta)
            if abs(dq_dtheta[j]) > 1e-8
                v_limits[j] = abs(w_max[j]/dq_dtheta[j])
            else
                v_limits = Inf
            end
        end
        velocity_profile[i] = minimum(v_limits)
    end
    return velocity_profile
end

function generate_joint_trajectory(
    robot::SerialManipulator,
    joint_path::AbstractJointPath,
    path_speed::Vector{Float64},
    time_step::Float64
)::TrajectoryResult
    N = length(path_speed)
    theta = range(0.0, 1.0, length=N)
    dtheta = theta[2] - theta[1]
    time = zeros(Float64, N)
    for i in 2:N-1
        v_avg = (path_speed[i-1] + path_speed[i]) / 2
        if v_avg > 0
            time[i] = time[i-1] + dtheta/v_avg
        else
            time[i] = time[i-1]
        end
    end
    if N >= 4
        A = hcat(ones(3), theta[end-3:end-1], theta[end-3:end-1].^2)
        coeffs = A \ time[end-3:end-1]
        time[end] = coeffs[1] + coeffs[2]*theta[end] + coeffs[3]*theta[end]^2
    else
        time[end] = time[end-1] + dtheta / path_speed[end-1]
    end

    t_final = time[end]
    time_points = 0.0:time_step:t_final
    n_points = length(time_points)

    dof = robot.dof
    positions = zeros(Float64, dof, n_points)
    velocities = zeros(Float64, dof, n_points)
    accelerations = zeros(Float64, dof, n_points)
    torques = zeros(Float64, dof, n_points)

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
    
    for (idx, t) in enumerate(time_points)
        θ = theta_of_t(t)
        idx_t = searchsortedlast(time, t)
        if idx_t == 0
            idx_t = 1
        elseif idx_t >= N-1
            idx_t = N-2
        end

        dt1 = time[idx_t+1] - time[idx_t]
        dt2 = time[idx_t+2] - time[idx_t+1]
        θ1 = theta[idx_t]
        θ2 = theta[idx_t+1]
        θ3 = theta[idx_t+2]
        if idx_t == 1
            dθ_dt = (θ2 - θ1) / dt1
        elseif idx_t >= N-1
            dθ_dt = (θ2 - θ1) / dt1
        else
            dθ_dt = (θ3 - θ1) / (dt1 + dt2)
        end
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
