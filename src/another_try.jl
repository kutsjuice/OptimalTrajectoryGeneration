using LinearAlgebra
using StaticArrays

# Types
abstract type AbstractRobotManipulator end
abstract type AbstractLink end

struct Inertia
    mass::Float64
    Icm::SMatrix{3,3,Float64}
    c::SVector{3,Float64}
end

struct Link
    parent::Int
    pitch::Float64
    Xtree::SMatrix{6,6,Float64}
    inertia::Inertia
end

struct Model <: AbstractRobotManipulator
    N::Int
    links::Vector{Link}
    gravity::SVector{3,Float64}
end

dof(m::Model) = m.N

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
    theta::Vector{Float64}
    cartesian_trajectory::Matrix{Float64}
    feasible::Bool
end

abstract type AbstractCartesianPath end

struct BezierCartesianPath <: AbstractCartesianPath
    p0::SVector{3, Float64}
    p1::SVector{3, Float64}
    p2::SVector{3, Float64}
    p3::SVector{3, Float64}
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

function evaluate_path(path::AbstractCartesianPath, t::Float64)::Vector{Float64}
    error("evaluate_path not implemented for path type $(typeof(path))")
end

function evaluate_path(path::BezierCartesianPath, t::Float64)::Vector{Float64}
    @assert 0.0 ≤ t ≤ 1.0
    u = 1.0 - t
    p = u^3 * path.p0 + 3u^2*t * path.p1 + 3u*t^2 * path.p2 + t^3 * path.p3
    return p
end

abstract type AbstractJointPath end

struct BezierJointPath <: AbstractJointPath
    q0::Vector{Float64}
    q1::Vector{Float64}
    q2::Vector{Float64}
    q3::Vector{Float64}
end

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
        p = u^3 * path.q0 + 3u^2*t * path.q1 + 3u*t^2 * path.q2 + t^3 * path.q3
        return p
    elseif derivative == 1
        p = 3u^2 * (path.q1 - path.q0) + 6u*t * (path.q2 - path.q1) + 3t^2 * (path.q3 - path.q2)
        return p
    elseif derivative == 2
        p = 6u * (path.q2 - 2*path.q1 + path.q0) + 6t * (path.q3 - 2*path.q2 + path.q1)
        return p
    else
        error("Derivative type not supported")
    end
end

function quat_to_rot(q::SVector{4,Float64})
    w, x, y, z = q
    @SMatrix [
        1-2y^2-2z^2    2x*y - 2*w*z    2x*z + 2*w*y;
        2x*y + 2*w*z   1-2x^2-2z^2     2y*z - 2*w*x;
        2x*z - 2*w*y   2y*z + 2*w*x    1-2x^2-2y^2
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
    q0 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 0.0)), q_seed;
        max_iters=1000, damping_factor=0.01, tol=1e-4, verbose=false)
    q1 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 1/3  )), q0;
        max_iters=1000, damping_factor=0.01, tol=1e-4, verbose=false)
    q2 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 2/3  )), q1;
        max_iters=1000, damping_factor=0.01, tol=1e-4, verbose=false)
    q3 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 1.0  )), q2;
        max_iters=1000, damping_factor=0.01, tol=1e-4, verbose=false)
    return BezierJointPath(q0, q1, q2, q3)
end

function compute_limit_path_speed(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath,
    constraints::TrajectoryConstraints
)::Vector{Float64}
    N = 100
    theta = range(0.0, 1.0, length=N)
    velocity_profile = zeros(Float64, N)
    w_max = constraints.velocity_limits
    for i in 1:N
        dq_dtheta = evaluate_path(joint_path, theta[i], 1)
        v_limits = [abs(dq_dtheta[j]) > 1e-8 ? abs(w_max[j] / dq_dtheta[j]) : Inf for j in 1:length(dq_dtheta)]
        velocity_profile[i] = minimum(v_limits)
    end
    return velocity_profile
end

# Algebra
function skew(v::SVector{3,Float64})
    @SMatrix [
        0.0   -v[3]  v[2];
        v[3]   0.0  -v[1];
       -v[2]  v[1]   0.0
    ]
end

function unskew(S)
    Smat = SMatrix{3,3,Float64}(S)
    SVector(Smat[3,2], Smat[1,3], Smat[2,1])
end

const Z33 = @SMatrix zeros(3,3)
const I33 = SMatrix{3,3,Float64}(I)

# Spatial operators
function crm(v::SVector{6,Float64})
    ω   = SVector{3,Float64}(v[1:3])
    vlin = SVector{3,Float64}(v[4:6])
    top    = hcat(skew(ω),   Z33)
    bottom = hcat(skew(vlin), skew(ω))
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

crf(v::SVector{6,Float64}) = -crm(v)'

function spatial_transform(
    R::SMatrix{3,3,Float64},
    r::SVector{3,Float64}
)
    top = hcat(R, Z33)
    bottom = hcat(skew(r)*R, R)
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

function spatial_inertia(I::Inertia)
    m = I.mass
    c = I.c
    Ic = I.Icm
    C = skew(SVector{3,Float64}(c))    
    top_left = Ic + m * C * C'
    top_right = m * C
    bottom_left = m * C'
    bottom_right = m * I33
    top = hcat(top_left, top_right)
    bottom = hcat(bottom_left, bottom_right)
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

# Joint model
function jcalc(pitch::Float64, q::Float64)
    if pitch == 0.0
        E = @SMatrix [
            cos(q) -sin(q) 0.0
            sin(q)  cos(q) 0.0
            0.0     0.0    1.0
        ]
        XJ = spatial_transform(E, @SVector [0.0,0.0,0.0])
        S  = @SVector [0.0,0.0,1.0, 0.0,0.0,0.0]
    elseif pitch == Inf
        XJ = spatial_transform(I33, @SVector [0.0,0.0,q])
        S  = @SVector [0.0,0.0,0.0, 0.0,0.0,1.0]
    else
        E = @SMatrix [
            cos(q) -sin(q) 0.0
            sin(q)  cos(q) 0.0
            0.0     0.0    1.0
        ]
        XJ = spatial_transform(E, @SVector [0.0,0.0,pitch*q])
        S  = @SVector [0.0,0.0,1.0, 0.0,0.0,pitch]
    end
    return XJ, S
end

# Inverse dynamics (RNEA)
function inverse_dynamics(
    model::Model,
    q::Vector{Float64},
    qd::Vector{Float64},
    qdd::Vector{Float64}
)
    n = model.N
    v = Vector{SVector{6,Float64}}(undef, n)
    a = Vector{SVector{6,Float64}}(undef, n)
    f = Vector{SVector{6,Float64}}(undef, n)
    S = Vector{SVector{6,Float64}}(undef, n)
    Xup = Vector{SMatrix{6,6,Float64}}(undef, n)
    a0 = @SVector [0.0, 0.0, 0.0,
                   -model.gravity[1],
                   -model.gravity[2],
                   -model.gravity[3]]

    τ = zeros(n)
    # ----- forward recursion -----
    for i in 1:n
        XJ, S[i] = jcalc(model.links[i].pitch, q[i])
        Xup[i] = model.links[i].Xtree * XJ

        vJ = S[i] * qd[i]

        if model.links[i].parent == 0
            v[i] = vJ
            a[i] = Xup[i]*a0 + S[i]*qdd[i] + crm(v[i])*vJ
        else
            p = model.links[i].parent
            v[i] = Xup[i]*v[p] + vJ
            a[i] = Xup[i]*a[p] + S[i]*qdd[i] + crm(v[i])*vJ
        end
    end
    # ----- backward recursion -----
    for i in n:-1:1
        I = spatial_inertia(model.links[i].inertia)
        f[i] = I*a[i] + crf(v[i])*(I*v[i])

        if model.links[i].parent != 0
            p = model.links[i].parent
            f[p] += Xup[i]'*f[i]
        end

        τ[i] = S[i]' * f[i]
    end
    return τ
end

function compute_mass_matrix_and_force_terms(
    model::Model,
    q::Vector{Float64},
    qd::Vector{Float64}
)
    n = model.N
    M = zeros(n, n)
    bias = inverse_dynamics(model, q, qd, zeros(n))

    for j = 1:n
        qdd_unit = zeros(n); qdd_unit[j] = 1.0
        τ = inverse_dynamics(model, q, zeros(n), qdd_unit)
        M[:, j] .= τ
    end

    return M, bias
end

# Forward kinematics (spatial transform from base to body)
function forward_kinematics(model::Model, q::Vector{Float64}, body::Int=model.N)
    n = model.N
    Xa = Vector{SMatrix{6,6,Float64}}(undef, n)
    for i in 1:n
        XJ, _ = jcalc(model.links[i].pitch, q[i])
        Xup = model.links[i].Xtree * XJ
        if model.links[i].parent == 0
            Xa[i] = Xup
        else
            p = model.links[i].parent
            Xa[i] = Xup * Xa[p]
        end
    end
    Xa[body]
end

# Helper to convert homogeneous to Plucker (spatial transform)
function to_plucker(target_pose::SMatrix{4,4,Float64})
    R = SMatrix{3,3,Float64}(target_pose[1:3,1:3])
    p = SVector{3,Float64}(target_pose[1:3,4])
    spatial_transform(R, p)
end

# SE(3) log map (XtoV)
function XtoV(X::SMatrix{6,6,Float64})
    R = X[1:3,1:3]
    A = X[4:6,1:3]
    θ = acos(clamp((tr(R) - 1)/2, -1.0, 1.0))
    p = unskew(A * R')
    if θ < 1e-6
        ω = SVector{3,Float64}(0,0,0)
        v = SVector{3,Float64}(p)
    else
        ωsk = (θ / (2 * sin(θ))) * (R - R')
        ω = unskew(ωsk)
        a = (1 - cos(θ)) / θ^2
        b = (θ - sin(θ)) / θ^3
        Ginv = I33 - 0.5 * ωsk + (1/θ^2) * (1 - (θ * sin(θ)) / (2 * (1 - cos(θ)))) * ωsk^2
        v = Ginv * p
    end
    SVector{6,Float64}([ω; v])
end

# Inverse of a spatial transform (6x6 plucker/adjoint form)
# Compute inverse of spatial transform (Plücker coordinates)
function spatial_inverse(X::SMatrix{6,6,Float64})
    R = X[1:3,1:3]
    A = X[4:6,1:3]
    r = unskew(A * R')
    Rinv = R'
    top = hcat(Rinv, Z33)
    bottom = hcat(-Rinv * skew(r), Rinv)
    return SMatrix{6,6,Float64}(vcat(top, bottom))
end

# Body Jacobian
# Compute body-frame Jacobian: J[i] = adjoint(X_body^{-1}) * X[i] * S[i]
function body_jacobian(model::Model, body::Int, q::Vector{Float64})
    n = model.N
    Jb = zeros(6, n)
    Xa = Vector{SMatrix{6,6,Float64}}(undef, n)
    S_list = Vector{SVector{6,Float64}}(undef, n)
    # Forward pass: compute transforms and screw axes
    for i in 1:n
        XJ, S_list[i] = jcalc(model.links[i].pitch, q[i])
        Xup = model.links[i].Xtree * XJ
        if model.links[i].parent == 0
            Xa[i] = Xup
        else
            p = model.links[i].parent
            Xa[i] = Xup * Xa[p]
        end
    end
    # Jacobian: express each screw axis in body frame
    X_body = Xa[body]
    X_body_inv = spatial_inverse(X_body)
    for i in 1:n
        X_i = Xa[i]
        # Column i: adjoint(X_body^{-1}) * S[i]
        Jb[:,i] = X_body_inv * (X_i * S_list[i])
    end
    return Jb
end

# Forward kinematics - position only (end-effector)
function fk_position(model::Model, q::Vector{Float64})::SVector{3,Float64}
    X = forward_kinematics(model, q)
    R = X[1:3,1:3]
    p = unskew(X[4:6,1:3] * R')
    return SVector{3,Float64}(p)
end

# Forward kinematics - full SE(3) (position + orientation error in 6D)
function fk_full(model::Model, q::Vector{Float64})::Tuple{SVector{3,Float64}, SMatrix{3,3,Float64}}
    X = forward_kinematics(model, q)
    R = X[1:3,1:3]
    p = unskew(X[4:6,1:3] * R')
    return SVector{3,Float64}(p), R
end

# Orientation error (compact 3D representation: axis-angle)
function orientation_error(R_target::SMatrix{3,3,Float64}, R_current::SMatrix{3,3,Float64})::SVector{3,Float64}
    dR = R_target * R_current'
    # Extract axis-angle from rotation matrix
    θ = acos(clamp((tr(dR) - 1)/2, -1.0, 1.0))
    if θ < 1e-6
        return @SVector [0.0, 0.0, 0.0]
    else
        axis = @SVector [dR[3,2] - dR[2,3], dR[1,3] - dR[3,1], dR[2,1] - dR[1,2]] / (2 * sin(θ))
        return θ * axis
    end
end

# Jacobian - position and orientation (numerical, 6 x n)
function jacobian_full(model::Model, q::Vector{Float64}, h::Float64=1e-8)::Matrix{Float64}
    n = length(q)
    J = zeros(6, n)
    p0, R0 = fk_full(model, q)
    for i in 1:n
        q_plus = copy(q)
        q_plus[i] += rdcx 
        .
        p_plus, R_plus = fk_full(model, q_plus)cd
        .-
        . 
        J[1:3, i] = (p_plus - p0) / h
        J[4:6, i] = orientation_error(R_plus, R0) / h
    end
    return J
end

# Inverse kinematics (Newton method - position + orientation)
function inverse_kinematics(
    model::Model,
    target_pose::SMatrix{4,4,Float64},
    q0::Vector{Float64};
    tol::Float64=1e-6,
    max_iters::Int=100
)::Vector{Float64}
    target_pos = SVector{3,Float64}(target_pose[1:3, 4])
    target_R = SMatrix{3,3,Float64}(target_pose[1:3, 1:3])
    q = copy(q0)
    for _ in 1:max_iters
        p, R = fk_full(model, q)
        err_pos = p - target_pos
        err_orient = orientation_error(target_R, R)
        err = [err_pos; err_orient]
        if norm(err) < tol
            return q
        end
        J = jacobian_full(model, q)
        try
            q -= J \ err
        catch
            return fill(NaN, length(q0))
        end
    end
    return fill(NaN, length(q0))
end

function generate_joint_trajectory(
    robot::Model,
    joint_path::BezierJointPath,
    path_speed::Vector{Float64};
    time_step::Float64=0.01,
    constraints::TrajectoryConstraints
)::TrajectoryResult
    # Discretize path parameter
    N = length(path_speed)
    theta = range(0.0, 1.0, length=N)
    dtheta = theta[2] - theta[1]
    # Compute time from velocity profile (trapezoidal integration)
    time = zeros(Float64, N)
    for i in 2:N
        v_avg = (path_speed[i-1] + path_speed[i]) / 2
        if v_avg > 0
            time[i] = time[i-1] + dtheta / v_avg
        else
            time[i] = time[i-1]
        end
    end
    # Polynomial extrapolation for last point (if needed)
    if N >= 4
        A = hcat(ones(3), theta[end-3:end-1], theta[end-3:end-1].^2)
        coeffs = A \ time[end-3:end-1]
        time[end] = coeffs[1] + coeffs[2] * theta[end] + coeffs[3] * theta[end]^2
    end
    # Generate trajectory at specified time steps
    t_final = time[end]
    time_points = 0.0:time_step:t_final
    n_points = length(time_points)
    # Initialize result arrays
    dof = robot.N
    positions     = zeros(Float64, dof, n_points)
    velocities    = zeros(Float64, dof, n_points)
    accelerations = zeros(Float64, dof, n_points)
    torques       = zeros(Float64, dof, n_points)
    theta_traj    = zeros(Float64, n_points)
    # Simple linear interpolation for theta(t)
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
        # Get path parameter and its derivatives
        θ = theta_of_t(t)
        # Find nearest interval for finite differences
        idx_t = searchsortedlast(time, t)
        idx_t = clamp(idx_t, 1, N-2)
        dt1 = time[idx_t+1] - time[idx_t]
        dt2 = time[idx_t+2] - time[idx_t+1]
        θ1 = theta[idx_t]
        θ2 = theta[idx_t+1]
        θ3 = theta[idx_t+2]
        # First derivative (central difference when possible)
        if idx_t == 1
            dθ_dt = (θ2 - θ1) / dt1
        else
            dθ_dt = (θ3 - θ1) / (dt1 + dt2)
        end
        # Second derivative (approximation)
        if idx_t == 1 || idx_t >= N-2
            d2θ_dt2 = 0.0
        else
            d2θ_dt2 = 2 * ((θ3 - θ2)/dt2 - (θ2 - θ1)/dt1) / (dt1 + dt2)
        end
        # Joint configuration and derivatives w.r.t. path parameter θ
        q         = evaluate_path(joint_path, θ, 0)
        dq_dθ     = evaluate_path(joint_path, θ, 1)
        d2q_dθ2   = evaluate_path(joint_path, θ, 2)
        # Chain rule → joint velocities and accelerations
        dq_dt   = dq_dθ .* dθ_dt
        d2q_dt2 = d2q_dθ2 .* (dθ_dt^2) .+ dq_dθ .* d2θ_dt2
        # Compute joint torques using inverse dynamics
        τ = inverse_dynamics(robot, q, dq_dt, d2q_dt2)
        # Store results
        positions[:, idx]     .= q
        velocities[:, idx]    .= dq_dt
        accelerations[:, idx] .= d2q_dt2
        torques[:, idx]       .= τ
        theta_traj[idx]       = θ
    end
    # Compute cartesian trajectory (end-effector positions)
    cartesian_trajectory = zeros(Float64, 3, n_points)
    for i in 1:n_points
        X = forward_kinematics(robot, positions[:, i])
        R = X[1:3,1:3]
        p = unskew(X[4:6,1:3] * R')
        cartesian_trajectory[:, i] = p
    end
    # Check feasibility (torque limits only in this version)
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
        theta_traj,
        cartesian_trajectory,
        feasible
    )
end

# Example: SCARA
Icm1 = @SMatrix [
    0.01 0 0
    0    0.01 0
    0    0    0.02
]

Icm2 = @SMatrix [
    0.008 0 0
    0     0.008 0
    0     0     0.015
]

Icm3 = @SMatrix [
    0.002 0 0
    0     0.002 0
    0     0     0.001
]

link1 = Link(
    0, 0.0,
    spatial_transform(I33, @SVector [0.0,0.0,0.0]),
    Inertia(1.0, Icm1, @SVector [0.15,0.0,0.0])
)

link2 = Link(
    1, 0.0,
    spatial_transform(I33, @SVector [0.3,0.0,0.0]),
    Inertia(0.8, Icm2, @SVector [0.125,0.0,0.0])
)

link3 = Link(
    2, Inf,
    spatial_transform(I33, @SVector [0.25,0.0,0.0]),
    Inertia(0.5, Icm3, @SVector [0.0,0.0,-0.05])
)

scara = Model(
    3,
    [link1, link2, link3],
    @SVector [0.0,0.0,9.81]
)

# Tests
q   = [0.5, 0.3, 0.1]
qd  = [0.1, 0.2, 0.01]
qdd = [0.01,0.02,0.001]
τ = inverse_dynamics(scara, q, qd, qdd)
println("Joint efforts:")
println("τ1 = ", τ[1])
println("τ2 = ", τ[2])
println("f3 = ", τ[3])

q   = [0.0, 0.0, 0.0]
qd  = [0.0, 0.0, 0.0]
qdd = [0.0, 0.0, 0.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("Гравитационные моменты в покое:")
println(τ)

q   = [0.0, 0.0, 0.1]
qd  = [0.0, 0.0, 0.0]
qdd = [0.0, 0.0, 0.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("τ при q₃ = 0.1 м:", τ[3])

q   = [0.0, π/4, 0.0]
qd  = [1.0, 1.0, 0.0]
qdd = [0.0, 0.0, 0.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("Центробежные/кориолисовы:", round.(τ, digits=4))

q   = [0.0, 0.0, 0.0]
qd  = [0.0, 0.0, 0.0]
qdd = [2.0, 3.0, 4.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("Инерционные моменты:", round.(τ, digits=4))

q   = [π/6, π/3, 0.15]
qd  = [0.8, 1.2, 0.05]
qdd = [1.5, 2.0, 0.3]
τ = inverse_dynamics(scara, q, qd, qdd)
println("Полный пример:")
println(round.(τ, digits=4))


# Пример траектории
constraints = TrajectoryConstraints(
    [2.0, 2.0, 0.5],           # velocity limits
    [10.0, 10.0, 2.0],         # acceleration limits
    [50.0, 50.0, 20.0],        # torque limits
    [100.0, 100.0, 50.0],      # jerk limits
    ([-π, -π, -0.3], [π, π, 0.3])  # position limits
)

# Простая декартова кривая (от (0.4,0,0.1) до (0.4,0.2,0.1))
path_cart = BezierCartesianPath(
    SVector{3,Float64}(0.40, 0.00, 0.10),
    SVector{3,Float64}(0.40, 0.05, 0.15),
    SVector{3,Float64}(0.40, 0.15, 0.15),
    SVector{3,Float64}(0.40, 0.20, 0.10)
)

q_seed = [0.0, 0.0, 0.0]
joint_path = joint_path_from_cartesian_bezier(scara, path_cart, q_seed)

# Ограниченная скорость по пути (только по velocity limits)
path_speed = compute_limit_path_speed(scara, joint_path, constraints)

# Генерация траектории
result = generate_joint_trajectory(
    scara,
    joint_path,
    path_speed,
    time_step = 0.01,
    constraints = constraints
)

println("Точек в траектории: ", size(result.trajectory.positions, 2))
println("Макс. момент по осям: ", maximum(abs.(result.torques), dims=2))
println("Feasible: ", result.feasible)