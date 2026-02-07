using LinearAlgebra
using StaticArrays

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

abstract type AbstractRobotManipulator end
abstract type AbstractCartesianPath end
abstract type AbstractJointPath end

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
    time_vector::Vector{Float64}
    theta::Vector{Float64}
    cartesian_trajectory::Matrix{Float64}
    feasible::Bool
end

struct BezierCartesianPath <: AbstractCartesianPath
    p0::SVector{3,Float64}
    p1::SVector{3,Float64}
    p2::SVector{3,Float64}
    p3::SVector{3,Float64}
end

struct BezierJointPath <: AbstractJointPath
    q0::Vector{Float64}
    q1::Vector{Float64}
    q2::Vector{Float64}
    q3::Vector{Float64}
end

# ============================================================================
# BEZIER PATH EVALUATION
# ============================================================================

function evaluate_path(path::BezierCartesianPath, t::Float64)::Vector{Float64}
    @assert 0.0 ≤ t ≤ 1.0 "Path parameter t must be in [0, 1]"
    u = 1.0 - t
    p = u^3 * path.p0 + 3u^2*t * path.p1 + 3u*t^2 * path.p2 + t^3 * path.p3
    return Vector(p)
end

function evaluate_path(path::BezierJointPath, t::Float64, derivative::Int64=0)::Vector{Float64}
    @assert 0.0 ≤ t ≤ 1.0 "Path parameter t must be in [0, 1]"
    @assert derivative ≥ 0 "Derivative order must be non-negative"
    
    u = 1.0 - t
    if derivative == 0
        p = u^3 * path.q0 + 3u^2*t * path.q1 + 3u*t^2 * path.q2 + t^3 * path.q3
    elseif derivative == 1
        p = 3u^2 * (path.q1 - path.q0) + 6u*t * (path.q2 - path.q1) + 3t^2 * (path.q3 - path.q2)
    elseif derivative == 2
        p = 6u * (path.q2 - 2*path.q1 + path.q0) + 6t * (path.q3 - 2*path.q2 + path.q1)
    else
        error("Derivative $derivative not supported (only 0, 1, 2)")
    end
    return p
end

# ============================================================================
# SPATIAL ALGEBRA UTILITIES
# ============================================================================

function quat_to_rot(q::SVector{4,Float64})::SMatrix{3,3,Float64}
    w, x, y, z = q
    @SMatrix [
        1-2y^2-2z^2    2x*y - 2*w*z    2x*z + 2*w*y;
        2x*y + 2*w*z   1-2x^2-2z^2     2y*z - 2*w*x;
        2x*z - 2*w*y   2y*z + 2*w*x    1-2x^2-2y^2
    ]
end

function skew(v::SVector{3,Float64})::SMatrix{3,3,Float64}
    @SMatrix [
        0.0    -v[3]   v[2];
        v[3]    0.0   -v[1];
       -v[2]    v[1]    0.0
    ]
end

function unskew(S::SMatrix{3,3,Float64})::SVector{3,Float64}
    SVector{3,Float64}(S[3,2], S[1,3], S[2,1])
end

const Z33 = @SMatrix zeros(3,3)
const I33 = SMatrix{3,3,Float64}(I)

function crm(v::SVector{6,Float64})::SMatrix{6,6,Float64}
    ω = SVector{3,Float64}(v[1:3])
    vlin = SVector{3,Float64}(v[4:6])
    ωsk = skew(ω)
    vlisk = skew(vlin)
    top = hcat(ωsk, Z33)
    bottom = hcat(vlisk, ωsk)
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

function crf(v::SVector{6,Float64})::SMatrix{6,6,Float64}
    -crm(v)'
end

function spatial_transform(R::SMatrix{3,3,Float64}, r::SVector{3,Float64})::SMatrix{6,6,Float64}
    rsk = skew(r)
    top = hcat(R, Z33)
    bottom = hcat(rsk * R, R)
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

function spatial_inertia(I::Inertia)::SMatrix{6,6,Float64}
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

function spatial_inverse(X::SMatrix{6,6,Float64})::SMatrix{6,6,Float64}
    R = X[1:3,1:3]
    A = X[4:6,1:3]
    r = unskew(A * R')
    Rinv = R'
    top = hcat(Rinv, Z33)
    bottom = hcat(-Rinv * skew(r), Rinv)
    SMatrix{6,6,Float64}(vcat(top, bottom))
end

# ============================================================================
# JOINT CALCULATIONS
# ============================================================================

function jcalc(pitch::Float64, q::Float64)::Tuple{SMatrix{6,6,Float64}, SVector{6,Float64}}
    """Spatial joint transform and screw axis for revolute/prismatic/screw joints"""
    if pitch == 0.0  # revolute joint (rotation around z-axis)
        E = @SMatrix [
            cos(q) -sin(q) 0.0;
            sin(q)  cos(q) 0.0;
            0.0     0.0    1.0
        ]
        XJ = spatial_transform(E, @SVector [0.0, 0.0, 0.0])
        S = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    elseif pitch == Inf  # prismatic joint (translation along z-axis)
        XJ = spatial_transform(I33, @SVector [0.0, 0.0, q])
        S = @SVector [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    else  # screw joint (helical)
        E = @SMatrix [
            cos(q) -sin(q) 0.0;
            sin(q)  cos(q) 0.0;
            0.0     0.0    1.0
        ]
        XJ = spatial_transform(E, @SVector [0.0, 0.0, pitch*q])
        S = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, pitch]
    end
    return XJ, S
end

# ============================================================================
# KINEMATICS: FORWARD & INVERSE
# ============================================================================

function forward_kinematics(model::Model, q::Vector{Float64}, body::Int=model.N)::SMatrix{6,6,Float64}
    """Compute spatial transform from world to body frame"""
    @assert length(q) == model.N "Joint vector length doesn't match DOF"
    n = model.N
    Xa = Vector{SMatrix{6,6,Float64}}(undef, n)
    
    for i in 1:n
        XJ, _ = jcalc(model.links[i].pitch, q[i])
        Xup = model.links[i].Xtree * XJ
        
        if model.links[i].parent == 0
            Xa[i] = Xup
        else
            p = model.links[i].parent
            Xa[i] = Xa[p] * Xup  # Fixed: correct order for FK accumulation
        end
    end
    
    return Xa[body]
end

function fk_full(model::Model, q::Vector{Float64})::Tuple{SVector{3,Float64}, SMatrix{3,3,Float64}}
    """Compute end-effector position and rotation matrix"""
    X = forward_kinematics(model, q)
    R = X[1:3,1:3]
    A = SMatrix{3,3,Float64}(X[4:6,1:3] * R')
    p = unskew(A)
    return SVector{3,Float64}(p), R
end

function orientation_error(R_target::SMatrix{3,3,Float64}, R_current::SMatrix{3,3,Float64})::SVector{3,Float64}
    """Compute orientation error as axis-angle (3D compact form)"""
    dR = R_target * R_current'
    trace_dR = tr(dR)
    trace_dR = clamp(trace_dR, -1.0 + 1e-10, 3.0 - 1e-10)
    θ = acos((trace_dR - 1) / 2)
    
    if θ < 1e-6
        return @SVector [0.0, 0.0, 0.0]
    else
        sin2θ = 2 * sin(θ)
        axis = SVector(dR[3,2] - dR[2,3], dR[1,3] - dR[3,1], dR[2,1] - dR[1,2]) / sin2θ
        return θ * axis
    end
end

function jacobian_full(model::Model, q::Vector{Float64}, h::Float64=1e-8)::Matrix{Float64}
    """Compute numerical Jacobian: 6×n matrix coupling position & orientation"""
    n = length(q)
    J = zeros(6, n)
    p0, R0 = fk_full(model, q)
    
    for i in 1:n
        q_plus = copy(q)
        q_plus[i] += h
        p_plus, R_plus = fk_full(model, q_plus)
        
        J[1:3, i] = (p_plus - p0) / h
        J[4:6, i] = orientation_error(R_plus, R0) / h
    end
    
    return J
end

function inverse_kinematics(
    model::Model,
    target_pose::SMatrix{4,4,Float64},
    q0::Vector{Float64};
    tol::Float64=1e-6,
    max_iters::Int=100
)::Vector{Float64}
    """Solve IK using Newton-Raphson with damped least squares"""
    target_pos = SVector{3,Float64}(target_pose[1:3, 4])
    target_R = SMatrix{3,3,Float64}(target_pose[1:3, 1:3])
    q = copy(q0)
    best_err = Inf
    best_q = copy(q)
    
    for iter in 1:max_iters
        p, R = fk_full(model, q)
        err_pos = p - target_pos
        err_orient = orientation_error(target_R, R)
        err = [err_pos; err_orient]
        err_norm = norm(err)
        
        # Track best solution
        if err_norm < best_err
            best_err = err_norm
            best_q = copy(q)
        end
        
        if err_norm < tol
            return q
        end
        
        J = jacobian_full(model, q)
        
        try
            # Damped Least Squares (DLS) for better stability
            λ = 0.01  # damping factor
            H = J' * J + λ * I(length(q))
            dq = H \ (J' * err)
            
            # Adaptive step size with line search
            α = 1.0
            for LS in 1:5
                q_test = q - α * dq
                p_test, R_test = fk_full(model, q_test)
                err_test = [p_test - target_pos; orientation_error(target_R, R_test)]
                if norm(err_test) < err_norm
                    break
                end
                α *= 0.5
            end
            
            q = q - α * dq
        catch
            # Return best solution found if solver fails
            return best_q
        end
    end
    
    return best_q
end

# ============================================================================
# INVERSE DYNAMICS (RNEA) & MASS MATRIX
# ============================================================================

function inverse_dynamics(
    model::Model,
    q::Vector{Float64},
    qd::Vector{Float64},
    qdd::Vector{Float64}
)::Vector{Float64}
    """Recursive Newton-Euler Algorithm for inverse dynamics"""
    @assert length(q) == length(qd) == length(qdd) == model.N
    
    n = model.N
    v = Vector{SVector{6,Float64}}(undef, n)
    a = Vector{SVector{6,Float64}}(undef, n)
    f = Vector{SVector{6,Float64}}(undef, n)
    S = Vector{SVector{6,Float64}}(undef, n)
    Xup = Vector{SMatrix{6,6,Float64}}(undef, n)
    
    a0 = @SVector [
        0.0, 0.0, 0.0,
        -model.gravity[1], -model.gravity[2], -model.gravity[3]
    ]
    
    τ = zeros(n)
    
    # Forward pass: compute accelerations and velocities
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
    
    # Backward pass: compute forces and torques
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
)::Tuple{Matrix{Float64}, Vector{Float64}}
    """Compute mass matrix M and bias terms (Coriolis + gravity)"""
    n = model.N
    M = zeros(n, n)
    bias = inverse_dynamics(model, q, qd, zeros(n))
    
    for j = 1:n
        qdd_unit = zeros(n)
        qdd_unit[j] = 1.0
        τ = inverse_dynamics(model, q, zeros(n), qdd_unit)
        M[:, j] .= τ
    end
    
    return M, bias
end

# ============================================================================
# TRAJECTORY PLANNING & GENERATION
# ============================================================================

function compute_limit_path_speed(
    robot::AbstractRobotManipulator,
    joint_path::AbstractJointPath,
    constraints::TrajectoryConstraints
)::Vector{Float64}
    """Compute maximum path speed respecting joint velocity limits"""
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

function joint_path_from_cartesian_bezier(
    robot::AbstractRobotManipulator,
    cartesian_path::BezierCartesianPath,
    q_seed::Vector{Float64}
)::BezierJointPath
    """Convert Cartesian Bezier path to joint space using IK at 4 control points"""
    default_quat = @SVector [1.0, 0.0, 0.0, 0.0]  # identity quaternion (w,x,y,z)
    
    function make_full_pose(pos_vec::Vector{Float64})::SMatrix{4,4,Float64}
        pos = SVector{3,Float64}(pos_vec)
        R = quat_to_rot(default_quat)
        @SMatrix [
            R[1,1] R[1,2] R[1,3] pos[1];
            R[2,1] R[2,2] R[2,3] pos[2];
            R[3,1] R[3,2] R[3,3] pos[3];
            0.0    0.0    0.0    1.0
        ]
    end
    
    # Solve IK at 4 Bezier curve points
    q0 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 0.0)), q_seed; max_iters=1000, tol=1e-4)
    
    if any(isnan, q0)
        @warn "IK failed at Bezier point t=0.0, target may be unreachable"
        return BezierJointPath(q_seed, q_seed, q_seed, q_seed)
    end
    
    q1 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 1/3)), q0; max_iters=1000, tol=1e-4)
    if any(isnan, q1)
        @warn "IK failed at Bezier point t=1/3"
        q1 = q0
    end
    
    q2 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 2/3)), q1; max_iters=1000, tol=1e-4)
    if any(isnan, q2)
        @warn "IK failed at Bezier point t=2/3"
        q2 = q1
    end
    
    q3 = inverse_kinematics(robot, make_full_pose(evaluate_path(cartesian_path, 1.0)), q2; max_iters=1000, tol=1e-4)
    if any(isnan, q3)
        @warn "IK failed at Bezier point t=1.0"
        q3 = q2
    end
    
    return BezierJointPath(q0, q1, q2, q3)
end

function generate_joint_trajectory(
    robot::Model,
    joint_path::BezierJointPath,
    path_speed::Vector{Float64};
    time_step::Float64=0.01,
    constraints::TrajectoryConstraints
)::TrajectoryResult
    """Generate full trajectory: positions, velocities, accelerations, torques"""
    N = length(path_speed)
    theta = range(0.0, 1.0, length=N)
    dtheta = theta[2] - theta[1]
    
    # Integrate path speed to get time-to-parameter mapping
    time = zeros(Float64, N)
    for i in 2:N
        v_avg = (path_speed[i-1] + path_speed[i]) / 2
        if v_avg > 1e-10
            time[i] = time[i-1] + dtheta / v_avg
        else
            time[i] = time[i-1] + dtheta / 1e-6
        end
    end
    
    # Generate trajectory at regular time steps
    t_final = time[end]
    time_points = collect(0.0:time_step:t_final)
    if time_points[end] < t_final
        push!(time_points, t_final)
    end
    
    n_points = length(time_points)
    dof = robot.N
    
    positions = zeros(Float64, dof, n_points)
    velocities = zeros(Float64, dof, n_points)
    accelerations = zeros(Float64, dof, n_points)
    torques = zeros(Float64, dof, n_points)
    theta_traj = zeros(Float64, n_points)
    
    # Interpolate path parameter as function of time
    function theta_of_t(t_val)
        if t_val <= time[1]
            return theta[1]
        elseif t_val >= time[end]
            return theta[end]
        end
        
        idx = searchsortedlast(time, t_val)
        idx = clamp(idx, 1, N-1)
        
        if idx >= N
            return theta[end]
        end
        
        α = (t_val - time[idx]) / (time[idx+1] - time[idx])
        return (1 - α) * theta[idx] + α * theta[idx+1]
    end
    
    # Compute trajectory for each time point
    for (idx, t) in enumerate(time_points)
        θ = theta_of_t(t)
        
        # Find time interval and compute derivatives
        idx_t = searchsortedlast(time, t)
        idx_t = clamp(idx_t, 1, N-2)
        
        dt1 = time[idx_t+1] - time[idx_t]
        dt2 = time[idx_t+2] - time[idx_t+1]
        θ1 = theta[idx_t]
        θ2 = theta[idx_t+1]
        θ3 = theta[idx_t+2]
        
        # First and second derivatives of θ(t)
        if idx_t == 1
            dθ_dt = (θ2 - θ1) / dt1
        else
            dθ_dt = (θ3 - θ1) / (dt1 + dt2)
        end
        
        if idx_t == 1 || idx_t >= N-2
            d2θ_dt2 = 0.0
        else
            d2θ_dt2 = 2 * ((θ3 - θ2)/dt2 - (θ2 - θ1)/dt1) / (dt1 + dt2)
        end
        
        # Joint path parameters and derivatives w.r.t. θ
        q = evaluate_path(joint_path, θ, 0)
        dq_dθ = evaluate_path(joint_path, θ, 1)
        d2q_dθ2 = evaluate_path(joint_path, θ, 2)
        
        # Chain rule: convert path parameter derivatives to time derivatives
        dq_dt = dq_dθ .* dθ_dt
        d2q_dt2 = d2q_dθ2 .* (dθ_dt^2) .+ dq_dθ .* d2θ_dt2
        
        # Compute joint torques using RNEA
        τ = inverse_dynamics(robot, q, dq_dt, d2q_dt2)
        
        positions[:, idx] .= q
        velocities[:, idx] .= dq_dt
        accelerations[:, idx] .= d2q_dt2
        torques[:, idx] .= τ
        theta_traj[idx] = θ
    end
    
    # Compute cartesian end-effector trajectory
    cartesian_trajectory = zeros(Float64, 3, n_points)
    for i in 1:n_points
        X = forward_kinematics(robot, positions[:, i])
        R = X[1:3,1:3]
        A = SMatrix{3,3,Float64}(X[4:6,1:3] * R')
        p = unskew(A)
        cartesian_trajectory[:, i] = p
    end
    
    # Check feasibility against torque limits
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

# ============================================================================
# EXAMPLE: SCARA ROBOT
# ============================================================================

# Inertia matrices for 3 links
Icm1 = @SMatrix [
    0.01  0.0  0.0;
    0.0   0.01 0.0;
    0.0   0.0  0.02
]

Icm2 = @SMatrix [
    0.008 0.0   0.0;
    0.0   0.008 0.0;
    0.0   0.0   0.015
]

Icm3 = @SMatrix [
    0.002 0.0   0.0;
    0.0   0.002 0.0;
    0.0   0.0   0.001
]

# Define robot links
link1 = Link(
    0, 0.0,
    spatial_transform(I33, @SVector [0.0, 0.0, 0.0]),
    Inertia(1.0, Icm1, @SVector [0.15, 0.0, 0.0])
)

link2 = Link(
    1, 0.0,
    spatial_transform(I33, @SVector [0.3, 0.0, 0.0]),
    Inertia(0.8, Icm2, @SVector [0.125, 0.0, 0.0])
)

link3 = Link(
    2, Inf,
    spatial_transform(I33, @SVector [0.25, 0.0, 0.0]),
    Inertia(0.5, Icm3, @SVector [0.0, 0.0, -0.05])
)

scara = Model(
    3,
    [link1, link2, link3],
    @SVector [0.0, 0.0, 9.81]
)

# ============================================================================
# TEST & VALIDATION (disabled - run full_example.jl instead)
# ============================================================================

# Test 1: Inverse dynamics validation
#=
println("=== Inverse Dynamics Tests ===")
q   = [0.5, 0.3, 0.1]
qd  = [0.1, 0.2, 0.01]
qdd = [0.01, 0.02, 0.001]
τ = inverse_dynamics(scara, q, qd, qdd)
println("Joint efforts:\nτ1 = ", τ[1], "\nτ2 = ", τ[2], "\nf3 = ", τ[3])

# Test 2: Gravity at zero configuration
q   = [0.0, 0.0, 0.0]
qd  = [0.0, 0.0, 0.0]
qdd = [0.0, 0.0, 0.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("\nGravity load at HOME:")
println(τ)

# Test 3: Gravity compensation with extended link
q   = [0.0, 0.0, 0.1]
qd  = [0.0, 0.0, 0.0]
qdd = [0.0, 0.0, 0.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("\nGravity at q₃ = 0.1 m: ", τ[3])

# Test 4: Centrifugal/Coriolis effects
q   = [0.0, π/4, 0.0]
qd  = [1.0, 1.0, 0.0]
qdd = [0.0, 0.0, 0.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("\nCentrifugal/Coriolis effects: ", round.(τ, digits=4))

# Test 5: Inertial effects
q   = [0.0, 0.0, 0.0]
qd  = [0.0, 0.0, 0.0]
qdd = [2.0, 3.0, 4.0]
τ = inverse_dynamics(scara, q, qd, qdd)
println("\nInertial torques: ", round.(τ, digits=4))

# Test 6: Full dynamics test
q   = [π/6, π/3, 0.15]
qd  = [0.8, 1.2, 0.05]
qdd = [1.5, 2.0, 0.3]
τ = inverse_dynamics(scara, q, qd, qdd)
println("\nFull dynamics example: ", round.(τ, digits=4))

# ============================================================================
# TRAJECTORY PLANNING EXAMPLE
# ============================================================================

println("\n=== Trajectory Planning ===")

constraints = TrajectoryConstraints(
    [2.0, 2.0, 0.5],           # velocity limits (rad/s, rad/s, m/s)
    [10.0, 10.0, 2.0],         # acceleration limits
    [50.0, 50.0, 20.0],        # torque limits
    [100.0, 100.0, 50.0],      # jerk limits
    ([-π, -π, -0.3], [π, π, 0.3])  # position limits
)

# SCARA reachable workspace: links 0.3 + 0.25 = 0.55m maximum reach
# Create a simple horizontal path in reachable workspace  
# from end position [0.55, 0, 0.05] to [0.3, 0.25, 0.05]
path_cart = BezierCartesianPath(
    SVector{3,Float64}(0.55, 0.0,  0.05),
    SVector{3,Float64}(0.50, 0.08, 0.05),
    SVector{3,Float64}(0.40, 0.18, 0.05),
    SVector{3,Float64}(0.30, 0.25, 0.05)
)

q_seed = [0.0, 0.0, 0.0]

# Convert to joint space
println("Converting Cartesian path to joint space...")
joint_path = joint_path_from_cartesian_bezier(scara, path_cart, q_seed)

# Check if IK was successful
if any(isnan, [joint_path.q0; joint_path.q1; joint_path.q2; joint_path.q3])
    println("⚠ IK failed - trajectory not feasible with current robot workspace")
else
    # Compute path speed limits
    path_speed = compute_limit_path_speed(scara, joint_path, constraints)
    
    # Generate full trajectory
    println("Generating trajectory...")
    result = generate_joint_trajectory(
        scara,
        joint_path,
        path_speed,
        time_step = 0.01,
        constraints = constraints
    )
    
    println("✓ Trajectory points: ", size(result.trajectory.positions, 2))
    println("✓ Trajectory duration: ", result.time_vector[end], " s")
    println("✓ Max torques by axis: ", round.(maximum(abs.(result.torques), dims=2)[:,1], digits=4))
    println("✓ Feasible: ", result.feasible)
end
=#
