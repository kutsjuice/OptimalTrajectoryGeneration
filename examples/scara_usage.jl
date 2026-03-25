using ForwardDiff
using LinearAlgebra
using CairoMakie
using Interpolations
using Polynomials
using QuadGK
using Dierckx
using NPZ

# TestRobot будет наследовать от него после интеграции с пакетом.
# using OptimalTrajectoryGeneration

mutable struct TestRobot
    l1::Float64
    l2::Float64
    dof::Int
end

function forward_kinematics(robot::TestRobot, q::AbstractVector)
    q1 = q[1]
    q2 = q[2]
    l1, l2 = robot.l1, robot.l2
    x = l1 * cos(q1) + l1 * cos(q1 + q2) + l2 * cos(q1 + 0.5 * q2)
    y = l1 * sin(q1) + l1 * sin(q1 + q2) + l2 * sin(q1 + 0.5 * q2)
    return [x, y]
end

function body_jacobian(robot::TestRobot, q::Vector{Float64})
    q1, q2 = q
    s1  = sin(q1)
    s2  = sin(q1 + q2)
    s12 = sin(q1 + 0.5*q2)
    c1  = cos(q1)
    c2  = cos(q1 + q2)
    c12 = cos(q1 + 0.5*q2)
    return [
        -robot.l1*s1 - robot.l1*s2 - robot.l2*s12    -robot.l1*s2 - 0.5*robot.l2*s12
         robot.l1*c1 + robot.l1*c2 + robot.l2*c12     robot.l1*c2 + 0.5*robot.l2*c12
    ]
end

function ik(robot::TestRobot, target::AbstractVector, initial_guess::AbstractVector;
            max_iterations=100, tolerance=1e-6)
    q = copy(initial_guess)
    for _ in 1:max_iterations
        pos = forward_kinematics(robot, q)
        err = target - pos
        if norm(err) < tolerance
            return q
        end
        J = body_jacobian(robot, q)
        q += J \ err
    end
    return q
end

struct BezierCurve
    control_points::Matrix{Float64}
end

function make_bezier(p_start::AbstractVector, p_end::AbstractVector, t=0.7)
    P = hcat(p_start, [t, 0.0], [0.0, t], p_end)
    return BezierCurve(P)
end

function cartesian_traj(curve::BezierCurve, theta::AbstractVector)
    P = curve.control_points
    n = length(theta)
    traj = zeros(n, 2)
    for i in 1:n
        t = theta[i]; mt = 1 - t
        traj[i,:] = mt^3 * P[:,1] + 3*mt^2*t*P[:,2] + 3*mt*t^2*P[:,3] + t^3*P[:,4]
    end
    return traj
end

function max_theta_dot(jnt_traj::AbstractMatrix, theta::AbstractVector,
                       w1_max::Float64, w2_max::Float64)
    spl1 = cubic_spline_interpolation(theta, jnt_traj[:, 1], bc=Line(OnGrid()))
    spl2 = cubic_spline_interpolation(theta, jnt_traj[:, 2], bc=Line(OnGrid()))

    theta_dense = range(theta[1] + 0.001, theta[end] - 0.001, length=max(100, length(theta)*10))

    dq1  = [ForwardDiff.derivative(spl1, t) for t in theta_dense]
    dq2  = [ForwardDiff.derivative(spl2, t) for t in theta_dense]
    ddq1 = [ForwardDiff.derivative(t -> ForwardDiff.derivative(spl1, t), t) for t in theta_dense]
    ddq2 = [ForwardDiff.derivative(t -> ForwardDiff.derivative(spl2, t), t) for t in theta_dense]

    return (
        max_vel_q1   = maximum(abs.(dq1)),
        max_vel_q2   = maximum(abs.(dq2)),
        max_acc_q1   = maximum(abs.(ddq1)),
        max_acc_q2   = maximum(abs.(ddq2)),
        satisfies_constraints = maximum(abs.(dq1)) <= w1_max && maximum(abs.(dq2)) <= w2_max
    )
end

function joint_traj(cart_traj::Matrix{Float64}, robot::TestRobot, q_init::AbstractVector)
    N = size(cart_traj,1)
    q = copy(q_init)
    q_traj = zeros(N, robot.dof)
    for i in 1:N
        q = ik(robot, cart_traj[i,:], q; max_iterations=50, tolerance=1e-5)
        q_traj[i,:] = q
    end
    return q_traj
end

function time_parametrise(theta::AbstractVector, theta_dot_max::AbstractVector)
    T = eltype(theta)
    n = length(theta)
    idx = vcat(1:50:n, n)
    knots = theta_dot_max[idx]
    knots[1] = knots[end] = 0.0
    avg_step = sum(diff(theta[idx])) / length(diff(theta[idx]))
    if all(x -> isapprox(x, avg_step; rtol=1e-6), diff(theta[idx]))
        itp = interpolate(knots, BSpline(Cubic(Line(OnGrid()))))
        vel_prof = [itp(1 + (i-1)*(length(idx)-1)/(n-1)) for i in 1:n]
    else
        itp = linear_interpolation(theta[idx], knots)
        vel_prof = itp.(theta)
    end
    h = theta[2] - theta[1]
    time = zeros(T, n)
    time[2:end] = cumsum(h ./ vel_prof[2:end])
    if n >= 5
        idx_fit = (n-3):(n-1)
        p = fit(theta[idx_fit], time[idx_fit], 2)
        time[end] = p(theta[end])
    end
    return time, vel_prof
end

function plot_joints_vs_time(time_vec, jnt_traj_data)
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="t (s)", ylabel="q (rad)", title="Joint Angles vs Time")
    lines!(ax, time_vec, jnt_traj_data[:, 1], label="q1(t)")
    lines!(ax, time_vec, jnt_traj_data[:, 2], label="q2(t)")
    axislegend(ax)
    return fig
end

function time_step(ds::Float64, v0::Float64, a0::Float64, a1::Float64)
    j = (a1 - a0) / ds
    if abs(v0) < 1e-6
        dt = sqrt(2ds / a0)
    else
        dt, _ = quadgk(s -> 1/sqrt(v0^2 + 2a0*s + j*s^2), 0, ds)
    end
    dv = a0 * dt + (j * dt^2) / 2
    return dt, dv
end

function time_step2(v0, v1)
    return abs((log(v1) - log(v0)) / (v1 - v0))
end

function velocity_limit(theta, d_psi_1_d_th, d_psi_2_d_th, w1_max, w2_max)
    j1_lim = abs(w1_max / d_psi_1_d_th(theta))
    j2_lim = abs(w2_max / d_psi_2_d_th(theta))
    return min(j1_lim, j2_lim)
end

function acceleration_limits(theta, d_th_dt,
                             d_psi_1_d_th, d_psi_2_d_th,
                             dd_psi_1_d_th2, dd_psi_2_d_th2,
                             e1_max, e2_max, dtheta=1e-3)
    if abs(d_th_dt) < 1e-6
        denom = abs(dd_psi_1_d_th2(theta + dtheta/2) * dtheta^2 +
                    d_psi_1_d_th(theta + dtheta/2) * dtheta)
        return [-e1_max / denom, e1_max / denom]
    end

    if d_psi_1_d_th(theta) * d_th_dt > 0
        j1_lim = [
            (-e1_max - dd_psi_1_d_th2(theta) * d_th_dt^2) / (d_psi_1_d_th(theta) * d_th_dt),
            ( e1_max - dd_psi_1_d_th2(theta) * d_th_dt^2) / (d_psi_1_d_th(theta) * d_th_dt)
        ]
    else
        j1_lim = [
            ( e1_max - dd_psi_1_d_th2(theta) * d_th_dt^2) / (d_psi_1_d_th(theta) * d_th_dt),
            (-e1_max - dd_psi_1_d_th2(theta) * d_th_dt^2) / (d_psi_1_d_th(theta) * d_th_dt)
        ]
    end

    if d_psi_2_d_th(theta) * d_th_dt > 0
        j2_lim = [
            (-e2_max - dd_psi_2_d_th2(theta) * d_th_dt^2) / (d_psi_2_d_th(theta) * d_th_dt),
            ( e2_max - dd_psi_2_d_th2(theta) * d_th_dt^2) / (d_psi_2_d_th(theta) * d_th_dt)
        ]
    else
        j2_lim = [
            ( e2_max - dd_psi_2_d_th2(theta) * d_th_dt^2) / (d_psi_2_d_th(theta) * d_th_dt),
            (-e2_max - dd_psi_2_d_th2(theta) * d_th_dt^2) / (d_psi_2_d_th(theta) * d_th_dt)
        ]
    end

    low_lim = max(j1_lim[1], j2_lim[1])
    upp_lim = min(j1_lim[2], j2_lim[2])
    return low_lim, upp_lim
end

# ==================== MAIN SCRIPT ====================

L1 = 0.2
l2 = 0.3

w1_max = w2_max = 335 * π / 180
e1_max = e2_max = 2500
q0 = [0.01, -0.02]
testr = TestRobot(L1, l2, 2)
p0 = forward_kinematics(testr, q0)

for x_new in LinRange(p0[1], 0.65, 100)
    global q0
    q0 = ik(testr, [x_new, 0.0], q0)
end

p0 = forward_kinematics(testr, q0)
p1 = [p0[2], p0[1]]
k = 0.9
curve = make_bezier(p0, p1, k * p0[1])

N = 4001
theta = LinRange(0, 1, N)
ds = theta[2] - theta[1]
cart_traj_pts = cartesian_traj(curve, theta)

let
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="X", ylabel="Y", title="Cartesian Trajectory")
    lines!(ax, cart_traj_pts[:, 1], cart_traj_pts[:, 2])
    display(fig)
end

jnt_traj_pts = joint_traj(cart_traj_pts, testr, q0)

theta_vec = collect(theta)
spl1 = Spline1D(theta_vec, jnt_traj_pts[:, 1], k=3, s=0.0)
spl2 = Spline1D(theta_vec, jnt_traj_pts[:, 2], k=3, s=0.0)

psi1 = t -> spl1(t)
psi2 = t -> spl2(t)
psi3 = t -> -spl2(t)/2

d_psi1_dth   = t -> Dierckx.derivative(spl1, t)
d_psi2_dth   = t -> Dierckx.derivative(spl2, t)
d_psi3_dth   = t -> -Dierckx.derivative(spl2, t)/2

dd_psi1_dth2 = t -> Dierckx.derivative(spl1, t, 2)
dd_psi2_dth2 = t -> Dierckx.derivative(spl2, t, 2)
dd_psi3_dth2 = t -> -Dierckx.derivative(spl2, t, 2)/2

# Velocity limit from joint velocity constraints
v_lim1 = [abs(w1_max / d_psi1_dth(t)) for t in theta]
v_lim2 = [abs(w2_max / d_psi2_dth(t)) for t in theta]
vel_profile = min.(v_lim1, v_lim2)
vel_profile = max.(vel_profile, 1e-10)

let
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="θ", ylabel="θ̇_max", title="Velocity Profile")
    lines!(ax, theta_vec, vel_profile, label="Velocity limit")
    axislegend(ax)
    display(fig)
end

# Crude time estimate
time_init = zeros(length(theta))
h_step = theta[2] - theta[1]
for i in 2:length(theta)
    time_init[i] = time_init[i-1] + h_step / vel_profile[i]
end
n_len = length(theta)
if n_len >= 5
    idx_fit = n_len-4:n_len
    p_fit = fit(theta_vec[idx_fit], time_init[idx_fit], 2)
    time_init[end] = p_fit(theta_vec[end])
end

let
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="Time (s)", ylabel="θ", title="θ vs Time (initial)")
    lines!(ax, time_init, theta_vec)
    display(fig)
end

# Arrays for forward/backward integration
d_th_d_t_f   = fill(1000.0, N)
d_th_d_t_b   = fill(1000.0, N)
dd_th_d_t2_f = zeros(N)
dd_th_d_t2_b = zeros(N)

# Torques before optimization
torq_before_opt = zeros(2, N-1)
M    = diagm(0 => [1.0, 1.0, 1.0, 1.0])
h_dyn = zeros(4)

for i in 2:N
    theta_cur      = 0.5 * (theta[i-1] + theta[i])
    d_th_dt_cur    = 0.5 * (vel_profile[i-1] + vel_profile[i])
    dd_th_d_t2_cur = (vel_profile[i] - vel_profile[i-1]) / ds * d_th_dt_cur

    ddq_cur = [
        0.0,
        dd_psi1_dth2(theta_cur) * d_th_dt_cur^2 + d_psi1_dth(theta_cur) * dd_th_d_t2_cur,
        dd_psi2_dth2(theta_cur) * d_th_dt_cur^2 + d_psi2_dth(theta_cur) * dd_th_d_t2_cur,
        dd_psi3_dth2(theta_cur) * d_th_dt_cur^2 + d_psi3_dth(theta_cur) * dd_th_d_t2_cur
    ]

    torq_before_opt[:, i-1] = (M * ddq_cur + h_dyn)[2:3]
end

npzwrite("torq_before_opt_k=$(k).npz", Dict("torq" => torq_before_opt, "theta" => theta_vec))

# Reset integration arrays
d_th_d_t_f = fill(1000.0, N)
d_th_d_t_b = fill(1000.0, N)

# Determine torque limit from pre-optimization torques
mask = (theta_vec[1:end-1] .> 0.05) .& (theta_vec[1:end-1] .< 0.95)
p_frac = 1
Tmax_val = maximum(abs.(torq_before_opt[:, mask])) * p_frac
Tmax = fill(Tmax_val, 4)
Tmax[end] = Inf

# Boundary conditions
d_th_d_t_f[1] = 0.0
d_th_d_t_b[N] = 0.0

# Initial velocity bound at start — used as d_th_d_t_f[2]
let d_psi = [0.0, d_psi1_dth(0.0), d_psi2_dth(0.0), d_psi3_dth(0.0)],
    dd_psi = [0.0, dd_psi1_dth2(0.0), dd_psi2_dth2(0.0), dd_psi3_dth2(0.0)]
    a_b = M * dd_psi
    v_b = M * d_psi
    th_buf = Inf
    for i in 1:4
        if abs(v_b[i]) > 1e-12 && Tmax[i] < Inf
            th_buf = min(th_buf, sqrt(abs(Tmax[i] / (0.5*abs(v_b[i])/ds + 0.25*abs(a_b[i])))))
        end
    end
    global d_th_d_t_f
    d_th_d_t_f[2] = th_buf
end

# Initial velocity bound at end — used as d_th_d_t_b[N-1]
let d_psi = [0.0, d_psi1_dth(1.0), d_psi2_dth(1.0), d_psi3_dth(1.0)],
    dd_psi = [0.0, dd_psi1_dth2(1.0), dd_psi2_dth2(1.0), dd_psi3_dth2(1.0)]
    a_b = M * dd_psi
    v_b = M * d_psi
    th_buf = Inf
    for i in 1:4
        if abs(v_b[i]) > 1e-12 && Tmax[i] < Inf
            th_buf = min(th_buf, sqrt(abs(Tmax[i] / (0.5*abs(v_b[i])/ds + 0.25*abs(a_b[i])))))
        end
    end
    global d_th_d_t_b
    d_th_d_t_b[N-1] = th_buf
end

dd_th_d_t2_f[1] = d_th_d_t_f[2] / ds
dd_th_d_t2_b[N] = -d_th_d_t_b[N-1] / ds

# ========== FORWARD PASS ==========
for i in 2:N-1
    θ_cur = theta[i]
    θ_nxt = theta[i+1]
    dθ    = θ_nxt - θ_cur
    v_cur = d_th_d_t_f[i]
    θ_mid = θ_cur + dθ/2

    f1 = M * [0.0, d_psi1_dth(θ_mid)*v_cur, d_psi2_dth(θ_mid)*v_cur, d_psi3_dth(θ_mid)*v_cur]
    f2 = M * [0.0, dd_psi1_dth2(θ_mid)*v_cur^2, dd_psi2_dth2(θ_mid)*v_cur^2, dd_psi3_dth2(θ_mid)*v_cur^2]

    a_min = -Inf; a_max = Inf
    for j in 1:4
        (abs(f1[j]) < 1e-12 || isinf(Tmax[j])) && continue
        if f1[j] > 0
            a_min = max(a_min, (-Tmax[j] - f2[j]) / f1[j])
            a_max = min(a_max, ( Tmax[j] - f2[j]) / f1[j])
        else
            a_min = max(a_min, ( Tmax[j] - f2[j]) / f1[j])
            a_max = min(a_max, (-Tmax[j] - f2[j]) / f1[j])
        end
    end

    v_nxt = sqrt(max(0.0, v_cur^2 + 2*a_max*dθ))
    if v_nxt > vel_profile[i+1]
        a_req = (vel_profile[i+1]^2 - v_cur^2) / (2*dθ)
        if     a_req > a_max; v_nxt = v_nxt  # already candidate
        elseif a_req < a_min; v_nxt = sqrt(max(0.0, v_cur^2 + 2*a_min*dθ))
        else;                  v_nxt = vel_profile[i+1]
        end
    end

    d_th_d_t_f[i+1] = max(0.0, v_nxt)
    dd_th_d_t2_f[i] = (d_th_d_t_f[i+1]^2 - v_cur^2) / (2*dθ)
end
dd_th_d_t2_f[N] = 0.0

# ========== BACKWARD PASS ==========
# Starts at N-2 to preserve the boundary condition d_th_d_t_b[N-1] set above
for i in N-2:-1:2
    θ_cur = theta[i]
    θ_nxt = theta[i+1]
    dθ    = θ_nxt - θ_cur
    v_nxt = d_th_d_t_b[i+1]
    θ_mid = θ_cur + dθ/2

    f1 = M * [0.0, d_psi1_dth(θ_mid)*v_nxt, d_psi2_dth(θ_mid)*v_nxt, d_psi3_dth(θ_mid)*v_nxt]
    f2 = M * [0.0, dd_psi1_dth2(θ_mid)*v_nxt^2, dd_psi2_dth2(θ_mid)*v_nxt^2, dd_psi3_dth2(θ_mid)*v_nxt^2]

    a_min = -Inf; a_max = Inf
    for j in 1:4
        (abs(f1[j]) < 1e-12 || isinf(Tmax[j])) && continue
        if f1[j] > 0
            a_min = max(a_min, (-Tmax[j] - f2[j]) / f1[j])
            a_max = min(a_max, ( Tmax[j] - f2[j]) / f1[j])
        else
            a_min = max(a_min, ( Tmax[j] - f2[j]) / f1[j])
            a_max = min(a_max, (-Tmax[j] - f2[j]) / f1[j])
        end
    end

    # v_cur² = v_nxt² − 2·a_min·dθ  (most negative a → largest v_cur)
    v_cur = sqrt(max(0.0, v_nxt^2 - 2*a_min*dθ))
    if v_cur > vel_profile[i]
        a_req = (v_nxt^2 - vel_profile[i]^2) / (2*dθ)
        if     a_req < a_min; v_cur = v_cur  # already candidate
        elseif a_req > a_max; v_cur = sqrt(max(0.0, v_nxt^2 - 2*a_max*dθ))
        else;                  v_cur = vel_profile[i]
        end
    end

    d_th_d_t_b[i] = max(0.0, v_cur)
    dd_th_d_t2_b[i] = (v_nxt^2 - d_th_d_t_b[i]^2) / (2*dθ)
end
dd_th_d_t2_b[N-1] = (d_th_d_t_b[N]^2 - d_th_d_t_b[N-1]^2) / (2*ds)

# Final trajectory
traj = min.(d_th_d_t_f, d_th_d_t_b)

# Torques after optimization
torq_after_opt = zeros(2, N-1)
for i in 2:N
    theta_cur      = 0.5 * (theta[i-1] + theta[i])
    d_th_dt_cur    = 0.5 * (traj[i-1] + traj[i])
    dd_th_d_t2_cur = (traj[i] - traj[i-1]) / ds * d_th_dt_cur

    ddq_cur = [
        0.0,
        dd_psi1_dth2(theta_cur) * d_th_dt_cur^2 + d_psi1_dth(theta_cur) * dd_th_d_t2_cur,
        dd_psi2_dth2(theta_cur) * d_th_dt_cur^2 + d_psi2_dth(theta_cur) * dd_th_d_t2_cur,
        dd_psi3_dth2(theta_cur) * d_th_dt_cur^2 + d_psi3_dth(theta_cur) * dd_th_d_t2_cur
    ]

    torq_after_opt[:, i-1] = (M * ddq_cur + h_dyn)[2:3]
end

npzwrite("velocity_prof_opt_k=$(k)_p=$(p_frac).npz",
         Dict("velocity_profile" => traj, "theta" => theta_vec))
npzwrite("torq_after_opt_k=$(k)_p=$(p_frac).npz",
         Dict("torq" => torq_after_opt, "theta" => theta_vec))

# ==================== PLOTS ====================

let
    fig = Figure()
    ax = Axis(fig[1,1], xlabel="θ", ylabel="θ̇", title="Velocity Profiles")
    lines!(ax, theta_vec[2:end-1], vel_profile[2:end-1],  label="Velocity limit", color=:gray)
    lines!(ax, theta_vec[2:end-1], d_th_d_t_f[2:end-1],  label="Forward",        color=:blue)
    lines!(ax, theta_vec[2:end-1], d_th_d_t_b[2:end-1],  label="Backward",       color=:red)
    lines!(ax, theta_vec[2:end-1], traj[2:end-1],         label="Final",          color=:green, linewidth=2)
    axislegend(ax)
    display(fig)
    save("velocity_profiles.png", fig)
end

let
    theta_mid       = (theta_vec[1:end-1] .+ theta_vec[2:end]) ./ 2
    theta_mid_inner = theta_mid[2:end-1]
    tb = torq_before_opt[:, 2:end-1]
    ta = torq_after_opt[:,  2:end-1]

    fig = Figure()
    ax  = Axis(fig[1,1], xlabel="θ", ylabel="Torque", title="Torque Comparison")
    lines!(ax, theta_mid_inner, tb[1,:], label="Joint 1 before", color=:blue,  linestyle=:dash)
    lines!(ax, theta_mid_inner, tb[2,:], label="Joint 2 before", color=:red,   linestyle=:dash)
    lines!(ax, theta_mid_inner, ta[1,:], label="Joint 1 after",  color=:blue)
    lines!(ax, theta_mid_inner, ta[2,:], label="Joint 2 after",  color=:red)
    hlines!(ax, [ Tmax_val], color=:black, linestyle=:dot, linewidth=1.5,
            label="±Tmax = $(round(Tmax_val, digits=2))")
    hlines!(ax, [-Tmax_val], color=:black, linestyle=:dot, linewidth=1.5)
    axislegend(ax, position=:lt)
    display(fig)
    save("torque_comparison.png", fig)
end

let
    time_opt = zeros(N)
    for i in 2:N
        time_opt[i] = traj[i] > 1e-10 ? time_opt[i-1] + ds / traj[i] : time_opt[i-1]
    end
    fig = Figure()
    ax  = Axis(fig[1,1], xlabel="t (s)", ylabel="θ", title="θ(t) — optimized")
    lines!(ax, time_opt, theta_vec)
    display(fig)
    save("theta_time.png", fig)
end
