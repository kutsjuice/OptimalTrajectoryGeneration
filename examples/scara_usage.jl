using ForwardDiff
using LinearAlgebra
using Plots
using Interpolations
using Polynomials
using QuadGK
using Dierckx
using NPZ

abstract type AbstractRobotManipulator end

mutable struct TestRobot <: AbstractRobotManipulator
    l1::Float64
    l2::Float64
    dof::Int
    # gravity::SVector{3,Float64}
    # links::Vector{Link}
end

# struct Link
#     parent::Int
#     Xtree::SMatrix{6,6,Float64}
#     inertia::SMatrix{6,6,Float64}
#     pitch::Float64
# end

# struct SpatialInertia
#     mass::Float64
#     com::SVector{3,Float64}
#     inertia::SMatrix{3,3,Float64}
# end

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
# function body_jacobian(robot::TestRobot, q::Vector{Float64})
#     J = ForwardDiff.jacobian(q_vec -> forward_kinematics(robot, q_vec), q)
#     return J

    # q1 = q[1]
    # q2 = q[2]

    # s1  = sin(q1)
    # s2  = sin(q1 + q2)
    # s12 = sin(q1 + 0.5*q2)

    # c1  = cos(q1)
    # c2  = cos(q1 + q2)
    # c12 = cos(q1 + 0.5*q2)

    # return [
    #     -robot.l1 * s1 - robot.l1 * s2 - robot.l2 * s12     -robot.l1 * s2 - 0.5*robot.l2*s12
    #      robot.l1 * c1 + robot.l1 * c2 + robot.l2 * c12      robot.l1 * c2 + 0.5*robot.l2*c12
    # ]
# end

# function inverse_dynamics(
#     robot::TestRobot,
#     q::Vector{Float64},
#     qd::Vector{Float64},
#     qdd::Vector{Float64}
# )::Vector{Float64}
#     """Recursive Newton-Euler Algorithm for inverse dynamics"""
#     @assert length(q) == length(qd) == length(qdd) == model.N
    
#     n = model.N
#     v = Vector{SVector{6,Float64}}(undef, n)
#     a = Vector{SVector{6,Float64}}(undef, n)
#     f = Vector{SVector{6,Float64}}(undef, n)
#     S = Vector{SVector{6,Float64}}(undef, n)
#     Xup = Vector{SMatrix{6,6,Float64}}(undef, n)
    
#     a0 = @SVector [
#         0.0, 0.0, 0.0,
#         -robot.gravity[1], -robot.gravity[2], -robot.gravity[3]
#     ]
    
#     τ = zeros(n)
    
#     # Forward pass: compute accelerations and velocities
#     for i in 1:n
#         XJ, S[i] = jcalc(robot.links[i].pitch, q[i])
#         Xup[i] = robot.links[i].Xtree * XJ
        
#         vJ = S[i] * qd[i]
        
#         if robot.links[i].parent == 0
#             v[i] = vJ
#             a[i] = Xup[i]*a0 + S[i]*qdd[i] + crm(v[i])*vJ
#         else
#             p = robot.links[i].parent
#             v[i] = Xup[i]*v[p] + vJ
#             a[i] = Xup[i]*a[p] + S[i]*qdd[i] + crm(v[i])*vJ
#         end
#     end
    
#     # Backward pass: compute forces and torques
#     for i in n:-1:1
#         I = spatial_inertia(robot.links[i].inertia)
#         f[i] = I*a[i] + crf(v[i])*(I*v[i])
        
#         if robot.links[i].parent != 0
#             p = robot.links[i].parent
#             f[p] += Xup[i]'*f[i]
#         end
        
#         τ[i] = S[i]' * f[i]
#     end
    
#     return τ
# end

# function compute_mass_matrix_and_force_terms(
#     robot::TestRobot,
#     q::Vector{Float64},
#     qd::Vector{Float64}
# )::Tuple{Matrix{Float64}, Vector{Float64}}
#     """Compute mass matrix M and bias terms (Coriolis + gravity)"""
#     n = robot.dof
#     M = zeros(n, n)
#     bias = inverse_dynamics(robot, q, qd, zeros(n))
    
#     for j = 1:n
#         qdd_unit = zeros(n)
#         qdd_unit[j] = 1.0
#         τ = inverse_dynamics(robot, q, zeros(n), qdd_unit)
#         M[:, j] .= τ
#     end
    
#     return M, bias
# end

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
    # Create cubic spline interpolations with linear extrapolation
    spl1 = cubic_spline_interpolation(theta, jnt_traj[:, 1], bc=Line(OnGrid()))
    spl2 = cubic_spline_interpolation(theta, jnt_traj[:, 2], bc=Line(OnGrid()))
    
    # Sample at interior points only (avoid boundaries where extrapolation might fail)
    theta_dense = range(theta[1] + 0.001, theta[end] - 0.001, length=max(100, length(theta)*10))
    
    # Compute first derivatives (angular velocities)
    dq1 = [ForwardDiff.derivative(spl1, t) for t in theta_dense]
    dq2 = [ForwardDiff.derivative(spl2, t) for t in theta_dense]
    
    # Compute second derivatives (angular accelerations)
    ddq1 = [ForwardDiff.derivative(t -> ForwardDiff.derivative(spl1, t), t) for t in theta_dense]
    ddq2 = [ForwardDiff.derivative(t -> ForwardDiff.derivative(spl2, t), t) for t in theta_dense]
    
    # Return max velocities and accelerations
    return (
        max_vel_q1 = maximum(abs.(dq1)),
        max_vel_q2 = maximum(abs.(dq2)),
        max_acc_q1 = maximum(abs.(ddq1)),
        max_acc_q2 = maximum(abs.(ddq2)),
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
    if isapprox(diff(theta[idx]), fill(mean(diff(theta[idx])), length(idx)-1); rtol=1e-6)
        itp = interpolate(knots, BSpline(Akima(Line(OnGrid()))))  # Assumes Interpolations supports Akima; else use Cubic
        vel_prof = [itp(1 + (i-1)*(length(idx)-1)/(n-1)) for i in 1:n]
    else
        itp = LinearInterpolation(theta[idx], knots)
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

function plot_joints_vs_time(time, jnt_traj)
    plot(time, [jnt_traj[:, 1] jnt_traj[:, 2]], label=["q1(t)" "q2(t)"])
end

function time_step(ds::Float64, v0::Float64, a0::Float64, a1::Float64)
    j = (a1 - a0) / ds
    if abs(v0) < 1e-6
        dt = sqrt(2ds / a0)
    else
        dt, err = quadgk(s -> 1/sqrt(v0^2 + 2a0*s + j*s^2), 0, ds)
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
        # Special case - nearly zero velocity
        denom = abs(dd_psi_1_d_th2(theta + dtheta/2) * dtheta^2 +
                    d_psi_1_d_th(theta + dtheta/2) * dtheta)
        return [-e1_max / denom, e1_max / denom]
    end
    
    # Limits for first joint
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
    
    # Limits for second joint
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
    
    low_lim = max(j1_lim[1], j2_lim[1])  # Lower bound
    upp_lim = min(j1_lim[2], j2_lim[2])  # Upper bound
    return low_lim, upp_lim
end

# Example usage
L1 = 0.2
l2 = 0.3

w1_max = w2_max = 335 * π / 180
e1_max = e2_max = 2500
q0 = [0.01, -0.02]
testr = TestRobot(L1, l2, 2)
p0 = forward_kinematics(testr, q0)

# Iterative movement along X axis
for x_new in LinRange(p0[1], 0.65, 100)
    global q0
    q0 = ik(testr, [x_new, 0.0], q0)
end

p0 = forward_kinematics(testr, q0)
p1 = [p0[2], p0[1]]  # Swap x and y
k = 0.9
curve = make_bezier(p0, p1, k * p0[1])

N = 4001
theta = LinRange(0, 1, N)
ds = theta[2] - theta[1]
cart_traj = cartesian_traj(curve, theta)
display(plot(cart_traj[:, 1], cart_traj[:, 2], label="Cartesian Trajectory", xlabel="X", ylabel="Y"))
jnt_traj = joint_traj(cart_traj, testr, q0)

spl1 = Spline1D(theta, jnt_traj[:, 1], k=3, s=0.0)
spl2 = Spline1D(theta, jnt_traj[:, 2], k=3, s=0.0)

psi1 = theta -> spl1(theta)
psi2 = theta -> spl2(theta)
psi3 = theta -> -spl2(theta)/2

d_psi1_dth = theta -> Dierckx.derivative(spl1, theta)
d_psi2_dth = theta -> Dierckx.derivative(spl2, theta)
d_psi3_dth = theta -> -Dierckx.derivative(spl2, theta)/2

dd_psi1_dth2 = theta -> Dierckx.derivative(spl1, theta, 2)
dd_psi2_dth2 = theta -> Dierckx.derivative(spl2, theta, 2)
dd_psi3_dth2 = theta -> -Dierckx.derivative(spl2, theta, 2)/2

v_lim1 = [abs(w1_max / d_psi1_dth(t)) for t in theta]
v_lim2 = [abs(w2_max / d_psi2_dth(t)) for t in theta]
vel_profile = min.(v_lim1, v_lim2)
display(plot(theta, vel_profile, label="Velocity Profile", xlabel="Theta", ylabel="Max Theta Dot"))
time = zeros(length(theta))
h = theta[2] - theta[1]

for i in 2:length(theta)
    if vel_profile[i] > 1e-10
        time[i] = time[i-1] + h / vel_profile[i]
    else
        time[i] = time[i-1] + h / 1e-10
    end
end

n = length(theta)
if n >= 5
    idx_fit = n-4:n
    p = fit(theta[idx_fit], time[idx_fit], 2)
    time[end] = p(theta[end])
end
display(plot(time, theta, label="Theta vs Time", xlabel="Time (s)", ylabel="Theta"))
theta_to_t_spl = Spline1D(theta, time, k=3, s=0.0)
theta_to_t = t -> theta_to_t_spl(t)
d_th_to_t = t -> Dierckx.derivative(theta_to_t_spl, t)
dd_th_to_t2 = t -> Dierckx.derivative(theta_to_t_spl, t, 2)

d_th_d_t_f = ones(N) * 1e3; d_th_d_t_b = ones(N) * 1e3
dd_th_d_t2_f = zeros(N); dd_th_d_t2_b = zeros(N);; dd_th_d_t2_a = zeros(N)
t = zeros(N)
torq_before_opt = zeros(2, N-1)   # 2 строки × (N-1) столбцов



M = diagm(0 => [1.0, 1.0, 1.0, 1.0])
h = zeros(4)
for i in 2:N
    theta_cur     = 0.5 * (theta[i-1] + theta[i])
    d_th_dt_cur   = 0.5 * (vel_profile[i-1] + vel_profile[i])
    dd_th_d_t2_cur = (vel_profile[i] - vel_profile[i-1]) / ds * d_th_dt_cur

    q_cur = [
        0.0,
        psi1(theta_cur),
        psi2(theta_cur),
        -psi2(theta_cur)/2
    ]

    dq_cur = [
        0.0,
        d_psi1_dth(theta_cur) * d_th_dt_cur,
        d_psi2_dth(theta_cur) * d_th_dt_cur,
        -d_psi2_dth(theta_cur) * d_th_dt_cur / 2
    ]

    ddq_cur = [
        0.0,
        dd_psi1_dth2(theta_cur) * d_th_dt_cur^2 + d_psi1_dth(theta_cur) * dd_th_d_t2_cur,
        dd_psi2_dth2(theta_cur) * d_th_dt_cur^2 + d_psi2_dth(theta_cur) * dd_th_d_t2_cur,
        -(dd_psi2_dth2(theta_cur) * d_th_dt_cur^2 + d_psi2_dth(theta_cur) * dd_th_d_t2_cur)/2
    ]


    torq = M * ddq_cur + h

    torq_before_opt[:, i-1] = torq[2:3]
end
filename = "torq_before_opt_k=$(k).npz"
npzwrite(filename, Dict("torq" => torq_before_opt, "theta" => theta))
d_th_d_t_f = fill(1000.0, N)
d_th_d_t_b = fill(1000.0, N)
mask = (theta[1:end-1] .> 0.05) .& (theta[1:end-1] .< 0.95)
p = 0.75
Tmax_val = maximum(abs.(torq_before_opt[:, mask])) * p
Tmax = fill(Tmax_val, 4)
Tmax[end] = Inf
th0  = 0.0
dth0 = 0.0
q0 = [
    0.0,
    psi1(th0),
    psi2(th0),
    psi3(th0)
]
dq0 = [
    0.0,
    d_psi1_dth(th0) * dth0,
    d_psi2_dth(th0) * dth0,
    d_psi3_dth(th0) * dth0
]
# M, h = ComputeMassMatrixAndForceTerms(q0, dq0)

# Начальные условия
d_th_d_t_f[1] = 0.0;    t[1] = 0.0
d_th_d_t_b[end] = 0.0
d_psi_d_th = [
    0,
    d_psi1_dth(th0),
    d_psi2_dth(th0),
    d_psi3_dth(th0)
]
dd_psi_dth2 = [
    0,
    dd_psi1_dth2(th0),
    dd_psi2_dth2(th0),
    dd_psi3_dth2(th0)
]
a_buf  = M * dd_psi_dth2
v_buf = M * d_psi_d_th
th_buf = Inf
buf = []
for i in 1:4
    local th_buf = Inf
    th_i = sqrt(abs(Tmax[i] / (0.5 * v_buf[i] / ds + 0.25a_buf[i])))
    th_buf = min(th_buf, th_i)
end
d_th_d_t_f[2] = th_buf

d_psi_d_th = [
    0,
    d_psi1_dth(1),
    d_psi2_dth(1),
    d_psi3_dth(1)
]
d_psi_d_th2 = [
    0,
    dd_psi1_dth2(1),
    dd_psi2_dth2(1),
    dd_psi3_dth2(1)
]
a_buf  = M * d_psi_d_th2
v_buf = M * d_psi_d_th
th_buf = Inf
for i in 1:4
    local th_buf = Inf
    th_i = sqrt(abs(Tmax[i] / (0.5v_buf[i]/ds + 0.25a_buf[i])))
    th_buf = min(th_buf, th_i)
end
d_th_d_t_b[end-1] = th_buf
dd_th_d_t2_f[1] = (d_th_d_t_f[2])/ds
dd_th_d_t2_b[end] = -(d_th_d_t_b[end-1])/ds
for i in 3:N
    # Forward pass
    theta_cur, d_th_dt_cur = theta[i-1], d_th_d_t_f[i-1]
    ddtheta_prev = dd_th_d_t2_f[i-1] - d_th_d_t_f[i-2]
    d_th_dt_half = d_th_dt_cur + 0.5 * ddtheta_prev * ds
    # Compute current joint values and velocities
    q_cur = [
        0,
        d_psi1_dth(theta_cur + ds/2) * d_th_dt_half,
        d_psi2_dth(theta_cur + ds/2) * d_th_dt_half,
        d_psi3_dth(theta_cur + ds/2) * d_th_dt_half
    ]
    # M, h = ComputeMassMatrixAndForceTerms(q_cur, dq0)
    # Compute partial products f1 and f2
    f1 = M * [
        0,
        d_psi1_dth(theta_cur + ds/2) * d_th_dt_half,
        d_psi2_dth(theta_cur + ds/2) * d_th_dt_half,
        d_psi3_dth(theta_cur + ds/2) * d_th_dt_half,
    ]
    f2 = M * [
        0,
        dd_psi1_dth2(theta_cur + ds/2) * d_th_dt_half * d_th_dt_half,
        dd_psi2_dth2(theta_cur + ds/2) * d_th_dt_half * d_th_dt_half,
        dd_psi3_dth2(theta_cur + ds/2) * d_th_dt_half * d_th_dt_half,
    ]
    ddtheta_arr = zeros(3)
    ddtheta_arr[1] = Inf
    for j in 1:3
        if abs(f1[j]) < 1e-12
            ddtheta_arr[j] = Inf
        elseif f1[j] < 0
            Ti = -Tmax[j] - h[j]
            ddtheta_arr[j] = (Ti - f2[j]) / f1[j]
        else
            Ti = Tmax[j] - h[j]
            ddtheta_arr[j] = (Ti - f2[j]) / f1[j]
        end
    end
    ddtheta = minimum(ddtheta_arr)
    d_th_d_t_f[i] = min(vel_profile[i], d_th_dt_cur + ddtheta * ds) 
    # Backward pass
    theta_cur, d_th_dt_cur = theta[end-i+1], d_th_d_t_b[end-i+1]
    ddtheta_prev = dd_th_d_t2_b[end-i+1] - d_th_d_t_b[end-i+2]
    d_th_dt_half = d_th_dt_cur + 0.5 * ds * ddtheta_prev

    q_cur = [
        0,
        psi1(theta_cur - ds/2),
        psi2(theta_cur - ds/2),
        psi3(theta_cur - ds/2)
    ]
    dq0_cur = [
        0,
        d_psi1_dth(theta_cur - ds/2) * d_th_dt_half,
        d_psi2_dth(theta_cur - ds/2) * d_th_dt_half,
        d_psi3_dth(theta_cur - ds/2) * d_th_dt_half
    ]
    # M, h = ComputeMassMatrixAndForceTerms(q_cur, dq0_cur)
    f1 = M * [
        0,
        d_psi1_dth(theta_cur - ds/2) * d_th_dt_half,
        d_psi2_dth(theta_cur - ds/2) * d_th_dt_half,
        d_psi3_dth(theta_cur - ds/2) * d_th_dt_half,
    ]
    f2 = M * [
        0,
        dd_psi1_dth2(theta_cur - ds/2) * d_th_dt_half * d_th_dt_half,
        dd_psi2_dth2(theta_cur - ds/2) * d_th_dt_half * d_th_dt_half,
        dd_psi3_dth2(theta_cur - ds/2) * d_th_dt_half * d_th_dt_half,
    ]
    ddtheta_arr = zeros(3)
    ddtheta_arr = fill(-Inf, 3)
    for j in 1:3
        if abs(f1[j]) < 1e-12
            ddtheta_arr[j] = -Inf
        elseif f1[j] < 0
            Ti = -Tmax[j] - h[j]
            ddtheta_arr[j] = (Ti - f2[j]) / f1[j]
        else
            Ti = Tmax[j] - h[j]
            ddtheta_arr[j] = (Ti - f2[j]) / f1[j]
        end
    end
    ddtheta = maximum(ddtheta_arr)
    d_th_d_t_new = min(vel_profile[end-i+1], d_th_dt_cur + ddtheta * ds)

    d_th_d_t_b[end-i+1] = abs(d_th_d_t_new)
end

traj = min.(d_th_d_t_f, d_th_d_t_b)

torq_after_opt = zeros(2, N-1)

for i in 2:N
    theta_cur   = 0.5 * (theta[i-1] + theta[i])
    d_th_dt_cur = 0.5 * (traj[i-1] + traj[i])
    dd_th_d_t2_cur = (traj[i] - traj[i-1]) / ds * d_th_dt_cur

    q_cur = [
        0.0,
        psi1(theta_cur),
        psi2(theta_cur),
        psi3(theta_cur)
    ]

    dq_cur = [
        0.0,
        d_psi1_dth(theta_cur) * d_th_dt_cur,
        d_psi2_dth(theta_cur) * d_th_dt_cur,
        d_psi3_dth(theta_cur) * d_th_dt_cur
    ]

    ddq_cur = [
        0.0,
        dd_psi1_dth2(theta_cur) * d_th_dt_cur^2 +
        d_psi1_dth(theta_cur)  * dd_th_d_t2_cur,

        dd_psi2_dth2(theta_cur) * d_th_dt_cur^2 +
        d_psi2_dth(theta_cur)  * dd_th_d_t2_cur,

        dd_psi3_dth2(theta_cur) * d_th_dt_cur^2 +
        d_psi3_dth(theta_cur)  * dd_th_d_t2_cur
    ]

    torq = M * ddq_cur + h
    torq_after_opt[:, i-1] = torq[2:3]
end

npzwrite("velocity_prof_opt_k=$(k)_p=$(p).npz",
         Dict("velocity_profile" => traj,
              "theta" => theta))

npzwrite("torq_after_opt_k=$(k)_p=$(p).npz",
         Dict("torq" => torq_after_opt,
              "theta" => theta))

plot(theta, vel_profile, label="Velocity limit")
plot!(theta, d_th_d_t_f, label="Forward")
plot!(theta, d_th_d_t_b, label="Backward")
plot!(theta, traj, label="Final trajectory")
xlabel!("θ")
ylabel!("θ̇")
gr()

plot(theta[1:end-1], torq_before_opt[1,:],
     label="Joint 1 before")
plot!(theta[1:end-1], torq_before_opt[2,:],
     label="Joint 2 before")

plot!(theta[1:end-1], torq_after_opt[1,:],
     label="Joint 1 after")

plot!(theta[1:end-1], torq_after_opt[2,:],
     label="Joint 2 after")

xlabel!("θ")
ylabel!("Torque")
gr()

time = zeros(N)
for i in 2:N
    if traj[i] > 1e-10
        time[i] = time[i-1] + ds / traj[i]
    else
        time[i] = time[i-1]
    end
end

plot(time, theta,
     xlabel="t (s)",
     ylabel="θ",
     label="θ(t)")