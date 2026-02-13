using ForwardDiff
using LinearAlgebra
using Plots
using Interpolations
using Polynomials
using QuadGK
using Dierckx

abstract type AbstractRobotManipulator end

struct TestRobot <: AbstractRobotManipulator
    l1::Float64
    l2::Float64
    dof::Int
end

function forward_kinematics(
    robot::TestRobot,
    q::AbstractVector
)
    q1 = q[1]
    q2 = q[2]
    l1, l2 = robot.l1, robot.l2
    x = l1 * cos(q1) + l1 * cos(q1 + q2) + l2 * cos(q1 + 0.5 * q2)
    y = l1 * sin(q1) + l1 * sin(q1 + q2) + l2 * sin(q1 + 0.5 * q2)
    return [x, y]
end

function body_jacobian(robot::TestRobot, q::Vector{Float64})
    J = ForwardDiff.jacobian(q_vec -> forward_kinematics(robot, q_vec), q)
    return J
end

function ik(
    robot::TestRobot,
    target::AbstractVector,
    initial_guess::AbstractVector;
    max_iterations::Int = 100,
    tolerance::Float64 = 1e-6,
    verbose::Bool = false
)
    q = copy(initial_guess)
    for i in 1:max_iterations
        current_pos = forward_kinematics(robot, q)
        error = target - current_pos
        if norm(error) < tolerance
            verbose && println("Converged in $i iterations.")
            return q
        end
        J = body_jacobian(robot, q)
        delta_q = pinv(J) * error
        q += delta_q
    end
end

struct BezierCurve
    control_points::Matrix{Float64}
end

function make_bezier(
    p_start::AbstractVector,
    p_end::AbstractVector,
    t_param = 0.7
)
    # Control points for cubic Bezier curve (2, 4)
    P = hcat(p_start, [t_param, 0.0], [0.0, t_param], p_end)
    return BezierCurve(P)
end

function cartesian_traj(curve::BezierCurve, theta::AbstractVector, robot::TestRobot)
    # Evaluate Bezier curve at parameter values theta
    P = curve.control_points
    num_points = length(theta)
    trajectory = zeros(num_points, robot.dof)
    
    for i in 1:num_points
        t = theta[i]
        # Cubic Bezier formula: B(t) = (1-t)³P₀ + 3(1-t)²t P₁ + 3(1-t)t² P₂ + t³ P₃
        mt = 1 - t
        point = mt^3 * P[:, 1] + 3*mt^2*t * P[:, 2] + 3*mt*t^2 * P[:, 3] + t^3 * P[:, 4]
        trajectory[i, :] = point
    end
    
    return trajectory
end

function max_theta_dot(
    jnt_traj::AbstractMatrix,
    theta::AbstractVector,
    w1_max::Float64,
    w2_max::Float64
)
    # Create cubic spline interpolations with linear extrapolation
    if isa(theta, Vector)
        theta_range = range(theta[1], theta[end], length=length(theta))
    else
        theta_range = theta
    end
    
    spl1 = cubic_spline_interpolation(theta_range, jnt_traj[:, 1], bc=Line(OnGrid()))
    spl2 = cubic_spline_interpolation(theta_range, jnt_traj[:, 2], bc=Line(OnGrid()))
    
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

function joint_traj(
    cartesian_traj::Matrix{Float64},
    robot::TestRobot,
    initial_guess::AbstractVector
)
    N = size(cartesian_traj, 1)
    joint_trajectory = zeros(N, robot.dof)
    for i in 1:N
        target = cartesian_traj[i, :]
        joint_trajectory[i, :] = ik(robot, target, initial_guess, verbose=false)
    end
    return joint_trajectory
end

function time_parametrise(
    theta::AbstractVector,
    theta_dot_max::AbstractVector
)
    idx = vcat[1:50:length(theta), length(theta)]
    knots = copy(theta_dot_max)
    knots[0] = knots[end] = 0.0
    itp = interpolate(y, BSpline(Akima(Line(OnGrid()))))
    vel_prof = zeros(T, n)
    if isapprox(diff(theta[idx]), fill(mean(diff(theta[idx])), length(idx)-1), rtol=1e-6)
        for i in 1:length(theta)
            scaled_idx = 1 + (i-1) * (length(idx) - 1) / (n - 1)
            vel_prof[i] = itp(scaled_idx)
        end
    else
        itp = linear_interpolation(theta[idx], knots)
        vel_prof = itp.(theta)
    end
    h = theta[1] - theta[0]
    time = zeros(T, n)
    time[2:end-1] = cumsum(h./vel_prof[2:end-1])
    if n >= 5
        idx_fit = (n-3):(n-1)
        p = fit(theta[idx_fit], time[idx_fit], 2)
        time[end] = p(theta[end])
    end
    return time, vel_prof
end

# def plot_joints_vs_time(time, jnt_traj):
#     plt.plot(time, jnt_traj[:,0], label='q1(t)')
#     plt.plot(time, jnt_traj[:,1], label='q2(t)')
#     plt.legend(); plt.show()
function plot_joints_vs_time(time, jnt_traj)
    plot(time, [joint_traj[:, 0], joint_traj[:,1]], label=["q1(t)" "q2(t)"])
end

function time_step(
    ds::Float64,
    v0::Float64,
    a0::Float64,
    a1::Float64
    )
    j = (a1 - a0) / ds
    if abs(v0) < 1e-6
        dt = sqrt(2ds / a0)
    else
        dt, error = quadgk(s -> 1/sqrt(v0^2 + 2a0 * s + j * s^2), 0, ds)
    end
    dv = a0 * dt + (j * dt^2) / 2
    return dt, dv
end

function time_step2(v0, v1)
    return  abs((log(v1) - log(v1)) / (v1 - v0))
end

function velocity_limit(
    theta,
    d_psi_1_d_th,
    d_psi_2_d_th,
    w1_max,
    w2_max
)
    j1_lim = abs(w1_max/d_psi_1_d_th(theta))
    j2_lim = abs(w2_max/d_psi_2_d_th(theta))

    return min(j1_lim, j2_lim)
end

function acceleration_limits(
    theta,
    d_th_dt,
    d_psi_1_d_th,
    d_psi_2_d_th,
    dd_psi_1_d_th2,
    dd_psi_2_d_th2,
    e1_max,
    e2_max,
    dtheta=1e-3
)
    if abs(d_th_dt) < 1e-6
        return [
            -e1_max / abs(dd_psi_1_d_th2(theta+dtheta/2) * dtheta * dtheta + d_psi_1_d_th(theta+dtheta/2) * dtheta),
            e1_max / abs(dd_psi_1_d_th2(theta+dtheta/2) * dtheta * dtheta + d_psi_1_d_th(theta+dtheta/2) * dtheta)
        ]
    if (d_psi_1_d_th(theta) * d_th_dt > 0)
        j1_lim = [
            (- e1_max - dd_psi_1_d_th2(theta) * d_th_dt * d_th_dt) / d_psi_1_d_th(theta) / d_th_dt,
            (e1_max - dd_psi_1_d_th2(theta) * d_th_dt * d_th_dt) / d_psi_1_d_th(theta) / d_th_dt
        ]
    else
        j1_lim = [
            (e1_max - dd_psi_1_d_th2(theta) * d_th_dt * d_th_dt) / d_psi_1_d_th(theta) / d_th_dt,
            (- e1_max - dd_psi_1_d_th2(theta) * d_th_dt * d_th_dt) / d_psi_1_d_th(theta) / d_th_dt,
        ]
    if (d_psi_2_d_th(theta) * d_th_dt > 0)
        j2_lim = [
            (- e2_max - dd_psi_2_d_th2(theta)* d_th_dt * d_th_dt) / d_psi_2_d_th(theta) / d_th_dt,
            (e2_max - dd_psi_2_d_th2(theta) * d_th_dt * d_th_dt) / d_psi_2_d_th(theta) / d_th_dt,
        ]
    else
        j2_lim = [
            (e2_max - dd_psi_2_d_th2(theta) * d_th_dt * d_th_dt) / d_psi_2_d_th(theta) / d_th_dt,
            (- e2_max - dd_psi_2_d_th2(theta)* d_th_dt * d_th_dt) / d_psi_2_d_th(theta) / d_th_dt,
        ]
    low_lim = max([
        j1_lim[0],
        j2_lim[0]
    ])
    upp_lim = min([
        j1_lim[1],
        j2_lim[1]
    ])
    return low_lim, upp_lim
end

# Example usage
L1 = 0.2
l2 = 0.3

w1_max = w2_max = 335/360
e1_max = e2_max = 2500
q0 = [0.01, -0.02]
testr = TestRobot(L1, L2, 2)
p0 = forward_kinematics(
    testr, q0
)

for x_new in LinRange(p0[1], 0.65, 100)
    q0 = ik(testr, [x_new, 0.0], q0)
end

p0 = forward_kinematics(testr, q0)
p1 = [p0[2], p0[1]]
k = 0.9
curve = make_bezier(p0, p1, k*p0[0])
N = 4001
theta = LinRange(0, 1, N)
cart_traj = cartesian_traj(curve, theta, testr)
jnt_traj = joint_traj(cart_traj, testr)
spl1 = Spline1D(theta, jnt_traj[:,0], k=3, s=0.0)
spl1 = Spline1D(theta, jnt_traj[:,1], k=3, s=0.0)

