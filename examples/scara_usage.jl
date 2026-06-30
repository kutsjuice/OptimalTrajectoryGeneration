using OptimalTrajectoryGeneration
using CairoMakie
using NPZ
using LinearAlgebra

include("scara_robot.jl")  # SCARARobot + forward_kinematics/jacobian/dynamics

# ==================== ROBOT & PATH SETUP ====================

robot = SCARARobot(0.2, 0.3)

q0 = [0.01, -0.02]
p0 = OptimalTrajectoryGeneration.forward_kinematics(robot, q0)

# Sweep q0 along x until the body Jacobian is well-conditioned at the
# starting pose (same trick the original script used).
for x_new in LinRange(p0[1], 0.65, 100)
    global q0 = inverse_kinematics(robot, [x_new, 0.0], q0)
end
p0 = OptimalTrajectoryGeneration.forward_kinematics(robot, q0)
p1 = [p0[2], p0[1]]

k = 0.9
curve = make_bezier(p0, p1, k * p0[1])

# ==================== CONSTRAINTS ====================

w_max = 335 * π / 180   # rad/s, per joint
p_frac = 0.75            # fraction of an unconstrained-run peak torque to allow
# ==================== RUN TOPP ====================

N = 4001
constraints_vel_only = TrajectoryConstraints(
    fill(w_max, robot.dof),
    fill(Inf, robot.dof),
    fill(Inf, robot.dof),
    fill(Inf, robot.dof),
    (fill(-Inf, robot.dof), fill(Inf, robot.dof))
)
result_vel = generate_joint_trajectory(robot, curve, q0, constraints_vel_only; n_points=N)

# Exclude the boundary region: the TOPP forward/backward pass forces
# theta_dot = 0 at both path endpoints, which produces large but
# physically irrelevant acceleration/torque spikes there. Mask them out
# before estimating the torque limit, same as the original script did.
theta_vel = range(0.0, 1.0, length=N)
mask = (theta_vel .> 0.05) .& (theta_vel .< 0.95)
Tmax_val = maximum(abs.(result_vel.torques[mask, :])) * p_frac

constraints = TrajectoryConstraints(
    fill(w_max, robot.dof),                 # velocity_limits
    fill(100, robot.dof),                   # acceleration_limits (unused by TOPP yet)
    fill(Tmax_val, robot.dof),               # torque_limits
    fill(Inf, robot.dof),                   # jerk_limits (unused)
    (fill(-Inf, robot.dof), fill(Inf, robot.dof)),  # position_limits
)
result = generate_joint_trajectory(robot, curve, q0, constraints; n_points=N)

println("Feasible: ", result.feasible)

theta = range(0.0, 1.0, length=N)

# ==================== PLOTS ====================

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="θ", ylabel="θ̇", title="Path speed profile")
    lines!(ax, theta, result.path_velocity, label="Optimized θ̇(θ)", color=:green, linewidth=2)
    axislegend(ax)
    display(fig)
    save("velocity_profiles.png", fig)
end

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="θ", ylabel="Torque", title="Joint torques along path")
    lines!(ax, theta_vel, result_vel.torques[:, 1], label="Joint 1 (before opt)", color=(:blue, 0.35), linestyle=:dash)
    lines!(ax, theta_vel, result_vel.torques[:, 2], label="Joint 2 (before opt)", color=(:red, 0.35), linestyle=:dash)
    lines!(ax, theta, result.torques[:, 1], label="Joint 1", color=:blue)
    lines!(ax, theta, result.torques[:, 2], label="Joint 2", color=:red)
    hlines!(ax, [Tmax_val, -Tmax_val], color=:black, linestyle=:dot, label="±Tmax")
    axislegend(ax, position=:lt)
    y_pad = 0.1 * maximum(abs.(result.torques))
    ylims!(ax, -maximum(abs.(result.torques)) - y_pad, maximum(abs.(result.torques)) + y_pad)
    display(fig)
    save("torque_comparison.png", fig)
end

let
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="t (s)", ylabel="θ", title="θ(t) — optimized")
    lines!(ax, result.time_vector, theta)
    display(fig)
    save("theta_time.png", fig)
end

npzwrite("velocity_prof_opt_k=$(k)_p=$(p_frac).npz",
    Dict("velocity_profile" => result.path_velocity, "theta" => collect(theta)))
npzwrite("torq_after_opt_k=$(k)_p=$(p_frac).npz",
    Dict("torq" => result.torques, "theta" => collect(theta)))
