using Plots

# Run the example to produce `result` (this will execute `another_try.jl`)
include("another_try.jl")

# Extract data
positions = result.trajectory.positions
velocities = result.trajectory.velocities
accelerations = result.trajectory.accelerations
torques = result.torques
time = collect(result.time_vector)
theta = result.theta
cart = result.cartesian_trajectory

dir = joinpath(@__DIR__, "..")

# Plot torques vs time
p1 = plot(time, torques[1, :], label="τ1", xlabel="t (s)", ylabel="Torque (Nm)")
plot!(time, torques[2, :], label="τ2")
plot!(time, torques[3, :], label="τ3")
savefig("torques_vs_time.png")

# Cartesian path plots
p2 = plot(cart[1, :], cart[2, :], label="XY path", xlabel="x (m)", ylabel="y (m)", title="End-effector path (XY)")
savefig("cartesian_xy.png")

p3 = plot(cart[1, :], cart[3, :], label="XZ path", xlabel="x (m)", ylabel="z (m)", title="End-effector path (XZ)")
savefig("cartesian_xz.png")

# Joint positions vs theta
p4 = plot()
for i in 1:size(positions, 1)
    plot!(theta, positions[i, :], label="q$(i)")
end
xlabel!("θ (path parameter)")
ylabel!("Joint position (rad or m)")
title!("Joint positions vs θ")
savefig("q_vs_theta.png")

println("Plots saved: torques_vs_time.png, cartesian_xy.png, cartesian_xz.png, q_vs_theta.png")
