using Flexia
include(joinpath(pkgdir(Flexia), "src", "coordinate_reduction.jl"))
using LinearAlgebra
using GLMakie
using OptimalTrajectoryGeneration
using NPZ
include("serial.jl")
include("gear_constraint.jl")

# Same wafer-handler topology as examples/flexia/scara.jl (link 3 slaved to
# links 1,2 by θ3=(θ1+θ2)/2 via GearConstraint) and, in closed form, as
# examples/scara_robot.jl (whose q1,q1+q2,q1+0.5*q2 formula is this exact
# relation with q2 written as a RELATIVE angle). The robot below reproduces
# that same relative-angle q (see T_rel below FlexiaSCARAPlaceholder) and the
# same M = I(2), h = 0 placeholder dynamics, so forward_kinematics here is
# algebraically identical to scara_robot.jl's closed form and the TOPP result
# should match the root example's graphs up to Newton-solve tolerance.
l1 = 0.2
l2 = 0.2
l3 = 0.3
m1 = 1.0
m2 = 1.0
m3 = 1.0
izz1 = 0.01
izz2 = 0.01
izz3 = 0.01

lengths = [l1, l2, l3]
chain = build_serial_chain(lengths, [m1, m2, m3], [izz1, izz2, izz3]; with_motors=false, auto_assemble=false)
sys = chain.sys
link1, link2, link3 = chain.links

add!(sys, GearConstraint([link1, link2, link3], [-0.5, -0.5, 1.0]))
assemble!(sys)

"Straight-line pose (x, y, θ=0) of each link, chain laid out along +X from
the origin."
function straight_chain_poses(lengths)
    poses = Vector{SVector{3,Float64}}(undef, length(lengths))
    x = 0.0
    for i in eachindex(lengths)
        poses[i] = SA[x + lengths[i] / 2, 0.0, 0.0]
        x += lengths[i]
    end
    return poses
end
poses = straight_chain_poses(lengths)

initial = zeros(Float64, number_of_dofs(sys))
for (link, pose) in zip(chain.links, poses)
    set_initial_position!(initial, sys, link, pose)
end

sol = reshape(initial, :, 1)
total_reach = sum(lengths)
fig = Flexia.draw_static(sys, sol; limits=(-0.05, total_reach + 0.05, -total_reach / 2, total_reach / 2))
save(joinpath(@__DIR__, "out", "scara_placeholder_static.png"), fig)
display(fig)

# --- Reduce to 2 independent generalized coordinates q=(θ1,θ2); θ3 is dependent ---
rm = ReducedModel(sys, [angle_dof(link1), angle_dof(link2)])

p_guess = zeros(n_coords(rm))
for (link, pose) in zip(chain.links, poses)
    p_guess[collect(body_dofs(link))] .= pose
end

p_zero = assemble_position(rm, [0.0, 0.0]; p0=p_guess)
println("Reduction check at zero configuration (q̇ = [0.3, -0.2]):")
check_reduction(rm, p_zero, [0.3, -0.2])

# ==================== Placeholder robot: real Flexia kinematics/Jacobian, ====
# ==================== but M = I(2), h = 0 instead of real dynamics.       ====
mutable struct FlexiaSCARAPlaceholder <: OptimalTrajectoryGeneration.AbstractRobotManipulator
    rm::ReducedModel
    link3::Body2D
    l3::Float64
    dof::Int
    p_cache::Vector{Float64}
end
FlexiaSCARAPlaceholder(rm, link3, l3, p_init) = FlexiaSCARAPlaceholder(rm, link3, l3, 2, copy(p_init))

function tip_position(robot::FlexiaSCARAPlaceholder, p::AbstractVector{T}) where {T}
    x3, y3, θ3 = p[collect(body_dofs(robot.link3))]
    return [x3 + robot.l3 / 2 * cos(θ3), y3 + robot.l3 / 2 * sin(θ3)]
end

# The robot-facing q here is (θ1, q2) with q2 RELATIVE (θ2_abs = θ1+q2), the
# same convention as examples/scara_robot.jl — not ReducedModel's own q_abs =
# (θ1,θ2_abs). T is the (constant) Jacobian of that change of variables;
# q_abs = T*q. Composed with tip_position this makes forward_kinematics here
# algebraically identical to scara_robot.jl's closed-form formula, so M=I(2)
# means the same physical placeholder dynamics in both examples.
const T_rel = [1.0 0.0; 1.0 1.0]

function OptimalTrajectoryGeneration.forward_kinematics(robot::FlexiaSCARAPlaceholder, q::AbstractVector{Float64})::Vector{Float64}
    robot.p_cache = assemble_position(robot.rm, T_rel * q; p0=robot.p_cache)
    return tip_position(robot, robot.p_cache)
end

function OptimalTrajectoryGeneration.jacobian(robot::FlexiaSCARAPlaceholder, q::AbstractVector{Float64})::Matrix{Float64}
    robot.p_cache = assemble_position(robot.rm, T_rel * q; p0=robot.p_cache)
    red = reduce_at(robot.rm, robot.p_cache)
    Jp = ForwardDiff.jacobian(pp -> tip_position(robot, pp), robot.p_cache)
    return Jp * red.R * T_rel
end

function OptimalTrajectoryGeneration.compute_mass_and_force_terms(
    robot::FlexiaSCARAPlaceholder, q::AbstractVector{Float64}, dq::AbstractVector{Float64}
)::Tuple{Matrix{Float64},Vector{Float64}}
    return Matrix{Float64}(I, robot.dof, robot.dof), zeros(robot.dof)
end

robot = FlexiaSCARAPlaceholder(rm, link3, l3, p_zero)

# ==================== TOPP: same experiment as examples/scara_usage.jl and
# examples/flexia/scara.jl (bezier path, velocity-only TOPP, Tmax estimate,
# torque-limited TOPP, same k=0.9, p_frac=0.75), but through Flexia's own
# forward_kinematics/jacobian with placeholder M=I(2), h=0 dynamics. ====
q0 = [0.01, -0.02]
p0 = OptimalTrajectoryGeneration.forward_kinematics(robot, q0)
for x_new in LinRange(p0[1], 0.65, 100)
    global q0 = inverse_kinematics(robot, [x_new, 0.0], q0)
end
p0 = OptimalTrajectoryGeneration.forward_kinematics(robot, q0)
p1 = [p0[2], p0[1]]

k = 0.9
curve = make_bezier(p0, p1, k * p0[1])

w_max = 335 * π / 180
p_frac = 0.75
N = 4001
theta = range(0.0, 1.0, length=N)

constraints_vel_only = TrajectoryConstraints(
    fill(w_max, robot.dof), fill(Inf, robot.dof), fill(Inf, robot.dof), fill(Inf, robot.dof),
    (fill(-Inf, robot.dof), fill(Inf, robot.dof)))
result_vel = generate_joint_trajectory(robot, curve, q0, constraints_vel_only; n_points=N)

mask = (theta .> 0.05) .& (theta .< 0.95)   # boundary torque spikes are a known TOPP artifact
Tmax_val = maximum(abs.(result_vel.torques[mask, :])) * p_frac

constraints = TrajectoryConstraints(
    fill(w_max, robot.dof), fill(100, robot.dof), fill(Tmax_val, robot.dof), fill(Inf, robot.dof),
    (fill(-Inf, robot.dof), fill(Inf, robot.dof)))
result = generate_joint_trajectory(robot, curve, q0, constraints; n_points=N)

println("Feasible: ", result.feasible)

outdir = joinpath(@__DIR__, "out", "placeholder")
mkpath(outdir)

let
    fig2 = Figure()
    ax = Axis(fig2[1, 1], xlabel="θ", ylabel="θ̇", title="Path speed profile (Flexia geometry, M=I placeholder)")
    lines!(ax, theta, result.path_velocity, label="Optimized θ̇(θ)", color=:green, linewidth=2)
    axislegend(ax)
    save(joinpath(outdir, "velocity_profiles.png"), fig2)
end

let
    fig2 = Figure()
    ax = Axis(fig2[1, 1], xlabel="θ", ylabel="Torque", title="Joint torques along path (Flexia geometry, M=I placeholder)")
    lines!(ax, theta, result_vel.torques[:, 1], label="Joint 1 (before opt)", color=(:blue, 0.35), linestyle=:dash)
    lines!(ax, theta, result_vel.torques[:, 2], label="Joint 2 (before opt)", color=(:red, 0.35), linestyle=:dash)
    lines!(ax, theta, result.torques[:, 1], label="Joint 1", color=:blue)
    lines!(ax, theta, result.torques[:, 2], label="Joint 2", color=:red)
    hlines!(ax, [Tmax_val, -Tmax_val], color=:black, linestyle=:dot, label="±Tmax")
    axislegend(ax, position=:lt)
    y_pad = 0.1 * maximum(abs.(result.torques))
    ylims!(ax, -maximum(abs.(result.torques)) - y_pad, maximum(abs.(result.torques)) + y_pad)
    save(joinpath(outdir, "torque_comparison.png"), fig2)
end

let
    fig2 = Figure()
    ax = Axis(fig2[1, 1], xlabel="t (s)", ylabel="θ", title="θ(t) — optimized (Flexia geometry, M=I placeholder)")
    lines!(ax, result.time_vector, theta)
    save(joinpath(outdir, "theta_time.png"), fig2)
end

npzwrite(joinpath(outdir, "velocity_prof_opt_k=$(k)_p=$(p_frac).npz"),
    Dict("velocity_profile" => result.path_velocity, "theta" => collect(theta)))
npzwrite(joinpath(outdir, "torq_after_opt_k=$(k)_p=$(p_frac).npz"),
    Dict("torq" => result.torques, "theta" => collect(theta)))

println("Plots and .npz saved to: ", outdir)
