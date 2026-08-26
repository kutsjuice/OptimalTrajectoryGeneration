using Flexia
include(joinpath(pkgdir(Flexia), "src", "coordinate_reduction.jl"))
using GLMakie
using OptimalTrajectoryGeneration
using NPZ
using Dierckx
include("serial.jl")
include("gear_constraint.jl")
include("flexia_robot.jl")

l1 = 0.205
l2 = 0.205
l3 = 0.39
m1 = 6.936
m2 = 7.609
m3 = 0.533

# CAD data (assembly global frame, arm fully extended = reference pose).
# Motion plane is XY (all three joints rotate about vertical Z); the base
# column joint0->joint1 (300 mm in Z, com_cyl_global) is dropped — it has no
# mass of its own in this model and contributes no horizontal offset.
joint_xy = [[0.0, 0.0], [-l1, 0.0], [-(l1 + l2), 0.0]]
com_xy = [[0.0, 0.0], [-l1, 0.0], [-(l1 + l2), 0.0]]

# Per-link inertia tensor in CAD (X,Y,Z) axes, about each link's own COM
# (isolated-part CAD mass properties) -> CAD Z is global Z, so only
# Izz = tensor[3,3] is needed for the planar model.
inertiaTensor1 = [
    25749603.700 25062.125 9651359.305
    25062.125 92340828.791 24778.147
    9651359.305 24778.147 83172050.999
] ./ 1e9
inertiaTensor2 = [
    13449632.937 -24026.329 1660746.774
    -24026.329 84912103.395 27847.617
    1660746.774 27847.617 89804961.922
] ./ 1e9
inertiaTensor3 = [
    480197.752 835.157 2908.542
    835.157 14103107.261 0.0
    2908.542 0.0 14577958.019
] ./ 1e9

lengths = [l1, l2, l3]
masses = [m1, m2, m3]
tensors = [inertiaTensor1, inertiaTensor2, inertiaTensor3]
direction = [-1.0, 0.0]   # CAD reference pose points along -X

# Body2D assumes a link's COM sits exactly at the midpoint between its two
# joints (±length/2). This robot's real COM is offset toward the proximal
# joint (motor/gearbox on the axis) — accepted approximation (COM -> midpoint)
# — so Izz is moved from the true COM to that assumed midpoint via the
# parallel-axis theorem: Izz_mid = Izz_com + m·d², d = planar offset.
izz = Float64[]
for (jt, cm, len, m, t) in zip(joint_xy, com_xy, lengths, masses, tensors)
    mid = jt .+ (len / 2) .* direction
    d = cm .- mid
    push!(izz, t[3, 3] + m * (d[1]^2 + d[2]^2))
end

# coordinate_reduction.jl requires a system WITHOUT actuators (a Flexia motor
# is itself a kinematic constraint, which would leave d = 3n_b - m = 0 free
# coordinates). The link-3 coupling is added as its own constraint below, so
# no joint here gets a motor.
chain = build_serial_chain(lengths, masses, izz; with_motors=false, auto_assemble=false)
sys = chain.sys
link1, link2, link3 = chain.links

# Wafer-handler coupling: link 3 is not independently driven, its angle is
# slaved to links 1 and 2 by θ3 = (θ1+θ2)/2, i.e. -0.5·θ1 - 0.5·θ2 + θ3 = 0.
add!(sys, GearConstraint([link1, link2, link3], [-0.5, -0.5, 1.0]))
assemble!(sys)

"Straight-line pose (x, y, θ=0) of each link, chain laid out along +X from
the origin — the factory's zero-target pose (mirrored vs. the CAD -X pose)."
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
save(joinpath(@__DIR__, "out", "scara_static.png"), fig)
screen = display(fig)

# --- Reduce to 2 independent generalized coordinates q=(θ1,θ2); θ3 is dependent ---
rm = ReducedModel(sys, [angle_dof(link1), angle_dof(link2)])

p_guess = zeros(n_coords(rm))
for (link, pose) in zip(chain.links, poses)
    p_guess[collect(body_dofs(link))] .= pose
end

p_zero = assemble_position(rm, [0.0, 0.0]; p0=p_guess)
println("Reduction check at zero configuration (q̇ = [0.3, -0.2]):")
check_reduction(rm, p_zero, [0.3, -0.2])

# ==================== TOPP: same experiment as examples/scara_usage.jl
# (bezier path, velocity-only TOPP, Tmax estimate, torque-limited TOPP, same
# k=0.9, p_frac=0.75) but with the REAL M(q),h(q,q̇) from the CAD model
# instead of scara_robot.jl's kinematic placeholder (identity M, h=0). ====
robot = FlexiaSCARA(rm, link3, l3, p_zero)

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

outdir = joinpath(@__DIR__, "out", "topp")
mkpath(outdir)

let
    fig2 = Figure()
    ax = Axis(fig2[1, 1], xlabel="θ", ylabel="θ̇", title="Path speed profile (Flexia CAD dynamics)")
    lines!(ax, theta, result.path_velocity, label="Optimized θ̇(θ)", color=:green, linewidth=2)
    axislegend(ax)
    save(joinpath(outdir, "velocity_profiles.png"), fig2)
end

let
    fig2 = Figure()
    ax = Axis(fig2[1, 1], xlabel="θ", ylabel="Torque", title="Joint torques along path (Flexia CAD dynamics)")
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
    ax = Axis(fig2[1, 1], xlabel="t (s)", ylabel="θ", title="θ(t) — optimized (Flexia CAD dynamics)")
    lines!(ax, result.time_vector, theta)
    save(joinpath(outdir, "theta_time.png"), fig2)
end


# ==================== Full replay through Flexia's own actuators ====
# Build a SEPARATE Flexia model (its own bodies/joints — no aliasing with
# `sys`/`rm` above) with REAL PositionMotor2D actuators wired exactly like the
# physical robot: motor1 grounded on joint1 (drives θ1 absolute), motor2 on
# joint2 (drives the RELATIVE angle θ2-θ1, since a HingeJoint motor's
# constraint is θ_body2 - θ_body1 - target = 0). Feed it the optimized
# θ1(t), θ2(t) as time-varying motor targets and integrate Flexia's own DAE
# solver (`simulate`, a Rosenbrock/CROS stepper) forward in time — a full,
# independent replay through Flexia, not just a per-point M,h evaluation.
chain_act = build_serial_chain(lengths, masses, izz; with_motors=false, auto_assemble=false)
sys_act = chain_act.sys
l1a, l2a, l3a = chain_act.links
add!(sys_act, GearConstraint([l1a, l2a, l3a], [-0.5, -0.5, 1.0]))
motor1 = PositionMotor2D(chain_act.joints[1], 0.0)   # ground -> link1: target = θ1(t)
motor2 = PositionMotor2D(chain_act.joints[2], 0.0)   # link1 -> link2: target = θ2(t) - θ1(t)
add!(sys_act, motor1)
add!(sys_act, motor2)
assemble!(sys_act)

# Drop near-duplicate time samples (path speed ≈ 0 at the very ends) so the
# spline knots are strictly increasing, as Dierckx requires.
keep = [1; findall(diff(result.time_vector) .> 1e-9) .+ 1]
t_knots = result.time_vector[keep]
spl_th1 = Spline1D(t_knots, result.trajectory.positions[keep, 1], k=3, s=0.0)
spl_th2 = Spline1D(t_knots, result.trajectory.positions[keep, 2], k=3, s=0.0)
spl_tau1_topp = Spline1D(t_knots, result.torques[keep, 1], k=3, s=0.0)
spl_tau2_topp = Spline1D(t_knots, result.torques[keep, 2], k=3, s=0.0)

sys_act.prestep = (u) -> begin
    tt = clamp(u[end], t_knots[1], t_knots[end])
    θ1t, θ2t = spl_th1(tt), spl_th2(tt)
    settarget!(motor1, θ1t)
    settarget!(motor2, θ2t - θ1t)
end

# Consistent initial state: the same assembled pose as
# q0 = result.trajectory.positions[1,:], zero velocity (TOPP starts from rest).
p0_state = assemble_position(rm, result.trajectory.positions[1, :]; p0=p_zero)
initial_act = zeros(number_of_dofs(sys_act))
for body_act in sys_act.bodies
    initial_act[collect(get_body_position_dofs(sys_act, body_act))] .= p0_state[collect(body_dofs(body_act))]
end

n_steps = 5000
time_span = range(0.0, t_knots[end], length=n_steps)
sol_act = simulate(sys_act, initial_act, time_span)
t_sim = collect(time_span)

θ1_sim = sol_act[get_body_position_dofs(sys_act, l1a)[3], :]
θ2_sim = sol_act[get_body_position_dofs(sys_act, l2a)[3], :]

θ1_plan = spl_th1.(t_sim)
θ2_plan = spl_th2.(t_sim)

println("Actuator replay (Flexia DAE integration with real PositionMotor2D):")
println("  position tracking: max |θ1_sim - θ1_plan| = ", maximum(abs.(θ1_sim .- θ1_plan)))
println("                      max |θ2_sim - θ2_plan| = ", maximum(abs.(θ2_sim .- θ2_plan)))

# ==================== Honest torque: inverse_dynamics through the real
# actuator map B (motor1 grounded on link1, motor2 relative between link1
# and link2 — see the wiring comment above `motor1`/`motor2`), evaluated on
# the SAME time grid as the replay, using analytic derivatives of the
# planned θ(t) splines (no finite differences involved).
B = actuator_map(rm, [(nothing, link1), (link1, link2)])

τ1_real = zeros(length(t_sim))
τ2_real = zeros(length(t_sim))
p_cur2 = copy(p_zero)
for (i, tt) in enumerate(t_sim)
    q_t = [spl_th1(tt), spl_th2(tt)]
    dq_t = [Dierckx.derivative(spl_th1, tt, 1), Dierckx.derivative(spl_th2, tt, 1)]
    ddq_t = [Dierckx.derivative(spl_th1, tt, 2), Dierckx.derivative(spl_th2, tt, 2)]
    global p_cur2 = assemble_position(rm, q_t; p0=p_cur2)
    red = reduce_at(rm, p_cur2)
    τ = inverse_dynamics(rm, red, dq_t, ddq_t, B)
    τ1_real[i], τ2_real[i] = τ[1], τ[2]
end

println("  real motor torque (inverse_dynamics, B-mapped) vs. TOPP's Mq̈+h:")
println("    joint 1: max |τ| TOPP = ", maximum(abs.(spl_tau1_topp.(t_sim))), "   max |τ| real = ", maximum(abs.(τ1_real)))
println("    joint 2: max |τ| TOPP = ", maximum(abs.(spl_tau2_topp.(t_sim))), "   max |τ| real = ", maximum(abs.(τ2_real)))

# ==================== λ vs honest torque: Flexia's actuator reaction force
# used to be added to the POSITION dofs instead of the VELOCITY dofs (fixed
# upstream in commit 5dc0b84, "fix: correct lagrangian forces for
# actuators"). That fix made position tracking accurate (~1e-4 rad here), but
# λ read off sol_act is STILL not usable as a torque signal — confirmed
# empirically by rerunning this replay at n_steps=10k/50k/200k: max|λ| scales
# as ~1/h (grows 10x when h shrinks 10x) while position error shrinks as
# expected. This is the classic index-3 DAE symptom: the motor constraint
# (θ2-θ1-target=0) is enforced at the POSITION level with no Baumgarte/GGL
# velocity-level stabilization, so the multiplier diverges under mesh
# refinement instead of converging. `τ1_real`/`τ2_real` below (analytic,
# inverse_dynamics on the planned spline) remain the only reliable torque.
λ1_sim = sol_act[get_lms(sys_act, motor1)[1], :]
λ2_sim = sol_act[get_lms(sys_act, motor2)[1], :]

println("  λ (read directly off sol_act) vs. honest torque (inverse_dynamics):")
println("    joint 1: max |λ1 - τ1_real| = ", maximum(abs.(λ1_sim .- τ1_real)))
println("    joint 2: max |λ2 - τ2_real| = ", maximum(abs.(λ2_sim .- τ2_real)))

let
    fig4 = Figure()
    ax = Axis(fig4[1, 1], xlabel="t (s)", ylabel="Torque",
        title="Motor λ (from DAE state) vs. honest torque (inverse_dynamics)")
    lines!(ax, t_sim, τ1_real, label="Joint 1, honest (inverse_dynamics)", color=:blue)
    lines!(ax, t_sim, λ1_sim, label="Joint 1, λ (sol_act)", color=:blue, linestyle=:dash, linewidth=2)
    lines!(ax, t_sim, τ2_real, label="Joint 2, honest (inverse_dynamics)", color=:red)
    lines!(ax, t_sim, λ2_sim, label="Joint 2, λ (sol_act)", color=:red, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lt)
    save(joinpath(outdir, "lambda_vs_honest_torque.png"), fig4)
end

# ==================== Fixed: honest_constraint_forces (Flexia,
# src/constraint_forces.jl) reconstructs λ from the ALREADY-ACCURATE position
# trajectory instead of reading it out of the DAE state — sol_act's own λ
# diverges as ~1/h (above), and its own velocity dofs are unusable too (they
# stay pinned at ≈0 for a rigidly actuated body under this DAE formulation —
# a separate symptom of the same stiffness, even though position tracks the
# plan correctly). honest_constraint_forces sidesteps both: it re-derives v,
# v̇ from finite differences of sol_act's POSITION rows, then solves the
# well-posed index-1 KKT system M v̇ = F(q,v) + Φ_q(q)ᵀλ for λ directly. Works
# on the raw MBSystem2D — no ReducedModel required.
λ_reconstructed = honest_constraint_forces(sys_act, sol_act, step(time_span))
λ1_honest = λ_reconstructed[get_lms(sys_act, motor1)[1]-last_body_dof(sys_act), :]
λ2_honest = λ_reconstructed[get_lms(sys_act, motor2)[1]-last_body_dof(sys_act), :]

println("  λ reconstructed post-hoc (honest_constraint_forces) vs. honest torque (inverse_dynamics):")
println("    joint 1: max |λ1_honest - τ1_real| = ", maximum(abs.(λ1_honest .- τ1_real)))
println("    joint 2: max |λ2_honest - τ2_real| = ", maximum(abs.(λ2_honest .- τ2_real)))

let
    fig5 = Figure()
    ax = Axis(fig5[1, 1], xlabel="t (s)", ylabel="Torque",
        title="Reconstructed λ (post-hoc, position-FD) vs. honest torque (inverse_dynamics)")
    lines!(ax, t_sim, τ1_real, label="Joint 1, honest (inverse_dynamics)", color=:blue)
    lines!(ax, t_sim, λ1_honest, label="Joint 1, λ (reconstructed)", color=:blue, linestyle=:dash, linewidth=2)
    lines!(ax, t_sim, τ2_real, label="Joint 2, honest (inverse_dynamics)", color=:red)
    lines!(ax, t_sim, λ2_honest, label="Joint 2, λ (reconstructed)", color=:red, linestyle=:dash, linewidth=2)
    axislegend(ax, position=:lt)
    save(joinpath(outdir, "lambda_reconstructed_vs_honest.png"), fig5)
end

let
    fig2 = Figure()
    ax1 = Axis(fig2[1, 1], xlabel="t (s)", ylabel="θ", title="Position tracking: actuator replay vs. TOPP plan")
    lines!(ax1, t_sim, θ1_plan, label="θ1 plan", color=:blue)
    lines!(ax1, t_sim, θ1_sim, label="θ1 simulated", color=:blue, linestyle=:dash)
    lines!(ax1, t_sim, θ2_plan, label="θ2 plan", color=:red)
    lines!(ax1, t_sim, θ2_sim, label="θ2 simulated", color=:red, linestyle=:dash)
    axislegend(ax1, position=:lt)
    save(joinpath(outdir, "actuator_replay_position.png"), fig2)

    fig3 = Figure()
    ax2 = Axis(fig3[1, 1], xlabel="t (s)", ylabel="Torque",
        title="TOPP's Mq̈+h vs. real motor torque (inverse_dynamics)")
    lines!(ax2, t_sim, spl_tau1_topp.(t_sim), label="Joint 1, TOPP (Mq̈+h)", color=:blue)
    lines!(ax2, t_sim, τ1_real, label="Joint 1, real motor1 (inverse_dynamics)", color=:blue, linestyle=:dash, linewidth=2)
    lines!(ax2, t_sim, spl_tau2_topp.(t_sim), label="Joint 2, TOPP (Mq̈+h)", color=:red)
    lines!(ax2, t_sim, τ2_real, label="Joint 2, real motor2 (inverse_dynamics)", color=:red, linestyle=:dash, linewidth=2)
    axislegend(ax2, position=:lt)
    save(joinpath(outdir, "actuator_torque_real.png"), fig3)
end

npzwrite(joinpath(outdir, "velocity_prof_opt_k=$(k)_p=$(p_frac).npz"),
    Dict("velocity_profile" => result.path_velocity, "theta" => collect(theta)))
npzwrite(joinpath(outdir, "torq_after_opt_k=$(k)_p=$(p_frac).npz"),
    Dict("torq" => result.torques, "theta" => collect(theta)))

# --- Motion video: replay the optimized q(θ) trajectory through the full
# mechanism (θ3 reconstructed by Newton via assemble_position). Frames are
# spaced evenly in θ, not real time, since animate()'s record loop assumes a
# dense fixed-dt sol (stride 5) — oversample 5x so the stride doesn't waste
# most of our (sparser) samples. ---
function state_from_p(sys, bodies, p)
    st = zeros(number_of_dofs(sys))
    for body in bodies
        st[collect(get_body_position_dofs(sys, body))] .= p[collect(body_dofs(body))]
    end
    return st
end

n_frames = 750   # -> 150 rendered frames after animate()'s internal stride-5
frame_idx = round.(Int, range(1, N, length=n_frames))
sol_traj = zeros(number_of_dofs(sys), n_frames)
p_track = copy(p_zero)
for (k, i) in enumerate(frame_idx)
    global p_track = assemble_position(rm, result.trajectory.positions[i, :]; p0=p_track)
    sol_traj[:, k] = state_from_p(sys, sys.bodies, p_track)
end
video_path = joinpath(outdir, "scara_motion.mp4")
animate(sys, sol_traj, 1:n_frames, video_path;
        framerate=150, limits=(-0.05, total_reach + 0.05, -total_reach / 2, total_reach / 2))

println("Plots, .npz and motion video saved to: ", outdir)

wait(screen)
