"""
AbstractCartesianPath represents a geometric path in task space,
parametrized by t ∈ [0, 1].
"""
abstract type AbstractCartesianPath end

"""
AbstractJointPath represents a geometric path in joint space,
parametrized by θ ∈ [0, 1] (this θ plays the role of the path parameter
used by the time-optimal path parametrization algorithm, NOT time).
"""
abstract type AbstractJointPath end

"""
    evaluate_path(path::AbstractCartesianPath, t) -> Vector{Float64}

Point on the Cartesian path at parameter t ∈ [0, 1].
"""
function evaluate_path(path::AbstractCartesianPath, t::Float64)::Vector{Float64}
    error("evaluate_path not implemented for path type $(typeof(path))")
end

"""
    evaluate_path(path::AbstractJointPath, θ, derivative=0) -> Vector{Float64}

Joint configuration (derivative=0), or its `derivative`-th derivative
w.r.t. the path parameter θ, at θ ∈ [0, 1].
"""
function evaluate_path(path::AbstractJointPath, t::Float64, derivative::Int=0)::Vector{Float64}
    error("evaluate_path not implemented for path type $(typeof(path))")
end

# ---------------------------------------------------------------------
# Cartesian path: cubic Bezier curve (any task-space dimension)
# ---------------------------------------------------------------------

"""
A cubic Bezier curve through 4 control points (columns of `control_points`,
each of length = task-space dimension).
"""
struct BezierCurve <: AbstractCartesianPath
    control_points::Matrix{Float64}
end

"""
    make_bezier(p_start, p_end, t=0.7)

Build a cubic Bezier curve from `p_start` to `p_end` with two intermediate
control points placed at `(t, 0)` and `(0, t)` — this specific construction
matches the original 2D SCARA example. For other task-space dimensions or
shapes, construct `BezierCurve` directly with your own control points.
"""
function make_bezier(p_start::AbstractVector, p_end::AbstractVector, t=0.7)
    P = hcat(p_start, [t, 0.0], [0.0, t], p_end)
    return BezierCurve(P)
end

function evaluate_path(curve::BezierCurve, t::Float64)::Vector{Float64}
    P = curve.control_points
    mt = 1 - t
    return mt^3 .* P[:, 1] .+ 3 * mt^2 * t .* P[:, 2] .+ 3 * mt * t^2 .* P[:, 3] .+ t^3 .* P[:, 4]
end

# ---------------------------------------------------------------------
# Joint path: per-joint cubic splines, fit from IK samples along a
# Cartesian path. This replaces the hand-written psi1/psi2/psi3 splines
# in the original SCARA script with a generic, any-dof version.
# ---------------------------------------------------------------------

struct SplineJointPath <: AbstractJointPath
    splines::Vector{Spline1D}
end

function evaluate_path(path::SplineJointPath, t::Float64, derivative::Int=0)::Vector{Float64}
    if derivative == 0
        return [spl(t) for spl in path.splines]
    else
        return [Dierckx.derivative(spl, t, derivative) for spl in path.splines]
    end
end

"""
    build_joint_path(robot, cart_path, theta, q_init) -> (SplineJointPath, q_traj)

Sample `cart_path` at parameters `theta` (sorted, ⊂ [0,1]), solve inverse
kinematics at each sample (warm-started from the previous solution), and
fit a cubic spline per joint through the resulting joint trajectory.

`q_traj` (size length(theta) × dof) is also returned for inspection/plotting.
"""
function build_joint_path(
    robot::AbstractRobotManipulator,
    cart_path::AbstractCartesianPath,
    theta::AbstractVector{Float64},
    q_init::AbstractVector{Float64};
    ik_tolerance::Float64=1e-5,
    ik_max_iterations::Int=50,
)
    n = length(theta)
    d = length(q_init)
    q_traj = zeros(n, d)
    q = copy(q_init)
    for i in 1:n
        target = evaluate_path(cart_path, theta[i])
        q = inverse_kinematics(robot, target, q; tolerance=ik_tolerance, max_iterations=ik_max_iterations)
        q_traj[i, :] = q
    end
    theta_v = collect(theta)
    splines = [Spline1D(theta_v, q_traj[:, j], k=3, s=0.0) for j in 1:d]
    return SplineJointPath(splines), q_traj
end
