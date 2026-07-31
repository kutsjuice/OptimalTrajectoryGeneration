using OptimalTrajectoryGeneration, Flexia, ForwardDiff

"""
Wraps a reduced Flexia chain (2 independent generalized coordinates q) as an
OptimalTrajectoryGeneration robot. The task-space point is the far end of
`link3`. `p_cache` warm-starts the Newton position solve between calls;
relies on `ReducedModel`/`assemble_position`/`reduce_at`/`reduced_dynamics`
from coordinate_reduction.jl already being in scope.
"""
mutable struct FlexiaSCARA <: OptimalTrajectoryGeneration.AbstractRobotManipulator
    rm::ReducedModel
    link3::Body2D
    l3::Float64
    dof::Int
    p_cache::Vector{Float64}
end
FlexiaSCARA(rm, link3, l3, p_init) = FlexiaSCARA(rm, link3, l3, 2, copy(p_init))

function tip_position(robot::FlexiaSCARA, p::AbstractVector{T}) where {T}
    x3, y3, θ3 = p[collect(body_dofs(robot.link3))]
    return [x3 + robot.l3 / 2 * cos(θ3), y3 + robot.l3 / 2 * sin(θ3)]
end

function OptimalTrajectoryGeneration.forward_kinematics(robot::FlexiaSCARA, q::AbstractVector{Float64})::Vector{Float64}
    robot.p_cache = assemble_position(robot.rm, q; p0=robot.p_cache)
    return tip_position(robot, robot.p_cache)
end

# Analytic q-Jacobian via the implicit function theorem instead of
# differentiating through the Newton solve: ∂p/∂q = R (reduce_at's null-space
# basis), so ∂tip/∂q = ∂tip/∂p · R.
function OptimalTrajectoryGeneration.jacobian(robot::FlexiaSCARA, q::AbstractVector{Float64})::Matrix{Float64}
    robot.p_cache = assemble_position(robot.rm, q; p0=robot.p_cache)
    red = reduce_at(robot.rm, robot.p_cache)
    Jp = ForwardDiff.jacobian(pp -> tip_position(robot, pp), robot.p_cache)
    return Jp * red.R
end

function OptimalTrajectoryGeneration.compute_mass_and_force_terms(
    robot::FlexiaSCARA, q::AbstractVector{Float64}, dq::AbstractVector{Float64}
)::Tuple{Matrix{Float64},Vector{Float64}}
    robot.p_cache = assemble_position(robot.rm, q; p0=robot.p_cache)
    dyn = reduced_dynamics(robot.rm, robot.p_cache, dq)
    return Matrix(dyn.M), dyn.h
end
