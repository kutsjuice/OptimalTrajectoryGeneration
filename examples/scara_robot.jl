using OptimalTrajectoryGeneration
using LinearAlgebra

"""
2-DOF SCARA-like manipulator. This is the ONLY file you need to write to
plug a new robot into the package — everything else (IK, path building,
TOPP) is generic.
"""
struct SCARARobot <: OptimalTrajectoryGeneration.AbstractRobotManipulator
    l1::Float64
    l2::Float64
    dof::Int
end
SCARARobot(l1, l2) = SCARARobot(l1, l2, 2)

function OptimalTrajectoryGeneration.forward_kinematics(robot::SCARARobot, q::AbstractVector{Float64})::Vector{Float64}
    q1, q2 = q
    l1, l2 = robot.l1, robot.l2
    x = l1 * cos(q1) + l1 * cos(q1 + q2) + l2 * cos(q1 + 0.5 * q2)
    y = l1 * sin(q1) + l1 * sin(q1 + q2) + l2 * sin(q1 + 0.5 * q2)
    return [x, y]
end

function OptimalTrajectoryGeneration.jacobian(robot::SCARARobot, q::AbstractVector{Float64})::Matrix{Float64}
    q1, q2 = q
    l1, l2 = robot.l1, robot.l2
    s1, s2, s12 = sin(q1), sin(q1 + q2), sin(q1 + 0.5 * q2)
    c1, c2, c12 = cos(q1), cos(q1 + q2), cos(q1 + 0.5 * q2)
    return [
        -l1*s1-l1*s2-l2*s12 -l1*s2-0.5*l2*s12
        l1*c1+l1*c2+l2*c12 l1*c2+0.5*l2*c12
    ]
end

"""
Placeholder dynamics: identity mass matrix, no bias/Coriolis/gravity terms.
This matches what the original script used (a kinematics-only demo) and
keeps torque values directly comparable in magnitude to joint accelerations.
Replace with your robot's real `M(q)`/`h(q,q̇)` (e.g. from a Lagrangian
or URDF-based dynamics library) for physically meaningful torque limits.
"""
function OptimalTrajectoryGeneration.compute_mass_and_force_terms(
    robot::SCARARobot, q::AbstractVector{Float64}, dq::AbstractVector{Float64}
)::Tuple{Matrix{Float64},Vector{Float64}}
    M = Matrix{Float64}(I, robot.dof, robot.dof)
    h = zeros(robot.dof)
    return M, h
end
