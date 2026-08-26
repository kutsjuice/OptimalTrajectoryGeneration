"""
    inverse_kinematics(robot, target_pose, initial_guess; tolerance=1e-6, max_iterations=100)

Numerically solve `forward_kinematics(robot, q) == target_pose` for `q`,
starting from `initial_guess`, via damped Newton iteration using `jacobian`.
Works for any robot implementing `forward_kinematics` and `jacobian` —
no robot-specific code needed here.

For redundant or non-square Jacobians, `J \\ error` uses the least-squares
solution, which is the standard choice for this iteration.
"""
function inverse_kinematics(
    robot::AbstractRobotManipulator,
    target_pose::AbstractVector{Float64},
    initial_guess::AbstractVector{Float64};
    tolerance::Float64=1e-6,
    max_iterations::Int=100,
)::Vector{Float64}
    q = copy(initial_guess)
    for _ in 1:max_iterations
        pos = forward_kinematics(robot, q)
        err = target_pose - pos
        if norm(err) < tolerance
            return q
        end
        J = jacobian(robot, q)
        q += J \ err
    end
    return q
end
