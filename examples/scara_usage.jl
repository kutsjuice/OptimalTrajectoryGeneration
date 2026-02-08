using ForwardDiff
using LinearAlgebra

abstract type AbstractRobotManipulator end

struct TestRobot <: AbstractRobotManipulator
    l1::Float64
    l2::Float64
    dof::Int
end

function forward_kinematics(
    robot::TestRobot,
    q::AbstractMatrix
)
    N = size(q, robot.dof)
    q1 = q[1,:]
    q2 = q[2,:]
    l1, l2 = robot.l1, robot.l2
    x = l1 * cos.(q1) .+ l1 * cos.(q1 .+ q2) .+ l2 * cos.(q1 .+ 0.5 * q2)
    y = l1 * sin.(q1) .+ l1 * sin.(q1 .+ q2) .+ l2 * sin.(q1 .+ 0.5 * q2)
    return [x'; y']
end

function body_jacobian(robot::TestRobot, q::AbstractMatrix)
    _, N = size(q)
    J = zeros(robot.dof, robot.dof, N)

    for i in 1:N
        qi = q[:, i]
        J[:, :, i] = ForwardDiff.jacobian(qi_vec -> forward_kinematics(robot, reshape(qi_vec, :, 1))[:, 1], qi)
    end

    return J
end

# Example usage
robot = TestRobot(1.0, 0.5, 2)
q = [0.3 0.2;
    0.1 0.8]

end_effector_pos  = forward_kinematics(robot, q)
J = body_jacobian(robot, q)
println("Body jacobian:", J)
