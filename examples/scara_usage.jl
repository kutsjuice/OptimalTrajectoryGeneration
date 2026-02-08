using ForwardDiff
using LinearAlgebra
using Plots
using Interpolations

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

function cartesian_traj(curve::BezierCurve, theta::AbstractVector)
    # Evaluate Bezier curve at parameter values theta
    P = curve.control_points
    num_points = length(theta)
    trajectory = zeros(num_points, 2)
    
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




# Example usage
robot = TestRobot(1.0, 0.5, 2)
q = [0.3, 0.2]

end_effector_pos  = forward_kinematics(robot, q)
J = body_jacobian(robot, q)
println("Body jacobian:", J)
ik_solution = ik(robot, [1.0, 0.5], [0.0, 0.0], verbose=true)
println("IK solution:", ik_solution)
println("End-effector position from IK solution:", forward_kinematics(robot, ik_solution))

# Visualization
theta = collect(0:0.01:1)
bezier_curve = make_bezier([0.0, 0.0], [2.0, 1.5], 0.8)
trajectory = cartesian_traj(bezier_curve, theta)

P = bezier_curve.control_points

plot(trajectory[:, 1], trajectory[:, 2], 
     label="Bezier Curve", linewidth=2, color=:blue,
     xlabel="X", ylabel="Y", 
     title="SCARA Robot Trajectory and Bezier Curve",
     grid=true, legend=:topright)
scatter!(P[1, :], P[2, :], label="Control Points", color=:red, markersize=6)
scatter!([end_effector_pos[1]], [end_effector_pos[2]], label="Current EE", color=:green, markersize=8)
scatter!([1.0], [0.5], label="IK Target", color=:orange, markersize=8)

savefig("trajectory_visualization.png")
println("Plot saved as trajectory_visualization.png")

# Compute joint trajectory from cartesian trajectory
println("\nComputing joint trajectory...")
jnt_traj = joint_traj(trajectory, robot, ik_solution)

# Analyze maximum velocities and accelerations
println("\nAnalyzing trajectory constraints...")
result = max_theta_dot(jnt_traj, theta, 1.5, 2.0)
println("\nTrajectory Analysis Results:")
println("Max velocity q1: $(result.max_vel_q1) rad/s")
println("Max velocity q2: $(result.max_vel_q2) rad/s")
println("Max acceleration q1: $(result.max_acc_q1) rad/s²")
println("Max acceleration q2: $(result.max_acc_q2) rad/s²")
println("Constraints satisfied (w1_max=1.5, w2_max=2.0): $(result.satisfies_constraints)")