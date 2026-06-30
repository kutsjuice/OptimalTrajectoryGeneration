module OptimalTrajectoryGeneration

using LinearAlgebra
using Interpolations
using Polynomials
using QuadGK
using Dierckx

export
    # interfaces.jl — robot model
    AbstractRobotManipulator,
    TrajectoryConstraints,
    JointTrajectory,
    TrajectoryResult,
    forward_kinematics,
    jacobian,
    compute_mass_and_force_terms,
    dof,
    # kinematics.jl
    inverse_kinematics,
    # paths.jl
    AbstractCartesianPath,
    AbstractJointPath,
    BezierCurve,
    make_bezier,
    SplineJointPath,
    build_joint_path,
    evaluate_path,
    # topp.jl — the core algorithm
    compute_limit_path_speed,
    generate_joint_trajectory,
    # time_parametrization.jl
    time_parametrise,
    time_step,
    time_step2

include("interfaces.jl")
include("kinematics.jl")
include("paths.jl")
include("topp.jl")
include("time_parametrization.jl")

end # module
