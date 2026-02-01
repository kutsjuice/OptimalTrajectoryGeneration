# The example of basic usage of OptimalTrajectoryGeneration.jl package for planning trajectory for SCARA robot
include("../src/OptimalTrajectoryGeneration.jl")
using .OptimalTrajectoryGeneration
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Statistics
using DocStringExtensions
using Plots
using Printf

l1 = 0.4
l2 = 0.4
d3_limit = 0.2

screw1 = @SVector [0., 0., 1.,  0., 0., 0.]
screw2 = @SVector [0., 0., 1., -l1, 0., 0.]
screw3 = @SVector [0., 0., 0.,  0., 0., 1.]
screw4 = @SVector [0., 0., 1.,  0., 0., 0.]

X1 = one(SMatrix{4,4,Float64})
X2 = @SMatrix [
1.0 0.0 0.0 l1;
0.0 1.0 0.0 0.0;
0.0 0.0 1.0 0.0;
0.0 0.0 0.0 1.0
]
X3 = @SMatrix [
1.0 0.0 0.0 l2;
0.0 1.0 0.0 0.0;
0.0 0.0 1.0 0.0;
0.0 0.0 0.0 1.0
]
X4 = one(SMatrix{4,4,Float64})

mass = [2.0, 1.5, 1.0, 0.8]
com = [@SVector[0.2, 0.0, 0.0], @SVector[0.15, 0.0, 0.0], @SVector[0.0, 0.0, 0.1], @SVector[0.0, 0.0, 0.05]]
inertias = [
    Base.invokelatest(OptimalTrajectoryGeneration.spatial_inertia, mass[1], com[1], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01]),
    Base.invokelatest(OptimalTrajectoryGeneration.spatial_inertia, mass[2], com[2], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01]),
    Base.invokelatest(OptimalTrajectoryGeneration.spatial_inertia, mass[3], com[3], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01]),
    Base.invokelatest(OptimalTrajectoryGeneration.spatial_inertia, mass[4], com[4], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01])
]

links = OptimalTrajectoryGeneration.RigidBody[]
for i = 1:4
    push!(links, Base.invokelatest(OptimalTrajectoryGeneration.RigidBody,
        [screw1, screw2, screw3, screw4][i],
        [X1, X2, X3, X4][i],
        inertias[i]
    ))
end

gravity = @SVector [0.0, 0.0, -9.81]

SCARA = Base.invokelatest(OptimalTrajectoryGeneration.SerialManipulator, links, gravity, 4)

start_point = @SVector [0.5, 0.0, 0.1]
end_point = @SVector [0.2, 0.3, 0.1]

control1 = start_point + @SVector [0.1, 0.1, 0.0]
control2 = end_point + @SVector [-0.1, -0.1, 0.0]

cartesian_path = Base.invokelatest(OptimalTrajectoryGeneration.BezierCartesianPath,
    start_point,
    control1,
    control2,
    end_point
)

q_seed = [-0.7, 1.4, 0.1, -0.7]

joint_path = Base.invokelatest(OptimalTrajectoryGeneration.joint_path_from_cartesian_bezier,
    SCARA, cartesian_path, q_seed)

constraints = Base.invokelatest(OptimalTrajectoryGeneration.TrajectoryConstraints,
    [2.0, 2.0, 0.5, 3.0],
    [5.0, 5.0, 1.0, 8.0],
    [50.0, 30.0, 20.0, 10.0],
    [100.0, 100.0, 20.0, 50.0],
    ([-π, -π, 0.0, -π], [π, π, d3_limit, π]))

path_speed = Base.invokelatest(compute_limit_path_speed,
    SCARA, joint_path, constraints)

time_step = 0.01

trajectory_result = Base.invokelatest(generate_joint_trajectory,
    SCARA, joint_path, path_speed, time_step)