# The example of basic usage of OptimalTrajectoryGeneration.jl package for planning trajectory for SCARA robot
include("../src/OptimalTrajectoryGeneration.jl")
using .OptimalTrajectoryGeneration
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Statistics
using DocStringExtensions

l1 = 0.4
l2 = 0.4
d3_limit = 0.2

screw1 = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
screw2 = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
screw3 = @SVector [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
screw4 = @SVector [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

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
    OptimalTrajectoryGeneration.spatial_inertia(mass[1], com[1], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01]),
    OptimalTrajectoryGeneration.spatial_inertia(mass[2], com[2], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01]),
    OptimalTrajectoryGeneration.spatial_inertia(mass[3], com[3], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01]),
    OptimalTrajectoryGeneration.spatial_inertia(mass[4], com[4], 
        @SMatrix [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01])
]

links = OptimalTrajectoryGeneration.RigidBody[]
for i = 1:4
    push!(links, OptimalTrajectoryGeneration.RigidBody(
        [screw1, screw2, screw3, screw4][i],
        [X1, X2, X3, X4][i],
        inertias[i]
    ))
end

gravity = @SVector [0.0, 0.0, -9.81]

SCARA = OptimalTrajectoryGeneration.SerialManipulator(links, gravity, 4)

println(SCARA)