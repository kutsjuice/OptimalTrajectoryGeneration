using LinearAlgebra
using StaticArrays
using ForwardDiff

abstract type AbstractRobotManipulator end
abstract type AbstractLink end

function skew(v::AbstractVector)
    skew_of_v = @SMatrix [
        0 -v[3] v[2];
        v[3] 0 -v[1];
        -v[2] v[1] 0
    ]
    return skew_of_v
end

println(skew([1,2,3]))
