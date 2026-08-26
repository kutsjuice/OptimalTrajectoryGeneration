using Flexia, StaticArrays

"""
Linear angular constraint Σ coeffs[i]·θ(bodies[i]) = 0 (e.g. a belt/gear
coupling between joint angles). One λ row, standard Flexia joint shape.
"""
mutable struct GearConstraint <: AbstractJoint2D
    bodies::Vector{Body2D}
    coeffs::Vector{Float64}
    index::Int64
end
GearConstraint(bodies, coeffs) = GearConstraint(collect(bodies), collect(Float64, coeffs), -1)

Flexia.number_of_dofs(::GearConstraint) = 1
Flexia.get_lms(sys::MBSystem2D, c::GearConstraint) = SA[sys.lmdofs[c.index]]

function Flexia.add_to_rhs!(rhs, state, sys::MBSystem2D, c::GearConstraint)
    lm = Flexia.get_lms(sys, c)[1]
    λ = state[lm]
    Φ = 0.0
    for (body, coeff) in zip(c.bodies, c.coeffs)
        θ_dof = Flexia.get_body_position_dofs(sys, body)[3]
        ω_dof = Flexia.get_body_velocity_dofs(sys, body)[3]
        rhs[ω_dof] += coeff * λ
        Φ += coeff * state[θ_dof]
    end
    rhs[lm] = Φ
end

# Only needed for sys.kinematic_constrains!, which coordinate_reduction.jl
# doesn't use (it reads Φ from sys.rhs instead) — kept for API completeness.
function Flexia.compute_kinematic_residual!(residual::Vector{Float64}, coordinates::Vector{Float64},
                                             sys::MBSystem2D, c::GearConstraint)
    lm_local = Flexia.get_lms(sys, c)[1] - last_body_dof(sys)
    Φ = 0.0
    for (body, coeff) in zip(c.bodies, c.coeffs)
        θ_idx = Flexia.get_body_generalized_dofs(sys, body)[3]
        Φ += coeff * coordinates[θ_idx]
    end
    residual[lm_local] = Φ
    return nothing
end
