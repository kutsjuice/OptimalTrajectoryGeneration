using Flexia, StaticArrays, ForwardDiff

struct SerialChain
    sys::MBSystem2D
    ground::Body2D
    links::Vector{Body2D}
    joints::Vector{AbstractJoint2D}              
    motors::Vector{AbstractPositionActuator2D}   
    joint_types::Vector{Symbol}                 
    base::SVector{2,Float64}
end

njoints(chain::SerialChain) = length(chain.links)

function build_serial_chain(lengths::Vector{Float64}, masses::Vector{Float64},
                            inertias::Vector{Float64};
                            joint_types::Vector{Symbol} = fill(:revolute, length(lengths)),
                            base = SA[0.0, 0.0],
                            gravity::Real = 0.0,
                            with_motors::Bool = true,
                            auto_assemble::Bool = true)::SerialChain
    n = length(lengths)
    @assert length(masses) == n && length(inertias) == n && length(joint_types) == n
    base_sv = SVector{2,Float64}(base)

    sys = MBSystem2D()

    ground = Body2D(1.0, 1.0; length = 0.0)
    add!(sys, ground)

    links = Body2D[]
    for i in 1:n
        link = Body2D(masses[i], inertias[i]; length = lengths[i])
        if gravity != 0
            m = masses[i]
            link.forces[2] = (s, t) -> -m * gravity
        end
        add!(sys, link)
        push!(links, link)
    end

    anchor = FixedJoint(ground)
    setposition!(anchor, base_sv)
    setrotation!(anchor, 0.0)
    add!(sys, anchor)

    joints = AbstractJoint2D[]
    motors = AbstractPositionActuator2D[]
    for i in 1:n
        prev     = i == 1 ? ground : links[i-1]
        prev_end = i == 1 ? SA[0.0, 0.0] : SA[lengths[i-1] / 2, 0.0]
        near_end = SA[-lengths[i] / 2, 0.0]

        if joint_types[i] === :revolute
            joint = HingeJoint(prev, links[i])
            set_position_on_first_body!(joint, prev_end)
            set_position_on_second_body!(joint, near_end)
            motor = with_motors ? PositionMotor2D(joint, 0.0) : nothing
        elseif joint_types[i] === :prismatic
            joint = SliderJoint(prev, links[i])
            set_position_on_first_body!(joint, prev_end)
            set_position_on_second_body!(joint, near_end)
            set_direction_on_first_body!(joint, SA[1.0, 0.0])
            set_direction_on_second_body!(joint, SA[1.0, 0.0])
            motor = with_motors ? PositionLinearActuator2D(joint, 0.0) : nothing
        else
            error("неизвестный тип сустава $(joint_types[i]); допустимы :revolute и :prismatic")
        end
        add!(sys, joint)
        push!(joints, joint)
        if motor !== nothing
            add!(sys, motor)   # motor right after its joint
            push!(motors, motor)
        end
    end

    auto_assemble && assemble!(sys)
    return SerialChain(sys, ground, links, joints, motors, Vector{Symbol}(joint_types), base_sv)
end