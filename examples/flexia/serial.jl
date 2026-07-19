using Flexia, StaticArrays

"""
Последовательная цепь, построенная фабрикой `build_serial_chain`.
Хранит всё, что понадобится адаптеру OTG: систему, тела, суставы и моторы.
"""
struct SerialChain
    sys::MBSystem2D
    ground::Body2D
    links::Vector{Body2D}
    joints::Vector{AbstractJoint2D}              # HingeJoint или SliderJoint
    motors::Vector{AbstractPositionActuator2D}   # PositionMotor2D или PositionLinearActuator2D
    joint_types::Vector{Symbol}                  # :revolute | :prismatic
    base::SVector{2,Float64}
end

njoints(chain::SerialChain) = length(chain.links)

"""
    build_serial_chain(lengths, masses, inertias;
                       joint_types = fill(:revolute, n),
                       base = SA[0.0, 0.0], gravity = 0.0) -> SerialChain

Строит последовательный манипулятор из n звеньев во Flexia.
`joint_types[i]` — тип i-го сустава: `:revolute` (шарнир + PositionMotor2D,
координата qᵢ — относительный угол θᵢ−θᵢ₋₁) или `:prismatic` (ползун +
PositionLinearActuator2D, координата qᵢ — смещение вдоль оси предыдущего
звена; при qᵢ=0 ближний конец звена совпадает с дальним концом предыдущего).
"""
function build_serial_chain(lengths::Vector{Float64}, masses::Vector{Float64},
                            inertias::Vector{Float64};
                            joint_types::Vector{Symbol} = fill(:revolute, length(lengths)),
                            base = SA[0.0, 0.0],
                            gravity::Real = 0.0)::SerialChain
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
            motor = PositionMotor2D(joint, 0.0)
        elseif joint_types[i] === :prismatic
            joint = SliderJoint(prev, links[i])
            set_position_on_first_body!(joint, prev_end)
            set_position_on_second_body!(joint, near_end)
            set_direction_on_first_body!(joint, SA[1.0, 0.0])
            set_direction_on_second_body!(joint, SA[1.0, 0.0])
            motor = PositionLinearActuator2D(joint, 0.0)
        else
            error("неизвестный тип сустава $(joint_types[i]); допустимы :revolute и :prismatic")
        end
        add!(sys, joint)
        add!(sys, motor)   # мотор сразу за своим суставом
        push!(joints, joint)
        push!(motors, motor)
    end

    assemble!(sys)
    return SerialChain(sys, ground, links, joints, motors, Vector{Symbol}(joint_types), base_sv)
end

"""
    straight_state(chain) -> Vector{Float64}

Состояние системы для конфигурации q = 0: все звенья вытянуты вдоль x от base.
Верно и для призматических суставов (при q=0 точки крепления ползуна совпадают).
Скорости, множители Лагранжа и время — нули.
"""
function straight_state(chain::SerialChain)::Vector{Float64}
    sys = chain.sys
    s = zeros(number_of_dofs(sys))
    set_initial_position!(s, sys, chain.ground, SA[chain.base[1], chain.base[2], 0.0])
    x = chain.base[1]
    for link in chain.links
        set_initial_position!(s, sys, link, SA[x + link.length / 2, chain.base[2], 0.0])
        x += link.length
    end
    return s
end

"Индексы позиционных координат (x, y, θ) всех тел в векторе состояния."
function position_rows(chain::SerialChain)::Vector{Int}
    rows = Int[]
    for b in (chain.ground, chain.links...)
        append!(rows, get_body_position_dofs(chain.sys, b))
    end
    return rows
end

"""
Проверки собранной цепи:
1. длина вектора состояния = 9n+10: тела 6(n+1), λ-блок 3+3n, время 1;
2. λ-строки rhs на straight_state — нули (геометрия согласована со связями).
"""
function selfcheck(chain::SerialChain)::SerialChain
    sys = chain.sys
    n = njoints(chain)
    nstate = number_of_dofs(sys)
    @assert nstate == 9n + 10 "state length $nstate != 9n+10 = $(9n + 10)"

    s0 = straight_state(chain)
    lam_rows = (last_body_dof(sys) + 1):(nstate - 1)
    err = maximum(abs, sys.rhs(s0)[lam_rows])
    @assert err < 1e-12 "constraint residual at straight_state: $err"

    println("selfcheck ok: n=$n [$(join(chain.joint_types, ", "))], ",
            "state=$nstate, residual=$err")
    return chain
end

# ---- демонстрация и проверки (выполняются только при запуске как скрипт) ----
if abspath(PROGRAM_FILE) == @__FILE__
    # 1. Двухзвенник на шарнирах
    chain = build_serial_chain([1.0, 0.7], [1.0, 0.5], [0.1, 0.05]; gravity = 9.81)
    selfcheck(chain)

    # 2. Моторы держат q=0 под гравитацией. Проверяем только ПОЗИЦИИ тел:
    #    скоростные компоненты состояния при GGL-моторах под нагрузкой нефизичны —
    #    λ мотора входит в позиционные строки (θ̇ = ω ± λ), поэтому ω тел растут,
    #    компенсируя нескомпенсированный в скоростных строках момент гравитации,
    #    а позы при этом держатся на уровне машинной точности.
    ts = 0:0.001:1.0
    s0 = straight_state(chain)
    sol = simulate(chain.sys, s0, ts)
    prows = position_rows(chain)
    println("position drift after 1 s under gravity: ",
            maximum(abs, sol[prows, end] - s0[prows]))

    # 3. Смешанная цепь: шарнир + ползун
    mixed = build_serial_chain([1.0, 0.6], [1.0, 0.5], [0.1, 0.05];
                               joint_types = [:revolute, :prismatic], gravity = 9.81)
    selfcheck(mixed)
    sm0 = straight_state(mixed)
    solm = simulate(mixed.sys, sm0, ts)
    pm = position_rows(mixed)
    println("position drift (mixed chain):           ",
            maximum(abs, solm[pm, end] - sm0[pm]))
end
