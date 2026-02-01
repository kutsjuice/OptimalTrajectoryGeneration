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

start_point = @SVector [0.5, 0.0, 0.1]
end_point = @SVector [0.2, 0.3, 0.1]

control1 = start_point + @SVector [0.1, 0.1, 0.0]
control2 = end_point + @SVector [-0.1, -0.1, 0.0]

cartesian_path = OptimalTrajectoryGeneration.BezierCartesianPath(
    start_point,
    control1,
    control2,
    end_point
)

q_seed = [0.0, 0.0, 0.1, 0.0]  
joint_path = OptimalTrajectoryGeneration.joint_path_from_cartesian_bezier(
    SCARA,
    cartesian_path,
    q_seed
)

constraints = OptimalTrajectoryGeneration.TrajectoryConstraints(
    [2.0, 2.0, 0.5, 3.0],        # Максимальные скорости суставов [рад/с, рад/с, м/с, рад/с]
    [5.0, 5.0, 1.0, 8.0],        # Максимальные ускорения
    [50.0, 30.0, 20.0, 10.0],    # Максимальные моменты
    [100.0, 100.0, 20.0, 50.0],  # Максимальные рывки
    ([-π, -π, 0.0, -π], [π, π, d3_limit, π])  # Ограничения положений
)

path_speed = OptimalTrajectoryGeneration.compute_limit_path_speed(
    SCARA,
    joint_path,
    constraints
)

time_step = 0.01
trajectory_result = OptimalTrajectoryGeneration.generate_joint_trajectory(
    SCARA,
    joint_path,
    path_speed,
    time_step
)

default(size=(800, 600), legendfontsize=8, titlefontsize=10)
layout = @layout [a b; c d; e f]

p1 = plot(title="Декартова траектория")
plot!([cartesian_path.p0[1], cartesian_path.p1[1], cartesian_path.p2[1], cartesian_path.p3[1]],
      [cartesian_path.p0[2], cartesian_path.p1[2], cartesian_path.p2[2], cartesian_path.p3[2]],
      [cartesian_path.p0[3], cartesian_path.p1[3], cartesian_path.p2[3], cartesian_path.p3[3]],
      seriestype=:scatter, label="Контрольные точки", markersize=5)

# Вычисление точек вдоль кривой Безье для отображения
t_range = range(0, 1, length=50)
path_points = [OptimalTrajectoryGeneration.evaluate_path(cartesian_path, t) for t in t_range]
x_vals = [p[1] for p in path_points]
y_vals = [p[2] for p in path_points]
z_vals = [p[3] for p in path_points]

plot!(x_vals, y_vals, z_vals, label="Путь Безье", linewidth=2)
xlabel!("X [м]")
ylabel!("Y [м]")
zlabel!("Z [м]")

# График 2: Позиции суставов
p2 = plot(title="Позиции суставов", legend=:topright)
for i in 1:SCARA.dof
    plot!(trajectory_result.time_vector, trajectory_result.trajectory.positions[i, :], 
          label="Сустав $i", linewidth=2)
end
xlabel!("Время [с]")
ylabel!("Позиция")

# График 3: Скорости суставов
p3 = plot(title="Скорости суставов", legend=:topright)
for i in 1:SCARA.dof
    plot!(trajectory_result.time_vector, trajectory_result.trajectory.velocities[i, :], 
          label="Сустав $i", linewidth=2)
    # Добавляем ограничения скорости
    hline!([constraints.velocity_limits[i], -constraints.velocity_limits[i]], 
           linestyle=:dash, color=:red, alpha=0.3, label="")
end
xlabel!("Время [с]")
ylabel!("Скорость")

# График 4: Ускорения суставов
p4 = plot(title="Ускорения суставов", legend=:topright)
for i in 1:SCARA.dof
    plot!(trajectory_result.time_vector, trajectory_result.trajectory.accelerations[i, :], 
          label="Сустав $i", linewidth=2)
    # Добавляем ограничения ускорения
    hline!([constraints.acceleration_limits[i], -constraints.acceleration_limits[i]], 
           linestyle=:dash, color=:red, alpha=0.3, label="")
end
xlabel!("Время [с]")
ylabel!("Ускорение")

# График 5: Моменты на суставах
p5 = plot(title="Моменты на суставах", legend=:topright)
for i in 1:SCARA.dof
    plot!(trajectory_result.time_vector, trajectory_result.torques[i, :], 
          label="Сустав $i", linewidth=2)
    # Добавляем ограничения моментов
    hline!([constraints.torque_limits[i], -constraints.torque_limits[i]], 
           linestyle=:dash, color=:red, alpha=0.3, label="")
end
xlabel!("Время [с]")
ylabel!("Момент [Н·м]")

# График 6: Профиль скорости пути
p6 = plot(title="Профиль скорости пути")
plot!(range(0, 1, length=length(path_speed)), path_speed, 
      label="Скорость пути", linewidth=2, color=:purple)
xlabel!("Параметр пути θ")
ylabel!("Скорость пути [1/с]")

# Объединяем все графики
plot(p1, p2, p3, p4, p5, p6, layout=layout, size=(1200, 900))
savefig("scara_trajectory_summary.png")
println("   График сохранен как 'scara_trajectory_summary.png'")

# 7. Дополнительный анализ
println("\n7. Дополнительный анализ траектории...")

# Максимальные значения
max_positions = maximum(abs.(trajectory_result.trajectory.positions), dims=2)
max_velocities = maximum(abs.(trajectory_result.trajectory.velocities), dims=2)
max_accelerations = maximum(abs.(trajectory_result.trajectory.accelerations), dims=2)
max_torques = maximum(abs.(trajectory_result.torques), dims=2)

println("\nМаксимальные значения:")
for i in 1:SCARA.dof
    println("   Сустав $i:")
    println("     Позиция: $(@sprintf("%.4f", max_positions[i]))")
    println("     Скорость: $(@sprintf("%.4f", max_velocities[i])) (ограничение: $(constraints.velocity_limits[i]))")
    println("     Ускорение: $(@sprintf("%.4f", max_accelerations[i])) (ограничение: $(constraints.acceleration_limits[i]))")
    println("     Момент: $(@sprintf("%.4f", max_torques[i])) (ограничение: $(constraints.torque_limits[i]))")
end

# Проверка ограничений
violations = 0
for i in 1:SCARA.dof
    if max_velocities[i] > constraints.velocity_limits[i]
        println("   ВНИМАНИЕ: Сустав $i превысил ограничение скорости!")
        violations += 1
    end
    if max_accelerations[i] > constraints.acceleration_limits[i]
        println("   ВНИМАНИЕ: Сустав $i превысил ограничение ускорения!")
        violations += 1
    end
    if max_torques[i] > constraints.torque_limits[i]
        println("   ВНИМАНИЕ: Сустав $i превысил ограничение момента!")
        violations += 1
    end
end

if violations == 0
    println("   ✓ Все ограничения соблюдены!")
else
    println("   ⚠ Найдено $violations нарушений ограничений")
end