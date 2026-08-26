"""
Standalone time-parametrization utilities. These are generic numerical
helpers independent of any robot model. The full TOPP pipeline in `topp.jl`
already returns a proper time vector, so you normally won't need these
directly — they're kept for quick diagnostics or for use outside the main
pipeline (e.g. a crude time estimate from a velocity profile alone).
"""

"""
    time_parametrise(theta, theta_dot_max) -> (time, vel_profile)

Given a path-parameter grid `theta` and a pointwise speed bound
`theta_dot_max(θ)`, produce a smoothed velocity profile and the
corresponding cumulative time vector by integrating dt = dθ / θ̇.
"""
function time_parametrise(theta::AbstractVector, theta_dot_max::AbstractVector)
    T = eltype(theta)
    n = length(theta)
    idx = vcat(1:50:n, n)
    knots = theta_dot_max[idx]
    knots[1] = knots[end] = 0.0
    avg_step = sum(diff(theta[idx])) / length(diff(theta[idx]))
    if all(x -> isapprox(x, avg_step; rtol=1e-6), diff(theta[idx]))
        itp = interpolate(knots, BSpline(Cubic(Line(OnGrid()))))
        vel_prof = [itp(1 + (i - 1) * (length(idx) - 1) / (n - 1)) for i in 1:n]
    else
        itp = linear_interpolation(theta[idx], knots)
        vel_prof = itp.(theta)
    end
    h = theta[2] - theta[1]
    time = zeros(T, n)
    time[2:end] = cumsum(h ./ vel_prof[2:end])
    if n >= 5
        idx_fit = (n-3):(n-1)
        p = fit(theta[idx_fit], time[idx_fit], 2)
        time[end] = p(theta[end])
    end
    return time, vel_prof
end

"""
    time_step(ds, v0, a0, a1) -> (dt, dv)

Time and velocity increment to traverse a path-parameter step `ds`,
assuming linearly varying acceleration from `a0` to `a1` (constant jerk),
starting from speed `v0`.
"""
function time_step(ds::Float64, v0::Float64, a0::Float64, a1::Float64)
    j = (a1 - a0) / ds
    if abs(v0) < 1e-6
        dt = sqrt(2ds / a0)
    else
        dt, _ = quadgk(s -> 1 / sqrt(v0^2 + 2a0 * s + j * s^2), 0, ds)
    end
    dv = a0 * dt + (j * dt^2) / 2
    return dt, dv
end

"""
    time_step2(v0, v1)

Time increment between two speeds under exponential-style scaling
(used as a cheap alternative to `time_step` when jerk info isn't available).
"""
function time_step2(v0, v1)
    return abs((log(v1) - log(v0)) / (v1 - v0))
end
