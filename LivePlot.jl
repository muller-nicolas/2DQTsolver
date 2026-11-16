# Live visualisation of a GPE simulation with GLMakie (density and phase).
#
# This file is NOT part of the GPE2D package, so the package does not depend on
# Makie. run_gpe.jl includes it when `live = true`. It needs GLMakie in your
# environment:   julia -e 'using Pkg; Pkg.add("GLMakie")'
#
# Usage:
#     lp = LivePlot(model, grid)                       # opens the window
#     lp(state)                                        # draw the current state
#     evolve!(state, stepper, model, n; callback = every(20, lp))
#
# The callable returns `:stop` when the window is closed, which ends `evolve!`.
# Each update copies ψ from the device to the host, so do not update every step.

using GLMakie
using Printf
using GPE2D

struct LivePlot{M, S}
    model::M
    screen::S
    ρ::Observable{Matrix{Float32}}        # density / n0
    θ::Observable{Matrix{Float32}}        # phase of ψ
    title::Observable{String}
    pause::Float64                        # extra seconds to wait after each frame
end

# `isopen` should exist for GLMakie screens; if it does not, assume the window is open
function _isopen(screen)
    try
        return isopen(screen)
    catch
        return true
    end
end

function LivePlot(model::GPE, grid::Grid; density_range = (0.0, 1.5), size = (1100, 520),
                  pause::Real = 0.0, phase=false)
    x, y = Float32.(xcoords(grid)), Float32.(ycoords(grid))
    ρ = Observable(zeros(Float32, grid.nx, grid.ny))
    title = Observable("t = 0")

    fig = Figure(size = size)
    Label(fig[0, 1:4], title; fontsize = 18)

    ax1 = Axis(fig[1, 1]; title = "density |ψ|²/n₀", xlabel = "x", ylabel = "y", aspect = DataAspect())
    hm1 = heatmap!(ax1, x, y, ρ; colormap = :viridis, colorrange = density_range)
    Colorbar(fig[1, 2], hm1)

    # if phase
    θ = Observable(zeros(Float32, grid.nx, grid.ny))
    ax2 = Axis(fig[1, 3]; title = "phase arg ψ", xlabel = "x", aspect = DataAspect())
    hm2 = heatmap!(ax2, x, y, θ; colormap = :twilight, colorrange = (-π, π))
    Colorbar(fig[1, 4], hm2)
    # end

    screen = display(fig)
    println("LivePlot: window opened (", typeof(screen), "). Close it to stop the run.")
    return LivePlot(model, screen, ρ, θ, title, Float64(pause))
end

function (lp::LivePlot)(state)
    if !_isopen(lp.screen)
        println("LivePlot: window closed, stopping the run.")
        return :stop
    end

    update_physical!(state, lp.model)          # refresh ψ from ψk
    ψ = Array(state.ψ)                         # device → host
    n0 = lp.model.params.n0

    # Assign NEW arrays: this always triggers a redraw (mutating in place and calling
    # `notify` is not reliable in recent Makie versions).
    lp.ρ[] = Float32.(abs2.(ψ) ./ n0)
    lp.θ[] = Float32.(angle.(ψ))

    c = state.clock
    lp.title[] = @sprintf("t = %.3f   (iteration %d)", c.t, c.iteration)

    sleep(max(lp.pause, 1e-3))                 # yield to the render loop so the frame is drawn
    return nothing
end
