using GLMakie

# Test data
x = range(0, 10, length = 100)
y = sin.(x)

# Create figure and axis
fig = Figure()
ax = Axis(fig[1, 1], title = "Plot", xlabel = "X", ylabel = "Y")

# Draw line
lines!(ax, x, y, color = :blue, linewidth = 2)

# Display the figure in an interactive window
display(fig)

# Keep the script open if run from the terminal
isinteractive() || wait(fig.scene)
