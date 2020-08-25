using LinearAlgebra, Statistics, Plots
gr(fmt=:png);

using Pkg
Pkg.add("MAT")
Pkg.add("SparseArrays")

using MAT, SparseArrays


# Create a random 10 X 10 matrix
A = rand(10,10) 

# Transpose
Atranspose = A'

# Multipication
A = A*Atranspose

isposdef(A)

# Create random vector
b = rand(10);

# AX = b
X = A\b;

@show norm(A*X-b)
;

x_vals = [0 0 0 ; 2 -3 -4]
y_vals = [0 0 0 ; 4 3 -3.5]

plot(x_vals, y_vals, arrow = true, color = :blue,
     legend = :none, xlims = (-5, 5), ylims = (-5, 5),
     annotations = [(2.2, 4.4, "[2, 4]"),
                    (-3.3, 3.3, "[-3, 3]"),
                    (-4.4, -3.85, "[-4, -3.5]")],
     xticks = -5:1:5, yticks = -5:1:5,
     framestyle = :origin)

# scalar multiplication

x = [2]
scalars = [-2 1 2]
vals = [0 0 0; x * scalars]
labels = [(-3.6, -4.2, "-2x"), (2.4, 1.8, "x"), (4.4, 3.8, "2x")]

plot(vals, vals, arrow = true, color = [:red :red :blue],
     legend = :none, xlims = (-5, 5), ylims = (-5, 5),
     annotations = labels, xticks = -5:1:5, yticks = -5:1:5,
     framestyle = :origin)

f(x) = 0.6cos(4x) + 1.3
grid = range(-2, 2, length = 100)
y_min, y_max = extrema( f(x) for x in grid )
plt1 = plot(f, xlim = (-2, 2), label = "f")
hline!(plt1, [f(0.5)], linestyle = :dot, linewidth = 2, label = "")
vline!(plt1, [-1.07, -0.5, 0.5, 1.07], linestyle = :dot, linewidth = 2, label = "")
plot!(plt1, fill(0, 2), [y_min y_min; y_max y_max], lw = 3, color = :blue,
      label = ["range of f" ""])
plt2 = plot(f, xlim = (-2, 2), label = "f")
hline!(plt2, [2], linestyle = :dot, linewidth = 2, label = "")
plot!(plt2, fill(0, 2), [y_min y_min; y_max y_max], lw = 3, color = :blue,
      label = ["range of f" ""])
plot(plt1, plt2, layout = (2, 1), ylim = (0, 3.5))
