using Pkg
Pkg.add("InstantiateFromURL")
using InstantiateFromURL

github_project("QuantEcon/quantecon-notebooks-julia", version = "0.8.0")

using LinearAlgebra, Statistics

randn()

randn()

using Plots
gr(fmt=:png);

n = 100 
ϵ = randn(n)
plot(1:n, ϵ)

typeof(ϵ)

ϵ[1:5]

?typeof #For HELP

n = 100
ϵ = zeros(n)

for i in eachindex(ϵ)
    ϵ[i]=randn()
end

ϵ_sum = 0.0 
m = 5
for ϵ_val in ϵ[1:m]
    ϵ_sum = ϵ_sum + ϵ_val
end
ϵ_mean = ϵ_sum / m

ϵ_mean ≈ mean(ϵ[1:m])
ϵ_mean ≈ sum(ϵ[1:m]) / m

function generatedata(n)
    ϵ = zeros(n)
    for i in eachindex(ϵ)
        ϵ[i] = (randn())^2 # squaring the result
    end
    return ϵ
end

data = generatedata(10)
plot(data)

n = 100
f(x) = x^2

x = randn(n)
plot(f.(x), label="x^2")
plot!(x, label="x")       

using Distributions

function plothistogram(distribution, n)
    ϵ = rand(distribution, n)  # n draws from distribution
    histogram(ϵ)
end

lp = Laplace()
plothistogram(lp, 500)

plothistogram(lp, 1000)

plothistogram(lp, 5000)


