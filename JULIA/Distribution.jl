using Statistics
using StatsBase
using RDatasets
using Plots
using StatsPlots
using KernelDensity
using Distributions
using LinearAlgebra
using HypothesisTests
using PyCall
using MLBase
using CSV

P = CSV.read("E:/BQdata.csv")

using Plots
histogram(randn(200), bins = :sqrt,  label = "Random values") 

plot(randn(200), seriestype = :scatter, title = "Random values") 

gr()
x=randn(100)
y=randn(300)

plot(x, seriestype = :scatter, title = "Random values") 

d=Normal()
randomvector=randn(d,10000)
histogram(randomvector)
p=kde(randomvector)

plot!(p.x,p.density .* length(myrandomvector) .*0.1, linewidth=3,color=2,label="kde fit")

myrandomvector = randn(100_000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.1, linewidth=3,color=2,label="kde fit")

d = Normal()
myrandomvector = rand(d,100000)
histogram(myrandomvector)
p=kde(myrandomvector)
plot!(p.x,p.density .* length(myrandomvector) .*0.1, linewidth=3,color=2,label="kde fit")
