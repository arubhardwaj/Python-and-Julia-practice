using Plots
n = 20
x = 1:n

X = [ones(n,1) x]
beta = [10, -1]
e = 6.0*randn(n)
PopRegLine = X*beta
y = PopRegLine + e

b = inv(X'*X)*X'*y
fit = X*b

plot(x,PopRegLine, label = "Population Regression Line")
plot!(x, fit, label = "Line of Best Fit")
xlabel= "X"
scatter!(x, y, label = "values")
