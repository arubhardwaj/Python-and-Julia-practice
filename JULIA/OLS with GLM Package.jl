using DataFrames, GLM

x = randn(10000)
y = randn(10000)
z = randn(10000)

df = DataFrame(x=x,y=y,z=z)

OLS = GLM.lm(@formula(y~x+z), df)
OLS

using CSV # Package for importing CSV file

data = CSV.read("E:/BQdata.csv") # Importing clean data

reg = GLM.lm(@formula(GNP~Unrate), data)
reg

summary(reg)

describe(data)
