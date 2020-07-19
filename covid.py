#!/usr/bin/env python
# coding: utf-8

# In this notebook I will show the SIR model, using the methodology used by [Andrew Atkeson](https://sites.google.com/site/andyatkeson/home?authuser=0) in the [NBER Working Paper number 26867](nber.org/papers/w26867.pdf) and the codes are heavily based on the notbook availiable on [QuantEcon](https://python.quantecon.org/sir_model.html#The-SIR-Model)
# 
# Following are the required imports for performing this analysis.

# In[2]:


import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# In this analysis, we have assumed that there are four states and all the indviduals are assumed to be in one of these four states. These are: susceptible (S), exposed (E), infected (I) and removed (R).
# 
# People in the **R** state have been infected and either recovered or died. The **E** state is not yet infectious.
# 
# According to the framework, the flow across the states follows the path of $ S \rightarrow E \rightarrow I \rightarrow R $ and the transmission rate is positive, which means $ i(0) \gt 0 $
# 
# The following are the dynamics of the model:
# 
# $ \dot{s}(t) = \ - \beta(t) \ s(t) \ i(t) $ 
# 
# 
# $ \dot{e}(t) = \ - \beta(t) \ s(t) \ i(t) \ - \sigma e(t) $ 
# 
# 
# $ \dot{i}(t) = \ \sigma e(t) \ - \gamma i(t) $
# 
# Where, $ \beta $ is transmission rate, $ \sigma $ is infection rate and $ \gamma $ represents the recovery rate. The dot symobol on s, e and i means the $ dy / dt $.
# 
# Now, I will show the codes for the model.

# ## Model

# In[3]:


pop_size = 13.5e2 # Population of India.
σ = 1 / 5.2 # average incubation period of 5.2 days.
γ = 1 / 18 # average illness duration of 18 days.


# The system of equation above can be represented in the form of vector as
#     $$ \dot{x} = F(x,t), \ x:= (s,e,i) $$
#     
# Now, the model will be constructed as:

# In[19]:


def F(x, t, R0=1.6):

    s, e, i = x
    
    β = R0(t) * γ if callable(R0) else R0 * γ
    ne = β * s * i

    
    ds = - ne
    de = ne - σ * e
    di = σ * e - γ * i

    return ds, de, di

i_0 = 1e-7
e_0 = 4 * i_0
s_0 = 1 - i_0 - e_0

x_0 = s_0, e_0, i_0



def solve_path(R0, t_vec, x_init=x_0):
    G = lambda x, t: F(x, t, R0)
    s_path, e_path, i_path = odeint(G, x_init, t_vec).transpose()

    c_path = 1 - s_path - e_path       # cumulative cases
    return i_path, c_path


# ## Expreiments

# As done in the QuantEcon's notebook, we will now run some random experiments to test our code.

# ### Constant R0 Cases
# 
# We will calculate the time path of infected people under different assumptions for R0:

# In[25]:


t_length = 550
grid_size = 1000
t_vec = np.linspace(0, t_length, grid_size)

R0_vals = np.linspace(1.6, 3.0, 6)
labels = [f'$R0 = {r:.2f}$' for r in R0_vals]
i_paths, c_paths = [], []

for r in R0_vals:
    i_path, c_path = solve_path(r, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)
    
def plot_paths(paths, labels, times=t_vec):

    fig, ax = plt.subplots()

    for path, label in zip(paths, labels):
        ax.plot(times, path, label=label)

    ax.legend(loc='upper left')

    plt.show()
    
plot_paths(i_paths, labels)


# In[24]:


plot_paths(c_paths, labels)


# ### Changing Mitigation

# In[26]:


def R0_mitigating(t, r0=3, η=1, r_bar=1.6):
    R0 = r0 * exp(- η * t) + (1 - exp(- η * t)) * r_bar
    return R0
η_vals = 1/5, 1/10, 1/20, 1/50, 1/100
labels = [fr'$\eta = {η:.2f}$' for η in η_vals]

fig, ax = plt.subplots()

for η, label in zip(η_vals, labels):
    ax.plot(t_vec, R0_mitigating(t_vec, η=η), label=label)

ax.legend()
plt.show()


# Time path for infected people:

# In[28]:


i_paths, c_paths = [], []

for η in η_vals:
    R0 = lambda t: R0_mitigating(t, η=η)
    i_path, c_path = solve_path(R0, t_vec)
    i_paths.append(i_path)
    c_paths.append(c_path)
    
plot_paths(i_paths, labels)


# In[29]:


plot_paths(c_paths, labels)


# ## Ending of Lockdown
# 
# Scenerio 1: $R_t=0.5$ for 30 days and then $R_t=2$ for the remaining 17 months. This corresponds to lifting lockdown in 30 days.
# 
# 
# Scenerio 2: $R_t=0.5$ for 120 days and then $R_t=2$ for the remaining 14 months. This corresponds to lifting lockdown in 4 months.

# In[33]:


i_0 = 4_00000 / pop_size
e_0 = 10_80000 / pop_size
s_0 = 1 - i_0 - e_0
x_0 = s_0, e_0, i_0

R0_paths = (lambda t: 0.5 if t < 30 else 2,
            lambda t: 0.5 if t < 120 else 2)

labels = [f'scenario {i}' for i in (1, 2)]

i_paths, c_paths = [], []

for R0 in R0_paths:
    i_path, c_path = solve_path(R0, t_vec, x_init=x_0)
    i_paths.append(i_path)
    c_paths.append(c_path)
    
plot_paths(i_paths, labels)


# In[34]:


ν = 0.01

paths = [path * ν * pop_size for path in c_paths]
plot_paths(paths, labels)


# In[35]:


paths = [path * ν * γ * pop_size for path in i_paths]
plot_paths(paths, labels)

