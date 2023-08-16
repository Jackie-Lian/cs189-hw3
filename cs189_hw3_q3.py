#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from   scipy.stats import multivariate_normal


# In[2]:


#part 1
fig = plt.figure(figsize=(5,5))
delta = 0.05
x = np.arange(-2, 4, delta)
y = np.arange(-2, 4, delta)
X, Y = np.meshgrid(x, y)

pos  = np.dstack((X, Y))
rv   = multivariate_normal([1, 1], [[1, 0], [0, 2]])
Z    = rv.pdf(pos)

plt.contourf(X, Y, Z)
plt.colorbar()
plt.show()


# In[3]:


#part 2
fig = plt.figure(figsize=(5,5))
delta = 0.05
x = np.arange(-5, 7, delta)
y = np.arange(-5, 7, delta)
X, Y = np.meshgrid(x, y)

pos  = np.dstack((X, Y))
rv   = multivariate_normal([-1, 2], [[2, 1], [1, 4]])
Z    = rv.pdf(pos)

plt.contourf(X, Y, Z)
plt.colorbar()
plt.show()


# In[4]:


#part 3
fig = plt.figure(figsize=(5,5))
delta = 0.05
x = np.arange(-5, 7, delta)
y = np.arange(-5, 7, delta)
X, Y = np.meshgrid(x, y)

pos  = np.dstack((X, Y))
rv1  = multivariate_normal([0, 2], [[2, 1], [1, 1]])
rv2  = multivariate_normal([2, 0], [[2, 1], [1, 1]])
Z    = rv1.pdf(pos) - rv2.pdf(pos)

plt.contourf(X, Y, Z)
plt.colorbar()
plt.show()


# In[5]:


#part 4
fig = plt.figure(figsize=(5,5))
delta = 0.05
x = np.arange(-5, 7, delta)
y = np.arange(-5, 7, delta)
X, Y = np.meshgrid(x, y)

pos  = np.dstack((X, Y))
rv1  = multivariate_normal([0, 2], [[2, 1], [1, 1]])
rv2  = multivariate_normal([2, 0], [[2, 1], [1, 4]])
Z    = rv1.pdf(pos) - rv2.pdf(pos)

plt.contourf(X, Y, Z)
plt.colorbar()
plt.show()


# In[6]:


#part 5
fig = plt.figure(figsize=(5,5))
delta = 0.05
x = np.arange(-5, 7, delta)
y = np.arange(-5, 7, delta)
X, Y = np.meshgrid(x, y)

pos  = np.dstack((X, Y))
rv1  = multivariate_normal([1, 1], [[2, 0], [0, 1]])
rv2  = multivariate_normal([-1, -1], [[2, 1], [1, 2]])
Z    = rv1.pdf(pos) - rv2.pdf(pos)

plt.contourf(X, Y, Z)
plt.colorbar()
plt.show()


# In[7]:


#question 4 starts
from scipy.stats import norm


# In[8]:


# arr = [1, 2, 3]
# arr * 3


# In[9]:


#add random seed 
np.random.seed(7)
x1 = np.random.normal(3, 3, 100)
z = np.random.normal(4, 2, 100)
x2 = []
for i in range(100):
    x2.append((0.25) * x1[i] + z[i])
x2 = np.array(x2)


# In[10]:


sample_pts = []
for i in range(100):
    sample_pts.append((x1[i], x2[i]))
sample_pts = np.array(sample_pts)


# In[11]:


#part 1
sample_mean = np.mean(sample_pts, axis = 0)
sample_mean


# In[12]:


#part 2
sample_cov = np.cov(sample_pts.T)
sample_cov


# In[13]:


#part 3
from numpy import linalg as LA
eigen_val, eigen_vec = LA.eig(sample_cov)
eigen_val, eigen_vec


# In[14]:


#part 4 i
plt.figure(figsize=(10, 10))
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(sample_pts[:,0], sample_pts[:, 1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("x1 vs x2")


# In[15]:


# part 4 ii
x_mean = sample_mean[0]
y_mean = sample_mean[1]
plt.figure(figsize=(10, 10))
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(sample_pts[:,0], sample_pts[:, 1])
plt.quiver([x_mean, x_mean], [y_mean, y_mean], 
          [eigen_vec[0][0] * eigen_val[0], eigen_vec[0][1] * eigen_val[1]], 
          [eigen_vec[1][0] * eigen_val[0], eigen_vec[1][1] * eigen_val[1]],
           scale = 1,
           scale_units = 'xy')
plt.xlabel("x1")
plt.xlabel("x2")
plt.title("x1 vs x2 with eigenvector plotted")


# In[16]:


#part 5
x_centered = sample_pts - sample_mean
# print(x_centered)
sample_rotated = np.dot(eigen_vec.T, x_centered.T).T
plt.figure(figsize=(10, 10))
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter(sample_rotated[:,0], sample_rotated[:, 1])
plt.xlabel("x1")
plt.xlabel("x2")
plt.title("sample points centered and rotated")


# In[ ]:





# In[ ]:




