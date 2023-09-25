#!/usr/bin/env python
# coding: utf-8

# # Lab 1
# 
# Welcome to your first lab! This notebook contains all the code and comments that you need to submit. Labs are two weeks long and the places where you need to edit are highlighted in red. Feel free to add in your own markdown for additional comments.
# 
# __Submission details: make sure you have run all your cells from top to bottom (you can click _Kernel_ and _Restart Kernel and Run All Cells_). Submit this Jupyter Notebook and also submit the .py file that is generated.__

# In[1]:


## This code snippet does not need to be edited

from python_environment_check import check_packages
from python_environment_check import set_background

## Colour schemes for setting background colour
white_bgd = 'rgba(0,0,0,0)'
red_bgd = 'rgba(255,0,0,0.2)'


# In[2]:


set_background(red_bgd)

## Code snippets in red (similar to this) is where you need to edit your answer)
# Set your student ID and name here:

student_number = 31975658
student_name = "Sihan Ren"


# # Libraries
# 
# 

# In[3]:


## Checks for the minimum version requirements for a few libraries
## Credits to: https://sebastianraschka.com/blog/2022/ml-pytorch-book.html for the code related to checking version requirements


d = {
    'torch': '1.8.0',
    'torchvision': '0.9.0',
    'numpy': '1.21.2',
    'matplotlib': '3.4.3'
}

check_packages(d)

if 'google.colab' in str(get_ipython()):
  print('Running on CoLab')
else:
  print('Not running on CoLab')
print(d)


# Libraries are important as it saves us time from writing our own functions all the time such as graphing, or creating matrices. Brief library descriptions have been added for every library that we import. We can use the "import" function to import libraries, and use _as_ to rename the library within our workspace. For example, we could have used _import numpy_ by itself, and referenced Numpy functions by _numpy.function_, but instead we have an alias of Numpy as _np.function_ instead. 

# In[4]:


## Libraries, you do not need to import any additional libraries for this lab

import numpy as np ## Numpy is the fundamental building block of understanding tensor (matrices) within Python
import matplotlib.pyplot as plt ## Matplotlib.pyplot is the graphing library that we will be using throughout the semester
import random ## Useful for sampling 

from scipy.special import gamma ## Pre-built gamma function that we will use for this lab (for the final task)
import math # Basic math library

import os ## Useful for running command line within python
from IPython.display import Image ## For markdown purposes


# # Section 1 - Understanding Python basics

# In this section, you will be writing some procedural-type code which includes:
# - 1.1 Data types
# - 1.2 Conditional statements
# - 1.3 Loops 
# - 1.4 Random number generator and writing functions
# - 1.5 Classes
# 
# Throughout this lab, there will be code and written answers that need to be filled out by you.  The comments in the code snippet and markdown text will guide you on what you need to fill out. 

# ## 1.1  Data types
# 
# This task is a gentle introduction in creating a few of the data types offered by Python. Formatting the print statement can be done in several ways within Python. The print statement below follows the format of __print(f"      ")__ and you can add in variable types (automatically converted to string) within the print statement by wrapping it with __{}__

# In[5]:


print(f"My ID is: {student_number} and my name is {student_name} ")


# __Lists__ are the most versatile of Python's compound data types. A list contains items separated by commas and enclosed within square brackets ([]). To some extent, lists are similar to arrays in C. One of the differences between them is that all the items belonging to a list can be of different data type.
# 
# The values stored in a list can be accessed using the slice operator ([ ] and [:]) with indexes starting at 0 in the beginning of the list and working their way to end -1. The plus (+) sign is the list concatenation operator, and the asterisk (*) is the repetition operator.

# In[6]:


set_background(red_bgd)

## Manually make a list of values from 0 to 2 (inclusive) in intervals of 0.5. 

my_list = [0, 0.5, 1, 1.5, 2]
#my_list = np.arange(0, 2.1, 0.5).tolist()


# In[7]:


set_background(red_bgd)

## Next, append the string "This is a string" to the end of the list

my_list.append("This is a string")

print(f"Length of list is {len(my_list)} and the contents of my_list is: \n{my_list}")


# Python's __dictionaries__ are kind of hash-table type. They work like associative arrays or hashes found in Perl and consist of key-value pairs. A dictionary key can be almost any Python type, but are usually numbers or strings. Values, on the other hand, can be any arbitrary Python object.
# 
# Dictionaries are enclosed by curly braces ({ }) and values can be assigned and accessed using square braces ([]).

# In[8]:


set_background(red_bgd)


## Create a dictionary with the following mappings and print out the dictionary by iterating through it. Use the functions "iter" and "next" to iterate through the dictionary

## Unit : 4179
## Session: Lab

my_dict = {'Unit': 4179, 'Sesion': 'lab'}


# ## 1.2 Conditional statements
# 
# This task demonstrates the high level language Python uses for program control

# In[9]:


set_background(red_bgd)

# Write an IF statement to check if your student ID is divisble by 5 and 8

if student_number % 40 == 0:
    print('student number is divisble by 5 and 8')
elif student_number % 5 == 0:
    print('student number is divisble by 5')
elif student_number % 8 == 0:
    print('student number is divisble by 8')
else:
    print('student number is not divisble by 5 or 8')
    
    


# In[10]:


set_background(red_bgd)

# Check if your student number is a float
if type(student_number) == float:
    print('student number is a float')
else:
    print('student number is not a float')

#isinstance(student_number, float)


# ## 1.3 Loops
# 
# This task demonstrates an understanding of how loops work in Python. Remember, indexing starts from 0! 

# In[11]:


set_background(red_bgd)

# For the numbers between 90 and 210 (inclusive), print out the numbers that are multiples of 15

for num in range(90, 211):
    if num % 15 == 0:
        print(num)
        


# In[12]:


set_background(red_bgd)


# For your student ID, keep dividing by 4 until it is a value lower than 10. Print out the final value. 
# Note: Do not edit the original student_number variable.

temp_ID = student_number

while temp_ID >= 10:
    temp_ID /= 4

print("{:.2f}".format(temp_ID))


# ## 1.4 Random number generator and writing functions
# 
# This task tests your ability to read the official documentation from Python, and being able to define your own functions.
# 
# Before beginning this task, it is important to understand the purpose of randon-number generator (RNG) seeds. RNG can never be completely random, and we can use the concept of _pseudo-random_ to be able to reproduce results. When setting a seed, any numbers that are randomly generated will be the same random numbers if we were to generate again using the same seed, which means we can get the same random result. This is useful in being able to run and generate random numbers between machine, and to get the same consistent random results. 

# In[13]:


set_background(red_bgd)

## Set a random seed that is equal to your student_number
## Hint: Have you read the python documentation for the official python library: "random" ? 
random.seed(31975658)


# In[14]:


set_background(red_bgd)

## Generate a random integer between 5 and 10
random.seed(31975658)
rand_num = random.randint(5, 10)
print(rand_num)


# In[15]:


set_background(red_bgd)


## Write a factorial function and pass in the number you have randomly generated into the factorial function

def factorial_fn(number):
    if number == 1:
        return number
    else:
        number *= factorial_fn(number - 1)
        return number
        

factorial_fn(rand_num)


# ## 1.5 Classes
# 
# As Python is an object orientated programming (OOP) language, you need to understand how classes and objects work to program more efficiently in Python. In C, we have typically used procedural programming where you simply use function calls to achieve specific outcomes. With OOP, it becomes more efficient as you inherit methods from classes which are linked to the objects that you have created. Read more here for OOP:
# https://en.wikipedia.org/wiki/Object-oriented_programming
# 
# In this task, you need to create a class called Person which will have 3 attributes:
# - Name
# - Age
# - Income
# 
# This class will also have two methods:
# - A print statement that shows all the attributes of the person
# - Prints the person's age in 5 years time
# 
# After declaring the class, make a "Person" named "John" with age being 36, and income being 50000. Use the two methods of summarise() and future_age() after creating this "Person". 

# In[16]:


set_background(red_bgd)

## Fill out this template to define a class called Person

class Person:

    def __init__(self, name, age, income):
        self.name = name
        self.age = age
        self.income = income

    def summarise(self): ## Summarises all attributes for Person
        print(f"The name is {self.name}, age is {self.age} and income is {self.income}")
    def future_age(self): ## Prints future age
        print(self.age + 5)
 
        
p1 = Person('John', 36, 50000)
p1.summarise() 
p1.future_age()

#Note in Python str() wil convert a variable to a string without 
# any of the weird conversion issues (most of the time)


# Well done on making the class called "Person". In this next part, you will inherit the class that you have previously defined and create a new class called Student. This class will inherit all the attributes and methods from the "Person" class, and will have the following changes/additions:
# 
# - Inherit the "Person" class 
# - Add an attribute of the student's mark
# - Redefine the method of "future_age" to print out the student's age in 2 years time instead
# - Create a "Student" called "Jane" that is 20 years old, with 10000 as their income, and 89 as their mark
# 
# After you have inherited the class, display the student's mark and execute the two methods under this new class

# In[17]:


set_background(red_bgd)


## Fill out this template of class inheritance

class Student(Person):
    def __init__(self, name, age, income, mark):
        self.mark = mark
        super().__init__(name, age, income)

    def future_age(self): ## Prints future age
        print(self.age + 2)

        

sp1 = Student('Jane', 20, 10000, 89)
print(sp1.mark)
sp1.future_age() 
sp1.summarise() 


# # Section 2 - Using libraries

# In this section, you will be using the following libraries:
# - 2.1 numpy
# - 2.2 matplotlib

# ## 2.1 Numpy
# 

# In[18]:


Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/NumPy_logo.svg/775px-NumPy_logo.svg.png', width=300)


# NumPy (Numerical python) is a very popular Python library with a huge number of functions and utilities. There are many similarities between the functions found in Matlab and those in Numpy and so we can use our knowledge of Matlab to help with learning to use Numpy. Later when we introduce Pytorch (The Deep Learning Framework used in this unit) we will see that there are many similarities between it and Numpy. In this way Numpy is our Python stepping stone between Matlab and Pytorch!
# 
# In this task, we will be performing manipulations on numpy arrays (from now on we will refer to numpy arrays as tensors) and understanding how tensors interact with one another. There are also new concepts such as broadcasting which is useful in writing more efficient code for tensor arithmetic by reducing the number of for loops required!

# In[19]:


## Numpy arithmetic is similar to performing matrix arithmetic, except we are usually dealing with high dimensional tensors
## There are a few caveats of numpy arithmetic to consider. 
## look at the following code example and comment on what is happening

tensor_a = np.eye(2)
tensor_b = np.array([[2, 5], [1, 3]])

print(f"tensor_a: \n{tensor_a}")
print(f"tensor_b: \n{tensor_b}")

tensor_add = np.add(tensor_a, tensor_b)
print(f"tensor_add: \n{tensor_add}")

tensor_add2 = tensor_a + tensor_b
print(f"tensor_add2: \n{tensor_add2}")

tensor_mult = np.matmul(tensor_a, tensor_b)
print(f"tensor_mult: \n{tensor_mult}")

tensor_mult2 = tensor_a * tensor_b
print(f"tensor_mult2: \n{tensor_mult2}")

tensor_pow = tensor_b**2
print(f"tensor_pow: \n{tensor_pow}")

tensor_pow2 = np.matmul(tensor_b, tensor_b)
print(f"tensor_pow2: \n{tensor_pow2}")


# In[20]:


set_background(red_bgd)



## Use numpy to generate data x-values for a sine function. Calculate sin(x) for each of the x values. 
## You can use np.linspace to create linearly spaced values for your "x".
## Generate at least 5 cycles

x_values = np.linspace(-20, 20, 1000)
sine_x = np.sin(x_values)

plt.plot(x_values,sine_x)
plt.xlabel("x-values")
plt.ylabel("sine(x)")
plt.title("Sine wave")
# print(x_values)
# print(sine_x)


# It is common to have data that is stored as an external file. In this case we will be using data that has been stored as a .npy format (numpy format) and using the numpy library to read it. Plot out the result and describe what you see.

# In[21]:


set_background(red_bgd)

## Load the data using np.load
## Look at the dimensions of the data and plot it out accordingly
## Comment on the plot
toy_data = np.load('toy_data.npy')

print("Toy data shape is", toy_data.shape )

plt.plot(toy_data[:, 0], toy_data[:, 1])

## Comment on what you see:
''' A vertical sine wave'''


# In[22]:


set_background(red_bgd)

## Some other useful numpy methods that you can use are:
## tensor.flatten()
## tensor.reshape()

## Have a play around with these and comment on what each one does

# temp_array = np.random.rand(10,10)
# print(temp_array.flatten().shape)
# print(temp_array.reshape(5,-1).shape)

# eye_array = np.eye(5)
# eye_array



# Broadcasting is a powerful tool that lets us perform element wise matrix or vector operations across higher dimensional Tensors. <br>
# 
# ![alt text](https://media.geeksforgeeks.org/wp-content/uploads/numpy2.png)
# 
# Lets see what we mean by this by working through the example below:

# In[23]:


#Lets create 2 differently shaped 2D Tensors (Matrices)

Tensor1 = np.random.randint(0, 10, (1, 4))
Tensor2 = np.random.randint(0, 10, (2, 1))

print("Tensor 1:\n", Tensor1)
print("With shape:\n", Tensor1.shape)

print("\nTensor 2:\n", Tensor2)
print("With shape:\n", Tensor2.shape)


# We know from high school days that there is no way we can perform a normal matrix addition on these two matrices, so when we try Numpy should give us an error right?

# In[24]:


Tensor3 = np.add(Tensor1, Tensor2)

print("The resulting Tensor:\n", Tensor3)
print("The resulting shape is:\n", Tensor3.shape)

# set_background(red_bgd)


# WHAT!?! A 1x4 Matrix added a 2x1? resulting in a 2x4 Matrix, What did Numpy do here?<br>
# Well, as suggested Numpy is NOT performing a normal Matrix addition. Instead Numpy is performing a broadcast operation, THEN a Matrix addition. <br>
# So then, what is Broadcasting? <br>
# Let's look again at the shape of those two 2D Tensors and the resulting Tensor

# In[25]:


print("Tensor 1 shape:\n", Tensor1.shape)
print("Tensor 2 shape:\n", Tensor2.shape)
print("Resulting Tensor shape:\n", Tensor3.shape)


# We can see the resulting shape of the Tensor addition seems to come from the larger dimensions of the multiplication<br>
# 1x<b>4</b>+<b>2</b>x1 = <b>4x2</b> <br>
# During the "Broadcast" operation Numpy "repeats" (Broadcasts) dimensions of the two Tensors so that they are the same shape, and then performs the addition. <br>
# __Where is this useful?__ In many cases in data science, we may want to perform the same operation with various arguments. For example, we may want to add numbers stored in an array, one by one to another array. You can do this by using a for-loop. Thankfully, NumPy makes things simple for us via broadcasting!

# ## 2.2 Matplotlib

# In[26]:


Image(url='https://matplotlib.org/_static/logo2_compressed.svg', width=300)


# 
# 
# Matplotlib is a "Matlab-ish" plotting librariy that lets us create all sorts of figures and plots. As suggested it works very similarly to Matlab's plotting functunality (specifically matplotlib.pyplot). The matplotlib library works very closely wth numpy and can plot out numpy arrays.
# 

# In[27]:


# set_background(red_bgd)
random.seed(1)
#lets create a plot of a noisy parabola!
#create some empty lists
x_parab_values = []
y_parab_values = []

for x in range(1000):
    x_parab_values.append(x)
    y_parab_values.append(pow(x, 2) + random.random())
#In a for-loop create the x and y values
#use the math function "pow" to compute x^2
#and then add a random number from 0 - 1
print(y_parab_values)


# In[28]:


set_background(red_bgd)
## Make a scatterplot
## Make sure you add your x and y labels. Add a title as well. Play around with the colour and marker types. 
## For more information, you can refer to the official matplotlib documentation: https://matplotlib.org/stable/api/index.html

plt.plot(x_parab_values, y_parab_values, 'r')
plt.xlabel('x_parab_values')
plt.ylabel('y_parab_values')
plt.title('noisy parabola')


# In[29]:


set_background(red_bgd)

## Discuss what you see


# # Section 3 - Applying your python knowledge 
# 
# In this section, you will apply the python basics that you have learnt so far in mathematical based problems. This section has two tasks:
# 
# - Estimating $\pi$
# - Calculating the volume of an N-dimensional ball
# 
# You have to use numpy and matplotlib to complete this section.

# ## 3.1 - Estimating $\pi$
# 
# In this task, you will be estimating the value of pi. The way that we can estimate the value of pi is by performing **random uniform sampling** between the values of \[0,1) in 2 dimensions. Next, we apply L2 normalisation (ie. euclidean distance) to every datapoint in the 2 dimensions and the probablity that a randomly sampled point lies within the unit circle. That probability provides us with 1/4 of the area, so we take this area and multiply it by 4 to calculate $\pi$. The following figure shows 10 randomly sampled points:
# 
# 
# 

# In[30]:


Image(url='https://miro.medium.com/max/803/0*oWmkwPg771ISI_aW', width=300)


# The equation below demonstrates how $\pi$ can be calculated
# 
# \begin{equation*}
# \frac{(\pi r^2)}{4} = area
# \newline
# \pi = \frac{4 * area}{r^2}
# \newline
# \pi = 4 * area
# \newline
# \pi = 4 * \frac{8}{10} = 3.2
# \end{equation*}
# 
# Where _area_ is the probability of a point being within the unit circle. In the above example, we have 8 points (out of 10) within the unit circle
# 
# You can read this article to get a better understanding of what is happening: https://www.cantorsparadise.com/calculating-the-value-of-pi-using-random-numbers-a-monte-carlo-simulation-d4b80dc12bdf
# 
# Use Numpy to complete this task.

# In[31]:


dim = 2 ## Number of dimensions needed to estimate pi
nSamples = int(1e4) ## Number of samples to be generated to estimate pi


# In[32]:


# set_background(red_bgd)

# x_y = np.array([[np.random.uniform(0, 1), np.random.uniform(0, 1)] for _ in range(nSamples)])
# print(x_y)
# print(x_y[:, 0])

x = np.random.uniform(0, 1, nSamples)
y = np.random.uniform(0, 1, nSamples)
count = 0

for i in range(nSamples):
    if x[i]**2 + y[i]**2 <= 1:
        count += 1

pi_estimate = (count / nSamples) * 4

print(pi_estimate)
# print(count)

# np.array([[1, 2], [3, 4]])
## Write your code here to estimate the value of pi
## You should use np.random.uniform to generate your random numbers



# In[33]:


set_background(red_bgd)

## Is the estimate of pi close to the actual value? Why/why not? What could be potential influences?
'''Yes, it's close. Because we use 1e4 samples. The number of samples
The accuracy is two decimal places, number of samples play a big role.'''


# ## 3.2 - Calculating the volume of an N-dimensional ball

# In this task, we want to estimate and calculate the volume of an N-dimensional ball of radius one and discuss the implications of scaling to a higher dimensional space. 
# 
# For a 2-dimensional ball (circle), the volume is by $\pi r^2$
# 
# For a 3-dimensional ball (sphere), the volume is given by $\frac{4 \pi r^3}{3}$
# 
# For an N-dimensional ball, the volume is given by $V_n(r) = \frac{\pi ^ \frac{n}{2}}{\Gamma(\frac{n}{2} + 1)}r^n$
# 
# $\Gamma$ is called _gamma_ and it is analagous to factorials but across the continuous domain. In this case, we are using it for calculus to calculate the volume of an N-dimensional ball. You can simply use the _gamma()_ function directly (we have imported this for you) in order to calculate the volume of an N-dimensional ball. Feel free to read more here with regards to N-dimensional balls: https://en.wikipedia.org/wiki/Volume_of_an_n-ball
# 
# Your task is to:
# 
# 1. For dimensions from 2 to 50, do steps 2 and 3.
# 2. Estimate the volume of an N-dimensional ball (similar to question 3). You can do this by taking the probability of a point being inside the N-dimensional ball of radius 1 by sampling between \[0,1) across the N-dimensions and calculating its Euclidean distance. Afterwards, you can calculate the volume by doing 2^N * (fraction of points within unit circle in high dimensional space). This is similar to question 3.
# 3. Calculate the actual volume of an N-dimensional ball with the equation given before. 
# 4. Compare estimated and actual volume. Comment your findings.
# 5. Interpret the actual volume and see if there is anything unexpected. Comment on your findings. 
# 
# Use Numpy and matplotplib to complete this task.

# In[34]:


nSamples = int(10)
distance = 0
count = 0

## Calculate the estimated volume of an N-dimensional ball from 2 to the 50th dimensions

coordinate = np.array([np.random.uniform(0, 1, 2) for _ in range(2)])
print(coordinate)


# In[ ]:


# set_background(red_bgd)

first_dim = 2
last_dim = 50

nSamples = int(1e4)
count = 0
estimated_value = []

## Calculate the estimated volume of an N-dimensional ball from 2 to the 50th dimensions
for d in range(2, 51):
    count = 0
    coordinate = np.array([np.random.uniform(0, 1, nSamples) for _ in range(d)]).transpose()
    # print(coordinate)
    for i in range(nSamples):
        distance = 0
        for j in range(d):
            distance += coordinate[i][j] ** 2
        # print(distance)
        if distance <= 1:
            count += 1
    # print(count)
    estimated_value.append(count / nSamples * 2 ** d)
    print("{} - D : {:.15f}".format(d, estimated_value[d - 2]))
    
    

        





# In[ ]:


set_background(red_bgd)

## Calculate the actual volume of an N-dimensional ball from 2 to the 50th dimensions
actual_value = []
for n in range(2, 51):
    actual_value.append(np.pi ** (n / 2) / gamma(n /2 + 1))
    print(f"{n} - D :", actual_value[n - 2] )

    


# In[ ]:


set_background(red_bgd)

## Plot the estimated and actual volume of the N-dimensional ball as a line graph
## Hint: The points should align well between the estimated and actual values
plt.plot(list(range(2, 51)), actual_value, label = 'actual value')
plt.plot(list(range(2, 51)), estimated_value, 'r+', label = 'estimated value', markersize=10)
plt.legend()
plt.xlabel('dimension')
plt.ylabel('volume')
plt.title('first graph')

# set_background(red_bgd)


# In[ ]:


set_background(red_bgd)

## Discuss what you see with regards to the volume of an N-dimensional ball (with N going from 2 to 50). 
# Can't predict after dimension 13, and the volume keeps decreasing.

## What do you notice about the volume with respect to higher dimensions?
# The volumes are very small.


# # Do not remove or edit the following code snippet. 
# 
# When submitting your report, please ensure that you have run the entire notebook from top to bottom. You can do this by clicking "Kernel" and "Restart Kernel and Run All Cells". Make sure the last cell (below) has also been run. 

# In[ ]:


file_name = str(student_number) + '_Lab1_Submission.py'
cmd = "jupyter nbconvert --to script Lab_1_Student.ipynb --output " + file_name
if(os.system(cmd)):
    print("Error converting to .py")


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=3b41861c-05c0-423a-b965-83ee82dc8610' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
