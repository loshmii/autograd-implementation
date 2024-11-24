# AUTOGRAD implementatio from scratch 

Light implementation of reverse mode of autograd similar to those found in ML Python libraries 

## Features 
- Reverse mode autodiff
- Basic mathematical opeartions supported (addition, multiplication, relu and tanh)
- Computational graph visualization 
- Gradient computation using topological sort on directed acyclic graphs 

## Usage example
# Create nodes
a = Value(1)
b = Value(2)
c = a * b
d = a + b
e = c**d     
h = 2**e     
h.backward()  # Compute gradients

# Visualize computational graph
print_graph(h)

# Get gradient 
print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")