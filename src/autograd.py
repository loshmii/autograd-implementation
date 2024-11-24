import numpy as np

class Value : 
    def __init__(self, data, children = (), _op = '') : 
        self.data = float(data)
        self.grad = 0
        self.children = set(children)
        self._backward = lambda : None
        self._op = _op

    def _mul(self, other) : 
        out = Value(self.data * other.data, (self, other), '*')
        def _backward() : 
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other) :
        other = other if isinstance(other, Value) else Value(other)
        return self._mul(other)

    def _add(self, other) : 
        out = Value(self.data + other.data, (self, other), '+')
        def _backward() : 
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __add__(self, other) :
        other = other if isinstance(other, Value) else Value(other)
        return self._add(other)
    
    def _pow(self, other) :
        assert other.data > 0, 'power must be greater than 0'
        out = Value(self.data ** other.data, (self, other), '**')
        def _backward() : 
            self.grad += other.data * self.data ** (other.data - 1) * out.grad
            other.grad += self.data ** other.data * np.log(self.data) * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other) :
        other = other if isinstance(other, Value) else Value(other)
        return self._pow(other)
    
    def __rpow__(self, other) :
        #Handle case where base is not a Value object
        other = other if isinstance(other, Value) else Value(other)
        return other._pow(self)

    def tanh(self) : 
        data = self.data
        val = (np.exp(2*data) - 1) / (np.exp(2*data) + 1)
        out = Value(val, (self,), 'tanh')
        def _backward() : 
            self.grad += (1 - val**2) * out.grad
        out._backward = _backward
        return out


    def relu(self) : 
        out = Value(self.data if self.data > 0 else 0, (self,), 'relu')
        def _backward() : 
            self.grad += out.grad if self.data > 0 else 0
        out._backward = _backward
        return out

    def backward(self) : 
        # topological sort all the children
        self.grad = 1
        topo = []
        visited = set()
        def build_topo(v) : 
            if not v in visited : 
                visited.add(v)
                for child in v.children : 
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # backprop
        for node in reversed(topo) : 
            node._backward()

    def __repr__(self) -> str:
        return f'data: {self.data}, grad: {self.grad}'
        
def print_graph(root, prefix="", is_last=True, visited=None):
    """
    Prints a tree visualization of the compute graph using ASCII characters.
    
    Args:
        root: The root node of the graph
        prefix: String prefix for current line (used for indentation)
        is_last: Boolean indicating if this is the last child of its parent
        visited: Set of visited nodes to handle cycles
    """
    if visited is None:
        visited = set()
        
    if root not in visited:
        visited.add(root)
        
        # Create the connection line
        connector = "└── " if is_last else "├── "
        
        # Print current node with its data and gradient
        print(prefix + connector + 
              f"{root._op if root._op else 'input'} " +
              f"(data: {root.data:.4f}, grad: {root.grad:.4f})")
        
        # Prepare the prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Get list of children
        children = list(root.children)
        
        # Print all children
        for i, child in enumerate(children):
            print_graph(child, child_prefix, i == len(children)-1, visited)

def test_extended():
    # Test basic arithmetic
    x = Value(2.0)
    y = Value(-3.0)
    z = x * y + x.relu() + y.tanh()
    z.backward()
    print_graph(z)
    print(f"z = {z.data:.4f}")
    print(f"dx = {x.grad:.4f}")
    print(f"dy = {y.grad:.4f}")



if __name__ == '__main__' : 
    test_extended()

