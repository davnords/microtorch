class tensor:
    def __init__(self, data, _children = (), _op=''):
        self.grad = 0
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return str(self.data)
    
    def __add__(self, other):
        out = tensor(self.data+other.data, (self, other), "+")
        return out
    
    def __mul__(self, other):
        out = tensor(self.data*other.data, (self, other), "*")
        return out
    
a = tensor(2.0)
b = tensor(3.0)
c = tensor(1.0)
d = a*b+c
print(d)