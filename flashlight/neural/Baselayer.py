class BaseLayer:
    def __init__(self):
        pass
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        raise NotImplementedError("Subclasses must implement parameters method")