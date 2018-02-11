from pandas import Series

class Transformer():
    def __init__(self):
        pass
    def fit(self, data):
        return self
    def apply(self, data):
        return data
    def invert(self, data):
        return data

class Differentiate(Transformer):
    def __init__(self, offset=1):
        self.offset = offset
    def fit(self, data):
        self.y = data[0]
        return self
    def apply(self, data):
        #self.y = data[0] # offset
        return Series([ x1 - x0 for x1, x0 in zip(data[self.offset:], data[:-self.offset])])
    def invert(self, data, y = None):
        if y is None:
            y = self.y
        inverted = [y]
        for x in data:
            prev = inverted[-1]
            inverted.append(prev+x)
        return Series(inverted)
    


class Scale(Transformer):
    def __init__(self, range=(-1, 1)):
        self.range = range
        self.source_range = ()

    def fit(self, series):
        min = series.min()
        max = series.max()
        self.b = (self.range[1] - self.range[0]) / (max - min)
        self.a = self.range[0]-(self.b * min)
        return self

    def apply(self, series):
        return Series([self.a + self.b * x for x in series])

    def invert(self, series):
        return Series([self.invert_value(x) for x in series])
        
    def invert_value(self, val):
        return ( val - self.a) / self.b

