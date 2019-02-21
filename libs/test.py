def a(snr, length):
    print("calling noise generator A!")
    return 1000


def b(snr, length):
    print("calling noise generator B!")
    return 5000


class Generator():
    def __init__(self, funcs=[a]):
        print("calling ctor!")
        self.funcs = funcs
        print(self.funcs)
    
    def process(self, x):
        print("calling process function!")
        xs_noised = []
        for func in self.funcs:
            print(func)
            noise = func(1, 1)
            xs_noised.append(x + noise)
        return xs_noised


# gen = Generator()
# x_noised = gen.process(42)
# print("Noised data:", x_noised)

gen2 = Generator(funcs=[a, b])
x_noised = gen2.process(42)
print("Noised data:", x_noised)
