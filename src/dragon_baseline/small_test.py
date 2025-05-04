class test():
    def __init__(self):
        self.num = 1
    
    def add(self, add):
        self.num += add

    def print_num(self):
        print(self.num)

instance = test()
instance.add(2)
instance.add(3)
instance.print_num()