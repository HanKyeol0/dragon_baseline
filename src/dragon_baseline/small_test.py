class superClass():
    def __init__(self, super_num):
        self.super_num = super_num
    
    def super_add(self, add):
        self.super_num += add

    def print_super_num(self):
        print("super_num:", self.super_num)

class test(superClass):
    def __init__(self):
        self.num = 1

    def init_super(self):
        super().__init__(super_num=0)
    
    def add(self, add):
        self.num += add

    def print_num(self):
        print("num:", self.num)

instance = test()
instance.init_super()
instance.print_super_num()
instance.add(2)
instance.add(3)
instance.print_num()
instance.super_add(2)
instance.print_super_num()
instance.init_super()
instance.print_num()
instance.print_super_num()