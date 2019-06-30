class Zeynep:
    def __init__(self, yemek_yapmak):
        self.yemek_yapmak = yemek_yapmak

    def kakan(self):
        print("asd")


class durkan(Zeynep):
    def __init__(self, yemek_pisirmek):
        Zeynep.__init__(self, yemek_yapmak=yemek_pisirmek)

    def kakan(self):
        print("başka bir olay")


durkan_nesnesi = durkan("bunu göndermek zorundasın")
print(durkan_nesnesi.kakan())