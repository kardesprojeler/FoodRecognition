class object:
    def __init__(self):
        print("object class")


class Another_Layer():
    def __init__(self):
        print("another layer")

    def another_method(self):
        print("another method çalıştı")

    def __call__(self, *args, **kwargs):
        print("another method called")


class Tensor_Layer(object):
    def __init__(self):
        super(Tensor_Layer, self).__init__()
        print("tensor layer oluşturuldu")

    def deneme(self):
        print("başka bir olay")

    def __call__(self, *args, **kwargs):
        print("tensor layer called")

class Own_Layer(Tensor_Layer):
    def __init__(self):
        super(Own_Layer, self).__init__()

    def deneme(self):
        print("asd")

    def benim_layerde_olan_function(self):
        print("özel fonksiyon")


class Tensor_Activation(Another_Layer):
    def __init__(self):
        super(Tensor_Activation, self).__init__()
        print("tensor activation")

    def deneme(self):
        print("Tensor activation çalıştı")

    def __call__(self, *args, **kwargs):
        print("tensor activation çağrıldı")


class MyActivation(Own_Layer, Tensor_Activation):
    def __init__(self):
        super(MyActivation, self).__init__()

    def deneme(self):
        print("my activation function çalıştı")


if __name__ == '__main__':
    myactivation = MyActivation()
    myactivation()
    myactivation.deneme()
    myactivation.another_method()
    myactivation.benim_layerde_olan_function()