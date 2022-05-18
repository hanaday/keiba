import numpy as np
#import matplotlib.pyplot as plt


def cul_power(input_data, correct_data):
    #print(input_data, correct_data)
    class OutputLayer:
        def __init__(self, n_upper, n):
            self.w = wb_width * np.random.randn(n_upper, n)

        def forward(self, x):
            self.x = x
            u = np.dot(x, self.w)
            self.y = u

        def backward(self, t):
            delta = self.y - t

            self.grad_w = np.dot(self.x.T, delta)

            self.grad_x = np.dot(delta, self.w.T)

        def update(self, eta):
            self.w -= eta * self.grad_w


    n_data = len(correct_data)

    train_data = []
    for data in input_data:
        train_data.append([data[0]**3, data[0]**2, data[0]**1, data[0]**0, data[1]**3, data[1]**2, data[1]**1, data[1]**0,
                           data[2]**3, data[2]**2, data[2]**1, data[2]**0, data[3]**3, data[3]**2, data[3]**1, data[3]**0])

    input_data = train_data
    input_data = np.array(input_data)
    correct_data = np.array(correct_data)
    correct_data = correct_data - 1

    n_in = 4
    n_mid = 16 #4
    n_out = 1

    wb_width = 0.0001
    eta = 0.0001
    epoch = 1001
    interval = 1000

    output_layer = OutputLayer(n_mid, n_out)

    plot_error = []

    for i in range(epoch):
        index_random = np.arange(n_data)
        np.random.shuffle(index_random)

        total_error = 0
        plot_x, plot_y = [], []

        for idx in index_random:
            x = input_data[idx:idx+1]
            t = correct_data[idx:idx+1]

            output_layer.forward(x)
            output_layer.backward(t.reshape(1, 1))
            output_layer.update(eta)

            #if i%interval == 0:
            #    y = output_layer.y.reshape(-1)
            #    total_error += 1.0/2.0*np.sum(np.square(y - t))
            #    plot_x.append(x)
            #    plot_y.append(y)

        if i%interval == 0:
            #plot_error.append(total_error/n_data)

            if np.isnan(output_layer.w[0]):
                print("error")
            #print(output_layer.w)
            #print("epoch:" + str(i) + "/" + str(epoch),
            #      "error:" + str(total_error/n_data))

            #if i == epoch-1:
            #    plt.plot(plot_error)
            #    plt.show()

    return (output_layer.w[0, 0], output_layer.w[1, 0], output_layer.w[2, 0], output_layer.w[3, 0], output_layer.w[4, 0], output_layer.w[5, 0], output_layer.w[6, 0], output_layer.w[7, 0],
            output_layer.w[8, 0], output_layer.w[9, 0], output_layer.w[10, 0], output_layer.w[11, 0], output_layer.w[12, 0], output_layer.w[13, 0], output_layer.w[14, 0], output_layer.w[15, 0])