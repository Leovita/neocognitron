import numpy as np
import matplotlib.pyploy as plt

def relu(x):
    return np.maximum(0, x)

class SLayer:
    def __init__(self, filter_size, num_filters):
        self.filter_size = filter_size
        self.num_filters = num_filters

        #filtri random idc
        self.filters = np.random.randn(num_filters, filter_size, filter_size)

    def apply_filters(self, input_image):
        height, width = input_image.shape
        output = np.zeros((self.num_filters, height - self.filter_size + 1, width - self.filter_size + 1))
        
        for f in range(self.num_filters):
            filter_kernel = self.filters[f]
            for i in range(height - self.filter_size + 1):
                for j in range(width - self.filter_size + 1):
                    region = input_image[i:i + self.filter_size, j:j + self.filter_size]
                    output[f, i, j] = np.sum(region * filter_kernel)

        return relu(output)
    
    def pooling(x, pool_size=2):
        output_shape = (x.shape[0] // pool_size, x.shape[1] // pool_size)
        pooled_output = np.zeros(output_shape)
        for i in range(0, x.shape[0], pool_size):
            for j in range(0, x.shape[1], pool_size):
                pooled_output[i // pool_size, j // pool_size] = np.max(x[i:i + pool_size, j:j + pool_size])
        return pooled_output

class CLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def apply_pooling(self, input_image):
        height, width = input_image.shape
        # Calcola l'output del pooling
        pooled_height = height // self.pool_size
        pooled_width = width // self.pool_size
        pooled_output = np.zeros((pooled_height, pooled_width))

        for i in range(0, height, self.pool_size):
            for j in range(0, width, self.pool_size):
                region = input_image[i:i + self.pool_size, j:j + self.pool_size]
                pooled_output[i // self.pool_size, j // self.pool_size] = np.max(region)

        return pooled_output



class Neocognitron:
    def __init__(self, input_shape, layers_config):
        self.input_shape = input_shape
        self.layers_config = layers_config
        self.build_network()

    def build_network(self):
        self.network = []
        for layer in self.layers_config:
            if layer['type'] == 'S':
                self.network.append(SLayer(layer['filter_size'], layer['num_filters']))
            elif layer['type'] == 'C':
                self.network.append(CLayer(layer['pool_size']))

    def forward(self, input_image):
        output = input_image
        for layer in self.network:
            if isinstance(layer, SLayer):
                output = layer.apply_filters(output)
            elif isinstance(layer, CLayer):
                output = layer.apply_pooling(output)
        return output

layers_config = [
    {'type': 'S', 'filter_size': 3, 'num_filters': 8},  
    {'type': 'C', 'pool_size': 2},                      
    {'type': 'S', 'filter_size': 3, 'num_filters': 16}, 
    {'type': 'C', 'pool_size': 2},                      
]

neocognitron = Neocognitron(input_shape=(28, 28), layers_config=layers_config)
input_image = np.random.rand(28, 28)
output = neocognitron.forward(input_image)

plt.imshow(output[0], cmap='gray')
plt.title("out")
plt.show()


