class ANNModel:
    """
    Simple Artificial Neural Network placeholder.
    """

    def __init__(self, layers=[64, 64, 10]):
        self.layers = layers

    def forward(self, input_data):
        # Dummy forward: scale input sum by layer size
        output = input_data
        for layer_size in self.layers:
            output = [sum(output) * 0.1] * layer_size
        return output
