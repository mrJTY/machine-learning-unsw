import numpy as np
import pdb

class NeuralNetwork(object):
    def __init__(self, learning_rate=0.1, epochs=1000, batch_size=None,neural_numbers=[10]):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.neural_numbers=neural_numbers
        self.layers=len(self.neural_numbers)+1
        np.random.seed(77)

    def fit(self,X,y):
        self.X,self.y = X,y
        self.initial_weight()
        self.backpropagate(X,y)

    def forward(self,X):
        output_list = []
        input_x = X

        for layer in range(self.layers):

            cur_weight = self.weight_list[layer]
            cur_bias = self.bias_list[layer]
            # Calculate the output for current layer
            output = self.neuron_output(cur_weight,input_x,cur_bias)
            # The current output will be the input for the next layer.
            input_x =  output

            output_list.append(output)
        return output_list

    def backpropagate(self,train_x,train_y):
        acc_list=[]
        for iteration in range(self.epochs):
            if self.batch_size:
                n=train_x.shape[0]
                # Sample batch_size number of sample for n samples
                sample_index=np.random.choice(n, self.batch_size, replace=False)
                x=train_x[sample_index,:]
                y=train_y[sample_index,:]
            else:
                x=train_x
                y=train_y

            # Two output lists
            output_list=self.forward(x)

            # 10x10 prediction
            y_pred=output_list.pop()

            # Record the accuracy every 5 iteration.
            if iteration%5==0:
                acc=self.accuracy(self.softmax(y),self.softmax(y_pred))
                acc_list.append(acc)

            loss_last=y-y_pred

            output=y_pred
            foo = range(self.layers-1,-1,-1)

            # range(2, -1, -1)
            for layer in range(self.layers-1,-1,-1):
                pdb.set_trace()
                if layer!=0:
                    input_last=output_list.pop()
                else:
                    input_last=x

                if layer==self.layers-1:
                    loss,dw,db=self.der_last_layer(loss_last,output,input_last)
                else:
                    weight=self.weight_list[layer+1]
                    loss,dw,db=self.der_hidden_layer(loss_last,output,input_last,weight)

                output=input_last
                self.weight_list[layer] +=dw*self.learning_rate
                self.bias_list[layer] +=db*self.learning_rate
                loss_last=loss
        self.acc_list=acc_list

    def predict(self,X):
        output_list = self.forward(X)
        pred_y = self.softmax(output_list[-1])
        return pred_y

    def accuracy(self, pred, y_test):
        assert len(pred) == len(y_test)
        true_pred=np.where(pred==y_test)
        if true_pred:
            true_n = true_pred[0].shape[0]
            return true_n/len(pred)
        else:
            return 0

    def initial_weight(self):
        if self.X is not None and self.y is not None:
            x=self.X
            y=self.y
            input_dim = x.shape[1]
            output_dim = y.shape[1]

            number_NN = self.neural_numbers+[output_dim]

            weight_list,bias_list = [],[]
            last_neural_number = input_dim     

            for cur_neural_number in number_NN:
                # The dimension of weight matrix is last neural number * current neural number
                weights = np.random.randn(last_neural_number, cur_neural_number)
                # The number of dimension for bias is 1 and the number of current neural
                bias = np.zeros((1, cur_neural_number))

                last_neural_number=cur_neural_number

                weight_list.append(weights)
                bias_list.append(bias)

            self.weight_list=weight_list
            self.bias_list=bias_list

    # Classical sigmoid activation functions are used in every layer in this network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Derivation of the sigmoid activation function
    def sigmoid_der(self, x):
        return (1 - x) * x

    # Calculate the output for this layer
    def neuron_output(self,w,x,b):
        wx=np.dot(x, w)
        return self.sigmoid( wx + b)

    def der_last_layer(self,loss_last,output,input_x):
        sigmoid_der=self.sigmoid_der(output)
        loss = sigmoid_der*loss_last
        dW = np.dot(input_x.T, loss)
        db = np.sum(loss, axis=0, keepdims=True)
        return loss,dW,db

    def der_hidden_layer(self,loss_last,output,input_x,weight):
        loss = self.sigmoid_der(output) * np.dot(loss_last,weight.T)
        db = np.sum(loss, axis=0, keepdims=True)
        dW = np.dot(input_x.T, loss)
        return loss,dW,db

    def softmax(self,y):
        return np.argmax(y,axis=1)

def make_digit(raw_digit):
    return [1 if c == '1' else 0
            for row in raw_digit.split("\n")
            for c in row.strip()]

if __name__ == "__main__":

    # Each picture is a 5x5 pixel = 25 pixel

    raw_digits = [
      """11111
         1...1
         1...1
         1...1
         11111""",

      """..1..
         ..1..
         ..1..
         ..1..
         ..1..""",

      """11111
         ....1
         11111
         1....
         11111""",

      """11111
         ....1
         11111
         ....1
         11111""",

      """1...1
         1...1
         11111
         ....1
         ....1""",

      """11111
         1....
         11111
         ....1
         11111""",

      """11111
         1....
         11111
         1...1
         11111""",

      """11111
         ....1
         ....1
         ....1
         ....1""",

      """11111
         1...1
         11111
         1...1
         11111""",

      """11111
         1...1
         11111
         ....1
         11111"""]

    inputs = np.array(list(map(make_digit, raw_digits)))

    targets = np.eye(10)

    pdb.set_trace()

    Learning_rate=0.05
    nn=NeuralNetwork(learning_rate=Learning_rate)
    nn.fit(inputs,targets)
