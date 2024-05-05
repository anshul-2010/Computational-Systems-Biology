import torch.nn as nn

class NN(nn.Module):
    """
    This is a Neural Network module. It contains the definition of function approximators in the RL problems

    Args:
        input_dim (int): It has the dimensions of input, states, tensor
        output_dim (int): It has the dimension of the output tensor
        hidden_dim (int): It has the dimension of each hidden layer, default is 20
        activation : It is the activation layer. We have used nn.ReLU
        n_hidden (int): number of hidden layers in the NN (We have set it to 10)

    """
    def __init__(self,input_dim, output_dim, hidden_dim=20, activation=nn.ReLU, n_hidden=8):
        super(NN,self).__init__()
        self.inlayer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation()
        )

        self.hidden=nn.Sequential(*[
            nn.Linear(hidden_dim,hidden_dim),
            activation()]*n_hidden)

        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs):
        out=self.inlayer(obs)
        out=self.hidden(out)
        out=self.output(out)
        return out
  
"""
The code is completely inspired from
https://github.com/chan-csu/SPAM-DFBA.git
"""