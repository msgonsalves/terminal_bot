import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import pprint

NUM_OUTPUTS = 1236
GENERATION = 0
SWITCH_BREEDING = 20
# the answer slowly converges on optimal weights
BREEDING_SD = 5 / (GENERATION + 1)




def breed_conv(mom_conv, dad_conv, features):
    # these features must be the same size
    # just choosing one weight
    if GENERATION < SWITCH_BREEDING:
        final_data = []
        for i in range(0, features[1]):
            temp_output = []
            for j in range(0, features[0]):
                temp_column = []
                for k in range(0, features[2]):
                    temp_row = []
                    for l in range(0, features[2]):

                        if .5 > random.uniform(0, 1):
                            temp_row.append(mom_conv[i][j][k][l].item())
                        else:
                            temp_row.append(dad_conv[i][j][k][l].item())
                    temp_column.append(temp_row)
                temp_output.append(temp_column)
            final_data.append(temp_output)

        return torch.Tensor(final_data)

    # probably narrowing down on something so averaging them together
    else:
        final_data = []
        for i in range(0, mom_conv.conv1_features[1]):
            temp_output = []
            for j in range(0, mom_conv.conv1_features[0]):
                temp_column = []
                for k in range(0, mom_conv.conv1_features[2]):
                    temp_row = []
                    for l in range(0, mom_conv.conv1_features[2]):
                        random_average = random.gauss(
                            (mom_conv[i][j][k][l].item() + dad_conv[i][j][k][l].item()) / 2), BREEDING_SD
                        while True:
                            if -1 < random_average < 1:
                                temp_row.append(random_average)
                                break
                            else:
                                random_average = random.gauss(
                                    (mom_conv[i][j][k][l] + dad_conv[i][j][k][l]) / 2), BREEDING_SD
                    temp_column.append(temp_row)
                temp_output.append(temp_column)
            final_data.append(temp_output)
        print(final_data)
        return torch.Tensor(final_data)


def linear_breeding(mom_lin, dad_lin, m_input, d_input, output):
    pass


class TerminalCNN(nn.Module):

    def __init__(self):
        super(TerminalCNN, self).__init__()

        # stop at 4 possible convolution layers
        self.conv1_features = [1, 1, 3, True]
        self.conv1 = nn.Conv2d(self.conv1_features[0], self.conv1_features[1], self.conv1_features[2])
        self.conv2_features = [1, 1, 3, False]
        self.conv2 = nn.Conv2d(self.conv2_features[0], self.conv2_features[1], self.conv2_features[2])
        self.conv3_features = [1, 1, 3, False]
        self.conv3 = nn.Conv2d(self.conv3_features[0], self.conv3_features[1], self.conv3_features[2])
        self.conv4_features = [1, 1, 3, False]
        self.conv4 = nn.Conv2d(self.conv3_features[0], self.conv3_features[1], self.conv3_features[2])
        self.pool = nn.MaxPool2d(2, 2)
        # stop at 10 possible feed forward layers
        self.fc1_features = [1 * 26 * 26, NUM_OUTPUTS, True]
        self.fc1 = nn.Linear(self.fc1_features[0], self.fc1_features[1])
        self.fc2_features = [1, 1, False]
        self.fc2 = nn.Linear(self.fc2_features[0], self.fc2_features[1])
        self.fc3_features = [1, 1, False]
        self.fc3 = nn.Linear(self.fc3_features[0], self.fc3_features[1])
        self.fc4_features = [1, 1, False]
        self.fc4 = nn.Linear(self.fc4_features[0], self.fc4_features[1])
        self.fc5_features = [1, 1, False]
        self.fc5 = nn.Linear(self.fc5_features[0], self.fc5_features[1])
        self.fc6_features = [1, 1, False]
        self.fc6 = nn.Linear(self.fc6_features[0], self.fc6_features[1])
        self.fc7_features = [1, 1, False]
        self.fc7 = nn.Linear(self.fc7_features[0], self.fc7_features[1])
        self.fc8_features = [1, 1, False]
        self.fc8 = nn.Linear(self.fc8_features[0], self.fc8_features[1])
        self.fc9_features = [1, 1, False]
        self.fc9 = nn.Linear(self.fc9_features[0], self.fc9_features[1])
        self.fc10_features = [1, 1, False]
        self.fc10 = nn.Linear(self.fc10_features[0], self.fc10_features[1])

    # this sets up the network so that it ignores layers that have yet to be introduced and matches layers properly.
    # self the network we are dealing with.
    # conv_layers a list of lists for every convolution layer with [[num_inputs, num_outputs, size fo convolution], ... []]
    # fc_layers is a list of lists for every feed forward layer[[num_inputs, num_outputs]...[]]
    def set_up_net(self, conv_layers, fc_layers):

        for i in range(0, conv_layers):
            if i == 0:
                self.conv1_features = conv_layers[i]
                self.conv1 = nn.Conv2d(conv_layers[i][0], conv_layers[i][1], conv_layers[i][2])
            elif i == 1:
                self.conv2_features = conv_layers[i]
                self.conv2 = nn.Conv2d(conv_layers[i][0], conv_layers[i][1], conv_layers[i][2])
            elif i == 2:
                self.conv3_features = conv_layers[i]
                self.conv3 = nn.Conv2d(conv_layers[i][0], conv_layers[i][1], conv_layers[i][2])
            elif i == 3:
                self.conv4_features = conv_layers[i]
                self.conv4 = nn.Conv2d(conv_layers[i][0], conv_layers[i][1], conv_layers[i][2])

        for i in range(0, len(fc_layers)):
            if i == 0:
                self.fc1 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 1:
                self.fc2 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 2:
                self.fc3 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 3:
                self.fc4 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 4:
                self.fc5 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 5:
                self.fc6 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 6:
                self.fc7 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 7:
                self.fc8 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 8:
                self.fc9 = nn.Linear(fc_layers[i][0], fc_layers[i][1])
            elif i == 9:
                self.fc10 = nn.Linear(fc_layers[i][0], fc_layers[i][1])

    # fills the layer with the wanted parameters
    # layer  determines the layer that is going to be corrected [layer_type, layer_num]
    # weight the weight value thats being filled into the parameter, pass in as tensor
    # bias the bias value thats being filled into the parameter, pass in as tensor
    def fill_layer_weights(self, layer, value, bias):
        # shape of value tensor (out_channels, in_channels, kernel_size[0], kernel_size[1])
        # value tensor = [[[[kernel_size[0] x kernel_size[1] x in_channels] x out_channels]]]
        # shape of bias tensor (out_channels)
        if layer[0] == "conv":
            if layer[1] == 0:

                self.conv1.weight.data = value
                self.conv1.bias.data = bias
            elif layer[1] == 1:
                self.conv2.weight.data = value
                self.conv2.bias.data = bias
            elif layer[1] == 2:
                self.conv3.weight.data = value
                self.conv3.bias.data = bias
            elif layer[1] == 3:
                self.conv4.weight.data = value
                self.conv4.bias.data = bias
        # weight – the learnable weights of the module of shape (out_features x in_features)
        # weight – the learnable weights of the module of shape (out_features x in_features)
        elif layer[0] == "fc":
            if layer[1] == 0:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            elif layer[1] == 1:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            elif layer[1] == 2:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            elif layer[1] == 3:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            if layer[1] == 4:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            elif layer[1] == 5:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            elif layer[1] == 6:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            elif layer[1] == 7:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            if layer[1] == 8:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias
            elif layer[1] == 9:
                self.fc1.weight.data = value
                self.fc1.bias.data = bias

    # mom the smaller parent network, dad the larger parent network, probability mom is stronger than dad
    # mom and dad must have the same number of layers and same size convolution
    def breed(self, mom, dad):

        if mom.conv1_features[3]:
            mom_weight = mom.conv1.weight.data
            mom_bias = mom.conv1.bias.data
            features = mom.conv1_features
            dad_weight = dad.conv1.weight.data
            dad_bias = dad.conv1.bias.data

            for j in range(0, mom.conv2_features[1]):
                if GENERATION < SWITCH_BREEDING:
                    who_bias = random.uniform(0, 1)
                    if who_bias < .5:
                        self.conv2.bias.data = mom_bias[j]
                    else:
                        self.conv2.bias.data = dad_bias[j]
                else:
                    child_bias = random.gauss((mom_bias[j].item() + dad_bias[j].item()) / 2, BREEDING_SD)
                    child_bias = torch.Tensor(child_bias)
                    self.conv1.bias.data = child_bias

            self.conv1.weight.data = breed_conv(mom_weight, dad_weight, features)

        if mom.conv2_features[3]:
            mom_weight = mom.conv2.weight.data
            mom_bias = mom.conv2.bias.data
            features = mom.conv2_features
            dad_weight = dad.conv2.weight.data
            dad_bias = dad.conv2.bias.data

            for j in range(0, mom.conv2_features[1]):
                if GENERATION < SWITCH_BREEDING:
                    who_bias = random.uniform(0, 1)
                    if who_bias < .5:
                        self.conv2.bias.data = mom_bias[j]
                    else:
                        self.conv2.bias.data = dad_bias[j]
                else:
                    child_bias = random.gauss((mom_bias[j].item() + dad_bias[j].item()) / 2, BREEDING_SD)
                    child_bias = torch.Tensor(child_bias)
                    self.conv1.bias.data = child_bias

            print("staring weight breeding")
            self.conv1.weight.data = breed_conv(mom_weight, dad_weight, features)

        if mom.conv3_features[3]:
            mom_weight = mom.conv3.weight.data
            mom_bias = mom.conv3.bias.data
            features = mom.conv3_features
            dad_weight = dad.conv3.weight.data
            dad_bias = dad.conv3.bias.data

            for j in range(0, mom.conv3_features[1]):
                if GENERATION < SWITCH_BREEDING:
                    who_bias = random.uniform(0, 1)
                    if who_bias < .5:
                        self.conv3.bias.data = mom_bias[j]
                    else:
                        self.conv3.bias.data = dad_bias[j]
                else:
                    child_bias = random.gauss((mom_bias[j].item() + dad_bias[j].item()) / 2, BREEDING_SD)
                    child_bias = torch.Tensor(child_bias)
                    self.conv1.bias.data = child_bias

            self.conv1.weight.data = breed_conv(mom_weight, dad_weight, features)

        if mom.conv4_features[3]:
            mom_weight = mom.conv4.weight.data
            mom_bias = mom.conv4.bias.data
            features = mom.conv4_features
            dad_weight = dad.conv4.weight.data
            dad_bias = dad.conv4.bias.data

            for j in range(0, mom.conv4_features[1]):
                if GENERATION < SWITCH_BREEDING:
                    who_bias = random.uniform(0, 1)
                    if who_bias < .5:
                        self.conv4.bias.data = mom_bias[j]
                    else:
                        self.conv4.bias.data = dad_bias[j]
                else:
                    child_bias = random.gauss((mom_bias[j].item() + dad_bias[j].item()) / 2, BREEDING_SD)
                    child_bias = torch.Tensor(child_bias)
                    self.conv1.bias.data = child_bias

            print("staring weight breeding")
            self.conv1.weight.data = breed_conv(mom_weight, dad_weight, features)

        # parents can have a different number of nodes in a layer as long as the have the same number of layers
        # accept dads number of output nodes

        is_final_layer = not mom.fc2_features[2]
        if mom.fc1_features[2]:
            mom_weight = mom.fc1.weight.data
            mom_bias = mom.fc1.bias.data
            dad_weight = dad.fc1.weight.data
            dad_bias = dad.fc1.bias.data

            self.fc1_features = [mom.fc1_features[0], dad.fc1_features[1], True]
            if is_final_layer:
                self.fc1.weight.data = linear_breeding(mom_weight, dad_weight, mom.fc1_features,
                                                       dad.fc1_features, NUM_OUTPUTS)
            else:
                self.fc1.weight.data = linear_breeding(mom_weight, dad_weight, mom.fc1_features,
                                                       dad.fc1_features, dad.fc1_features[1])

            child_bias = []
            for j in range(0, mom.fc1_features[1]):
                if GENERATION < SWITCH_BREEDING:
                    if .5 > random.uniform(0, 1):
                        child_bias.append(torch.Tensor(mom_bias[j].item()))
                    else:
                        child_bias.append(torch.Tensor(dad_bias[j].item()))

    def get_num_features(self):
        # only doing square convolutions
        num_features = 28
        depth = 1
        if self.conv1_features[3]:
            num_features -= self.conv1_features[2] - 1
            num_features /= 2
            depth = self.conv1_features[1]

        if self.conv2_features[3]:
            num_features -= self.conv2_features[2] - 1
            num_features /= 2
            depth = self.conv1_features[1]
        if self.conv3_features[3]:
            num_features -= self.conv3_features[2] - 1
            num_features /= 2
            depth = self.conv1_features[1]
        if self.conv4_features[3]:
            num_features -= self.conv4_features[2] - 1
            num_features /= 2
            depth = self.conv1_features[1]

        return num_features * num_features * depth


if __name__ == "__main__":
    tempCNN = TerminalCNN()
    other_CNN = TerminalCNN()
    child_CNN = TerminalCNN()
    child_CNN.breed(tempCNN, other_CNN)
    print(tempCNN.fc1.weight.data)
    print(other_CNN.conv1.weight.data)
    print(child_CNN.conv1.weight.data)
    # print(tempCNN.conv2.weight.data)
    # print(tempCNN.conv3.weight.data)
    # print(tempCNN.conv3.weight.data[0][0][0][0].item())
