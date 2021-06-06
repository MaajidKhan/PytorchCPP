#include "../include/network.h"

#include <iostream>
#include <torch/torch.h>

using namespace torch;

int main() {
    Net network(50,10);
    std::cout << "network: " << network << "\n\n";
    Tensor x, output;
    x = torch::randn({2, 50});
    output = network->forward(x);
    std::cout << "final output: " << output << std::endl;
}