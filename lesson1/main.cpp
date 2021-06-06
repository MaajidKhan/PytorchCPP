#include "iostream"
#include "torch/script.h"

int main() {
  torch::jit::script::Module net = torch::jit::load("../models/net.pt");
  torch::Tensor x = torch::randn({1, 100});
  std::vector<torch::jit::IValue> input;
  std::vector<torch::jit::IValue>::iterator itr;
  input.push_back(x);

  for(itr = input.begin(); itr!= input.end(); itr++ ) {
    std::cout << "itr: " << *itr << std::endl;
  }

  auto out = net.forward(input);
  std::cout << "out: " << out << std::endl;
  std::cout << typeid(out).name();
}
