#include <torch/torch.h>
#include <iostream>

int main() {
  Net net(4, 5);
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }
}