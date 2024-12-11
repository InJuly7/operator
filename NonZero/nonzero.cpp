#include <torch/extension.h>
torch::Tensor nonzero(torch::Tensor d_input0);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nonzero", &nonzero, "Nonzero Kernel Implement");
}
