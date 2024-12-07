#include <torch/extension.h>
torch::Tensor bmm(torch::Tensor d_input0, torch::Tensor d_input1);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm", &bmm, "Batch Matrix multiplication");
}
