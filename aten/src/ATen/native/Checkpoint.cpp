#include <ATen/ATen.h>
#include <ATen/CheckpointTensorImpl.h>

namespace at { namespace native {

Tensor checkpoint(const Tensor& t) {
  return Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t.detach()));
}

Tensor decheckpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  CHECK(cpti != nullptr);
  return cpti->ref->value->t;
}

bool is_checkpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

Tensor checkpoint_add(const Tensor& a, const Tensor& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::add(vec.at(0), vec.at(1), c)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("add", rt, s)[0];
}

Tensor& checkpoint_add_(Tensor& a, const Tensor& b, Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).add_(vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("add_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_abs(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::abs(vec.at(0))};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("abs", rt, s)[0];
}

Tensor checkpoint_div(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec.at(0), vec.at(1))};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("div", rt, s)[0];
}

Tensor& checkpoint_div_(Tensor& a, const Tensor& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).div_(vec.at(1));
    };
  CheckpointTensorImpl::mutate("div_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_constant_pad_nd(Tensor const& a, c10::ArrayRef<long> b, c10::Scalar c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::constant_pad_nd(vec.at(0), b_, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("constant_pad_nd", rt, s)[0];
}

Tensor checkpoint_binary_cross_entropy(const Tensor& a, const Tensor& b, const Tensor& c, long d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  return CheckpointTensorImpl::make("binary_cross_entropy", rt, s)[0];
}

Tensor& checkpoint_binary_cross_entropy_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_out(self, vec.at(1), vec.at(2), vec.at(3), e);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {a, b, c, d}, {0});
  return a;
}

Tensor checkpoint_binary_cross_entropy_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e) { 
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), e)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(d)};
  return CheckpointTensorImpl::make("binary_cross_entropy_backward", rt, s)[0];
}

Tensor& checkpoint_binary_cross_entropy_backward_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_backward_out(self, vec.at(1), vec.at(2), vec.at(3), vec.at(4), f);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_backward_out", mt, {a, b, c, d, e}, {0});
  return a;
}

Tensor checkpoint_embedding(const Tensor& a, const Tensor& b, long c, bool d, bool e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding(vec.at(0), vec.at(1), c, d, e)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("embedding", rt, s)[0];
}

Tensor checkpoint_embedding_backward(const Tensor& a, const Tensor& b, long c, long d, bool e, bool f) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding_backward(vec.at(0), vec.at(1), c, d, e, f)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("embedding", rt, s)[0];
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
checkpoint_cudnn_batch_norm(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, bool f, double g, double h) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_batch_norm(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), f, g, h);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(d), from_tensor(e)};
  auto ret = CheckpointTensorImpl::make("cudnn_batch_norm", rt, s);
  return {ret[0], ret[1], ret[2], ret[3]};
}

Tensor checkpoint_as_strided(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::optional<long> d) {
  std::vector<long> b_ = b.vec(), c_ = c.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::as_strided(vec.at(0), b_, c_, d)};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("as_strided", rt, s)[0];
}

Tensor checkpoint__masked_scale(const Tensor& a, const Tensor& b, double c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_masked_scale(vec.at(0), vec.at(1), c)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("_masked_scale", rt, s)[0];
}

Tensor checkpoint_cudnn_convolution(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("cudnn_convolution", rt, s)[0];
}

Tensor checkpoint_cudnn_convolution_transpose(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose(vec.at(0), vec.at(1), c_, d_, e_, f_, g, h, i)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("cudnn_convolution_transpose", rt, s)[0];
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, std::array<bool, 2ul> j) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_convolution_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g, h, i, j);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  auto ret = CheckpointTensorImpl::make("cudnn_convolution_backward", rt, s);
  return {ret[0], ret[1]};
}

std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_transpose_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, long h, bool i, bool j, std::array<bool, 2ul> k) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_convolution_transpose_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g_, h, i, j, k);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  auto ret = CheckpointTensorImpl::make("cudnn_convolution_transpose_backward", rt, s);
  return {ret[0], ret[1]};
}

Tensor checkpoint_cudnn_convolution_backward_input(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_backward_input(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  strongs s = {from_tensor(b), from_tensor(c)};
  return CheckpointTensorImpl::make("cudnn_convolution_backward_input", rt, s)[0];
}

Tensor checkpoint_cudnn_convolution_transpose_backward_input(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose_backward_input(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_input", rt, s)[0];
}

Tensor checkpoint_cudnn_convolution_backward_weight(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  strongs s = {from_tensor(b), from_tensor(c)};
  return CheckpointTensorImpl::make("cudnn_convolution_backward_weight", rt, s)[0];
}

Tensor checkpoint_cudnn_convolution_transpose_backward_weight(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i) {
  std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i)};
    };
  strongs s = {from_tensor(b), from_tensor(c)};
  return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_weight", rt, s)[0];
}

Tensor checkpoint_relu(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::relu(vec.at(0))};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("relu", rt, s)[0];
}

Tensor& checkpoint_relu_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).relu_();
    };
  CheckpointTensorImpl::mutate("relu_", mt, {a}, {0});
  return a;
}

std::tuple<Tensor&, Tensor&> checkpoint_max_pool2d_with_indices_out(Tensor& a, Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0), b_ = vec.at(1);
      at::max_pool2d_with_indices_out(a_, b_, vec.at(2), d_, e_, f_, g_, h);
    };
  CheckpointTensorImpl::mutate("max_pool2d_with_indices_out", mt, {a, b, c}, {0, 1});
  return {a, b};
}

Tensor checkpoint_avg_pool2d(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, bool e, bool f, c10::optional<long> g) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d(vec.at(0), b_, c_, d_, e, f, g)};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("avg_pool2d", rt, s)[0];
}

Tensor checkpoint_avg_pool2d_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d_backward(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("avg_pool2d_backward", rt, s)[0];
}

Tensor& checkpoint_avg_pool2d_out(Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::avg_pool2d_out(a_, vec.at(1), c_, d_, e_, f, g, h);
    };
  CheckpointTensorImpl::mutate("avg_pool2d_out", mt, {a, b}, {0});
  return a;
}

Tensor& checkpoint_avg_pool2d_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, bool h, c10::optional<long> i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::avg_pool2d_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, i);
    };
  CheckpointTensorImpl::mutate("avg_pool2d_backward_grad_input", mt, {a, b, c}, {0});
  return a;
}

std::tuple<Tensor, Tensor> checkpoint_max_pool2d_with_indices(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::max_pool2d_with_indices(vec.at(0), b_, c_, d_, e_, f);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  strongs s = {from_tensor(a)};
  auto ret = CheckpointTensorImpl::make("max_pool2d_backward", rt, s);
  return {ret[0], ret[1]};
}

Tensor& checkpoint_max_pool2d_with_indices_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h, const Tensor& i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::max_pool2d_with_indices_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, vec.at(3));
    };
  CheckpointTensorImpl::mutate("max_pool2d_with_indices_backward_grad_input", mt, {a, b, c, i}, {0});
  return a;

}

Tensor checkpoint_max_pool2d_with_indices_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, const Tensor& h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::max_pool2d_with_indices_backward(vec.at(0), vec.at(1), c_, d_, e_, f_, g, vec.at(2))};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(h)};
  return CheckpointTensorImpl::make("max_pool2d_with_indices_backward", rt, s)[0];
}

}}
