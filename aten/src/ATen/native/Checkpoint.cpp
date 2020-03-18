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

Tensor checkpoint_add(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
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
  CheckpointTensorImpl::mutate("add_", mt, {a, b});
  return a;
}

Tensor checkpoint_abs(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::abs(vec.at(0))};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("abs", rt, s)[0];
}

Tensor checkpoint_div(at::Tensor const& a, at::Tensor const& b) {
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
  CheckpointTensorImpl::mutate("div_", mt, {a, b});
  return a;
}

Tensor checkpoint_constant_pad_nd(Tensor const& a, c10::ArrayRef<long> b, c10::Scalar c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::constant_pad_nd(vec[0], b_, c)};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("constant_pad_nd", rt, s)[0];
}

Tensor checkpoint_binary_cross_entropy(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, long d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy(vec[0], vec[1], vec[2], d)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c)};
  return CheckpointTensorImpl::make("binary_cross_entropy", rt, s)[0];
}

Tensor& checkpoint_binary_cross_entropy_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c, at::Tensor const& d, long e) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_out(self, vec.at(1), vec.at(2), vec.at(3), e);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {a, b, c, d});
  return a;
}

Tensor checkpoint_binary_cross_entropy_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, at::Tensor const& d, long e) { 
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy_backward(vec[0], vec[1], vec[2], vec[3], e)};
    };
  strongs s = {from_tensor(a), from_tensor(b), from_tensor(c), from_tensor(d)};
  return CheckpointTensorImpl::make("binary_cross_entropy_backward", rt, s)[0];
}

Tensor& checkpoint_binary_cross_entropy_backward_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c, at::Tensor const& d, at::Tensor const& e, long f) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::binary_cross_entropy_backward_out(self, vec.at(1), vec.at(2), vec.at(3), vec.at(4), f);
    };
  CheckpointTensorImpl::mutate("binary_cross_entropy_backward_out", mt, {a, b, c, d, e});
  return a;
}

Tensor checkpoint_embedding(at::Tensor const& a, at::Tensor const& b, long c, bool d, bool e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding(vec[0], vec[1], c, d, e)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("embedding", rt, s)[0];
}

Tensor checkpoint_embedding_backward(at::Tensor const& a, at::Tensor const& b, long c, long d, bool e, bool f) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding_backward(vec[0], vec[1], c, d, e, f)};
    };
  strongs s = {from_tensor(a), from_tensor(b)};
  return CheckpointTensorImpl::make("embedding", rt, s)[0];
}

}}
