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

Tensor checkpoint_abs(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::abs(vec.at(0))};
    };
  strongs s = {from_tensor(a)};
  return CheckpointTensorImpl::make("abs", rt, s)[0];
}

Tensor& checkpoint_add_(Tensor& a, const Tensor& b, Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).add_(vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("add_", mt, {a, b});
  return a;
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

}}
