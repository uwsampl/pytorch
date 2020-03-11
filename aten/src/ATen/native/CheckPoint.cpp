#include <ATen/ATen.h>
#include <ATen/CheckPointTensorImpl.h>

namespace at { namespace native {

Tensor checkpoint(const Tensor& t) {
  return Tensor(intrusive_ptr<CheckPointTensorImpl>::make(t.detach()));
}

Tensor decheckpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckPointTensorImpl*>(t.unsafeGetTensorImpl());
  CHECK(cpti != nullptr);
  return cpti->t;
}

bool is_checkpoint(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckPointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

Tensor checkpoint_add(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
  return checkpoint(at::add(decheckpoint(a), decheckpoint(b), c));
}

}}
