#include <ATen/CheckPointTensorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace at {

void checkpoint_auto_evict() {
  CheckPointPool::singleton().auto_evict();
}

RegisterEvict::RegisterEvict() {
  c10::set_evict_func(checkpoint_auto_evict);
}

inline void gdb() {
  std::cout << *static_cast<int*>(nullptr) << std::endl;
}

CheckPointPool& CheckPointPool::singleton() {
  static CheckPointPool cpp;
  return cpp;
}

void CheckPointPool::clear() {
  tensors.clear();
  ec.clear();
}

CheckPointInfo merge(CheckPointInfo l, CheckPointInfo r) {
  TORCH_CHECK(l.cpis == CheckPointInfoState::Evicted);
  TORCH_CHECK(r.cpis == CheckPointInfoState::Evicted);
  return CheckPointInfo::evicted(l.memory + r.memory,
                                 l.compute_cost + r.compute_cost,
                                 std::max(l.last_used_time, r.last_used_time));
}

c10::optional<std::vector<EquivalentClassNode<CheckPointInfo>*>> CheckPointTensorCell::neighbors() {
  // todo: if invalid refresh
  using value_type = std::vector<EquivalentClassNode<CheckPointInfo>*>;
  using ret_type = c10::optional<value_type>;
  value_type neighbors;
  for (const auto& weak : remat->inputs) {
    auto strong = weak.lock();
    if (!strong.defined()) {
      return ret_type();
    }
    if (strong->ecn != nullptr && strong->ecn->get_t().cpis == CheckPointInfoState::Evicted) {
      neighbors.push_back(strong->ecn);
    }
  }
  for (const auto& weak : remat->outputs) {
    auto strong = weak.lock();
    if (strong.defined() && strong->ecn->get_t().cpis == CheckPointInfoState::Evicted) {
      neighbors.push_back(strong->ecn);
    }
  }
  for (const auto& weak : dependents) {
    auto strong = weak.lock();
    if (strong.defined() && strong->ecn->get_t().cpis == CheckPointInfoState::Evicted) {
      neighbors.push_back(strong->ecn);
    }
  }
  std::sort(neighbors.begin(), neighbors.end());
  neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  return neighbors;
}

void CheckPointPool::evict() {
  TORCH_CHECK(tensors.size() > 0);
  bool shrinked = false;
  int evict_idx = -1;
  double evict_score = INFINITY;
  time_t current_time = std::chrono::system_clock::now();
  auto remove_from_tensors = [&](size_t i) {
    tensors[i] = tensors[tensors.size() - 1];
    tensors.resize(tensors.size() - 1);
  };
  for (size_t i = 0; i < tensors.size();) {
    {
      auto cannot_evict = [&]() {
       shrinked = true;
       remove_from_tensors(i);
      };
      auto ti_strong = tensors[i].lock();
      if (!ti_strong.defined()) {
        cannot_evict();
        goto end_of_loop;
      }
      CheckPointTensorCell& cptc = *ti_strong;
      // calculate the cost of evicting.
      auto neighbors = cptc.neighbors();
      if (!neighbors.has_value()) {
        cannot_evict();
        goto end_of_loop;
      }
      CheckPointInfo cpi = cptc.evict_cpi();
      for (const auto& ecn : neighbors.value()) {
        if (ecn != nullptr) {
          cpi = merge(cpi, ecn->get_t());
        }
      }
      double cptc_evict_score = cpi.score(current_time);
      if (cptc_evict_score < evict_score) {
        evict_idx = i;
        evict_score = cptc_evict_score;
      }
      ++i;
    }
  // evict the node with the minimum score.
  end_of_loop:;
  }
  if (evict_idx < 0) {
    CHECK(shrinked);
  } else {
    auto ti_strong = tensors[evict_idx].lock();
    TORCH_CHECK(ti_strong.defined());
    ti_strong->evict();
    remove_from_tensors(evict_idx);
  }
}

CheckPointInfo CheckPointTensorCell::evict_cpi() {
  auto cpi = ecn->get_t();
  if (cpi.cpis == CheckPointInfoState::Unevicted) {
    return CheckPointInfo::evicted(memory, remat->compute_cost, last_used_time);
  } else if (cpi.cpis == CheckPointInfoState::Evicted) {
    return CheckPointInfo::evicted(memory + cpi.memory, cpi.compute_cost, std::max(last_used_time, cpi.last_used_time));
  }
  else {
    TORCH_CHECK(cpi.cpis == CheckPointInfoState::Invalid);
    TORCH_CHECK(false);
  }
}

void CheckPointTensorCell::evict() {
  TORCH_CHECK(can_evict);
  TORCH_CHECK(t, "t in evict is invalid");
  TORCH_CHECK(remat);
  remat->prepare();
  t.reset();
  auto neighbors = this->neighbors();
  TORCH_CHECK(neighbors.has_value());
  ecn->update_t(evict_cpi());
  for (const auto& n_ecn : neighbors.value()) {
    CheckPointPool::singleton().ec.merge(merge, ecn, n_ecn);
  }
}

Tensor CheckPointTensorCell::get(const weak_intrusive_ptr<CheckPointTensorCell>& self) {
  // it seems very strange to not just return *t.however auto evict might throw it out.
  Tensor ret;
  if (!t) {
    remat->remat();
    TORCH_CHECK(t, "t in get is invalid");
    ret = *t;
    TORCH_CHECK(ecn != nullptr);
    // ecn->get_t() = CheckPointInfo::invalid();
  } else {
    ret = *t;
  }
  last_used_time = std::chrono::system_clock::now();
  if (t && ecn != nullptr) {
    ecn->get_t().last_used_time = last_used_time;
  }
  return ret;
}

void CheckPointTensorCell::fill(const Tensor& t, const weak_intrusive_ptr<CheckPointTensorCell>& self, time_t before_call, time_t after_call) {
  if (!this->t) {
    this->t = std::make_unique<Tensor>(t);
    last_used_time = before_call;
    if (t.defined()) {
      memory = t.numel() * t.itemsize();
      if (memory > 0) {
        CheckPointPool::singleton().tensors.push_back(self);
      }
    }
    update_metadata();
    ecn = remat->get_ecn();
  }
}

void CheckPointTensorCell::release_resources() {
  t.reset();
  remat.reset();
}

void CheckPointTensorImpl::release_resources() {
  ref.reset();
}

TensorTypeSet convert_type_set(const TensorTypeSet& t) {
  return t.add(TensorTypeId::CheckPointTensorId).remove(TensorTypeId::VariableTensorId);
}

CheckPointTensorImpl::CheckPointTensorImpl(const Tensor& t) :
  TensorImpl(convert_type_set(t.type_set()), t.dtype(), t.optional_device()),
  ref(intrusive_ptr<CheckPointTensorImplCell>::make(intrusive_ptr<CheckPointTensorCell>::make(t))) {
}

CheckPointTensorImpl::CheckPointTensorImpl(const intrusive_ptr<CheckPointTensorCell>& cell) :
  TensorImpl(cell->type_set, cell->dtype, cell->device),
  ref(intrusive_ptr<CheckPointTensorImplCell>::make(cell)) {
  // This code must be put here instead of at CheckPointTensorCell
  // because in the constructor use_count is 0.
  if (const auto remat = ref->value->remat) {
    for (const auto& i : remat->inputs) {
      auto strong = i.lock();
      TORCH_CHECK(strong.defined());
      TORCH_CHECK(ref->value->ecn != nullptr);
      strong->dependents.push_back(weak(ref->value));
    }
  }
}

void CheckPointPool::auto_evict() {
  while (should_evict()) {
    evict();
  }
}

CheckPointTensorCell::CheckPointTensorCell(const intrusive_ptr<Rematerializer>& remat, const Unsafe&) :
  can_evict(true),
  remat(remat),
  ecn(remat->get_ecn()) {
  for (const auto& i : remat->input_values) {
    remat->inputs.push_back(weak(i));
  }
}

void CheckPointTensorCell::update_metadata() {
  if (t->defined()) {
    is_undefined = false;
    dim = t->dim();
    numel = t->numel();
    strides = t->strides().vec();
    sizes = t->sizes().vec();
  } else {
    numel = 0;
  }
  type_set = convert_type_set(t->type_set());
  dtype = t->dtype();
  device = t->optional_device();
}

CheckPointTensorCell::CheckPointTensorCell(const Tensor& t) :
  can_evict(false),
  t(std::make_unique<Tensor>(t)) {
  update_metadata();
}

CheckPointTensorCell::CheckPointTensorCell(const UndefinedTensorImpl&) :
  can_evict(false),
  t(std::make_unique<Tensor>()) {
}

// TODO: implement.
bool CheckPointPool::should_evict() {
  return tensors.size() > 10;
}

bool CheckPointTensorImpl::has_storage() const {
  return false;
}

bool CheckPointTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  // todo: fix
  return true;
}

int64_t CheckPointTensorImpl::dim() const {
  if(ref->value->is_undefined) {
    gdb();
  }
  TORCH_CHECK(!ref->value->is_undefined);
  return ref->value->dim;
}

const Storage& CheckPointTensorImpl::storage() const {
  gdb();
  AT_ERROR("checkpoint tensors do not have storage");
}

IntArrayRef CheckPointTensorImpl::sizes() const {
  if(ref->value->is_undefined) {
    gdb();
  }
  TORCH_CHECK(!ref->value->is_undefined);
  return ref->value->sizes;
}

int64_t CheckPointTensorImpl::size(int64_t d) const {
  AT_ERROR("size(dim) called on an checkpoint Tensor");
}

int64_t CheckPointTensorImpl::storage_offset() const {
  gdb();
  AT_ERROR("checkpoint tensors do not have storage_offset");
}

void CheckPointTensorImpl::resize_dim(int64_t ndim) {
  AT_ERROR("checkpoint tensors do not have resize_dim");
}

void CheckPointTensorImpl::set_size(int64_t dim, int64_t new_size) {
  AT_ERROR("checkpoint tensors do not have set_size");
}

void CheckPointTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  AT_ERROR("checkpoint tensors do not have set_stride");
}

void CheckPointTensorImpl::set_storage_offset(int64_t storage_offset) {
  AT_ERROR("checkpoint tensors do not have set_storage_offset");
}

IntArrayRef CheckPointTensorImpl::strides() const {
  if(ref->value->is_undefined) {
    gdb();
  }
  TORCH_CHECK(!ref->value->is_undefined);
  return ref->value->strides;
}

void CheckPointTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  AT_ERROR("checkpoint tensors do not have shallow_copy_from");
}

int64_t CheckPointTensorImpl::numel() const {
  if(ref->value->is_undefined) {
    gdb();
  }
  TORCH_CHECK(!ref->value->is_undefined);
  return ref->value->numel;
}

// is it safe to ignore version_counter?
intrusive_ptr<TensorImpl> CheckPointTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                                        bool allow_tensor_metadata_change) const {
  return intrusive_ptr<CheckPointTensorImpl>::make(ref);
}

CheckPointTensorImpl::CheckPointTensorImpl(const intrusive_ptr<CheckPointTensorImplCell>& ref) :
  TensorImpl(ref->value->type_set, ref->value->dtype, ref->value->device),
  ref(ref) {
}

Tensors CheckPointTensorImpl::make(const rematerialize_function_t& remat,
                                   const strongs& input_values) {
  auto rematerializer = intrusive_ptr<Rematerializer>::make(remat, input_values);
  auto res = rematerializer->calculate();
  Tensors ret;
  for (const Tensor& t : std::get<0>(res)) {
    strong cptc = strong::make(rematerializer, Unsafe{});
    cptc->fill(t, weak(cptc), std::get<1>(res), std::get<2>(res));
    rematerializer->outputs.push_back(weak(cptc));
    ret.push_back(Tensor(intrusive_ptr<CheckPointTensorImpl>::make(cptc)));
  }
  return ret;
}

void Rematerializer::prepare() {
  if (!prepared()) {
    for (const auto& i : inputs) {
      auto strong = i.lock();
      TORCH_CHECK(strong.defined());
      input_values.push_back(strong);
    }
  }
}

void Rematerializer::release_resources() {
  rematerialize_function = rematerialize_function_t();
  inputs.clear();
  outputs.clear();
  input_values.clear();
}

std::tuple<std::vector<Tensor>, time_t, time_t> Rematerializer::calculate() {
  TORCH_CHECK(prepared());
  std::vector<Tensor> args;
  for (const auto& iv : input_values) {
    args.push_back(iv->get(weak(iv)));
  }
  time_t before_call = std::chrono::system_clock::now();
  auto ret = rematerialize_function(args);
  time_t after_call = std::chrono::system_clock::now();
  unprepare();
  compute_cost = after_call - before_call;
  return {ret, before_call, after_call};
}

void Rematerializer::remat() {
  auto res = calculate();
  const auto& values = std::get<0>(res);
  TORCH_CHECK(values.size() == outputs.size());
  for (size_t i = 0; i < values.size(); ++i) {
    weak output = outputs[i];
    if (strong s = output.lock()) {
      s->fill(values[i], output, std::get<1>(res), std::get<2>(res));
    }
  }
}

Rematerializer::Rematerializer(const rematerialize_function_t& remat, weaks inputs) :
  rematerialize_function(remat),
  inputs(inputs),
  ecn(CheckPointPool::singleton().ec.create_node(CheckPointInfo::unevicted())) {
}

Rematerializer::Rematerializer(const rematerialize_function_t& remat, strongs inputs) :
  rematerialize_function(remat),
  input_values(inputs),
  ecn(CheckPointPool::singleton().ec.create_node(CheckPointInfo::unevicted())) {
  for (const strong& input : inputs) {
    this->inputs.push_back(weak(input));
  }
}

EquivalentClassNode<CheckPointInfo>* Rematerializer::get_ecn() {
  if (ecn == nullptr || ecn->get_t().cpis != CheckPointInfoState::Unevicted) {
    ecn = CheckPointPool::singleton().ec.create_node(CheckPointInfo::unevicted());
  }
  return ecn;
}

namespace native {
  int clear_checkpointpool() {
    CheckPointPool::singleton().clear();
    return 0;
  }
}

} // namespace c10
