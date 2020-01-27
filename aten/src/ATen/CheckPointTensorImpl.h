#pragma once

#include <atomic>
#include <memory>
#include <numeric>

#include <c10/core/Backend.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>
#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/native/CheckPoint.h>

namespace at {

template<typename T>
struct EquivalentClass;
template<typename T>
struct CAFFE2_API EquivalentClassNode {
  friend EquivalentClass<T>;
  explicit EquivalentClassNode(const T& t) : t(t) { }
  EquivalentClassNode* parent = nullptr;
  bool is_root() {
    return parent == nullptr;
  }
  EquivalentClassNode* find_root() {
    if (is_root()) {
      return this;
    } else {
      parent = parent->find_root();
      return parent;
    }
  }
  T& get_t() {
    return find_root()->t;
  }
  void update_t(const T& t) {
    this->t = t;
  }
 private:
  T t;
};

template<typename T>
struct CAFFE2_API EquivalentClass {
  std::vector<EquivalentClassNode<T>*> nodes;
  EquivalentClassNode<T>* merge(const std::function<T(const T&, const T&)>& merge_t,
                                EquivalentClassNode<T>* lhs,
                                EquivalentClassNode<T>* rhs) {
    auto l = lhs->find_root();
    auto r = rhs->find_root();
    if (l == r) {
      return l;
    }
    l->parent = r;
    r->t = merge_t(l->t, r->t);
    return r;
  }
  EquivalentClassNode<T>* create_node(const T& t) {
    nodes.push_back(new EquivalentClassNode<T>(t));
    return nodes.back();
  }
  void clear() {
    for (auto* ecn : nodes) {
      delete ecn;
    }
    nodes.clear();
  }
  ~EquivalentClass() {
    clear();
  }
};

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;

// data CPI = Unevicted | Evicted Compute Memoy LastAccess | Invalid.
enum class CheckPointInfoState {
  Unevicted, // It is not evicted and what is in it is the output of a function. Might be more then one as function can return tuple.
  Evicted, // It is evicted and we store some summary about all the tensors.
  Invalid // It is evicted but the summary got out of date.
};
struct CAFFE2_API CheckPointInfo {
  CheckPointInfoState cpis;
  std::size_t memory;
  duration_t compute_cost;
  time_t last_used_time;
  // @ZACH: Floating Point instability?
  double score(time_t current_time) const {
    TORCH_CHECK(cpis == CheckPointInfoState::Evicted);
    TORCH_CHECK(memory > 0);
    auto staleness = (current_time - last_used_time).count();
    TORCH_CHECK(staleness > 0);
    return compute_cost.count() / static_cast<double>(memory * staleness);
  }
  static CheckPointInfo unevicted() {
    CheckPointInfo ret;
    ret.cpis = CheckPointInfoState::Unevicted;
    return ret;
  }
  static CheckPointInfo invalid() {
    CheckPointInfo ret;
    ret.cpis = CheckPointInfoState::Invalid;
    return ret;
  }
  static CheckPointInfo evicted(size_t memory, duration_t compute_cost, time_t last_used_time) {
    CheckPointInfo ret;
    ret.memory = memory;
    ret.compute_cost = compute_cost;
    ret.last_used_time = last_used_time;
    ret.cpis = CheckPointInfoState::Evicted;
    return ret;
  }
};

class Rematerializer;
class CheckPointTensorCell;
using strong = intrusive_ptr<CheckPointTensorCell>;
using strongs = std::vector<strong>;
using weak = weak_intrusive_ptr<CheckPointTensorCell>;
using weaks = std::vector<weak>;
using Tensors = std::vector<Tensor>;
using rematerialize_function_t = std::function<Tensors(const Tensors&)>;

struct RegisterEvict {
  RegisterEvict();
};
static RegisterEvict register_evict;

// A global CheckPointPool.
// Keep track of CheckPoint Tensor, and decide what to evict.
// As it hold lots of weak pointer, it is recommended to call clear()
// after every forward/backward pass to improve performance.
struct CAFFE2_API CheckPointPool {
  CheckPointPool() {
    tensors.reserve(10 * 1000);
  }
  static CheckPointPool& singleton();
  long epoch = 0;
  // Clear all bookkeeping.
  void clear();
  weaks tensors;
  EquivalentClass<CheckPointInfo> ec;
  void evict();
  bool should_evict();
  void auto_evict();
  bool has_banishing = true;
  bool has_memory_budget = false;
  long memory_budget;
  bool has_batched_eviction_factor = false;
  double batched_eviction_factor;
  bool has_ignore_small_tensor = false;
  long ignore_small_tensor;
};

// The Rematerializer can be in two state: prepared, and unprepared.
// remat() go from prepared to unprepared, while prepare() go from unprepared to prepared.
// However, a call to prepare() might fail while a call to remat() should not.
struct Rematerializer : intrusive_ptr_target {
  rematerialize_function_t rematerialize_function;
  weaks inputs, outputs;
  strongs input_values;
  duration_t compute_cost;
  EquivalentClassNode<CheckPointInfo>* ecn = nullptr;
  EquivalentClassNode<CheckPointInfo>* get_ecn();
  bool prepared() { return !input_values.empty(); }
  void prepare();
  void unprepare() { input_values.clear(); }
  // helper function for remat.
  // result, before_call, after_call
  std::tuple<std::vector<Tensor>, time_t, time_t> calculate();
  void remat();
  void release_resources() override;
  Rematerializer(const rematerialize_function_t& remat, weaks inputs);
  Rematerializer(const rematerialize_function_t& remat, strongs inputs);
};

struct Unsafe {};
// A CheckPoint Tensor.
// Hold a Tensor Inside, that it is free to release and rematerialize later if needed.
// The tensor inside cannot has any mutation, alias, or autodiff, as they interfere with rematerialization.
// A CheckPoint Tensor can has two state:
// 0: It hold the value
// 1: It hold a function that can compute those value.
struct CAFFE2_API CheckPointTensorCell final : public intrusive_ptr_target {
  long epoch = CheckPointPool::singleton().epoch;
  c10::optional<std::vector<EquivalentClassNode<CheckPointInfo>*>> neighbors();
  void fill(const Tensor& t, const weak_intrusive_ptr<CheckPointTensorCell>& self, time_t before_call, time_t after_call);
  explicit CheckPointTensorCell(const Tensor& t);
  explicit CheckPointTensorCell(const UndefinedTensorImpl&);
  /* This function is unsafe because it cannot obtain a weak_ptr of itself inside the constructor.
   * Thus, it is unable to register itself at tensors.
   * It cannot be private because make_intrusive require public constructor.
   */
  explicit CheckPointTensorCell(const intrusive_ptr<Rematerializer>& remat, bool is_evictable, const Unsafe&);
  // A Tensor might be an UndefinedTensorImpl!
  bool can_evict;
  DispatchKeySet key_set;
  caffe2::TypeMeta dtype;
  optional<Device> device;
  bool is_undefined = true;
  int64_t dim;
  int64_t numel;
  // IntArrayRef might go out of life.
  std::vector<int64_t> strides;
  std::vector<int64_t> sizes;
  // The following fields may be invalid data.
  // Technically speaking, we dont have to wrap it in another smart pointer,
  // as Tensor is merely a smart pointer to TensorImpl.
  // However, this had helped code organization/debugging a lot and get rid of all the undefined tensor issues.
  std::unique_ptr<Tensor> t;
  // Invariant: !t -> remat
  intrusive_ptr<Rematerializer> remat;
  // There is no compute as it is shared between tuple outputs. It is only stored in Remterializer.
  size_t memory;
  time_t last_used_time;
  // This ecn is the core of the dynamic tensor rematerialization.
  // Every CPTC should have a non nullptr ecn, and if there is two CPTC where,
  // in order to recalculate one we must recalculate the other,
  // (e.g. they are both evicted, and one is the input of another, or they are output of a tuple).
  // they will be in a same ecn.
  // We assume when a CPTC in ecn get rematerialized, every single CPTC in it will also do so.
  // This is not strictly true, but:
  // It greatly simplify design and implementation
  // 0: It is true in linear evolution
  // 1: Neural Network is mostly linear evolution so it is mostly true
  // 2: We have written code for when the assumption break. We optimize under the assumption so it is not as fast, but it should be ok.
  // Note that ecn will be a nullptr iff it is an input, iff can_evict is false.
  EquivalentClassNode<CheckPointInfo>* ecn = nullptr;
  weaks dependents;
  // Assert that this is in a valid state.
  void check();
  // Evict the Tensor.
  void evict();
  // Get the tensor, rematerialize it if necessary.
  Tensor get(const weak_intrusive_ptr<CheckPointTensorCell>& self);
  void update_metadata();
  void release_resources() override;
  // The cpi if this tensor got evicted.
  CheckPointInfo evict_cpi();
};

struct CheckPointTensorImplCell : intrusive_ptr_target {
  mutable intrusive_ptr<CheckPointTensorCell> value;
  explicit CheckPointTensorImplCell(const intrusive_ptr<CheckPointTensorCell>& value) : value(value) { }
  void release_resources() final {
    value.reset();
  }
};

using mutate_function_t = std::function<void(const Tensors&)>;
// A CheckPointTensorImpl is simply a reference to CheckPointTensorCell.
// As it provide the same interface (without mutable View, which we will not use),
// one could just wrap Variable around this for ad.
struct CAFFE2_API CheckPointTensorImpl final : public TensorImpl {
  intrusive_ptr<CheckPointTensorImplCell> ref;
  void release_resources() final;
  explicit CheckPointTensorImpl(const intrusive_ptr<CheckPointTensorImplCell>& ref);
  explicit CheckPointTensorImpl(const Tensor& t);
  static Tensors make(const rematerialize_function_t& remat,
                      const strongs& input_values,
                      bool is_evictable=true);
  // Just like make, but try to do inplace update to save memory/efficiency.
  // assume the zeroth argument get inplace updated, but nothing else, which is pytorch calling convention.
  // There is a slow path and a fast path.
  // The slow path call Tensor::make, with the rematerialization function being a clone and a in place update on the clone.
  // Then the zeroth argument will update the pointer to point to the new value.
  // The fast path happens when, upon taking the slow path, the raw tensor of the original zeroth input will get gced.
  // It mean it will happens when banishing is on, the zero argument's cell only has one use_count (from the reference),
  // and the raw tensor only have one use_count (no split or view)
  // If so we just steal the tensor, banish the cell, then apply the function.
  static void mutate(const mutate_function_t& mutate,
                     const Tensors& input_values,
                     bool is_evictable=true);
  explicit CheckPointTensorImpl(const intrusive_ptr<CheckPointTensorCell>& cell);
  explicit CheckPointTensorImpl(const UndefinedTensorImpl&);
  IntArrayRef strides() const override;
  bool is_contiguous(at::MemoryFormat memory_format=at::MemoryFormat::Contiguous) const override;
  void resize_dim(int64_t ndim) override;
  void set_size(int64_t dim, int64_t new_size) override;
  void set_stride(int64_t dim, int64_t new_stride) override;
  void set_storage_offset(int64_t storage_offset) override;
  int64_t dim() const override;
  bool has_storage() const override;
  const Storage& storage() const override;
  int64_t storage_offset() const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;
  intrusive_ptr<TensorImpl> shallow_copy_and_detach(const VariableVersion& version_counter,
                                                    bool allow_tensor_metadata_change) const override;
  int64_t numel() const override;
  IntArrayRef sizes() const override;
  int64_t size(int64_t d) const override;
};

inline CheckPointTensorImpl* get_cpti(const Tensor& t) {
  auto* cpti = dynamic_cast<CheckPointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  return cpti;
}

inline strong from_tensor(const Tensor& t) {
  auto* cpt = dynamic_cast<CheckPointTensorImpl*>(t.unsafeGetTensorImpl());
  if(cpt != nullptr) {
    return cpt->ref->value;
  } else {
    return get_cpti(native::checkpoint(t))->ref->value;
  }
}

inline strong from_tensor_maybe(const Tensor& t) {
  auto* ut = dynamic_cast<UndefinedTensorImpl*>(t.unsafeGetTensorImpl());
  if (ut != nullptr) {
    return strong::make(*ut);
  }
  return from_tensor(t);
}

inline Tensor get(const strong& s) {
  return s->get(weak(s));
}

inline intrusive_ptr<CheckPointTensorImplCell> cell_from_tensor(const Tensor& t) {
  return get_cpti(t)->ref;
}

} // namespace at
