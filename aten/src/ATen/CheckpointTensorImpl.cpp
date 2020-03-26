#include <ATen/CheckpointTensorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace at {

struct DTRLogger {
  std::ofstream out;
  static std::string get_filename() {
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    std::string str =
      std::to_string(1900+tm->tm_year) + "-" +
      std::to_string(1+tm->tm_mon) + "-" +
      std::to_string(tm->tm_mday) + "-" +
      std::to_string(tm->tm_hour) + "-" +
      std::to_string(tm->tm_min) + "-" +
      std::to_string(tm->tm_sec) + ".log";
    return str;
  }
  DTRLogger() : out(get_filename()) { }
};

void DTRLog(const std::string& str) {
  static DTRLogger logger;
  logger.out << str << std::endl;
}

int CheckpointTensorImpl::counter = 0;

Tensor checkpoint_raw(const Tensor& t) {
  return Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t.detach()));
}

std::tuple<Tensors, duration_t> make_raw(const rematerialize_function_t& remat,
                 const strongs& input_values) {
  std::vector<Tensor> input;
  for (const strong& s: input_values) {
    CHECK(!s->t.key_set().has(DispatchKey::CheckpointTensorId));
    input.push_back(s->t);
  }
  time_t pre = std::chrono::system_clock::now();
  auto output = remat(input);
  time_t post = std::chrono::system_clock::now();
  Tensors ret;
  for (const Tensor& o: output) {
    ret.push_back(checkpoint_raw(o));
  }
  return {ret, post - pre};
}

std::string from_time(duration_t t) {
  return std::to_string(std::chrono::nanoseconds(t).count());
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   const Tensors& input) {
  strongs input_values;
  std::string arg = name + "(";
  for (const Tensor& t: input) {
    auto ft = from_tensor(t);
    input_values.push_back(std::get<0>(ft));
    arg += std::get<1>(ft);
    arg += ", ";
  }
  arg += ")";
  std::string log = "(";
  auto ret = make_raw(remat, input_values);
  for (const Tensor& t: std::get<0>(ret)) {
    log += get_cpti(t)->counter_name();
    log += ", ";
  }
  log += ") = ";
  log += arg;
  log += " TIME: ";
  log += from_time(std::get<1>(ret));
  DTRLog(log);
  for (const Tensor& t: std::get<0>(ret)) {
    auto cpti = get_cpti(t);
    DTRLog(cpti->counter_name() + " MEMORY: " + std::to_string(cpti->ref->value->memory()));
  }
  return std::get<0>(ret);
}

void CheckpointTensorImpl::mutate(const std::string& name,
                                  const mutate_function_t& mutate,
                                  const Tensors& inputs,
                                  const std::vector<size_t>& mutate_idx) {
  auto remat = [=](const Tensors& t) -> Tensors {
                 Tensors new_input_values = t;
                 for (size_t idx: mutate_idx) {
                   new_input_values[idx] = t[idx].clone();
                 }
                 mutate(new_input_values);
                 return new_input_values;
               };
  strongs input_values;
  std::string log = name;
  log += "(";
  for (const Tensor& t : inputs) {
    auto ft = from_tensor(t);
    log += std::get<1>(ft);
    log += ", ";
    input_values.push_back(std::get<0>(ft));
  }
  log += ")";
  auto ret = make_raw(remat, input_values);
  log += " TIME: ";
  log += from_time(std::get<1>(ret));
  DTRLog(log);
  const auto& modified = std::get<0>(ret);
  for (size_t idx: mutate_idx) {
    cell_from_tensor(inputs[idx])->value = cell_from_tensor(modified[idx])->value;
  }
}

}
