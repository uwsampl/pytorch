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

int CheckpointTensorCell::counter = 0;

Tensors make_raw(const rematerialize_function_t& remat,
                 const strongs& input_values) {
  std::vector<Tensor> input;
  for (const strong& s: input_values) {
    CHECK(!s->t.key_set().has(DispatchKey::CheckpointTensorId));
    input.push_back(s->t);
  }
  auto output = remat(input);
  Tensors ret;
  for (const Tensor& o: output) {
    ret.push_back(native::checkpoint(o));
  }
  return ret;
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   const strongs& input_values) {
  Tensors ret = make_raw(remat, input_values);
  std::string log("(");
  for (const Tensor& t: ret) {
    log += cell_from_tensor(t)->value->name();
    log += ", ";
  }
  log += ") = ";
  log += name;
  log += "(";
  for (const strong& s: input_values) {
    log += s->name();
    log += ", ";
  }
  log += ")";
  DTRLog(log);
  return ret;
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
  for (const Tensor& t : inputs) {
    input_values.push_back(from_tensor(t));
  }
  auto modified = make_raw(remat, input_values);
  DTRLog(name);
  for (size_t idx: mutate_idx) {
    cell_from_tensor(inputs[idx])->value = cell_from_tensor(modified[idx])->value;
  }
}

}
