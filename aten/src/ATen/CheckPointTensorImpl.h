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
#include <ATen/ATen.h>
#include <ATen/native/CheckPoint.h>

namespace at {

struct CAFFE2_API CheckPointTensorImpl final : public TensorImpl {
  Tensor t;
  explicit CheckPointTensorImpl(const Tensor& t) :
    TensorImpl(t.key_set(), t.dtype(), t.optional_device()),
    t(t) { }
};

}
