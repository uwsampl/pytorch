#pragma once

#include <c10/util/variant.h>

#include "basic.hpp"
#include "bag.hpp"
#include "fake_kinetic.hpp"
#include "kinetic.hpp"


enum class KineticHeapImpl {
  Bag = 0,
  Heap,
  Hanger
};


template<KineticHeapImpl impl, typename T, typename NotifyIndexChanged=NotifyKineticHeapIndexChanged<T>, typename GetRepresentative=GetTRepresentative<T>>
using KineticHeap = c10::variant_alternative_t<
  (size_t)impl,
  c10::variant<
    HeapImpls::Bag<T, NotifyIndexChanged>,
    HeapImpls::KineticMinHeap<T, false, NotifyIndexChanged, GetRepresentative>,
    HeapImpls::KineticMinHeap<T, true , NotifyIndexChanged, GetRepresentative>
  >
>;