#pragma once

// Language-version portability shims (plan of record, section 1).
// Prefer ISO feature-test macros; degrade to compiler builtins; never
// substitute a hard failure for a weaker contract.

#include <cassert>
#include <cstdint>

namespace lo {

#if defined(__cpp_contracts)
#define LO_ASSERT(cond) [[assert: cond]]
#else
#define LO_ASSERT(cond) assert(cond)
#endif

#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(assume)
#define LO_ASSUME(cond) [[assume(cond)]]
#endif
#endif

#if !defined(LO_ASSUME)
#if defined(__GNUC__)
#define LO_ASSUME(cond)        \
    do {                       \
        if (!(cond))           \
            __builtin_unreachable(); \
    } while (0)
#else
#define LO_ASSUME(cond) assert(cond)
#endif
#endif

inline constexpr std::uint32_t kApiVersion = 0;

}  // namespace lo
