#pragma once

// Bounded lock-free single-producer/single-consumer ring (Vyukov bounded
// queue). Contract follows the REQUIREMENTS/CLAIMS discipline of
// WyattAu/shm-rings: power-of-two capacity, cache-line-separated cursors,
// acquire/release ordering, non-blocking push/pop, sequential consistency
// never required. Implementation is original C++ (docs/PROVENANCE.md).

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace lo {

template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

    // Each cell owns a full cache line: the sequence word, the payload, and
    // the cursors never share a line (WyattAu/shm-rings contract). Without
    // alignas here, cells_ packs directly after tail_ (offset 72 in practice)
    // and the first cells share the consumer cursor's line.
    struct alignas(64) Cell {
        std::atomic<std::size_t> seq;
        T value;
    };

    alignas(64) std::atomic<std::size_t> head_{0};  // producer cursor
    alignas(64) std::atomic<std::size_t> tail_{0};  // consumer cursor
    Cell cells_[Capacity];

public:
    SpscRing() {
        for (std::size_t i = 0; i < Capacity; ++i)
            cells_[i].seq.store(i, std::memory_order_relaxed);
    }

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    // Producer only. Returns false when the ring is full.
    bool push(const T& v) {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        Cell& c = cells_[h & (Capacity - 1)];
        const std::size_t seq = c.seq.load(std::memory_order_acquire);
        if (static_cast<std::intptr_t>(seq) != static_cast<std::intptr_t>(h)) return false;
        c.value = v;
        c.seq.store(h + 1, std::memory_order_release);
        head_.store(h + 1, std::memory_order_relaxed);
        return true;
    }

    // Consumer only. Returns false when the ring is empty.
    bool pop(T& out) {
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        Cell& c = cells_[t & (Capacity - 1)];
        const std::size_t seq = c.seq.load(std::memory_order_acquire);
        if (static_cast<std::intptr_t>(seq) != static_cast<std::intptr_t>(t + 1)) return false;
#if defined(__GNUC__)
        // Compiler barrier. Optimizers speculate plain loads across branches;
        // a hoisted value copy would execute unsynchronized on failed
        // iterations — a machine-level race TSan correctly reports. The
        // clobber pins the copy below the seq guard.
        asm volatile("" ::: "memory");
#endif
        out = c.value;
        c.seq.store(t + Capacity, std::memory_order_release);
        tail_.store(t + 1, std::memory_order_relaxed);
        return true;
    }
};

}  // namespace lo
