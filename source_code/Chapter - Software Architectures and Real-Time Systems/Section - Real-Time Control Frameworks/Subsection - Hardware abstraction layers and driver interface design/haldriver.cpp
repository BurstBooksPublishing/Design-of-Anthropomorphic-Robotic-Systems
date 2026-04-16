cpp
#include <atomic>
#include <cstdint>
#include <cstddef>

// Assuming these are defined elsewhere in the project
struct SensorSample;
struct ActCmd;
constexpr size_t BUFFER_SIZE = 256;
extern "C" SensorSample read_hw_registers();
extern "C" uint64_t hw_timestamp();
extern ActCmd translate_sample_to_command(const SensorSample&);
extern ActCmd last_cmd;

class JointDriver {
  // Pre-allocated buffers and atomic flags (no heap in RT path)
  alignas(64) SensorSample samples[BUFFER_SIZE]; // pad for cache
  std::atomic<size_t> write_idx{0};              // RT writer index
  std::atomic<size_t> read_idx{0};               // non-RT reader index

public:
  // Real-time ISR: minimal, bounded WCET, no locks, no heap
  void isr_handler() noexcept {
    SensorSample s = read_hw_registers();        // ~C_driver
    s.t = hw_timestamp();                        // monotone timestamp
    size_t w = write_idx.load(std::memory_order_relaxed);
    samples[w % BUFFER_SIZE] = s;                // overwrite if full
    write_idx.store(w + 1, std::memory_order_release);
    // schedule offload to non-RT thread via eventfd (bounded)
  }

  // Non-RT HAL translation: safe conversions, calibration applied
  ActCmd get_command() {
    // atomic snapshot of indices
    size_t r = read_idx.load(std::memory_order_acquire);
    size_t w = write_idx.load(std::memory_order_acquire);
    if (r == w) return last_cmd;                 // no new sample
    SensorSample s = samples[r % BUFFER_SIZE];   // safe copy
    read_idx.store(r + 1, std::memory_order_release);
    // Convert to SI units, apply calibration (non-RT cost)
    return translate_sample_to_command(s);
  }
};