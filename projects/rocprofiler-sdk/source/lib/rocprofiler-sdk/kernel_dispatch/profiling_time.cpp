// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/tracing/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <string_view>

namespace
{
// Disabled by default. The enqueue timestamp, not callback time, determines
// whether a dispatch belongs to the selected strict window.
std::atomic<uint64_t> selected_start{std::numeric_limits<uint64_t>::max()};
std::atomic<uint64_t> selected_end{0};
constexpr uint64_t strict_bound_wait_limit_ns = 100000;
std::atomic<uint64_t> future_bound_count{0};
std::atomic<uint64_t> future_bound_max_gap_ns{0};
std::atomic<uint64_t> future_bound_total_wait_ns{0};
}

extern "C" __attribute__((visibility("default"))) void
rocprofiler_experimental_set_dispatch_strict_window(uint64_t start, uint64_t end)
{
    if(start == 0 || start > end)
    {
        selected_start.store(std::numeric_limits<uint64_t>::max(), std::memory_order_release);
        selected_end.store(0, std::memory_order_release);
        return;
    }
    if(selected_start.load(std::memory_order_acquire) != start)
    {
        future_bound_count.store(0, std::memory_order_release);
        future_bound_max_gap_ns.store(0, std::memory_order_release);
        future_bound_total_wait_ns.store(0, std::memory_order_release);
    }
    selected_end.store(end, std::memory_order_release);
    selected_start.store(start, std::memory_order_release);
}

extern "C" __attribute__((visibility("default"))) void
rocprofiler_experimental_get_dispatch_strict_window(uint64_t* start, uint64_t* end)
{
    if(start) *start = selected_start.load(std::memory_order_acquire);
    if(end) *end = selected_end.load(std::memory_order_acquire);
}

extern "C" __attribute__((visibility("default"))) int
rocprofiler_experimental_dispatch_strict_window_contains(uint64_t enqueue_ts)
{
    const auto start = selected_start.load(std::memory_order_acquire);
    const auto end = selected_end.load(std::memory_order_acquire);
    return start <= end && enqueue_ts >= start && enqueue_ts <= end;
}

extern "C" __attribute__((visibility("default"))) void
rocprofiler_experimental_get_dispatch_strict_bound_wait(uint64_t* count,
                                                         uint64_t* max_gap_ns,
                                                         uint64_t* total_wait_ns,
                                                         uint64_t* limit_ns)
{
    if(count) *count = future_bound_count.load(std::memory_order_acquire);
    if(max_gap_ns) *max_gap_ns = future_bound_max_gap_ns.load(std::memory_order_acquire);
    if(total_wait_ns) *total_wait_ns = future_bound_total_wait_ns.load(std::memory_order_acquire);
    if(limit_ns) *limit_ns = strict_bound_wait_limit_ns;
}

namespace rocprofiler
{
namespace kernel_dispatch
{
profiling_time
get_dispatch_time(hsa_agent_t             _hsa_agent,
                  hsa_signal_t            _signal,
                  rocprofiler_kernel_id_t _kernel_id,
                  std::optional<uint64_t> _baseline)  // NOLINT(performance-unnecessary-value-param)
{
    auto dispatch_time        = hsa_amd_profiling_dispatch_time_t{};
    auto dispatch_time_status = hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
        _hsa_agent, _signal, &dispatch_time);
    // Bound the observation after the runtime query has returned. In the
    // opt-in selected window, give the system clock at most 100 us to catch up
    // with a GPU end timestamp; never change the raw GPU interval.
    const auto first_ts = common::timestamp_ns();
    auto ts = first_ts;
    const bool selected_strict =
        _baseline && rocprofiler_experimental_dispatch_strict_window_contains(*_baseline);
    if(selected_strict && dispatch_time_status == HSA_STATUS_SUCCESS && dispatch_time.end > ts)
    {
        const auto gap = dispatch_time.end - ts;
        future_bound_count.fetch_add(1, std::memory_order_relaxed);
        auto maximum = future_bound_max_gap_ns.load(std::memory_order_relaxed);
        while(maximum < gap && !future_bound_max_gap_ns.compare_exchange_weak(
                                   maximum, gap, std::memory_order_relaxed))
        {}
        const auto deadline = first_ts + strict_bound_wait_limit_ns;
        do
        {
            ts = common::timestamp_ns();
        } while(ts < dispatch_time.end && ts < deadline);
        future_bound_total_wait_ns.fetch_add(ts - first_ts, std::memory_order_relaxed);
        // A scheduling pause may make the next clock read overshoot the limit.
        // Keep the comparison bound at the deadline in that case.
        if(ts > deadline) ts = deadline;
    }

    auto _profile_time =
        tracing::profiling_time{dispatch_time_status, dispatch_time.start, dispatch_time.end};

    if(_profile_time.status == HSA_STATUS_SUCCESS)
    {
        _profile_time = tracing::adjust_profiling_time(
            "dispatch",
            "hsa_amd_profiling_get_dispatch_time",
            _profile_time,
            tracing::profiling_time{
                HSA_STATUS_SUCCESS, _baseline.value_or(dispatch_time.start), ts},
            selected_strict);
    }
    else
    {
        ROCP_CI_LOG(ERROR) << fmt::format(
            "hsa_amd_profiling_get_dispatch_time for kernel id={} on agent-{} returned status={} "
            ":: {}",
            _kernel_id,
            CHECK_NOTNULL(agent::get_rocprofiler_agent(_hsa_agent))->node_id,
            static_cast<int>(dispatch_time_status),
            hsa::get_hsa_status_string(dispatch_time_status));
    }

    return _profile_time;
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
