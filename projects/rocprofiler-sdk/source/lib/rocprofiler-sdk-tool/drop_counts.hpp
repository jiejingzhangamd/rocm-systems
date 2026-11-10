#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace rocprofiler_clock_audit
{
struct row { uint64_t buffer, dropped, callbacks, regressions; };
// SDK buffer::drop_count is cumulative for each buffer, not a per-callback delta.
class drop_counts
{
    std::mutex mutex;
    std::map<uint64_t, row> rows;
public:
    void observe(uint64_t buffer, uint64_t cumulative)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto itr = rows.find(buffer);
        if(itr == rows.end()) itr = rows.emplace(buffer,row{buffer,0,0,0}).first;
        auto& r = itr->second;
        if(cumulative < r.dropped) ++r.regressions;
        if(cumulative > r.dropped) r.dropped = cumulative;
        ++r.callbacks;
    }
    std::vector<row> snapshot()
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<row> result;
        for(const auto& entry : rows) result.push_back(entry.second);
        return result;
    }
};
}
