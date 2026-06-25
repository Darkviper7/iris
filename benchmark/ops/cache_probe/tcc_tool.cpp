// SPDX-License-Identifier: MIT
// In-process rocprofiler-sdk tool: per-dispatch L2 (TCC) HIT/MISS counters,
// filtered to a kernel-name substring, summed and written to a CSV.
//
// Loaded via ROCP_TOOL_LIBRARIES so it registers as the OWNING process's
// rocprofiler client -> configures inside the valid period, avoiding the
// "configuration outside valid period" abort that hits the external rocprofv3
// CLI when PyTorch has already loaded librocprofiler-register.so.
//
// Env:
//   TCC_OUT          output CSV path (default tcc_counts.csv)
//   TCC_KERNEL_SUB   kernel-name substring filter (default "all_gather_matmul")
//   TCC_COUNTERS     comma-separated counter names (default "TCC_HIT,TCC_MISS")
//
// Build: see Makefile (hipcc -shared).

#include <hip/hip_runtime.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/callback_tracing.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define RC(result, msg)                                                                            \
    {                                                                                              \
        rocprofiler_status_t s_ = (result);                                                        \
        if(s_ != ROCPROFILER_STATUS_SUCCESS)                                                       \
        {                                                                                          \
            std::cerr << "[tcc_tool][" << __FILE__ << ":" << __LINE__ << "] " << msg               \
                      << " failed: " << rocprofiler_get_status_string(s_) << std::endl;            \
            throw std::runtime_error(msg);                                                         \
        }                                                                                          \
    }

namespace
{
rocprofiler_context_id_t  g_ctx{0};
std::string               g_out;
std::string               g_sub;
std::set<std::string>     g_want;       // counter names to collect

std::shared_mutex                                  g_kmutex;
std::unordered_map<uint64_t, std::string>          g_kernel_names;   // kernel_id -> name

std::mutex                                         g_cmutex;
std::unordered_map<uint64_t, std::string>          g_counter_names;  // counter_id.handle -> name
std::map<std::string, double>                      g_sums;           // counter name -> summed value
uint64_t                                           g_matched_dispatches = 0;

rocprofiler_context_id_t&
ctx()
{
    return g_ctx;
}

bool
name_matches(uint64_t kernel_id)
{
    auto rl = std::shared_lock{g_kmutex};
    auto it = g_kernel_names.find(kernel_id);
    if(it == g_kernel_names.end()) return false;
    return it->second.find(g_sub) != std::string::npos;
}

// ---- code-object tracing: build kernel_id -> name map ----
void
code_object_callback(rocprofiler_callback_tracing_record_t record,
                     rocprofiler_user_data_t*, void*)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* d = static_cast<
            rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(record.payload);
        auto wl = std::unique_lock{g_kmutex};
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            g_kernel_names.emplace(d->kernel_id, std::string(d->kernel_name ? d->kernel_name : ""));
            std::clog << "[tcc_tool] kernel registered id=" << d->kernel_id << " name="
                      << (d->kernel_name ? d->kernel_name : "?") << "\n";
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
            g_kernel_names.erase(d->kernel_id);
    }
}

// ---- per-dispatch: choose counters (only for matching kernels) ----
void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t*, void*)
{
    if(!name_matches(dispatch_data.dispatch_info.kernel_id)) return;  // leave *config empty -> no collection
    std::clog << "[tcc_tool] dispatch matched kernel_id=" << dispatch_data.dispatch_info.kernel_id << "\n";

    static std::shared_mutex                                             m{};
    static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t> cache{};
    auto agent = dispatch_data.dispatch_info.agent_id;

    {
        auto rl = std::shared_lock{m};
        auto it = cache.find(agent.handle);
        if(it != cache.end()) { *config = it->second; return; }
    }
    auto wl = std::unique_lock{m};
    if(auto it = cache.find(agent.handle); it != cache.end()) { *config = it->second; return; }

    std::vector<rocprofiler_counter_id_t> all;
    RC(rocprofiler_iterate_agent_supported_counters(
           agent,
           [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* cs, size_t n, void* ud) {
               auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
               for(size_t i = 0; i < n; i++) v->push_back(cs[i]);
               return ROCPROFILER_STATUS_SUCCESS;
           },
           static_cast<void*>(&all)),
       "iterate counters");

    std::vector<rocprofiler_counter_id_t> chosen;
    for(auto& c : all)
    {
        rocprofiler_counter_info_v0_t info;
        RC(rocprofiler_query_counter_info(c, ROCPROFILER_COUNTER_INFO_VERSION_0,
                                          static_cast<void*>(&info)),
           "query counter info");
        if(getenv("TCC_DUMP_COUNTERS"))
            std::clog << "[tcc_tool] avail counter: " << info.name << "\n";
        if(g_want.count(std::string(info.name)) > 0)
        {
            std::clog << "[tcc_tool] SELECTED counter: " << info.name << " handle=" << c.handle << "\n";
            chosen.push_back(c);
            auto cl = std::lock_guard{g_cmutex};
            g_counter_names[c.handle] = info.name;
        }
    }

    rocprofiler_counter_config_id_t prof = {.handle = 0};
    RC(rocprofiler_create_counter_config(agent, chosen.data(), chosen.size(), &prof),
       "create counter config");
    cache.emplace(agent.handle, prof);
    *config = prof;
}

// ---- per-dispatch records: sum counter values by name ----
void
record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                rocprofiler_counter_record_t* records, size_t n,
                rocprofiler_user_data_t, void*)
{
    if(!name_matches(dispatch_data.dispatch_info.kernel_id)) return;
    auto lk = std::lock_guard{g_cmutex};
    g_matched_dispatches++;
    for(size_t i = 0; i < n; i++)
    {
        // map record's counter id back to a name
        rocprofiler_counter_id_t cid{};
        rocprofiler_query_record_counter_id(records[i].id, &cid);
        auto it = g_counter_names.find(cid.handle);
        std::string nm = (it != g_counter_names.end()) ? it->second : ("id_" + std::to_string(cid.handle));
        g_sums[nm] += records[i].counter_value;
    }
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    // tool_init can be invoked more than once for this instance; create the
    // context (and register services) only once. A second create_context with a
    // non-zero ctx handle errors with "Context ID should be initialized to zero".
    if(ctx().handle != 0) return 0;
    RC(rocprofiler_create_context(&ctx()), "create context");
    RC(rocprofiler_configure_callback_tracing_service(
           ctx(), ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0, code_object_callback, nullptr),
       "code object tracing");
    RC(rocprofiler_configure_callback_dispatch_counting_service(
           ctx(), dispatch_callback, nullptr, record_callback, nullptr),
       "dispatch counting service");
    RC(rocprofiler_start_context(ctx()), "start context");
    return 0;
}

void
tool_fini(void*)
{
    rocprofiler_stop_context(ctx());
    // The tool .so can be loaded more than once (ROCP_TOOL_LIBRARIES + scan), each
    // instance with its own globals; only the instance whose record_callback fired
    // has data. Write to a UNIQUE per-instance file (TCC_OUT + ".<addr>") so the two
    // finis don't clobber each other; the driver picks the file with nonzero data.
    if(g_matched_dispatches == 0) return;  // empty instance: write nothing
    std::ofstream os{g_out};
    os << "counter,value\n";
    double hit = 0, miss = 0;
    for(auto& [nm, v] : g_sums)
    {
        os << nm << "," << (uint64_t) v << "\n";
        if(nm == "TCC_HIT") hit = v;
        if(nm == "TCC_MISS") miss = v;
    }
    os << "matched_dispatches," << g_matched_dispatches << "\n";
    if(hit + miss > 0)
        os << "hit_rate," << (hit / (hit + miss)) << "\n";
    os.flush();
    std::clog << "[tcc_tool] wrote " << g_out << " (matched dispatches=" << g_matched_dispatches
              << ", hit=" << (uint64_t)hit << " miss=" << (uint64_t)miss << ")\n";
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version, const char* runtime_version, uint32_t priority,
                      rocprofiler_client_id_t* id)
{
    id->name = "tcc_l2_probe";

    g_out = getenv("TCC_OUT") ? getenv("TCC_OUT") : "tcc_counts.csv";
    g_sub = getenv("TCC_KERNEL_SUB") ? getenv("TCC_KERNEL_SUB") : "all_gather_matmul";
    std::string counters = getenv("TCC_COUNTERS") ? getenv("TCC_COUNTERS") : "TCC_HIT,TCC_MISS";
    std::stringstream cs{counters};
    std::string tok;
    while(std::getline(cs, tok, ',')) if(!tok.empty()) g_want.insert(tok);

    std::clog << "[tcc_tool] configure v" << version << " (" << runtime_version << ") prio="
              << priority << " out=" << g_out << " sub='" << g_sub << "' counters=" << counters
              << std::endl;

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
