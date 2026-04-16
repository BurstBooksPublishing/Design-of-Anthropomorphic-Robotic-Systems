cpp
#include <ctime>
#include <cstdint>
#include <iostream>

// Assuming these are defined elsewhere in the project
extern bool running;
extern bool check_assumption();
extern void compute_control();
extern uint64_t now_usec();
extern int clock_nanosleep(clockid_t clk_id, int flags, const struct timespec *request, struct timespec *remain);

// Convert period to priority (Rate Monotonic: shorter period = higher priority)
static inline int period_to_priority(int period_ms) {
    // Implementation depends on RTOS; this is a placeholder
    return 1000 / period_ms; 
}

// Emit C++ RT task skeleton from contract specification
void emit_task(const std::map<std::string, std::any>& contract) {
    std::string name = std::any_cast<std::string>(contract.at("name"));
    int period_ms = std::any_cast<int>(contract.at("timing").at("period_ms"));
    int wcet_ms = std::any_cast<int>(contract.at("timing").at("wcet_ms"));
    
    // Priority derived by RM: shorter period -> higher priority
    int priority = period_to_priority(period_ms);
    
    std::cout << "// Task skeleton for " << name << std::endl;
    std::cout << "void " << name << "_task() {" << std::endl;
    std::cout << "  struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next);" << std::endl;
    std::cout << "  while(running) {" << std::endl;
    std::cout << "    // precondition check (assumption)" << std::endl;
    std::cout << "    if(!check_assumption()) { /* signal contract violation */ }" << std::endl;
    std::cout << "    auto t0 = now_usec();" << std::endl;
    std::cout << "    // user compute: implement guarantee contract here" << std::endl;
    std::cout << "    compute_control(); // should complete <= " << wcet_ms << " ms" << std::endl;
    std::cout << "    auto dt = now_usec() - t0; if(dt > " << (wcet_ms * 1000) << ") { /* WCET breach */ }" << std::endl;
    std::cout << "    // sleep until next period" << std::endl;
    std::cout << "    next.tv_nsec += " << static_cast<long>(period_ms * 1e6) << "; clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);" << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << "}" << std::endl;
}