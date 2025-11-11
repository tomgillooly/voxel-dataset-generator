/**
 * OptiX function table definition.
 *
 * This file must be compiled exactly once in the project.
 * It defines the global OptiX function table (g_optixFunctionTable_*)
 * and provides the implementation of optixInit() and optixGetErrorName/String().
 *
 * The function table is filled at runtime by loading the OptiX driver
 * (libnvoptix.so.1) via dlopen/dlsym. The driver is located by the system
 * at runtime (typically /usr/lib64/libnvoptix.so.1).
 */

#include <optix_function_table_definition.h>

// That's it! This single include creates:
// - g_optixFunctionTable_* (the global function table)
// - optixInit() implementation (loads driver and fills function table)
// - optixGetErrorName/String() implementations
//
// At runtime, optixInit() will:
// 1. Use dlopen() to load libnvoptix.so.1
// 2. Use dlsym() to get function pointers from the driver
// 3. Fill g_optixFunctionTable with those pointers
