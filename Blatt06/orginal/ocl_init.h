#ifndef _OCL_INIT_H
#define _OCL_INIT_H

#include <stdbool.h>
#include <CL/cl.h>

/*
 * Abort the program after printig an OpenCL error code.
 */
void error_and_abort(const char *msg, cl_int err);

/*
 * Find an OpenCL context and device (returned in 'devid').
 * If 'want_gpu' is true, a GPU is searched for, otherwise a CPU.
 */
cl_context create_cl_context(bool want_gpu, cl_device_id *devid);

/*
 * Load OpenCL kernel source from 'filename' and compile it
 * for the given context and device.
 */
cl_program build_program(const char *filename, cl_context context, cl_device_id devid);

/*
 * Return the start or end time in nano seconds of the given OpenCL event.
 */
cl_ulong get_event_start_nanos(cl_event event);
cl_ulong get_event_end_nanos(cl_event event);

#endif
