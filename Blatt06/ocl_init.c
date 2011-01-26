#include <CL/cl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ocl_init.h"

/*
 * For simplicity, we abort when we encounter an error...
 */
void error_and_abort(const char *msg, cl_int err) {
    if (err != 0)
        fprintf(stderr, "%s: %d\n", msg, (int)err);
    else
        fprintf(stderr, "%s\n", msg);
    exit(1);
}

cl_context create_cl_context(bool want_gpu, cl_device_id *devid) {
    cl_platform_id platforms[1];
    cl_uint num_platforms;
    cl_int res;

    res = clGetPlatformIDs(1, platforms, &num_platforms);
    if (res != CL_SUCCESS)
        error_and_abort("Could not get platform IDs", res);

    for (cl_uint platf=0; platf<num_platforms; platf++) {
        cl_context_properties props[] = 
            {
                CL_CONTEXT_PLATFORM, 
                (cl_context_properties)platforms[platf],
                0
            };

        cl_int dev_type = want_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
        res = clGetDeviceIDs(platforms[platf], dev_type, 1, devid, 0);
        if (res != CL_SUCCESS)
            error_and_abort("Could not get device IDs", res);
        
        // create a compute context with the device
        cl_context context;
        context = clCreateContext(props, 1, devid, NULL, NULL, NULL);
        if (context != NULL) {
            char buf[256];
            size_t len;
            res = clGetDeviceInfo(*devid, CL_DEVICE_NAME, sizeof(buf), buf, &len);
            if (res != CL_SUCCESS)
                strncpy(buf, "(unknown)", sizeof(buf));
            buf[sizeof(buf)-1] = '\0';
            fprintf(stderr, "Using device \"%s\"\n", buf);
            return context;
        }
    }
    error_and_abort("Could not create any context", 0);
    return NULL;
}

static char *read_file(const char *filename) {
    FILE *f;
    struct stat st;
    char *buf;

    f = fopen(filename, "rt");
    if (f == NULL)
        error_and_abort("Could not open program file", 0);

    if (fstat(fileno(f), &st) != 0) {
        fclose(f);
        error_and_abort("Could not stat program file", 0);
    }
     
    buf = malloc(st.st_size+1);
    if (buf == NULL) {
        fclose(f);
        error_and_abort("Could not allocate buffer for program file", 0);
    }

    if (fread(buf, 1, st.st_size, f) != st.st_size) {
        fclose(f);
        error_and_abort("Could not read program file", 0);
    }
     
    fclose(f);
    buf[st.st_size] = '\0';
    return buf;
}

static void print_compiler_messages(cl_int err, cl_program program, cl_device_id devid) {
    if (err != CL_SUCCESS)
        fprintf(stderr, "Could not build program: %d\n", (int)err);
    size_t len;
    clGetProgramBuildInfo(program, devid, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    if (err != CL_SUCCESS || len > 0) {
        char msg[len];
        clGetProgramBuildInfo(program, devid, CL_PROGRAM_BUILD_LOG, len, msg, NULL);
        fprintf(stderr, "Compiler messages:\n%s\n", msg);
    }
}

cl_program build_program(const char *filename, cl_context context, cl_device_id devid) {
    const char *compiler_options = "-I.";
    char *text;
    cl_program program;
    cl_int err;
    size_t lengths[1];
    const char *lines[1];

    text = read_file(filename);
    if (text == NULL)
        return NULL;

    lines[0] = text;
    lengths[0] = strlen(text);
    program = clCreateProgramWithSource(context, 1, lines, lengths, &err);
    if (program == NULL) {
        free(text);
        error_and_abort("Could create CL program from source", err);
    }
 
    // build the compute program executable
    err = clBuildProgram(program, 1, &devid, compiler_options, NULL, NULL);
    print_compiler_messages(err, program, devid);
    if (err != CL_SUCCESS) {
        /*
         * Print the (first) binary generated.
         * On NVIDIA, the "binary" is actually the PTX assembly code.
         * (Useful for debugging the NVIDIA compiler when it generates
         * invalid PTX...)
         */
/*
        size_t n_sizes;
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 0, NULL, &n_sizes);
        size_t blens[n_sizes];
        clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(*blens)*n_sizes, blens, NULL);
        unsigned char bin[blens[0]+1];
        unsigned char *bptrs[n_sizes];
        for (int i=1; i<n_sizes; i++)
            bptrs[i] = NULL;
        bptrs[0] = bin;
        clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(*bptrs)*n_sizes, bptrs, NULL);
        bin[blens[0]] = '\0';
        fprintf(stderr, "Binary:\n%s\n", bin);
*/
        exit(1);
    }

    return program;
}

static cl_ulong get_event_nanos(cl_event event, cl_ulong prof) {
    cl_ulong nanos;
    cl_int res;
    res = clGetEventProfilingInfo(event, prof,
                                  sizeof(nanos), &nanos, NULL);
    if (res != CL_SUCCESS) {
        fprintf(stderr, "Could not get event time\n");
        return 0;
    }
    return nanos;
}

cl_ulong get_event_start_nanos(cl_event event) {
    return get_event_nanos(event, CL_PROFILING_COMMAND_START);
}

cl_ulong get_event_end_nanos(cl_event event) {
    return get_event_nanos(event, CL_PROFILING_COMMAND_END);
}
