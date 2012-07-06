#include <CL/cl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ocl_init.h"

/* More convenient error code detection */
char *print_cl_errstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                          return strdup("Success!");
        case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
        case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
        case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
        case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
        case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
        case CL_MAP_FAILURE:                      return strdup("Map failure");
        case CL_INVALID_VALUE:                    return strdup("Invalid value");
        case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
        case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
        case CL_INVALID_DEVICE:                   return strdup("Invalid device");
        case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
        case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
        case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
        case CL_INVALID_BINARY:                   return strdup("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
        case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
        case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
        case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
        case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
        case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
        case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
        case CL_INVALID_EVENT:                    return strdup("Invalid event");
        case CL_INVALID_OPERATION:                return strdup("Invalid operation");
        case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
        default:                                  return strdup("Unknown");
    }
}

/*
 * For simplicity, we abort when we encounter an error...
 */
void error_and_abort(const char *msg, cl_int err) {
    if (err != 0)
        fprintf(stderr, "%s: %s\n", msg, print_cl_errstring(err));
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
