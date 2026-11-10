#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char *const args[] = {
        "powermetrics",
        "--samplers", "gpu_power",
        "--show-process-gpu",
        "--show-process-samp-norm",
        "-i", "1000",
        NULL
    };

    execvp(args[0], args);

    fprintf(stderr, "execvp(powermetrics) failed: %s\n", strerror(errno));
    return EXIT_FAILURE;
}
