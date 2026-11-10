#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    char interval_ms[32];
    const long default_interval_ms = 30000; // 30 seconds 
    long interval = default_interval_ms;

    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr, "Usage: %s [interval_ms]\n", argv[0]);
        fprintf(stderr, "Default interval: %ld ms\n", default_interval_ms);
        return argc == 2 && strcmp(argv[1], "--help") == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (argc == 2) {
        char *end = NULL;
        errno = 0;
        interval = strtol(argv[1], &end, 10);
        if (errno == ERANGE || end == argv[1] || *end != '\0' ||
            interval <= 0 || interval > INT_MAX) {
            fprintf(stderr, "Interval must be a positive integer in milliseconds: %s\n", argv[1]);
            return EXIT_FAILURE;
        }
    }

    snprintf(interval_ms, sizeof(interval_ms), "%ld", interval);

    char *const args[] = {
        "powermetrics",
        "--samplers", "gpu_power",
        "--show-process-gpu",
        "--show-process-samp-norm",
        "-i", interval_ms,
        NULL
    };

    execvp(args[0], args);

    fprintf(stderr, "execvp(powermetrics) failed: %s\n", strerror(errno));
    return EXIT_FAILURE;
}
