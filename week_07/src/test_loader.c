/*
 * test_loader.c - Load JSON test cases for DeepSeekV3 MoE verification
 *
 * Week 7: Communication - Pure C Implementation
 *
 * Implements a minimal JSON parser sufficient for loading our test cases.
 * No external libraries needed - just standard C.
 */

#include "test_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* ============================================================
 * File I/O
 * ============================================================ */

char *read_file(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", filepath);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = (char *)malloc(len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t nread = fread(buf, 1, len, f);
    (void)nread;
    buf[len] = '\0';
    fclose(f);

    return buf;
}

/* ============================================================
 * JSON Parsing Helpers
 * ============================================================
 * These are minimal parsers that work for our specific JSON format.
 * Not a general-purpose JSON library.
 */

/*
 * Find the position after "key": in the JSON string.
 * Returns pointer to the value part, or NULL if not found.
 */
static const char *find_key_value(const char *json, const char *key) {
    char search[512];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char *pos = json;
    while ((pos = strstr(pos, search)) != NULL) {
        /* Move past the key string */
        pos += strlen(search);
        /* Skip whitespace */
        while (*pos && isspace(*pos)) pos++;
        /* Expect colon */
        if (*pos == ':') {
            pos++;
            /* Skip whitespace */
            while (*pos && isspace(*pos)) pos++;
            return pos;
        }
        /* If no colon, this wasn't our key, keep searching */
    }
    return NULL;
}

int json_get_int(const char *json, const char *key) {
    const char *val = find_key_value(json, key);
    if (!val) {
        fprintf(stderr, "WARNING: Key '%s' not found, returning 0\n", key);
        return 0;
    }
    return atoi(val);
}

float json_get_float(const char *json, const char *key) {
    const char *val = find_key_value(json, key);
    if (!val) {
        fprintf(stderr, "WARNING: Key '%s' not found, returning 0.0\n", key);
        return 0.0f;
    }
    return (float)atof(val);
}

int json_get_bool(const char *json, const char *key) {
    const char *val = find_key_value(json, key);
    if (!val) return 0;
    if (strncmp(val, "true", 4) == 0) return 1;
    return 0;
}

char *json_get_string(const char *json, const char *key) {
    const char *val = find_key_value(json, key);
    if (!val || *val != '"') return NULL;
    val++; /* skip opening quote */
    const char *end = strchr(val, '"');
    if (!end) return NULL;
    int len = (int)(end - val);
    char *result = (char *)malloc(len + 1);
    strncpy(result, val, len);
    result[len] = '\0';
    return result;
}

/*
 * Extract a flat float array from a nested JSON array.
 *
 * Strategy: Find the opening '[' after the key, then walk through
 * the entire nested structure collecting all numeric values.
 * Ignores brackets and commas, just parses numbers.
 */
float *json_get_float_array(const char *json, const char *key, int *out_count) {
    const char *val = find_key_value(json, key);
    if (!val || *val != '[') {
        fprintf(stderr, "WARNING: Key '%s' not found or not an array\n", key);
        *out_count = 0;
        return NULL;
    }

    /* First pass: count numbers */
    int count = 0;
    int depth = 0;
    const char *p = val;

    while (*p) {
        if (*p == '[') {
            depth++;
        } else if (*p == ']') {
            depth--;
            if (depth == 0) break;
        } else if (*p == '-' || isdigit(*p)) {
            /* Found a number - count it */
            count++;
            /* Skip past the number */
            if (*p == '-') p++;
            while (*p && (isdigit(*p) || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) {
                /* Handle 'e' followed by '+' or '-' */
                if ((*p == 'e' || *p == 'E') && (*(p+1) == '+' || *(p+1) == '-')) {
                    p += 2;
                    continue;
                }
                p++;
            }
            continue; /* Don't advance p again */
        }
        p++;
    }

    /* Second pass: extract numbers */
    float *result = (float *)malloc(count * sizeof(float));
    if (!result) {
        *out_count = 0;
        return NULL;
    }

    int idx = 0;
    depth = 0;
    p = val;

    while (*p && idx < count) {
        if (*p == '[') {
            depth++;
        } else if (*p == ']') {
            depth--;
            if (depth == 0) break;
        } else if (*p == '-' || isdigit(*p)) {
            /* Parse the number */
            char *endptr;
            result[idx++] = (float)strtod(p, &endptr);
            p = endptr;
            continue;
        }
        p++;
    }

    *out_count = idx;
    return result;
}

/*
 * Extract a flat int array from a nested JSON array.
 */
int *json_get_int_array(const char *json, const char *key, int *out_count) {
    const char *val = find_key_value(json, key);
    if (!val || *val != '[') {
        fprintf(stderr, "WARNING: Key '%s' not found or not an array\n", key);
        *out_count = 0;
        return NULL;
    }

    /* First pass: count numbers */
    int count = 0;
    int depth = 0;
    const char *p = val;

    while (*p) {
        if (*p == '[') depth++;
        else if (*p == ']') { depth--; if (depth == 0) break; }
        else if (*p == '-' || isdigit(*p)) {
            count++;
            if (*p == '-') p++;
            while (*p && isdigit(*p)) p++;
            continue;
        }
        p++;
    }

    /* Second pass: extract */
    int *result = (int *)malloc(count * sizeof(int));
    int idx = 0;
    depth = 0;
    p = val;

    while (*p && idx < count) {
        if (*p == '[') depth++;
        else if (*p == ']') { depth--; if (depth == 0) break; }
        else if (*p == '-' || isdigit(*p)) {
            char *endptr;
            result[idx++] = (int)strtol(p, &endptr, 10);
            p = endptr;
            continue;
        }
        p++;
    }

    *out_count = idx;
    return result;
}

/* ============================================================
 * Test Result Comparison
 * ============================================================ */

TestResult compare_outputs(const char *test_name,
                            const float *expected, const float *actual,
                            int n, float tolerance) {
    TestResult result;
    result.test_name = test_name;
    result.num_values = n;
    result.max_error = 0.0f;
    result.mean_error = 0.0f;
    result.passed = 1;

    float sum_error = 0.0f;

    for (int i = 0; i < n; i++) {
        float err = fabsf(expected[i] - actual[i]);
        sum_error += err;
        if (err > result.max_error) {
            result.max_error = err;
        }
    }

    result.mean_error = (n > 0) ? sum_error / n : 0.0f;
    result.passed = (result.max_error <= tolerance) ? 1 : 0;

    return result;
}

void print_test_result(const TestResult *result, float tolerance) {
    printf("  %-30s  %s  max_err=%.2e  mean_err=%.2e  (n=%d, tol=%.0e)\n",
           result->test_name,
           result->passed ? "PASS" : "FAIL",
           result->max_error,
           result->mean_error,
           result->num_values,
           tolerance);
}
