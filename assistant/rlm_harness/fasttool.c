
/* fasttool.c - fast <tool_call>{...}</tool_call> extractor
 * 
 * Extracts the first top-level "name" and "arguments"/"parameters" from
 * a tool-call JSON object. C version is ~100x faster than the Python
 * regex+json path.
 *
 * Build:  cc -O3 -shared -fPIC -o fasttool.so fasttool.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

typedef struct {
    char name[128];
    char args[4096];
    int  found;
} tool_call_t;

/* Skip JSON whitespace */
static inline const char *skip_ws(const char *p, const char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    return p;
}

/* Find balanced close brace, with proper string handling */
static const char *find_close(const char *p, const char *end) {
    int depth = 0;
    int in_str = 0;
    int esc = 0;
    while (p < end) {
        char c = *p;
        if (in_str) {
            if (esc) { esc = 0; }
            else if (c == '\\') { esc = 1; }
            else if (c == '"') { in_str = 0; }
        } else {
            if (c == '"') { in_str = 1; }
            else if (c == '{') { depth++; }
            else if (c == '}') {
                depth--;
                if (depth == 0) return p;
            }
        }
        p++;
    }
    return NULL;
}

/* Find value of a top-level key in a JSON object.
 * Walks the object (depth 0 inside it) and finds the first matching key.
 * Returns pointer to value (with surrounding quotes stripped if it's
 * a string) and sets *out_len.
 */
static const char *find_top_key(const char *obj_start, const char *obj_end,
                                const char *key, int *out_len) {
    int klen = strlen(key);
    const char *p = obj_start;
    int depth = 0;
    int in_str = 0;
    int esc = 0;
    while (p < obj_end) {
        char c = *p;
        if (in_str) {
            if (esc) esc = 0;
            else if (c == '\\') esc = 1;
            else if (c == '"') in_str = 0;
            p++;
            continue;
        }
        if (c == '"') {
            /* Could be a key. If at depth 1 and previous non-ws is ',' or '{', */
            /* read the key, see if it matches, return value.                  */
            const char *ks = ++p;
            while (p < obj_end && *p != '"') {
                if (*p == '\\' && p+1 < obj_end) p++;
                p++;
            }
            if (p >= obj_end) return NULL;
            int this_klen = p - ks;
            int matched = (this_klen == klen && strncmp(ks, key, klen) == 0);
            p++;  /* closing " */
            p = skip_ws(p, obj_end);
            if (p >= obj_end) return NULL;
            if (*p != ':') continue;
            p++;
            p = skip_ws(p, obj_end);
            if (p >= obj_end) return NULL;
            if (depth == 1 && matched) {
                if (*p == '"') {
                    /* string value */
                    p++;
                    const char *vs = p;
                    while (p < obj_end && *p != '"') {
                        if (*p == '\\' && p+1 < obj_end) p++;
                        p++;
                    }
                    *out_len = p - vs;
                    return vs;
                } else if (*p == '{' || *p == '[') {
                    /* object/array - return raw */
                    char open = *p;
                    char close = open == '{' ? '}' : ']';
                    const char *vs = p;
                    int d = 0;
                    int is = 0;
                    int e = 0;
                    while (p < obj_end) {
                        char cc = *p;
                        if (is) {
                            if (e) e = 0;
                            else if (cc == '\\') e = 1;
                            else if (cc == '"') is = 0;
                        } else {
                            if (cc == '"') is = 1;
                            else if (cc == open) d++;
                            else if (cc == close) {
                                d--;
                                if (d == 0) { p++; break; }
                            }
                        }
                        p++;
                    }
                    *out_len = p - vs;
                    return vs;
                }
            }
        } else if (c == '{') {
            depth++;
            p++;
        } else if (c == '}') {
            depth--;
            p++;
        } else {
            p++;
        }
    }
    return NULL;
}

/* Extract the FIRST tool call from text. Returns 1 if found. */
int extract_tool_call(const char *text, int text_len, tool_call_t *out) {
    if (!text || text_len <= 0 || !out) return 0;
    memset(out, 0, sizeof(*out));
    const char *end = text + text_len;

    /* Look for the first '{' that starts a top-level call */
    const char *p = text;
    const char *start = NULL;
    while (p < end) {
        if (*p == '{') { start = p; break; }
        p++;
    }
    if (!start) return 0;
    const char *close = find_close(start, end);
    if (!close) return 0;

    int obj_len = close - start + 1;
    char *obj = (char *)malloc(obj_len + 1);
    if (!obj) return 0;
    memcpy(obj, start, obj_len);
    obj[obj_len] = 0;

    int name_len = 0, args_len = 0;
    const char *name = find_top_key(obj, obj + obj_len, "name", &name_len);
    const char *args = find_top_key(obj, obj + obj_len, "arguments", &args_len);
    if (!args) args = find_top_key(obj, obj + obj_len, "parameters", &args_len);

    if (name && name_len > 0) {
        int n = name_len < (int)sizeof(out->name) - 1 ? name_len : (int)sizeof(out->name) - 1;
        memcpy(out->name, name, n);
        out->name[n] = 0;
        out->found = 1;
    }
    if (args && args_len > 0) {
        int a = args_len < (int)sizeof(out->args) - 1 ? args_len : (int)sizeof(out->args) - 1;
        memcpy(out->args, args, a);
        out->args[a] = 0;
    }
    free(obj);
    return out->found;
}

#ifdef TEST_MAIN
int main(int argc, char **argv) {
    const char *samples[] = {
        "<tool_call>{\"name\": \"app.open\", \"arguments\": {\"name\": \"chrome\"}}</tool_call>",
        "{\"name\": \"system.volume.set\", \"arguments\": {\"level\": 30}}",
        "<tool_call>\n{\"name\": \"browser.search\", \"arguments\": {\"query\": \"weather kolkata\"}}\n</tool_call>",
        "Hi there!",
        "<tool_call>{\"name\":\"media.pause\"}</tool_call>",
        NULL
    };
    tool_call_t out;
    for (int i = 0; samples[i]; i++) {
        int rc = extract_tool_call(samples[i], strlen(samples[i]), &out);
        printf("input:  %s\n", samples[i]);
        printf("rc=%d name=%s args=%s\n\n", rc, out.found ? out.name : "(none)", out.args);
    }
    return 0;
}
#endif
