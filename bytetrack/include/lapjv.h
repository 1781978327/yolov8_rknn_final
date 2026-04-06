#pragma once

#define LARGE 1000000

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define NEW(x, t, n) if ((x = (t *)malloc(sizeof(t) * (n))) == 0) { return -1; }
#define FREE(x) if (x != 0) { free(x); x = 0; }
#define SWAP_INDICES(a, b) { int_t _temp_index = a; a = b; b = _temp_index; }

#if 0
#include <assert.h>
#define ASSERT(cond) assert(cond)
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define ASSERT(cond)
#define PRINTF(fmt, ...)
#endif

typedef signed int   int_t;
typedef unsigned int uint_t;
typedef double       cost_t;
typedef char         boolean;

extern int_t lapjv_internal(
	const uint_t n, cost_t *cost[],
	int_t *x, int_t *y);
