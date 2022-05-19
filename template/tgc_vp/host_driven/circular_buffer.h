#ifndef CIRCULAR_BUFFER_H_
#define CIRCULAR_BUFFER_H_

#include <stdbool.h>
#include <stdint.h>

typedef struct circular_buf_t circular_buf_t;

typedef circular_buf_t* cbuf_handle_t;

cbuf_handle_t circular_buf_init(uint8_t* buffer, size_t size);

void circular_buf_free(cbuf_handle_t me);

void circular_buf_reset(cbuf_handle_t me);

void circular_buf_put(cbuf_handle_t me, uint8_t data);

int circular_buf_try_put(cbuf_handle_t me, uint8_t data);

int circular_buf_get(cbuf_handle_t me, uint8_t* data);

bool circular_buf_empty(cbuf_handle_t me);

bool circular_buf_full(cbuf_handle_t me);

size_t circular_buf_capacity(cbuf_handle_t me);

size_t circular_buf_size(cbuf_handle_t me);

#endif // CIRCULAR_BUFFER_H_
