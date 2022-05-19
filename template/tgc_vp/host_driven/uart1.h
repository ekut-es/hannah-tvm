#ifndef _UART1_H_
#define _UART1_H_

#include "platform.h"

#include <stdio.h>
#include <stdint.h>

static void uart1_init(size_t baud_rate){
  GPIO_REG(GPIO_IOF_SEL) &= ~IOF0_UART1_MASK;
  GPIO_REG(GPIO_IOF_EN) |= IOF0_UART1_MASK;
  UART1_REG(UART_REG_DIV) = get_cpu_freq() / baud_rate - 1;
  UART1_REG(UART_REG_TXCTRL) |= UART_TXEN;
  UART1_REG(UART_REG_RXCTRL) |= UART_RXEN;
}


static ssize_t uart1_write(const void* ptr, size_t len){
  const uint8_t * current = (const char *)ptr;

  for (size_t jj = 0; jj < len; jj++) {
  	while(UART1_REG(UART_REG_TXFIFO) & 0x80000000) ;
  	UART1_REG(UART_REG_TXFIFO) = current[jj];
  }
  return len;
}

static ssize_t uart1_read(void* ptr, size_t len)
{
  uint8_t * current = (uint8_t *)ptr;
  volatile uint32_t * uart_rx = (uint32_t *)(UART1_CTRL_ADDR + UART_REG_RXFIFO);


  ssize_t result = 0;

  for (current = (uint8_t *)ptr;
      current < ((uint8_t *)ptr) + len;
      current ++) {
    uint32_t current_data = *uart_rx;
    if((current_data & 0x80000000)){
        break;
    }
    *current = current_data;
    result++;
  }
  return result;
}

#endif // _UART1_H_
