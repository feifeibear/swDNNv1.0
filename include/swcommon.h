#ifndef _SW_COMMON_H_
#define _SW_COMMON_H_
#include <athread.h>
#include <sys/time.h>

void start_use_slave();

void end_use_slave();

#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))
#endif
