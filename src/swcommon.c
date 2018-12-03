#include "include/swcommon.h"
void start_use_slave() {
  athread_init();
}

void end_use_slave() {
  athread_halt();
}


