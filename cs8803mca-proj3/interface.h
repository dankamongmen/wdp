#ifndef _interface_h
#define _interface_h

#include "router.h"

typedef enum {
    SDD_READ_MISS,
    SDD_READ_MISS_FORWARD,
    SDD_READ_MISS_REPLY_DATA,
    SDD_WRITE_HIT_UPDATE_DATA,
    SDD_WRITE_MISS,
    SDD_WRITE_MISS_FORWARD,
    SDD_WRITE_MISS_REPLY,
    SDD_WRITE_MISS_REPLY_DATA,
    SDD_WRITE_MISS_UPDATE_MEM,
    SDD_INVALIDATE_FORWARD,
    SDD_READ_DATA_REPLY_DEFER,
    SDD_WRITE_DATA_REPLY_DEFER
} COHERENCE_MSG;


//global variable to turn network enqueue/dequeue debugging on/off; off by default
extern unsigned int NETWORK_DEBUG;

//new network parameters passed in via command line
extern int input_link_width;
extern int input_virtual_channels;
extern int input_buffers_per_vc;

//useful debug printing function
void print_msg (unsigned int src_id, unsigned int dest_id,
           unsigned int fwd_id, unsigned int addr, COHERENCE_MSG msg_type);

int get_message_count();
float get_ave_message_latency();

// Network Interface Prototypes
void network_register (unsigned int core_id);
void network_enqueue (unsigned int src_id, unsigned int dest_id,
                      unsigned int fwd_id, unsigned int addr,
                      COHERENCE_MSG msg_type);
unsigned int network_dequeue (unsigned int core_id, unsigned int *src_id,
                              unsigned int *dest_id, unsigned int *fwd_id,
                              unsigned int *addr, COHERENCE_MSG *msg_type);
unsigned int network_tick ();

void final_hook();

//Router * get_Router ( int router_id );

#endif
