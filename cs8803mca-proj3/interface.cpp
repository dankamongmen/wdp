#include <iostream>
#include <iomanip>
#include <cmath>
#include "router.h"
#include "interface.h"
#include <cassert>


using namespace std;

static unsigned int network_time = 0;
static unsigned int num_messages = 0; 
static unsigned int msg_latency = 0;
Router ** network_routers = NULL;

int get_message_count() {
    return num_messages;
}

float get_ave_message_latency() {

    return ((float)msg_latency / num_messages );

}
void final_hook() {

    for( int i = 0; i < 16; i++ ) {
        delete network_routers[i];
    }
    delete [] network_routers;

}


void network_register (unsigned int core_id) { 

  // kinda stupid, but it works
  // initialize all routers on core_id == 0, all other
  // calls are meaningless

  //TODO make this generic to the number of PE's  ***will more than
  // 16 (or less) ever happen?!?!?
  if( core_id == 0 ) {
      
      NETWORK_DEBUG = 1; 
      network_routers = new Router *[16];

      for( int i = 0; i < 16; i++ ) {
          network_routers[i] = new Router(input_link_width, 
                                          input_virtual_channels,
                                          input_buffers_per_vc,
                                          i );
      }
  }
}

void network_enqueue (unsigned int src_id, unsigned int dest_id,
                      unsigned int fwd_id, unsigned int addr,
                      COHERENCE_MSG msg_type) {

    const int cache_line_size = 32;
    // TODO verify this....is the input_link_width equal to the flit_width?
    // assuming it is for now.
    int flit_size = input_link_width;
    int num_bits;
    int num_flits_per_msg;
    
    
    num_bits = 64; // fixed constant from spec
    
    switch( msg_type ) {
    // message types that do not carry data
    case SDD_READ_MISS:
    case SDD_READ_MISS_FORWARD:
    case SDD_WRITE_MISS:
    case SDD_WRITE_MISS_FORWARD:
    case SDD_WRITE_MISS_REPLY:
    case SDD_INVALIDATE_FORWARD:
    case SDD_READ_DATA_REPLY_DEFER:
    case SDD_WRITE_DATA_REPLY_DEFER:
        break;

    // message types that do carry data, increment the number of bits appropriately
    case SDD_READ_MISS_REPLY_DATA:
    case SDD_WRITE_HIT_UPDATE_DATA:
    case SDD_WRITE_MISS_REPLY_DATA:
    case SDD_WRITE_MISS_UPDATE_MEM:
        num_bits += cache_line_size * 8;
        break;
    };

    num_flits_per_msg = num_bits / flit_size + !!(num_bits % flit_size);
    //num_flits_per_msg = 20;

    if( ROUTER_DEBUG ) {
        cout << "creating " << num_flits_per_msg << " flits for msg " << hex << addr << dec << endl;
    }

    // this message will contain "num_flits_per_msg" number of flits
    // encode each of the flits with the appropriate information...
    for( int i = 0; i < num_flits_per_msg; i++ ) {

        flit_t * new_flit = new flit_t;
        new_flit->msg_type = msg_type;
        new_flit->timestamp = network_time;
        new_flit->message_id = num_messages;
        new_flit->head_flit = false;
        new_flit->tail_flit = false;
        new_flit->src = src_id;
        new_flit->dst = dest_id;
        new_flit->fwd = fwd_id;
        new_flit->addr = addr;
        new_flit->length = i;
        new_flit->state = NAS;
        new_flit->enter_network_time = network_time;
        // we don't know the direction yet...that will get calculated
        // in routeCalculate
        new_flit->out_direction = NOT_COMPUTED;
        new_flit->in_direction = NOT_COMPUTED;
        
        new_flit->VCA_stall = false;

        // the flit can be the head and the tail all at once               
        if( i == 0 ) {
            new_flit->head_flit = true;
        } 
        if( i == num_flits_per_msg-1 ) {
            new_flit->tail_flit = true;
        }
        
        if( ROUTER_DEBUG ) {
            cout << "chopping flit in enqueue " << new_flit->message_id << endl;
        }
                
        network_routers[src_id]->push_network_enqueue_waiting_flit( new_flit );
    }

    num_messages++;        

    if( NETWORK_DEBUG ) {
        cout << "\tNetwork Enqueue  (cycle ";
        cout.width(6);
        cout << network_time << ") - ";
        print_msg( src_id, dest_id, fwd_id, addr, msg_type );
    }



    // take the message and dice it up into flits
    // synonymous with the VC Alloc Generator in the network diagram

}

unsigned int network_dequeue (unsigned int core_id, unsigned int *src_id,
                              unsigned int *dest_id, unsigned int *fwd_id,
                              unsigned int *addr, COHERENCE_MSG *msg_type) {


    // assemble the flit and reconstruct before printing this bitch out
    // don't forget to delete the flits as they are being popped off the queue
    flit_t * msg = network_routers[ core_id ] ->getCompletedMsg();
    
    if( msg ) {
            
        *src_id = msg->src;
        *dest_id = msg->dst;
        *fwd_id = msg->fwd;
        *msg_type = msg->msg_type;
        *addr = msg->addr;

        assert( network_time - msg->enter_network_time > 0 );
        msg_latency += network_time - msg->enter_network_time;

        /*cout << "MESSAGE LATENCY " << network_time - msg->enter_network_time << endl;
        cout << "ENTER " << msg->enter_network_time << endl;
        cout << "EXIT  " << network_time << endl;*/

		delete msg;        // no memory leaks here...

        if( NETWORK_DEBUG ) {
            cout << "\tNetwork Dequeue  (cycle ";
            cout.width(6);
            cout << dec << network_time << ") - ";
            print_msg( *src_id, *dest_id, *fwd_id, *addr, *msg_type );
        }
        return 1;
    } else {
        return 0;
    }    
}

unsigned int network_tick () {


    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->injectNewFlit();
    }

    network_time++;

    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->router_tick(); // increment router time
    }

    //TODO, make this generic to the number of processing elements!!!!
    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->linkTraversal2();
    }

    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->linkTraversal();
    }
    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->switchTraversal();
    }
    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->switchAllocation();
    }
    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->vcAllocation();
    }
    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->routeCalculation();
    }
    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->issueBuffer();
//        network_routers[i]->injectNewFlit();
    }
    for( int i = 0; i < 16; i++ ) {
        network_routers[i]->injectNewVCA();
    }

    return network_time;

}

