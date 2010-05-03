#include "router.h"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>

using namespace std;

const char * direction_str[10] = {
    "NORTH",
    "WEST",
    "LOCAL",
    "EAST",
    "SOUTH",
    "NOT_COMPUTED"
};

// it's ugly, but it's fast and it works...we gotta get to the neighbors somehow
extern Router ** network_routers;


InputBuffer::InputBuffer( int channels,
                          int buffer_size,
                          direction_t direction):
    num_vc(channels),
    max_size(buffer_size),
    vc_contents(NULL),
    size(NULL),
    dir(direction),
    msg_id(NULL),
    busy(NULL)
{

    
    vc_contents = new list<flit_t*> [channels];

    size = new int[channels];
    busy = new bool[channels];
    msg_id = new unsigned[channels];
    
    
    for( int i = 0; i < channels; i++ ) {
        size[i] = 0;
        busy[i] = false;
        msg_id[i] = -1;
        vc_contents[i].clear();
    }
    
    
}

InputBuffer::~InputBuffer() {
    delete [] size;
    delete [] vc_contents;
    delete [] busy;
    delete [] msg_id;
}


// get a free vc for a specified message id

int InputBuffer::get_free_vc( ) {

    for( int i = 0; i < num_vc; i++ ) {
        if( !busy[i] ) {
            busy[i] = true;
            return i;
        }
    }    
    return -1;
}

void InputBuffer::set_msg_id( int vc, unsigned int id ) {
    assert( vc < num_vc );
    assert( vc >= 0 );

    msg_id[vc] = id;
}


bool InputBuffer::vc_full( int vc ) {

    int size = vc_contents[vc].size();
    return size == max_size;
}


bool InputBuffer::enter_flit( flit_t * flit, int vc ) {


    if( flit->message_id != (unsigned int)msg_id[vc] ||
        vc_full( vc ) ) {
        return false;
    }
    
    if( vc_contents[vc].size() > 0 ) {
        assert( is_pending( vc, flit->message_id ) );
    }

    vc_contents[vc].push_back( flit );
    return true;
}


flit_t * InputBuffer::exit_flit( int vc ) {

    flit_t * ret_flit = NULL;

    if( vc_contents[vc].size() > 0 ) {
        ret_flit = vc_contents[vc].front();
//        cout << " ret_flit " << ret_flit->message_id << endl;

        vc_contents[vc].pop_front();
        
    }    
//    if( ret_flit )
//        cout << " ret_flit " << ret_flit->message_id << endl;

    if( ret_flit->tail_flit ) {
        busy[vc] = false;
        msg_id[vc] = -1;
    }

    return ret_flit;
}

bool InputBuffer::is_pending( int vc, unsigned int message_id ) {

    return( msg_id[vc] == message_id );
//    if( vc_contents[vc].size() == 0 ) {
 //       return false;
  //  }

   // return vc_contents[vc].front()->message_id == message_id;    
}

// top returns a reference for STL list, doesn't actually reduce
// the list or anything

flit_t * InputBuffer::peek( int vc ) {
    if( vc_contents[vc].size() > 0 ) {        
        return vc_contents[vc].front();
    }
    else
        return NULL;
}

void InputBuffer::print( ) {

    cout << direction_str[dir] << endl;
    for( int i = 0; i < num_vc; i++ ) {
        cout << "vc " << i << " size " << vc_contents[i].size() << endl;
    }
}




Router::Router( int link_width,
                int virtual_channels,
                int buffers_per_vc,
                int router_id ):
    link_width(link_width),
    num_vc(virtual_channels),
    buffer_size_per_vc(buffers_per_vc),
    input_buffers(NULL),
    waiting_IB_flits(NULL),
    time(0),
    id(router_id),
    SA_XBAR_NORTH(NULL),
    SA_XBAR_SOUTH(NULL),
    SA_XBAR_EAST(NULL),
    SA_XBAR_WEST(NULL),
    SA_XBAR_LOCAL(NULL),
    ST_XBAR_NORTH(NULL),
    ST_XBAR_SOUTH(NULL),
    ST_XBAR_EAST(NULL),
    ST_XBAR_WEST(NULL),
    ST_XBAR_LOCAL(NULL),
    LT_NORTH(NULL),
    LT_SOUTH(NULL),
    LT_EAST(NULL),
    LT_WEST(NULL),
    LT_LOCAL(NULL),
    LT2_NORTH(NULL),
    LT2_SOUTH(NULL),
    LT2_EAST(NULL),
    LT2_WEST(NULL),
    LT2_LOCAL(NULL)
 {


    //5 inputs for N, S, E, W, and local
    input_buffers = new InputBuffer * [5];

    // initialize input buffers for all directions: N,S,E,W,local
    for( int i = 0; i < 5; i++ ) { 
        input_buffers[i] = new InputBuffer(
            num_vc,
            buffer_size_per_vc,
            static_cast<direction_t>(i) );
    }    

}

Router::~Router() {

    for( int i = 0; i < 5; i++ ) {
        delete input_buffers[i];
    }

    delete[] input_buffers;
    
}


flit_t * Router::getCompletedMsg( ) {

    flit_t * arrival_msg = NULL;
    if( completed_msg.size() > 0 ) {
        arrival_msg = completed_msg.front();
        completed_msg.pop_front();       
        LT2_LOCAL = NULL;
//         LT_EAST = NULL;
//         LT_WEST = NULL;
//         LT_SOUTH = NULL;
//         LT_NORTH = NULL;
    }
    return arrival_msg;           
}

void Router::linkTraversal2( ) {

    LT2_NORTH = LT_NORTH;
    LT2_SOUTH = LT_SOUTH;
    LT2_EAST = LT_EAST;
    LT2_WEST = LT_WEST;
//    LT2_LOCAL = LT_LOCAL;

    LT_NORTH = NULL;
    LT_SOUTH = NULL;
    LT_EAST = NULL;
    LT_WEST = NULL;
    LT_LOCAL = NULL;




    if( ROUTER_DEBUG ) {

        if( LT2_NORTH || 
            LT2_SOUTH || 
            LT2_EAST ||
            LT2_WEST ||
            LT2_LOCAL ) {
        
            st_and_lt_print( true, NORTH, LT2_NORTH );
            st_and_lt_print( true, SOUTH, LT2_SOUTH );
            st_and_lt_print( true, EAST, LT2_EAST );
            st_and_lt_print( true, WEST, LT2_WEST );
            st_and_lt_print( true, LOCAL, LT2_LOCAL );
        }
    }
}

void Router::linkTraversal() {

    if( ST_XBAR_NORTH && ST_XBAR_NORTH->state == ST ) {
        assert( ST_XBAR_NORTH->state == ST );
        LT_NORTH = ST_XBAR_NORTH;
        LT_NORTH->state = LT;
    }

    if( ST_XBAR_SOUTH && ST_XBAR_SOUTH->state == ST ) {
        assert( ST_XBAR_SOUTH->state == ST );
        LT_SOUTH = ST_XBAR_SOUTH;
        LT_SOUTH->state = LT;
    }

    if( ST_XBAR_EAST && ST_XBAR_EAST->state == ST ) {
        assert( ST_XBAR_EAST->state == ST );
        LT_EAST = ST_XBAR_EAST;
        LT_EAST->state = LT;
    }

    if( ST_XBAR_WEST && ST_XBAR_WEST->state == ST ) {
        assert( ST_XBAR_WEST->state == ST );
        LT_WEST = ST_XBAR_WEST;
        LT_WEST->state = LT;
    }
    
//    if( ST_XBAR_LOCAL && ST_XBAR_LOCAL->state == ST ) {
//        LT2_LOCAL = ST_XBAR_LOCAL;
//        LT2_LOCAL->state = LT;
//    }

    if( ST_XBAR_LOCAL && ST_XBAR_LOCAL->state == ST ) {
        if( ST_XBAR_LOCAL->tail_flit ) {
            // all of the message information is embedded into all flits, so
            // really all we need to save is the last one
            completed_msg.push_back( ST_XBAR_LOCAL );
        } else {
            // delete the flits after they have arrived at their destination
            // (only the case for multi-flit messages)
            delete ST_XBAR_LOCAL;
            ST_XBAR_LOCAL = NULL;
        }
    }



    ST_XBAR_NORTH = NULL;
    ST_XBAR_SOUTH = NULL;
    ST_XBAR_EAST = NULL;
    ST_XBAR_WEST = NULL;
    ST_XBAR_LOCAL = NULL;


}

void Router::switchTraversal() {

    
    if( SA_XBAR_NORTH && SA_XBAR_NORTH->state == SA ) {
        ST_XBAR_NORTH = SA_XBAR_NORTH;
        ST_XBAR_NORTH->state = ST;
    }
    
    if( SA_XBAR_SOUTH && SA_XBAR_SOUTH->state == SA ) {
        ST_XBAR_SOUTH = SA_XBAR_SOUTH;
        ST_XBAR_SOUTH->state = ST;
    }

    if( SA_XBAR_EAST && SA_XBAR_EAST->state == SA ) {
        ST_XBAR_EAST = SA_XBAR_EAST;
        ST_XBAR_EAST->state = ST;
    }

    if( SA_XBAR_WEST && SA_XBAR_WEST->state == SA ) {
        ST_XBAR_WEST = SA_XBAR_WEST;
        ST_XBAR_WEST->state = ST;
    }
    
    if( SA_XBAR_LOCAL && SA_XBAR_LOCAL->state == SA ) {
        ST_XBAR_LOCAL = SA_XBAR_LOCAL;
        ST_XBAR_LOCAL->state = ST;
    }

//     if( SA_XBAR_LOCAL && SA_XBAR_LOCAL->state == SA ) {
//         if( SA_XBAR_LOCAL->tail_flit ) {
//             // all of the message information is embedded into all flits, so
//             // really all we need to save is the last one
//             completed_msg.push_back( SA_XBAR_LOCAL );
//         } else {
//             // delete the flits after they have arrived at their destination
//             // (only the case for multi-flit messages)
//             delete SA_XBAR_LOCAL;
// 	    SA_XBAR_LOCAL = NULL;
//         }
//     }



    SA_XBAR_LOCAL = NULL;
    SA_XBAR_EAST = NULL;
    SA_XBAR_WEST = NULL;
    SA_XBAR_NORTH = NULL;
    SA_XBAR_SOUTH = NULL;


    if( ROUTER_DEBUG ) {

        if( ST_XBAR_NORTH || 
            ST_XBAR_SOUTH || 
            ST_XBAR_EAST ||
            ST_XBAR_WEST ||
            ST_XBAR_LOCAL ) {
        
            st_and_lt_print( false, NORTH, ST_XBAR_NORTH );
            st_and_lt_print( false, SOUTH, ST_XBAR_SOUTH );
            st_and_lt_print( false, EAST, ST_XBAR_EAST );
            st_and_lt_print( false, WEST, ST_XBAR_WEST );
            st_and_lt_print( false, LOCAL, ST_XBAR_LOCAL );
        }
    }
}

void Router::st_and_lt_print( bool LT, direction_t dir, flit_t * flit ) {


    if( flit != NULL ) {
        cout << "time " << time << " flit " << hex << flit->addr << dec 
             << " " << flit->length << " " 
             << " is in the ";
        if( LT ) 
            cout << "LT";
        else
            cout << "ST";
        
        cout << " stage on node " << id << " and is clear to go " 
             << direction_str[dir] << endl;
    }
}


//

flit_t * InputBuffer::get_first_VCA ( int vc ) {
    list<flit_t*>::iterator iter;
    flit_t * flit = NULL;

    if( vc_contents[vc].size() > 0 ) {
        for( iter = vc_contents[vc].begin();
             iter != vc_contents[vc].end();
             iter++ ) {
            if( (*iter)->state == RC )
                flit = (*iter);
        }
    }

    return flit;
}

void Router::switchAllocation() {


    // look over all of the VCs for all the inputs and find the lowest timestamp
    // if 2 VCs have the same timestamp, then the tie will be broken by the 
    // direction of origin (the way it came in) based upon N, W, Local, E, and S
    // 
    // perform the crossbar arbitration here where multiple things could be sent at
    // once...because ST and LT can't stall!
    //
    // don't forget to call exit flit here to remove the sent flits from the VC
    // they will be put into the SA_XBAR temporary one-flit buffers

    // also check to see if flits have been stalled in the VCA stage
    // flits can also stall here if there is crossbar contention
    SA_XBAR_NORTH = NULL;
    SA_XBAR_SOUTH = NULL;
    SA_XBAR_EAST = NULL;
    SA_XBAR_WEST = NULL;
    SA_XBAR_LOCAL = NULL;
    
    // find minimum timestamp of pending flit waiting to be sent that hasn't
    // been stalled due to VCA...     
  

    vector< flit_t* >     schedule_list;
    bool XBAR[5][5];

    for( int i = 0; i < 5; i++ ) {
        for( int j = 0; j < 5; j++ ) {
            XBAR[i][j] = false;
        }
    }

    

    for( int i = 0; i < 5; i++ ) {
        for( int j = 0; j < num_vc; j++ ) {
            flit_t * flit = input_buffers[i]->peek(j);
//            flit_t * flit = input_buffers[i]->get_first_VCA(j);
            
            if( flit && !flit->VCA_stall && 
                flit->out_direction != NOT_COMPUTED &&
                flit->state == VCA ) {
                
//                cout << flit->length << " " << i << " " << j << endl;
                schedule_list.push_back( flit );
//                cout << "paul " << hex << flit->addr << dec << endl;
            }
        }
    }

    while( schedule_list.size() > 0 ) {
        int min = get_min_schedule_flit( schedule_list );
        flit_t * flit = schedule_list.at(min);

        assert(min != -1 );


        // is this valid on the crossbar?
        bool valid = true;
        

        for( int input = 0; input < 5; input++ ) {
            if( XBAR[input][flit->out_direction] )
                valid = false;
        }

        for( int output = 0; output < 5; output++ ) {
            if ( XBAR[flit->in_direction][output] ) 
                valid = false;
        }

        // remove from consideration of the schedule list either way
        // I fucking hate the STL right now

        vector<flit_t*>::iterator iter;
        int index=0;

        for( iter = schedule_list.begin(); iter != schedule_list.end(); iter++ ) {
            if( index == min ) {
//                cout << " erasing " << (*iter)->length << endl;
                schedule_list.erase(iter);
                break;
            }
            index++;
        }           

        if( !valid ) {
            cout << " SA stall " << endl;
        }
        
        if( valid ) {

            XBAR[flit->in_direction][flit->out_direction] = true;
            flit->state = SA;
            switch( flit->out_direction ) {
            case NORTH: SA_XBAR_NORTH = flit; break;
            case SOUTH: SA_XBAR_SOUTH = flit; break;
            case EAST:  SA_XBAR_EAST = flit; break;
            case WEST:  SA_XBAR_WEST = flit; break;
            case LOCAL: SA_XBAR_LOCAL = flit; break;
            case NOT_COMPUTED: assert(0); break;
            }

            if( ROUTER_DEBUG ) {
                cout << "time " << time << " flit " << hex << flit->addr << dec
                     << " " << flit->length << " " 
                     << " is in SA stage on node " << id << " and is clear to go " << endl;
//                     << direction_str[flit->out_direction] <<  " " << input << direction_str[LOCAL] << " VC " << vc << endl;
            }

            if( JESSE_DEBUG ) {
                cout << "\t\t" << time << ", node " << id << ": Flit SA going ";
                cout.width(5);
                     cout << left << direction_str[flit->in_direction] << " to ";
                     cout.width(5);
                cout << direction_str[flit->out_direction] << " - " << right;
                if( flit->head_flit ) cout << "is HEAD, ";
                if( flit->tail_flit ) cout << "is TAIL, ";
                cout << "src: " << flit->src << " ";
                cout << "dest: " << flit->dst << " ";
                cout << "fwd: " << flit->fwd << " ";
                cout << "addr 0x" << hex << flit->addr << dec << " ";

                const char * msg_str[50] = {

                    "READ_MISS",
                    "READ_MISS_FORWARD",
                    "READ_MISS_REPLY_DATA",
                    "WRITE_HIT_UPDATE_DATA",
                    "WRITE_MISS",
                    "WRITE_MISS_FORWARD",
                    "WRITE_MISS_REPLY",
                    "WRITE_MISS_REPLY_DATA",
                    "WRITE_MISS_UPDATE_MEM",
                    "INVALIDATE_FORWARD",
                    "READ_DATA_REPLY_DEFER",
                    "WRITE_DATA_REPLY_DEFER"
                };

                
                cout << "msg: " << msg_str[flit->msg_type] << endl;

            }
                

            for( int i = 0; i < 5; i++ ) {
                for( int j = 0; j < num_vc; j++ ) {
                    if( input_buffers[i]->is_pending( j, flit->message_id ) ) {
                        input_buffers[i]->exit_flit( j );
                        break;
                    }
                }                                
            }        
        }
    }   
}

int Router::get_min_schedule_flit( vector< flit_t * > schedule_list ) {

    int index = 0;
    int min_index = -1;
    direction_t min_direction = NOT_COMPUTED;

    unsigned int min_timestamp = time;
    unsigned int min_network_timestamp = time;
    
//    cout << "size " << schedule_list.size() << endl;
    
    vector<flit_t *>::iterator i;

//     if( time == 96 && id == 5 ) {
//         for( i = schedule_list.begin();
//              i != schedule_list.end(); i++ ) {
            
//             cout << hex << (*i)->addr << " " << dec << direction_str[(*i)->in_direction]
//                  << " " << direction_str[(*i)->out_direction] 
//                  << " " << (*i)->timestamp << " " << (*i)->enter_network_time 
//                  << endl;
//         }
//     }



    for( i = schedule_list.begin();
         i != schedule_list.end();
         i++ ) {

//        cout << "schedule min timestamp " << (*i)->timestamp << endl;

        if( (*i)->timestamp < min_timestamp ) {
            min_index = index;
            min_timestamp = (*i)->timestamp;
            min_direction = (*i)->in_direction;
            min_network_timestamp = (*i)->enter_network_time;

        } else if ( (*i)->timestamp == min_timestamp ) {
            
            if( (*i)->in_direction < min_direction ) {
                min_timestamp = (*i)->timestamp;
                min_direction = (*i)->in_direction;
                min_network_timestamp = (*i)->enter_network_time;
                min_index = index;

            } else if( (*i)->in_direction == min_direction ) {
                
                if( (*i)->enter_network_time < min_network_timestamp ) {
                    min_network_timestamp = (*i)->enter_network_time;
                    min_timestamp = (*i)->timestamp;
                    min_direction = (*i)->in_direction;
                    min_index = index;
                } else {
                    assert(0);
                }
            }
        }

        index++;

    }
//    cout << "returning index " << min_index << endl;
    return min_index;
}




void Router::vcAllocation() {

    int row = (id & 0xc) >> 2;  // upper 2 bits of id
    int col = id & 0x3;         // lower 2 bits of id
        
    int north_neighbor = -1;
    int south_neighbor = -1;
    int east_neighbor  = -1;
    int west_neighbor  = -1;


    if( row > 0 ) {
        north_neighbor = ((row-1) << 2) | col;
    }

    if( row < 3 ) {
        south_neighbor = ((row+1) << 2) | col;
    }

    if( col > 0 ) {
        west_neighbor = (row << 2) | (col-1);
    }

    if( col < 3) {
        east_neighbor = (row << 2) | (col+1);
    }
    
//    cout << "id " << id << " checking south vc of node " << north_neighbor << endl;
    

    if( north_neighbor != -1 )
        vcAllocHelper( SOUTH, north_neighbor );

    if( south_neighbor != -1 )
        vcAllocHelper( NORTH, south_neighbor );

    if( east_neighbor != -1 )
        vcAllocHelper( WEST, east_neighbor );
    
    if( west_neighbor != -1 )
        vcAllocHelper( EAST, west_neighbor );

    vcAllocHelper( LOCAL, id );
}

//TODO i think i forgot about local node VCA...what happens if a flit is going
//to the current destination node?  probably doesn't matter, i think this happens
//in the SA stage

// we may need to add an additional variable to the Input Buffers to indicate
// that a VC has been claimed by the previous

flit_t * InputBuffer::get_first_RC ( int vc ) {
    list<flit_t*>::iterator iter;
    flit_t * flit = NULL;

    if( vc_contents[vc].size() > 0 ) {
        for( iter = vc_contents[vc].begin();
             iter != vc_contents[vc].end();
             iter++ ) {
            if( (*iter)->state == RC )
                flit = (*iter);
        }
    }

    return flit;

    
}


void Router::vcAllocHelper( direction_t remote_dir, unsigned int neighbor ) {


    assert( neighbor <= 15 );


    flit_t * local_flit;
    int vc;

    direction_t local_dir;
    if( remote_dir == NORTH ) local_dir = SOUTH;
    else if( remote_dir == SOUTH ) local_dir = NORTH;
    else if( remote_dir == EAST ) local_dir = WEST;
    else if( remote_dir == WEST) local_dir = EAST;

    // check local VCs to see if a packet is ready to be transferred
    // an out_direction of -1 means that it hasn't gone through the 
    // RC stage yet

    // check all local direction input buffers VCs for the neighbor specified
    // by the direction dir...

    //if the flit is going towards the local node, no VC needs to be checked with a remote neighbor
    else{
        
        for( int dir = 0; dir < 5; dir ++ ) {
            for( int i = 0; i < num_vc; i++ ) {
                //local_flit = input_buffers[local_dir]->peek(i);
                local_flit = input_buffers[dir]->get_first_RC(i);

                // a flit going to local in VCA can't stall
                if( local_flit && local_flit->out_direction == LOCAL &&
                    local_flit->state == RC ) {

                    local_flit->VCA_stall = false;
                    local_flit->state = VCA;
                    
                    if( ROUTER_DEBUG ) {
                        cout << "time " << time << " flit " << hex << local_flit->addr << dec 
                             << " " << local_flit->length << " " 
                             << " is in VCA stage "
                             << "on node " << id << " and is clear to go "
                             << direction_str[remote_dir] 
                             << " " << "input" << direction_str[dir] << " VC " << i << endl;
                    }
                }                
            }
        }

        return;
    }
   
    for( int dir = 0; dir < 5; dir++ ) {        
        for( int i = 0; i < num_vc; i++ ) {

            //local_flit = input_buffers[dir]->peek(i);
            local_flit = input_buffers[dir]->get_first_RC(i);

            if( !local_flit )
                continue;

            assert( local_flit->state == RC );	 	 
            //
            if( local_flit->out_direction == local_dir && local_flit->state == RC) {

                InputBuffer * ib = network_routers[neighbor]->getInputBuffer(remote_dir);
                vc = -1;

                for( int remote_vc = 0; remote_vc < num_vc; remote_vc++ ) {

                    if( ib->is_pending( remote_vc, local_flit->message_id ) ) {
                        vc = remote_vc;
                        break;
                    }
                    if( remote_vc == num_vc-1 ) {
                        vc = ib->get_free_vc();
                        if( vc != -1 ) {
	                        // set this VC as busy and allocated with the current message
	                       ib->set_msg_id( vc, local_flit->message_id );
                        }
                    }
                }

                // vc is -1 if none are free and if the message isn't currently occupying
                // the neighbor's input buffer
                if( vc != -1 ) {
                    if( !ib->vc_full( vc ) ) {
                        // we're a go for xfer...  need to figure out a way to stall if
                        // any of these conditions aren't true
                        local_flit->VCA_stall = false;
                        local_flit->state = VCA;

                        if( ROUTER_DEBUG ) {
                            cout << "time " << time << " flit " << hex << local_flit->addr 
                                 << " " << local_flit->length << " " 
                                 << dec << " is in VCA stage "
                                 << "on node " << id << " and is clear to go "
                                 << direction_str[local_dir]
                                 << " " << "input" << direction_str[dir] << " VC " << vc << endl;
                        }
                    } else {
                        // stall because the neighbors VC is already full....only possible
                        // if the message was already pending.
                        local_flit->VCA_stall = true;
//                        cout << " VCA stall " << endl;
                    }
                } else {
                    // stall if no free VCs are available
                    local_flit->VCA_stall = true;
//                    cout << " VCA stall " << endl;
                }                
            }            
        }
    }
}

flit_t * InputBuffer::get_first_IB ( int vc ) {
    list<flit_t*>::iterator iter;
    flit_t * flit = NULL;

    if( vc_contents[vc].size() > 0 ) {
        for( iter = vc_contents[vc].begin();
             iter != vc_contents[vc].end();
             iter++ ) {
            if( (*iter)->state == IB )
                flit = (*iter);
        }
    }

    return flit;

    
}

void Router::routeCalculation() {
    
    flit_t * head_flit;
    
    unsigned int thisRow, thisCol;
    unsigned int dstRow, dstCol;
    

    thisRow = (id & 0xc) >> 2;
    thisCol = (id & 0x3);
    
    // compute the route for all head flits for all VCs on all inputs
    for( int dir = 0; dir < 5; dir++ ) {
        for( int vc = 0; vc < num_vc; vc++ ) {

            //updates the output direction
            //head_flit = input_buffers[dir]->peek( vc );
            head_flit = input_buffers[dir]->get_first_IB( vc );

            if( !head_flit )
                continue;

            if( head_flit->state == IB ) {

                
            
                //row resolves firs1t because flits travel N/S first before
                //going E/W (column)
                if( head_flit->dst == id ) {  // the flit has reached its destination

                    head_flit->out_direction = LOCAL;

                } else {

                    dstRow = ((head_flit->dst) & 0xc) >> 2;
                    dstCol = ((head_flit->dst) & 0x3);
                    
                    if( dstRow != thisRow ) { // resolve N/S traffic destination
                        if( dstRow > thisRow ) {
                            head_flit->out_direction = SOUTH;
                        } else { //(dstRow < thisRow)
                            head_flit->out_direction = NORTH;
                        }
                        
                    } else if ( dstCol != thisCol ) { // resolve E/W traffic destination
                        if( dstCol > thisCol ) {
                            head_flit->out_direction = EAST;
                        } else { // (dstCol < thisCol)
                            head_flit->out_direction = WEST;
                        }
                    } else { // route must be local
                        assert(0);
                    }
                }
                head_flit->state = RC;
                
                if( ROUTER_DEBUG ) {
//                    cout << "flit dest " << head_flit->dst << endl;
                    cout << "time " << time << " flit " << hex << head_flit->addr << dec 
                         << " " << head_flit->length << " " 
                         << " is in RC stage"
                         << " on node " << id
                         << " calculated direction " 
                         << direction_str[head_flit->out_direction]
                         << " " << "input" << direction_str[dir] << " VC " << vc << endl;
//                cout << "located in vc " << vc << endl;
//                for( unsigned int i = 0; i < 5; i++ ) {
//                    input_buffers[i]->print();
//                }
                }
            }
        }        
    }
}

void Router::issueBuffer() {

    // try to request flits to all VCs from all directions, including local

    unsigned int row = (id & 0xc) >> 2;  // upper 2 bits of id
    unsigned int col = id & 0x3;  // lower 2 bits of id
        
    int north_neighbor = -1;
    int south_neighbor = -1;
    int east_neighbor  = -1;
    int west_neighbor  = -1;


    if( row > 0 ) {
        north_neighbor = ((row-1) << 2) | col;
    }

    if( row < 3 ) {
        south_neighbor = ((row+1) << 2) | col;
    }

    if( col > 0 ) {
        west_neighbor = (row << 2) | (col-1);
    }

    if( col < 3) {
        east_neighbor = (row << 2) | (col+1);
    }

//    cout << "north router neighbor " << north_neighbor << endl;                        
//    cout << "south router neighbor " << south_neighbor << endl;
//    cout << "east router neighbor " << east_neighbor << endl;
//    cout << "west router neighbor " << west_neighbor << endl;


    // no tie breakers should really be happening here...but I'm still
    // resolving IB entries in the order of travel: N, S, Local, E, W.
    if( north_neighbor != -1 )
        issueBufferHelper( SOUTH, north_neighbor ); //northbound traffic

    if( south_neighbor != -1 )
        issueBufferHelper( NORTH, south_neighbor ); //southbound traffic

    //INJECT flits into the local node is not part of IB!!!!

    if( east_neighbor != -1 )
        issueBufferHelper( WEST, east_neighbor );   //eastbound traffic

    if( west_neighbor != -1 )
        issueBufferHelper( EAST, west_neighbor );   //westbound traffic  
}


void Router::issueBufferHelper( direction_t dir, unsigned int neighbor ) {

    // check the direction of the neighbor and get the pending LT stage
    // flit via the get_LT_flit member function
    
    // this should never stall!
    // I'm checking the condition of the VC to make sure that everything
    // is sane!  go asserts!

    assert( neighbor <= 15 );

    direction_t local_dir;
    if( dir == NORTH ) local_dir = SOUTH;
    else if( dir == SOUTH ) local_dir = NORTH;
    else if( dir == EAST ) local_dir = WEST;
    else if( dir == WEST ) local_dir = EAST;
    else{ assert( 0 ); }

    flit_t * local_flit;

    if( dir == LOCAL ) {
        local_flit = get_LT_flit( dir );
    } else {
        local_flit = network_routers[neighbor]->get_LT_flit(dir);               
    }

    int vc = -1;

    if( local_flit && (local_flit->state == LT || local_flit->state == NAS )) {

        // use the existing VC if the message has already been started
        for( int i = 0; i < num_vc; i++ ) {
            if( input_buffers[ local_dir ]->is_pending( i, local_flit->message_id ) ) {
                vc = i;
                break;
            }

            // if this is a new message, then get a free VC
            if( i == num_vc-1 ) {
                vc = input_buffers[ local_dir ]->get_free_vc();
            }
        }
        
        // don't execute if there are no free VCs or the VC is already full
        if( vc != -1 && !input_buffers[local_dir]->vc_full( vc ) ) {

            
            // when a flit is entered into this router, update it's time
            local_flit->timestamp = time;
            
            // reset the output direction so we can recognize of a flit
            // has actually gone through the RC stage
            local_flit->out_direction = NOT_COMPUTED;
            local_flit->in_direction = local_dir;
            local_flit->state = IB;

            input_buffers[local_dir]->set_msg_id( vc, local_flit->message_id );
            
            bool retval = input_buffers[local_dir]->enter_flit( local_flit, vc );      
            assert( retval );
            
            if( dir == LOCAL ) {
                clear_LT_flit( dir );
            } else {
                network_routers[neighbor]->clear_LT_flit( dir );
            }

            if( ROUTER_DEBUG ) {            
                cout << "time " << time << " flit " << hex << local_flit->addr << dec 
                     << " " << local_flit->length << " " << " is in " 
                     << "IB stage " << " on node " << id
                     << " " << "input" << direction_str[local_dir] << " VC " << vc << endl;
            }                                         
        } else {
            // cout << " IB stall " << endl;
        }
    }       
}


void Router::injectNewVCA( ) {

   int vc = -1;
   flit_t * flit = NULL;


    if( waiting_IB_flits.size() > 0 ) {

        flit = waiting_IB_flits.front();

        if( time == flit->timestamp ) {
            return;
        }
        
        for( int i = 0; i < num_vc; i++ ) {
            if( input_buffers[LOCAL]->is_pending( i, flit->message_id ) ) {
                vc = i;
                break;
            }

            
            if( i == num_vc-1 ) {
                vc = input_buffers[LOCAL]->get_free_vc();
                if(vc >= 0){
                    //cout <<" allocating local VCA at time " << time << endl;        
                    input_buffers[LOCAL]->set_msg_id( vc, flit->message_id );                
                }
            }
        }
        
    }

}

void Router::injectNewFlit( ) {

    flit_t * flit = NULL;
    int vc = -1;

    if( waiting_IB_flits.size() > 0 ) {

        flit = waiting_IB_flits.front();


        if( time == flit->timestamp + 1 ) {
            return;
        }

//        cout << "injecting new local flit at time " << time << endl;

        for( int i = 0; i < num_vc; i++ ) {
            if( input_buffers[LOCAL]->is_pending( i, flit->message_id ) ) {
                vc = i;
                break;
            }
        }


        if( vc != -1 && !input_buffers[LOCAL]->vc_full(vc) ) {

            bool retval = input_buffers[LOCAL]->enter_flit( flit, vc );
            assert( retval );
            waiting_IB_flits.pop_front();

            
            flit->timestamp = time;
            flit->out_direction = NOT_COMPUTED;
            flit->in_direction = LOCAL;
            flit->state = IB;
            
            
            if( ROUTER_DEBUG ) {            
                cout << "time " << time << " flit " << hex << flit->addr << dec 
                     << " " << flit->length << " " 
                     << " is in " 
                     << "IB stage " << " on node " << id
                     << " " << "input" << direction_str[LOCAL] << " VC " << vc << endl;
                    
            } 
        } else {
            // cout << " IB stall " << endl;
        }
    }    
}



flit_t * Router::get_LT_flit( direction_t dir ) {

    flit_t * flit = NULL;
    if( dir == NORTH ) {         
        flit = LT2_NORTH;
    } else if ( dir == SOUTH ) {
        flit =  LT2_SOUTH;
    } else if ( dir == EAST ) {
        flit =  LT2_EAST;
    } else if ( dir == WEST ) {
        flit =  LT2_WEST;
    } else if ( dir == LOCAL ) {
//        if( waiting_IB_flits.size() > 0 ) {
//            flit =  waiting_IB_flits.front(); // get the first element
//        }
        assert(0); // don't think this should ever happen
    } else {
        cout << "get_LT_flit called with invalid direction!" << endl;
        throw;
    }

//if( flit && flit->timestamp < time ) {
    if( flit ) {
        return flit;
    }
    return NULL;
}

void Router::clear_LT_flit( direction_t dir ) {
    if( dir == NORTH ) {
        LT2_NORTH = NULL;
    } else if ( dir == SOUTH ) {
        LT2_SOUTH = NULL;
    } else if ( dir == EAST ) {
        LT2_EAST = NULL;
    } else if ( dir == WEST ) {
        LT2_WEST = NULL;
    } else if( dir == LOCAL ) {
//        LT_LOCAL = NULL;
        // pop off the waiting_IB_flits buffer
//        if( waiting_IB_flits.size() > 0 )
//            waiting_IB_flits.pop_front();
    } else {
        cout << "clear_LT_flit called with invalid direction!" << endl;
        throw;
    }
}

void Router::push_network_enqueue_waiting_flit( flit_t * flit ) {

    waiting_IB_flits.push_back( flit );

//    cout << "Adding flit to IB waiting queue" << endl;
}

void Router::router_tick() {
    time++;
}

InputBuffer * Router::getInputBuffer( direction_t dir ) {
    
    return input_buffers[dir];
}
