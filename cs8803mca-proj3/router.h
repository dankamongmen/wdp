#ifndef __router_h__
#define __router_h__

#include <list>
#include <vector>
#include "interface.h"

using namespace std;

// enable router debugging information
#define ROUTER_DEBUG 0
#define JESSE_DEBUG 0


typedef enum {
    NORTH = 0,
    WEST,
    LOCAL,
    EAST,
    SOUTH,
    NOT_COMPUTED
} direction_t;

extern const char * direction_str[10];

typedef enum {    
    IB,
    RC,
    VCA,
    SA,
    ST,
    LT,
    NAS,
} state_t;

typedef struct {
    
    COHERENCE_MSG msg_type;

    unsigned int enter_network_time;
    unsigned int timestamp;

    // message id necessary to make sure that the buffered flits 
    // (going into the VC) are part of the same message
    unsigned int message_id; 

    // added this to determine if particular flits are the head 
    // (start of the message)
    bool head_flit;          

    // added this to determine if particular flits are the tail 
    // (end of the message)
    bool tail_flit;          

    unsigned int src;    // id of the source network router
    unsigned int dst;    // id of the destination network router
    unsigned int fwd;    // id of the forwarded address used on the other side
    unsigned int addr;   // address of the message
    unsigned int length; // length of the message that gets decremented

    direction_t out_direction; // direction the flit is travelling to
    direction_t in_direction;  // direction the flit just came from

    state_t state;
    bool VCA_stall;

} flit_t;


class InputBuffer {

private:        

    int num_vc;
    int max_size; // max buffer size of VC    

    list<flit_t *> * vc_contents;

    int * size; // number of flits in each of the VCs
    direction_t dir;
    
    unsigned * msg_id; // associates a message id with a VC
    bool * busy;  // indicates if the VC has already been allocated an id
    
    
    
public:
    InputBuffer( int channels, 
                 int buffer_size,  
                 direction_t direction );

    ~InputBuffer( );
    
    // return -1 if you can't find one, o/w return the vc channel
    // also, request a vc from the direction where IB called for the
    // first flit of a message
    int get_free_vc( ); 

    // while we are writing to a particular VC, it may become full due to stalls
    // this is use to queury if there is more buffer space....maybe not needed.
    bool vc_full( int vc_num );

    // called by the IB router stage to enter a flit into a particular VC direction
    //  (N, E, local, S, W)
    bool enter_flit( flit_t * flit, int vc_num );

    // called by the SA router stage to remove a flit from the VC.  This should
    // only be called if the flit can actually travel across the link
    flit_t * exit_flit( int vc_num );

    // returns the VC number if the specified input buffer direction 
    // if it has a message with the same id 
    // (it was already processing this message), o/w it returns -1
    bool is_pending( int vc, unsigned int message_id );

    // used by other route stages to get the head flit information for a given VC
    flit_t * peek( int vc );

    // stupid hack job
    flit_t * get_first_IB ( int vc );
    flit_t * get_first_RC( int vc );
    flit_t * get_first_VCA( int vc );
    
    
    // used in the VCA stage to allocate a VC for the neighbor, should
    // be used with get_free_vc...!
    void set_msg_id( int vc, unsigned int msg_id );  

    void print();

};



class Router {


private:
    int link_width;
    int num_vc;
    int buffer_size_per_vc;
    InputBuffer ** input_buffers;
    // buffer flits local flits waiting to be put in the VC at the IB stage
    list<flit_t *> waiting_IB_flits;

    // list of messages that have arrived to this router
    list<flit_t *> completed_msg; 

    unsigned int time;
    unsigned int id; // id of the router, specified by network_register
    
    // temporary buffer space to hold the flits while they are being
    // sent through the pipeline (and taken off the input buffer VC list)
    // if any of these are assigned (they should be NULL)
    // a flit will be travelling to the ST stage.  They will be allocated
    // by SA, and ST (switch traversal) will move them to the ST version
    flit_t * SA_XBAR_NORTH;
    flit_t * SA_XBAR_SOUTH;
    flit_t * SA_XBAR_EAST;
    flit_t * SA_XBAR_WEST;
    flit_t * SA_XBAR_LOCAL;

    // These will be loaded by ST state.                                    
    flit_t * ST_XBAR_NORTH;
    flit_t * ST_XBAR_SOUTH;
    flit_t * ST_XBAR_EAST;
    flit_t * ST_XBAR_WEST;
    flit_t * ST_XBAR_LOCAL;

    // These will be loaded by the LT state.  The IB of the neighbor
    // will check each of these via the accessor function get_LT_flit()
    flit_t * LT_NORTH;
    flit_t * LT_SOUTH;
    flit_t * LT_EAST;
    flit_t * LT_WEST;
    flit_t * LT_LOCAL;  

    flit_t * LT2_NORTH;
    flit_t * LT2_SOUTH;
    flit_t * LT2_EAST;
    flit_t * LT2_WEST;
    flit_t * LT2_LOCAL;  


public:
    Router( int link_width, 
            int virtual_channels,
            int buffers_per_vc,
            int router_id         );

    ~Router( );    
    
    // each time that a router stage function is called, then
    // iterate over all of the directions: N, E, Local, S, W.


    // The LT stage of the router
    // 
    // Stalls if: never
    // 
    // TODO lookup up cross-bar...up to N flits....
    void linkTraversal( );

    // The ST stage of the router
    //
    // Stalls if: never
    //
    // TODO 2 flit_t buffer....dependent upon previous
    //  NS / EW connection output router
    // updates the input/in-going direction at the flit_t level flit->in_direction
    void switchTraversal( );

    // The SA stage of the router
    // 
    // Stalls if: ?? not entirely certain about this one.
    //
    // responsible for removing flits that have been cleared for travel by the
    // VCA router stage.  Switch allocation looks at all incoming flits and 
    // schedules the oldest one.  If two arrive at the same time then ties 
    // are resolved based upon the direction of the incoming flit.  Priorities
    // are as follows:  going North, going West, coming in from Local, going
    // East, and going South.  If ties are still not broken, then the flit
    // with the lowest enqueue age (flit->timestamp) will be sent first.
    
    // The crossbar determines how many flits can be broadcast to the switch 
    // simultaneously.  No two packets can travel the same wire, but as many 
    // wires can be used individually... from the local router's perspective
    // a North packet could be going South and the local could send to East. 
    // But the local can't send to North and South because there's only one
    // wire to local.
    // pseudo code: (this is wrong now.  We should update the pseudo code...)
    //    flit_t * NSout = NULL, * EWout = NULL;
    //    foreach pending flit_t * p
    //    if( flit->out_direction == NORTH || flit->out_direction == SOUTH )
    //      if( !NWout ) {
    //        NSout = p;
    //        remove flit_t from list/VC/whatev
    //      }
    //    } else if ( ... do the same for east and west )
    //
    // TODO FIX THE PSEUDO CODE THAT IS WRONG
    
    void switchAllocation( );

    
    // The VCA stage of the router
    // 
    // Stalls if: the flit->out_direction router cannot get a free VC
    //            the flit->out_direction router's allocated VC is full
    //            
    // This will update the InputBuffer of the output direction router to
    // allocate a particular VC for the flit_t that is being transferred
    // 
    void vcAllocation( );

    // The RC stage of the router
    // 
    // Stalls if: never
    // 
    // This will determine the direction of travel given the current router id
    // and the destination id.  Routes are determistic to go North, South first
    // and then East, West to the appropriate device.  This will update the 
    // flit->out_direction
    void routeCalculation( );

    
    // The IB stage of the router
    //
    // Stalls if: a VC cannot be allocated for a head flit.
    //            the vc that it has been allocated is full
    //            the flit_t is not part of the same message that is currently occupying the vc
    // If all of these conditions are good, then it writes a flit_t * into the VC
    // of the appropriate direction.    
    //
    // Routers will poll their neighbors to see if a flit_t is waiting in the LT
    // stage for this router to accept.  This can be done with the get_LT_flit
    //
    // Also timestamp the flits as they arrive in the IB stage
    void issueBuffer( );


    // N, S, E, W, will get the flit_t located in the LT_NORTH, LT_SOUTH, etc.
    // based upon the direction
    flit_t * get_LT_flit( direction_t dir );

    // after the flit_t has been transferred, clear its state (back to NULL)
    void clear_LT_flit( direction_t dir );

    // push flits generated by local onto a queue to wait to be added to the
    // appropriate VC in the IB stage
    void push_network_enqueue_waiting_flit( flit_t * flit );


    void issueBufferHelper( direction_t dir, unsigned int neighbor );
    void vcAllocHelper( direction_t dir, unsigned int neighbor );

    // increment the timestamp of the router by one
    void router_tick();

    InputBuffer * getInputBuffer( direction_t dir );

    void st_and_lt_print( bool stage, direction_t dir, flit_t * flit );

    flit_t * getCompletedMsg( );

    void injectNewFlit();
    void injectNewVCA();
    
    void linkTraversal2();


    int get_min_schedule_flit( vector< flit_t * > schedule_list );

};


#endif
