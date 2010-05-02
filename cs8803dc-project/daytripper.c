// http://dank.qemfd.net/dankwiki/index.php/Daytripper
// initial structure based off DynamoRIO's inc2add client

#include "dr_api.h"

#ifdef WINDOWS
# define ATOMIC_INC(var) _InterlockedIncrement((volatile LONG *)(&(var)))
#else
# define ATOMIC_INC(var) __asm__ __volatile__("lock incl %0" : "=m" (var) : : "memory")
#endif

static enum {
	DAYTRIPPER_NOLSD,
	DAYTRIPPER_CONROE,
	DAYTRIPPER_NEHALEM,
} enable;

/* use atomic operations to increment these to avoid the hassle of locking. */
static int num_examined, num_converted;

static dr_emit_flags_t
bbcb(void *drcontext,void *tag,instrlist_t *bb,bool tracep,bool transp){
	if(dr_is_notify_on()){
		dr_printf("in dynamorio_trace(tag="PFX")\n", tag);
		instrlist_disassemble(drcontext, tag, bb, STDOUT);
	}
	return DR_EMIT_DEFAULT;
}

static void 
event_exit(void){
	dr_log(NULL, LOG_ALL, 1, "daytripper: did nothing\n");
	if(dr_is_notify_on()){
		dr_fprintf(STDERR, "daytripper: did nothing\n");
	}
}

DR_EXPORT void 
dr_init(client_id_t id){
	dr_log(NULL, LOG_ALL, 1, "Client 'daytripper' initializing\n");
	dr_register_bb_event(bbcb);
	dr_register_exit_event(event_exit);
	// LSD was introduced on the Conroe, and improved on Nehalem.
	if(proc_get_family() == FAMILY_CORE_2 &&
			(proc_get_model() == MODEL_CORE_2)){
		dr_log(NULL, LOG_ALL, 1, "daytripper: Found a Core 2 processor\n");
		enable = DAYTRIPPER_CONROE;
	}else if(proc_get_family() == FAMILY_CORE_I7 &&
			(proc_get_model() == MODEL_I7_GAINESTOWN ||
			 proc_get_model() == MODEL_I7_CLARKSFIELD)){
		dr_log(NULL, LOG_ALL, 1, "daytripper: Found a Core i7 processor\n");
		enable = DAYTRIPPER_NEHALEM;
	}else{
		dr_log(NULL, LOG_ALL, 1, "daytripper: Unsupported processor\n");
		enable = DAYTRIPPER_NOLSD;
	}
	if(dr_is_notify_on()){
		dr_fprintf(STDERR, "Client daytripper is %s\n",
			enable == DAYTRIPPER_NOLSD ? "inactive" : "active");
	}
	num_examined = 0;
	num_converted = 0;
}

/* static dr_emit_flags_t
event_trace(void *drcontext, void *tag, instrlist_t *trace, bool translating)
{
    instr_t *instr, *next_instr;
    int opcode;

    if (!enable)
	return DR_EMIT_DEFAULT;

#ifdef SHOW_RESULTS
    dr_printf("in dynamorio_trace(tag="PFX")\n", tag);
    instrlist_disassemble(drcontext, tag, trace, STDOUT);
#endif

    for (instr = instrlist_first(trace); instr != NULL; instr = next_instr) {
	// grab next now so we don't go over instructions we insert
	next_instr = instr_get_next(instr);
	opcode = instr_get_opcode(instr);
	if (opcode == OP_inc || opcode == OP_dec) {
            if (!translating)
                ATOMIC_INC(num_examined);
	    if (replace_inc_with_add(drcontext, instr, trace)) {
                if (!translating)
                    ATOMIC_INC(num_converted);
            }
	}
    }

#ifdef SHOW_RESULTS
    dr_printf("after dynamorio_trace(tag="PFX"):\n", tag);
    instrlist_disassemble(drcontext, tag, trace, STDOUT);
#endif

    return DR_EMIT_DEFAULT;
}*/

/* replaces inc with add 1, dec with sub 1 
 * returns true if successful, false if not
static bool
replace_inc_with_add(void *drcontext, instr_t *instr, instrlist_t *trace)
{
    instr_t *in;
    uint eflags;
    int opcode = instr_get_opcode(instr);
    bool ok_to_replace = false;

    DR_ASSERT(opcode == OP_inc || opcode == OP_dec);
#ifdef SHOW_RESULTS
    dr_print_instr(drcontext, STDOUT, instr, "in replace_inc_with_add:\n\t");
#endif

    // add/sub writes CF, inc/dec does not, make sure that's ok
    for (in = instr; in != NULL; in = instr_get_next(in)) {
	eflags = instr_get_eflags(in);
	if ((eflags & EFLAGS_READ_CF) != 0) {
#ifdef SHOW_RESULTS
            dr_print_instr(drcontext, STDOUT, in,
                           "\treads CF => cannot replace inc with add: ");
#endif
	    return false;
	}
	if (instr_is_exit_cti(in)) {
	    // to be more sophisticated, examine instructions at
	    // target of exit cti (if it is a direct branch).
	    // for this example, we give up if we hit a branch.
	    return false;
	}
	// if writes but doesn't read, ok
	if ((eflags & EFLAGS_WRITE_CF) != 0) {
	    ok_to_replace = true;
	    break;
	}
    }
    if (!ok_to_replace) {
#ifdef SHOW_RESULTS
        dr_printf("\tno write to CF => cannot replace inc with add\n");
#endif
	return false;
    }
    if (opcode == OP_inc) {
#ifdef SHOW_RESULTS
        dr_printf("\treplacing inc with add\n");
#endif
	in = INSTR_CREATE_add(drcontext, instr_get_dst(instr, 0),
			      OPND_CREATE_INT8(1));
    } else {
#ifdef SHOW_RESULTS
        dr_printf("\treplacing dec with sub\n");
#endif
	in = INSTR_CREATE_sub(drcontext, instr_get_dst(instr, 0),
			      OPND_CREATE_INT8(1));
    }
    if (instr_get_prefix_flag(instr, PREFIX_LOCK))
        instr_set_prefix_flag(in, PREFIX_LOCK);
    instr_set_translation(in, instr_get_app_pc(instr));
    instrlist_replace(trace, instr, in);
    instr_destroy(drcontext, instr);
    return true;
} */
