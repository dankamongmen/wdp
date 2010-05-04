#include "timing.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/****************************************************************
 * clock_gettime: POSIX high-resolution timer
 ****************************************************************/

#include <time.h>

#if !defined(HAVE_TIMER) && (defined(CLOCK_HIGHRES) || defined(CLOCK_REALTIME))
#  define TIMER_DESC "clock_gettime: POSIX high-resolution timer"

#  if defined(CLOCK_HIGHRES)
#    define CLOCK CLOCK_HIGHRES
#  else /* defined(CLOCK_REALTIME) */
#    define CLOCK CLOCK_REALTIME
#  endif

#define USE_STD_CREATE
#define USE_STD_DESTROY

static
long double
timespec_to_ldbl (struct timespec x)
{
  return x.tv_sec + 1.0E-9 * x.tv_nsec;
}

static
long double
timespec_diff (struct timespec start, struct timespec finish)
{
  long double out;
  out = finish.tv_nsec - (double)start.tv_nsec;
  out *= 1.0E-9L;
  out += finish.tv_sec - (double)start.tv_sec;
  return out;
}

static
long double
timer_resolution (void)
{
  struct timespec x;
  clock_getres (CLOCK, &x);
  return timespec_to_ldbl (x);
}

static
void
get_time (struct timespec* x)
{
  clock_gettime (CLOCK, x);
}

/* ======= */

struct stopwatch_t
{
  struct timespec t_start_;
  struct timespec t_stop_;
  int is_running_;
};

void
stopwatch_init (void)
{
  fprintf (stderr, "Timer: %s\n", TIMER_DESC);
  fprintf (stderr, "Timer resolution: %Lg\n", timer_resolution ());
  fflush (stderr);
}

void
stopwatch_start (struct stopwatch_t* T)
{
  assert (T);
  T->is_running_ = 1;
  get_time (&(T->t_start_));
}

long double
stopwatch_stop (struct stopwatch_t* T)
{
  if (T && T->is_running_) {
    get_time (&(T->t_stop_));
    T->is_running_ = 0;
  }
  return stopwatch_elapsed (T);
}

long double
stopwatch_elapsed (struct stopwatch_t* T)
{
  long double dt = 0;
  if (T) {
    if (T->is_running_) {
      struct timespec t_cur;
      get_time (&t_cur);
      dt = timespec_diff (T->t_start_, t_cur);
    } else {
      dt = timespec_diff (T->t_start_, T->t_stop_);
    }
  }
  return dt;
}

#  define HAVE_TIMER 1
#endif

/****************************************************************
 * gettimeofday: Better than nothing, I suppose.
 ****************************************************************/
#if !defined(HAVE_TIMER)
#  define TIMER_DESC "gettimeofday"

#define USE_STD_CREATE
#define USE_STD_DESTROY

#include <sys/time.h>

struct stopwatch_t
{
  struct timeval t_start_;
  struct timeval t_stop_;
  int is_running_;
};

void
stopwatch_init (void)
{
  fprintf (stderr, "Timer: %s\n", TIMER_DESC);
  fprintf (stderr, "Timer resolution: ~ 1 us (?)\n");
  fflush (stderr);
}

void
stopwatch_start (struct stopwatch_t* T)
{
  assert (T);
  T->is_running_ = 1;
  gettimeofday (&(T->t_start_), 0);
}

long double
stopwatch_stop (struct stopwatch_t* T)
{
  long double dt = 0;
  if (T) {
    if (T->is_running_) {
      gettimeofday (&(T->t_stop_), 0);
      T->is_running_ = 0;
    }
    dt = stopwatch_elapsed (T);
  }
  return dt;
}

static
long double
elapsed (struct timeval start, struct timeval stop)
{
  return (long double)(stop.tv_sec - start.tv_sec)
    + (long double)(stop.tv_usec - start.tv_usec)*1e-6;
}

long double
stopwatch_elapsed (struct stopwatch_t* T)
{
  long double dt = 0;
  if (T) {
    if (T->is_running_) {
      struct timeval stop;
      gettimeofday (&stop, 0);
      dt = elapsed (T->t_start_, stop);
    } else {
      dt = elapsed (T->t_start_, T->t_stop_);
    }
  }
  return dt;
}

#  define HAVE_TIMER 1
#endif

/****************************************************************
 * Base-case: No portable timer found.
 ****************************************************************/
#if !defined(HAVE_TIMER)
#  error "Can't find a suitable timer for this platform! Edit 'timer.c' to define one."
#endif

/****************************************************************
 * "Generic" methods that many timers can re-use.
 * (A hack to emulates C++-style inheritance.)
 ****************************************************************/

#if defined(USE_STD_CREATE)
struct stopwatch_t *
stopwatch_create (void)
{
  struct stopwatch_t* new_timer =
    (struct stopwatch_t *)malloc (sizeof (struct stopwatch_t));
  if (new_timer)
    memset (new_timer, 0, sizeof (struct stopwatch_t));
  return new_timer;
}
#endif

#if defined(USE_STD_DESTROY)
void
stopwatch_destroy (struct stopwatch_t* T)
{
  if (T)
    free (T);
}
#endif

/* eof */

