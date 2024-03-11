#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/delayed.h>
#include <parlay/parallel.h>
namespace parlay {
#define PARLAY_USE_STD_ALLOC 1

  using scheduler_type = internal::scheduler_type;

  template <typename F>
  long tabulate_reduce(long n, const F& f) {
    return parlay::reduce(parlay::delayed::tabulate(n, [&] (size_t i) {
	     return f(i);}));
  }
}