set xrange [1000:1000000]
set terminal pdf
set title "LRUMAP vs Classic True LRU, Order 8"
set xtics 300000
set xlabel "LRU sets"
set ylabel "Bytes used"
set key left box
# can we not parameterize these two? ugh
set out "out/lrumap8.pdf"
lruorder = 8
lg2(x) = log(x) / log(2)
fac(n) = int(n)!
lruspace(n,r) = n * r * lg2(r)
lrumapspace(n,r) = fac(r) * r * lg2(r) + n * lg2(fac(r))
plot lrumapspace(x,lruorder) title "LRUMAP" with points, \
	lruspace(x,lruorder) title "Classic LRU" with points
