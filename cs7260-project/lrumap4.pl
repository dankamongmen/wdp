set xrange [0:10000000]
set terminal pdf
set title "LRUMAP vs Classic True LRU, Order 4"
set xtics 3000000
set xlabel "LRU sets"
set ylabel "Bytes used"
set key left box
# can we not parameterize these two? ugh
set out "out/lrumap4.pdf"
lruorder = 4
lg2(x) = log(x) / log(2)
fac(n) = int(n)!
lruspace(n,r) = n * r * lg2(r)
lrumapspace(n,r) = fac(r) * r * lg2(r) + n * lg2(fac(r))
plot lrumapspace(x,lruorder) title "LRUMAP" with points, \
	lruspace(x,lruorder) title "Classic LRU" with points
