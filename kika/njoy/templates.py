NJOY_INPUT_TEMPLATE = """\
moder
 20 -25
reconr
 -25 -21
'{title}'/
 {mat} 0 0
 0.001 0 0.01 5e-08
 0 /
broadr
 -25 -21 -22
 {mat} 1 0 0 0
 0.001 1e+06 0.01 5e-08
 {T}
 0 /
heatr
 -25 -22 -21 /
 {mat} 5 0 0 0 0 /
 302 318 402 442 444 /
heatr
 -25 -22 -23 /
 {mat} 6 0 1 0 2 /
 302 303 402 442 443 444 /
purr
 -25 -21 -24
 {mat} 1 9 20 64 1 0 /
 {T} /
 1e+10 1e+8 1e+6 1e+4 1e+3 3e+2 1e+2 3e+1 1e+1 /
 0 /
gaspr
 -25 -24 -21  /
acer
 -25 -21 0 40 41
 1 0 1 {suff} /
'{title}'/
 {mat} {T}
 1 1 1
 /
acer
 0 40 0 43 44
 7 1 1 -1 /
'{title}'/
stop
"""