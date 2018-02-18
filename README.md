## Proyecto ##

### Useful commands: ###

Compilation:
`make`

Run:
`./test DetectorName DescriptorName MatcherName [show]`

### Available Detectors: ###
- FAST
- ORB
- BRISK
- AGAST
- GFTT
- BAFT
- LOCKY(S)

### Available Descriptors: ###
- BRIEF
- BRISK
- FREAK
- ORB
- LATCH
- LATCHK\*
- LDB
- BAFT
- BOLD

### Available Matchers: ###
- BFM
- FLANN
- GMS

### Observations: ###

**LATCHK** requires **-mavx2** and **-mfma** flags to work, they're currently disabled and
LATCHK's code commented as one of our laptops didn't support **avx2** nor **fma**. If
your hardware supports them, you can add the flags to the Makefile an uncomment
that part of the code. (**BAFT** and **LOCKY** also compiled with these optimizations
and didn't work properly, code is not commented though as they work just fine
without this. Again, if **avx2** and **fma** *are* supported, add them to the Makefile).

### TODO: ###
- [x] Add Makefile
- [x] Try OpenCv Matching implementation
- [ ] Add other features to compare
- [ ] Don't know
