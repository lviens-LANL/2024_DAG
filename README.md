# Python codes to analyze the healing phase of the medium following the DAG-2 chemical explosion

The codes are for the following manuscript:
- Viens L. and B. G. Delbridge (Submitted to JGR), Shallow Soil Response to a Buried Chemical Explosion with Geophones and Distributed Acoustic Sensing


# Description:

- The **Codes** folder contains the codes to reproduce most Figures of the manuscript.
  - The LowLevel_callback_healing folder in the **Codes** contains a C routine written By Kurama Okubo (NIED). The original version can be found [here](https://github.com/kura-okubo/SeisMonitoring_Paper/tree/master/Post/ModelFit/code/LowLevel_callback_healing_distributed). The C library of the kernel of integration needs to be compiled with before running the Fig_3.py and Fig_5.py codes with:
  < align="center">gcc -shared -o healing_int.so healing_int.c<>
  - Once compiled, the path of the compiled file needs to be changed in the python codes.
   
- The **Data** folder contains the pre-processed data needed to reproduce the Figures.
- The **Figures** folder contains all the figures that are output by the codes.

# Copyright

© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

# Licence

Copyright 2024
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
