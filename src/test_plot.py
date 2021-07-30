#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This must come first
from __future__ import print_function, unicode_literals, division

# Standard library imports
# ========================
# %matplotlib inline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


#######################################################################


plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
m.bluemarble(scale=0.5);

# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html


"""

In [53]: from pathlib import Path
In [54]: dirroot = "output"                                                                              

In [55]: dirroot.joinpath("test")                                                                        
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-55-725f4aaad8c6> in <module>
----> 1 dirroot.joinpath("test")

AttributeError: 'str' object has no attribute 'joinpath'

In [56]: dirroot = Path("output")                                                                        

In [57]: dirroot.joinpath("test")                                                                        
Out[57]: PosixPath('output/test')

In [58]: dirold = dirroot.joinpath("exemples")                                                           

In [59]: dirnew = dirroot.joinpath("2008", "02")                                                         

In [60]: dirnew                                                                                          
Out[60]: PosixPath('output/2008/02')

In [61]: dirold                                                                                          
Out[61]: PosixPath('output/exemples')

In [62]: fileold = dirold.joinpath(filename)                                                             

In [63]: filenew = dirnew.joinpath(filename)                                                             

In [64]: fileold, filenew                                                                                
Out[64]: 
(PosixPath('output/exemples/unmasked_ERA5_AIRS_V6_L2_P_surf_daily_average.20080220.AM_05'),
 PosixPath('output/2008/02/unmasked_ERA5_AIRS_V6_L2_P_surf_daily_average.20080220.AM_05'))

In [65]: f_old = FortranFile(fileold, "r")                                                               
/data/anaconda3/lib/python3.7/site-packages/fortio.py:123: UserWarning: byteorder of the file is set to '>' by auto-detection.
  warnings.warn(msg)

In [66]: f_new = FortranFile(filenew, "r")                                                               
/data/anaconda3/lib/python3.7/site-packages/fortio.py:123: UserWarning: byteorder of the file is set to '>' by auto-detection.
  warnings.warn(msg)

In [67]: rec_old = f_old.read_record(dtype=">f4")                                                        

In [68]: rec_old                                                                                         
Out[68]: 
array([ 687.80334,  684.7539 ,  677.43365, ..., 1004.3322 , 1004.6122 ,
       1004.8122 ], dtype=float32)

In [69]: rec_old                                                                                         
Out[69]: 
array([ 687.80334,  684.7539 ,  677.43365, ..., 1004.3322 , 1004.6122 ,
       1004.8122 ], dtype=float32)

In [70]: rec_new = f_new.read_record(dtype=">f4")                                                        

In [71]: rec_new                                                                                         
Out[71]: 
array([1004.3    , 1004.29443, 1004.28876, ...,  686.09344,  686.094  ,
        686.0945 ], dtype=float32)

In [72]: rec_old.min()                                                                                   
Out[72]: 488.59805

In [73]: rec_old.min(), rec_old.max(), rec_old.mean()                                                    
Out[73]: (488.59805, 1042.9492, 964.976)

In [74]: rec_new.min(), rec_new.max(), rec_new.mean()                                                    
Out[74]: (488.58032, 1042.9574, 964.97986)

In [75]: rec_old.reshape((721, 1440))                                                                    
Out[75]: 
array([[ 687.80334,  684.7539 ,  677.43365, ..., 1004.5839 , 1004.6248 ,
        1004.4165 ],
       [1004.4469 , 1004.4472 ,  687.80365, ..., 1004.9512 , 1004.7619 ,
        1004.5745 ],
       [1004.6166 , 1004.4191 , 1004.45056, ..., 1005.47015, 1005.17914,
        1004.9302 ],
       ...,
       [ 652.1422 ,  650.86884,  651.93634, ...,  679.4272 ,  671.95526,
         666.79376],
       [ 660.7422 ,  655.4207 ,  652.15814, ..., 1004.80115,  682.50806,
         679.44745],
       [ 672.0465 ,  666.906  ,  660.8356 , ..., 1004.3322 , 1004.6122 ,
        1004.8122 ]], dtype=float32)

In [76]: rec_old                                                                                         
Out[76]: 
array([ 687.80334,  684.7539 ,  677.43365, ..., 1004.3322 , 1004.6122 ,
       1004.8122 ], dtype=float32)
"""


