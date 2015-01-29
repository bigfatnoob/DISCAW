"""
# https://code.google.com/p/promisedata/source/browse/#svn%2Ftrunk%2Feffort%2Fcocomo-sdr
Standard header:

"""
from __future__ import division,print_function
import  sys
sys.dont_write_bytecode = True
from lib import *

"""
@attribute PREC {'Very High','Extra High',Nominal,'Very Low',High,Low}
@attribute FLEX {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute RESL {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute TEAM {High,'Very High','Extra High',Nominal,'Very Low',Low}
@attribute PMAT {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute RELY {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute DATA {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute CPLX {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute RUSE {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute DOCU {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute TIME {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute STOR {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute PVOL {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute ACAP {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute PCAP {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute PCON {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute AEXP {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute PEXP {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute LTEX {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute TOOL {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute SITE {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute SCED {Low,'Very Low',High,'Extra High','Very High',Nominal}
@attribute LOC numeric
@attribute 'ACTUAL EFFORT' numeric
"""

def cocomo(weighFeature = False, 
           split = "variance"):
  vl=1;l=2;n=3;h=4;vh=5;xh=6;_=0
  return data(indep= [ 
     # 0..6
     'Prec','Flex','Resl','Team','Pmat','rely','cplx','data','ruse',
     # 9 .. 17
     'time','stor','pvol','acap','pcap','pcon','aexp','plex','ltex',
     # 18 .. 25
     'tool','sced','site','docu','kloc'],
    less = ['Effort'],
    _rows=[
      [vh,n,l,h,vl,n,n,l,l,l,vh,h,n,h,vh,vh,vh,h,h,h,l,n,3,1.2],
      [vh,n,l,vh,vl,h,n,l,n,l,vh,h,n,h,h,vh,vh,vh,vh,h,vl,h,2,2],
      [xh,n,l,h,vl,h,vh,l,n,l,vh,vh,n,h,h,vh,vh,vh,vh,h,vl,h,4.25,4.5],
      [xh,h,h,h,n,h,h,vh,l,n,n,n,h,h,h,vl,h,vh,h,n,h,h,10,3],
      [n,n,l,h,n,l,n,vh,n,h,h,n,n,h,h,h,n,n,n,n,h,l,15,4],
      [vh,l,xh,xh,h,l,l,n,l,vl,n,n,l,vh,vh,h,n,h,h,vl,xh,l,40.53,22],
      [xh,vh,xh,n,h,l,l,l,n,vl,n,n,l,vh,vh,vh,n,vh,vh,vl,h,l,4.05,2],
      [n,vl,vh,n,l,l,vh,vl,h,vh,vh,vh,l,l,h,vh,n,n,n,n,vh,n,31.845,5],
      [vl,vl,n,vl,l,vl,n,l,vh,l,n,n,l,l,h,vh,h,h,h,n,h,n,114.28,18],
      [n,vl,vh,l,l,vl,h,n,vh,vl,vh,vh,l,l,vh,vh,h,vh,h,n,n,l,23.106,4],
      [h,n,vh,h,l,l,n,n,n,l,n,n,l,vh,h,vh,n,n,h,vh,vh,h,1.369,1],
      [l,n,n,h,n,l,l,h,h,n,n,n,l,l,h,vh,h,n,h,vh,vh,h,1.611,2.1]
    ],
    _tunings =[[
    #         vlow  low   nom   high  vhigh xhigh
    #scale factors:
    'Prec',   6.20, 4.96, 3.72, 2.48, 1.24, _ ],[
    'Flex',   5.07, 4.05, 3.04, 2.03, 1.01, _ ],[
    'Resl',   7.07, 5.65, 4.24, 2.83, 1.41, _ ],[
    'Pmat',   7.80, 6.24, 4.68, 3.12, 1.56, _ ],[
    'Team',   5.48, 4.38, 3.29, 2.19, 1.01, _ ]],
    weighFeature = weighFeature,
    _split = split
    )

def _cocomo(): print(cocomo())