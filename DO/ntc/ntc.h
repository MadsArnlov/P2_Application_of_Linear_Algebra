// ntc.h
 
float ntcToTemp(float r);
float tempToNtc(float t);
/*
NTC NTCLE100E3
--------------
If you have a 3.3 ohm then define ..   #define NTC2880

See Steinhart & Hart parameters below

AND LIFT THIS TO TOP OF FILE :-)

/Jens
Ohm B-25/85
3.3 2880
4.7 2880
6.8 2880
10 2990
15 3041
22 3136
33 3390
47 3390
68 3390
100 3560
150 3560
220 3560
330 3560
470 3560
680 3560
1000 3528
1500 3528
2000 3528
2200 3977
2700 3977
3300 3977
4700 3977
5000 3977
6800 3977
10000 3977
12000 3740
15000 3740
22000 3740
33000 4090
47000 4090
50000 4190
68000 4190
100000 4190
150000 4370
220000 4370
330000 4570
470000 4570
*/

#ifndef NTCH
#define NTCH

#ifdef NTC_R_S33 
#define NTC2880
#define NTC_REF_R 3.3
#endif

#ifdef NTC_R_S47 
#define NTC2880
#define NTC_REF_R 4.7 
#endif

#ifdef NTC_R_S68 
#define NTC2880
#define NTC_REF_R 6.8 
#endif

#ifdef NTC_R_10 
#define NTC2990
#define NTC_REF_R 10 
#endif

#ifdef NTC_R_15 
#define NTC3041
#define NTC_REF_R 15 
#endif

#ifdef NTC_R_22 
#define NTC3136
#define NTC_REF_R 22 
#endif

#ifdef NTC_R_33 
#define NTC3390
#define NTC_REF_R 33 
#endif

#ifdef NTC_R_47 
#define NTC3390
#define NTC_REF_R 47 
#endif

#ifdef NTC_R_68 
#define NTC3390
#define NTC_REF_R 68 
#endif

#ifdef NTC_R_100 
#define NTC3560
#define NTC_REF_R 100 
#endif

#ifdef NTC_R_150 
#define NTC3560
#define NTC_REF_R 150 
#endif

#ifdef NTC_R_220 
#define NTC3560
#define NTC_REF_R 220.0
#endif

#ifdef NTC_R_330 
#define NTC3560
#define NTC_REF_R 330.0 
#endif

#ifdef NTC_R_470 
#define NTC3560
#define NTC_REF_R 470.0 
#endif

#ifdef NTC_R_680 
#define NTC3560
#define NTC_REF_R 680.0 
#endif

#ifdef NTC_R_1000 
#define NTC3528
#define NTC_REF_R 1000.0 
#endif

#ifdef NTC_R_1500 
#define NTC3528
#define NTC_REF_R 1500.0 
#endif

#ifdef NTC_R_2000 
#define NTC3528
#define NTC_REF_R 2000.0 
#endif

#ifdef NTC_R_2200 
#define NTC3977
#define NTC_REF_R 2200.0 
#endif

#ifdef NTC_R_2700 
#define NTC3977
#define NTC_REF_R 2700.0 
#endif

#ifdef NTC_R_3300 
#define NTC3977
#define NTC_REF_R 3300.0 
#endif

#ifdef NTC_R_4700 
#define NTC3977
#define NTC_REF_R 4700.0 
#endif

#ifdef NTC_R_5000 
#define NTC3977
#define NTC_REF_R 5000.0 
#endif

#ifdef NTC_R_6800 
#define NTC3977
#define NTC_REF_R 6800.0 
#endif

#ifdef NTC_R_10000 
#define NTC3977
#define NTC_REF_R 10000.0 
#endif

#ifdef NTC_R_12000 
#define NTC3740
#define NTC_REF_R 12000.0
#endif

#ifdef NTC_R_15000 
#define NTC3740
#define NTC_REF_R 15000.0 
#endif

#ifdef NTC_R_22000 
#define NTC3740
#define NTC_REF_R 22000.0 
#endif

#ifdef NTC_R_33000 
#define NTC4090
#define NTC_REF_R 33000.0 
#endif

#ifdef NTC_R_47000 
#define NTC4090
#define NTC_REF_R 47000.0 
#endif

#ifdef NTC_R_50000 
#define NTC4190
#define NTC_REF_R 50000.0 
#endif

#ifdef NTC_R_68000 
#define NTC4190
#define NTC_REF_R 68000.0 
#endif

#ifdef NTC_R_100000 
#define NTC4190
#define NTC_REF_R 100000.0 
#endif

#ifdef NTC_R_150000 
#define NTC4370
#define NTC_REF_R 150000.0 
#endif

#ifdef NTC_R_220000 
#define NTC4370
#define NTC_REF_R 220000.0 
#endif

#ifdef NTC_R_330000 
#define NTC4570
#define NTC_REF_R 330000.0 
#endif

#ifdef NTC_R_470000 
#define NTC4570
#define NTC_REF_R 470000.0 
#endif


#ifndef NTC_REF_R
#pragma error "bad or no NTC value - see ntc.h"
#endif

#ifdef NTC2880
#define NTC_A -9.094
#define NTC_B 2251.74
#define NTC_C 229098
#define NTC_D -2.744820E+07
#define NTC_A1 3.354016E-03
#define NTC_B1 3.495020E-04
#define NTC_C1 2.095959E-06
#define NTC_D1 4.260615E-07
//NTC2880 3 -9.094 2251.74 229098 -2.744820E+07 3.354016E-03 3.495020E-04 2.095959E-06 4.260615E-07
#define SETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 3.495020E-04 , 2.095959E-06 , 4.260615E-07)
#endif

#ifdef NTC2990
#define NTC_A -10.2296
#define NTC_B 2887.62
#define NTC_C 132336
#define NTC_D -2.502510E+07
#define NTC_A1 3.354016E-03
#define NTC_B1 3.415560E-04
#define NTC_C1 4.955455E-06
#define NTC_D1 4.364236E-07
//NTC2990 3 -10.2296 2887.62 132336 -2.502510E+07 3.354016E-03 3.415560E-04 4.955455E-06 4.364236E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 3.415560E-04 , 4.955455E-06 , 4.364236E-07)
#endif

#ifdef NTC3041
#define NTC_A -11.1334
#define NTC_B 3658.73
#define NTC_C -102895.0
#define NTC_D 5.166520E+05
#define NTC_A1 3.354016E-03
#define NTC_B1 3.349290E-04
#define NTC_C1 3.683843E-06
#define NTC_D1 7.050455E-07
//NTC3041 3 -11.1334 3658.73 -102895 5.166520E+05 3.354016E-03 3.349290E-04 3.683843E-06 7.050455E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 3.349290E-04 , 3.683843E-06 , 7.050455E-07)
#endif

#ifdef NTC3136
#define NTC_A -12.4493
#define NTC_B 4702.74
#define NTC_C -402687.0
#define NTC_D 3.196830E+07
#define NTC_A1 3.354016E-03
#define NTC_B1 3.243880E-04
#define NTC_C1 2.658012E-06
#define NTC_D1 -2.701560E-07
//NTC3136 3 -12.4493 4702.74 -402687 3.196830E+07 3.354016E-03 3.243880E-04 2.658012E-06 -2.701560E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 3.243880E-04 , 2.658012E-06 , -2.701560E-07)
#endif

#ifdef NTC3390
#define NTC_A -12.6814
#define NTC_B 4391.97
#define NTC_C -232807.0
#define NTC_D 1.509643E+07
#define NTC_A1 3.354016E-03
#define NTC_B1 2.993410E-04
#define NTC_C1 2.135133E-06
#define NTC_D1 -5.672000E-09
//NTC3390 3 -12.6814 4391.97 -232807 1.509643E+07 3.354016E-03 2.993410E-04 2.135133E-06 -5.672000E-09
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.993410E-04 , 2.135133E-06 , -5.672000E-09)
#endif

#ifdef NTC3528
#define NTC_A -12.0596
#define NTC_B 3687.667
#define NTC_C -7617.13
#define NTC_D -5.914730E+06
#define NTC_A1 3.354016E-03
#define NTC_B1 2.909670E-04
#define NTC_C1 1.632136E-06
#define NTC_D1 7.192200E-08
//NTC3528 0.5 -12.0596 3687.667 -7617.13 -5.914730E+06 3.354016E-03 2.909670E-04 1.632136E-06 7.192200E-08
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.909670E-04 , 1.632136E-06 , 7.192200E-08)
#endif

#ifdef NTC3528
#define NTC_A -21.0704
#define NTC_B 11903.95
#define NTC_C -2504699.0
#define NTC_D 2.470338E+08
#define NTC_A1 3.354016E-03
#define NTC_B1 2.933908E-04
#define NTC_C1 3.494314E-06
#define NTC_D1 -7.712690E-07
//NTC3528 0.5 -21.0704 11903.95 -2504699 2.470338E+08 3.354016E-03 2.933908E-04 3.494314E-06 -7.712690E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.933908E-04 , 3.494314E-06 , -7.712690E-07)
#endif

#ifdef NTC3560
#define NTC_A -13.0723
#define NTC_B 4190.574
#define NTC_C -47158.4
#define NTC_D -1.199256E+07
#define NTC_A1 3.354016E-03
#define NTC_B1 2.884193E-04
#define NTC_C1 4.118032E-06
#define NTC_D1 1.786790E-07
//NTC3560 1.5 -13.0723 4190.574 -47158.4 -1.199256E+07 3.354016E-03 2.884193E-04 4.118032E-06 1.786790E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.884193E-04 , 4.118032E-06 , 1.786790E-07)
#endif

#ifdef NTC3470
#define NTC_A -13.8973
#define NTC_B 4557.725
#define NTC_C -98275.0
#define NTC_D -7.522357E+06
#define NTC_A1 3.354016E-03
#define NTC_B1 2.744032E-04
#define NTC_C1 3.666944E-06
#define NTC_D1 1.375492E-07
//NTC3470 2.0  -13.8973 4557.725 -98275 -7.522357E+06 3.354016E-03 2.744032E-04 3.666944E-06 1.375492E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.744032E-04 , 3.666944E-06 , 1.375492E-07)
#endif

#ifdef NTC3977
#define NTC_A -14.6337
#define NTC_B 4791.842
#define NTC_C -115334.0
#define NTC_D -3.730535E+06
#define NTC_A1 3.354016E-03
#define NTC_B1 2.569850E-04
#define NTC_C1 2.620131E-06
#define NTC_D1 6.383091E-08
//NTC3977 0.75 -14.6337 4791.842 -115334 -3.730535E+06 3.354016E-03 2.569850E-04 2.620131E-06 6.383091E-08
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.569850E-04 , 2.620131E-06 , 6.383091E-08)
#endif

#ifdef NTC4090
#define NTC_A -15.5322
#define NTC_B 5229.973
#define NTC_C -160451.0
#define NTC_D -5.414091E+06
#define NTC_A1 3.354016E-03
#define NTC_B1 2.519107E-04
#define NTC_C1 3.510939E-06
#define NTC_D1 1.105179E-07
//NTC4090 1.5 -15.5322 5229.973 -160451 -5.414091E+06 3.354016E-03 2.519107E-04 3.510939E-06 1.105179E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.519107E-04 , 3.510939E-06 , 1.105179E-07)
#endif

#ifdef NTC4190
#define NTC_A -16.0349
#define NTC_B 5459.339
#define NTC_C -191141.0
#define NTC_D -3.328322E+06
#define NTC_A1 3.354016E-03
#define NTC_B1 2.460382E-04
#define NTC_C1 3.405377E-06
#define NTC_D1 1.034240E-07
//NTC4190 1.5 -16.0349 5459.339 -191141 -3.328322E+06 3.354016E-03 2.460382E-04 3.405377E-06 1.034240E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.460382E-04 , 3.405377E-06 , 1.034240E-07)
#endif

#ifdef NTC4370
#define NTC_A -16.8717
#define NTC_B 5759.15
#define NTC_C -194267.0
#define NTC_D -6.869149E+06
#define NTC_A1 3.354016E-03
#define NTC_B1 2.367720E-04
#define NTC_C1 3.585140E-06
#define NTC_D1 1.255349E-07
//NTC4370 2.5 -16.8717 5759.15  -194267 -6.869149E+06 3.354016E-03 2.367720E-04 3.585140E-06 1.255349E-07
#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.367720E-04 , 3.585140E-06 , 1.255349E-07)
#endif

#ifdef NTC4570
#define NTC_A -17.6439
#define NTC_B 6022.726
#define NTC_C -203157.0
#define NTC_D -7.183526E+06
#define NTC_A1 3.354016E-03
#define NTC_B1 2.264097E-04
#define NTC_C1 3.278184E-06
#define NTC_D1 1.097628E-07
//NTC4570 1.5 -17.6439 6022.726 -203157 -7.183526E+06 3.354016E-03 2.264097E-04 3.278184E-06 1.097628E-07


#define NTCSETPARMS setNtcParms(NTC_REF_R, 3.354016E-03 , 2.264097E-04 , 3.278184E-06 , 1.097628E-07)

#endif

// Steinhart and HArt formula and coefficients from above given
// by your define of NTC_R_10000 etc
// See top


#endif
 
 
 
float ntcToTemp(float r)
{
float logV,vv,t;
 
  //formula t = 1.0 / (NTC_A1 + NTC_B1 * logV + NTC_C1 * logV * logV + NTC_D1 * logV * logV * logV);
  // doing a little bit more efficient ...
  
  vv = logV = log(r / NTC_REF_R); 

  t = NTC_A1 + NTC_B1 * vv;
  
  vv *= logV;  // logV^2
  t += NTC_C1 * vv;  
 
  vv *= logV;   // logV^3
  t += NTC_D1 * vv;
  
  return (1.0 / t); // in Kelvin !!!
}
  
float tempToNtc(float t)
{
float tmp;
  // R = Rref * exp(A +B/T + C/T^2 + D/T^3)
  
  tmp = NTC_A;
  tmp += NTC_B/t;
  tmp += NTC_C/(t*t);
  tmp += NTC_D/(t*t*t);
  tmp = exp(tmp);
  
  return    NTC_REF_R * tmp;
}
 
