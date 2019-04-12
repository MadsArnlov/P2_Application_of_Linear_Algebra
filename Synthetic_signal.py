"Synthetic signal"
import numpy as np
J = 18                          # Indicates the length of the signal as 2^J
freq1 = 75                      # Frequency of the single sine wave
phase1 = 1                      # Phase of the first sine wave
freq2 = 250                     # Makes the 2nd sine zero
phase2 = -np.pi/5                 # Phase of the 2nd wave is zero
freq3 = 800                     # Makes the 3rd sine zero
phase3 = np.pi/3                # Phase of the 3rd wave is zero
freq4 = 2250                    # Frequency of the sine wave
phase4 = np.pi                  # Phase of the sine wave
imp_freq = 10                   # Frequency of the impulses
scaling1 = 2                    # The amplitude of the sine wave
shift = 2**14                   # The same signal is shifted
