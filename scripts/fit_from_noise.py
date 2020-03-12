'''
This script is needed when a noise file needs to be converted to frequency and
quality factor spectrum but the VNA file has been acquired in slightly different
contidions. The script tries to force a smooth result and should not be used for
diagnostic purposes.
'''

import numpy as np
import sys,os
import time
import argparse
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print ("Cannot find the pyUSRP package")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='\
    This script is needed when a noise file needs to be converted to frequency and\
    quality factor spectrum but the VNA file has been acquired in slightly different\
    contidions. The script tries to force a smooth result and should not be used for\
    diagnostic purposes.\
    ')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--noise', '-n', help='Name of the file containing the noise data', type=str, required = True)
    parser.add_argument('--vna', '-v', help='Name of the VNA file to initialize', type=str, required = True)
    parser.add_argument('--guard_tone', '-g', help='Guard tone in MHz', type=float)

    args = parser.parse_args()

    os.chdir(args.folder)

    if args.guard_tone is not None:
        guard_tones = [args.guard_tone]
    else:
        guard_tones = None

    u.init_from_noise(args.noise, args.vna, guard_tones = guard_tones)
    print_error("SILL IN DEVELOPEMENT")
