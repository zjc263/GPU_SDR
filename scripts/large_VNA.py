
import sys,os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

import argparse

def run(gain,iter,rate,freq,front_end, f0,f1, lapse, points, ntones, delay_duration, delay_over):

    try:
        if u.LINE_DELAY[str(int(rate/1e6))]: pass
    except KeyError:

        if delay_over is None:
            print("Cannot find line delay. Measuring line delay before VNA:")

            filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = delay_duration)

            delay = u.analyze_line_delay(filename, True)

            u.write_delay_to_file(filename, delay)

            u.load_delay_from_file(filename)

        else:

            u.set_line_delay(rate, delay_over)

        if ntones ==1:
            ntones = None

    vna_filename = u.Single_VNA(start_f = f0, last_f = f1, measure_t = lapse, n_points = points, tx_gain = gain, Rate=rate, decimation=True, RF=freq, Front_end=front_end,
               Device=None, output_filename=None, Multitone_compensation=ntones, Iterations=iter, verbose=False)

    return vna_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')
    '''
    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    parser.add_argument('--freq', '-f', help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default 300 MHz)', nargs='+')
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--frontend', '-rf', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--f0', '-f0', help='Baseband start frequrency in MHz', type=float, default=-45)
    parser.add_argument('--f1', '-f1', help='Baseband end frequrency in MHz', type=float, default=+45)
    parser.add_argument('--points', '-p', help='Number of points used in the scan', type=float, default=50e3)
    parser.add_argument('--time', '-t', help='Duration of the scan in seconds per iteration', type=float, default=10)
    parser.add_argument('--iter', '-i', help='How many iterations to perform', type=float, default=1)
    parser.add_argument('--gain', '-g', help='set the transmission gain. Multiple gains will result in multiple scans (per frequency). Default 0 dB',  nargs='+')
    parser.add_argument('--tones', '-tones', help='expected number of resonators',  type=int)
    parser.add_argument('--delay_duration', '-dd', help='Duration of the delay measurement',  type=float, default=0.01)
    parser.add_argument('--delay_over', '-do', help='Manually set line delay in nanoseconds. Skip the line delay measure.',  type=float)
    '''
    parser.add_argument('--rate', '-r', help='Sampling frequency in Msps', type=float, default = 100)
    parser.add_argument('--bandwidth', '-bw', help='Rate in which the single VNA is flat. considered plus/minus', type=float, default = 10)
    parser.add_argument('--start', '-f', help='Starting frequency in MHz', type=float, default = 200)
    parser.add_argument('--to', '-t', help='Ending frequency in MHz', type=float, default = 1000)
    parser.add_argument('--points', '-p', help='Number of points per scan', type=float, default = 10000)
    parser.add_argument('--iter', '-i', help='How many iterations to perform', type=float, default=1)
    parser.add_argument('--frontend', '-fr', help='front-end character: A or B', type=str, default="A")
    parser.add_argument('--time', '-tt', help='Duration of the scan in seconds per iteration', type=float, default=2)
    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored', type=str, default = "data")
    args = parser.parse_args()
    
    os.chdir(args.folder)
    
    #find number of intervals:
    n_int = np.abs(args.start - args.to)/args.bandwidth + 1
    rf_0 = args.start + args.bandwidth
    u.Connect()
    print("initializing delay...")
    blockPrint()
    try:
        if u.LINE_DELAY[str(int(args.rate))]: pass
    except KeyError:


        print("Cannot find line delay. Measuring line delay before VNA:")

        filename = u.measure_line_delay(
            args.rate*1e6, rf_0, args.frontend, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = 0.01)

        delay = u.analyze_line_delay(filename, True)

        u.write_delay_to_file(filename, delay)

        u.load_delay_from_file(filename)




    for i in range(100):
        blockPrint()
        vna_filename = u.Single_VNA(
            start_f = -args.bandwidth*1e6,
            last_f = args.bandwidth*1e6,
            measure_t = args.time,
            n_points = args.points,
            tx_gain = 0,
            Rate=args.rate*1e6,
            decimation=True,
            RF=(rf_0 + i*2*args.bandwidth)*1e6,
            Front_end=args.frontend,
            Device=None,
            output_filename=None,
            Multitone_compensation=1,
            Iterations=args.iter,
            verbose=False
        )
        enablePrint()
        print("iteration %d done" % i)


    u.Disconnect()
    # Data analysis and plotting will be in an other python script
