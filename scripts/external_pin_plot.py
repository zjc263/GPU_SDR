import sys,os,glob

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot the external dataset of a file')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data are stored', type=str, default = "data")
    parser.add_argument('--file', '-f', help='name of the file to plot, default is all files in the folder', type=str, default= "all")


    args = parser.parse_args()
    os.chdir(args.folder)

    if args.file == "all":
        target_files = glob.glob("*.h5")
    else:
        target_files = [args.file]


    for f in target_files:
        u.plot_external_dataset(f, ant = None, usrp_number = 0)
        u.plot_raw_data(f, decimation=1000, displayed_samples=None, low_pass=None, backend='plotly', output_filename=None,
                          channel_list=None, mode='PM', start_time=1, end_time=None, auto_open=True)
