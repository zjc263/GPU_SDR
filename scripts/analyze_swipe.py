###############################
# THE IDEA IS TO CREATE AN AUTOMATED ANALYSIS ROUTINE. THE CURRENT VERSION DOES NOT CONTEMPLATE
# TEMPERATURE BUT IS A THING THAT MUST BE ADDED IN THE FUTURE.
# THE OUTPUT OF THIS PROGRAM IS A NICE DATA EXPLORER TO LOOK AT POWER SWIPE
###############################
from multiprocessing import Pool
from functools import partial
import sys,os,random,glob
import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('..')
        import pyUSRP as u
    except ImportError:
        print ("Cannot find the pyUSRP package")

import argparse

# Varaibles definitions
PLOT_FOLDER = "plots"

# For clarity
class text_color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def make_bold(x):
    return(text_color.BOLD + str(x) + text_color.END)

def filedict_pp(filedict):
    '''
    Print the file structure
    '''
    print("Printing file structure:")
    for i in range(len(filedict['ordered'])):
        if filedict['ordered'][i] in filedict['vna']:
            print("Power per tone before att: " + make_bold("%.2f dB"%filedict["gain"][i]))
            print(filedict['ordered'][i])
        elif filedict['ordered'][i] in filedict['noise']:
            print("\t" + filedict['ordered'][i])

def intfromdate(x):
    '''
    Sorting purpose
    '''
    return int(x[-18:-3].split("_")[0] + x[-18:-3].split("_")[1])

def append_full_path(files):
    '''
    Append full path to file structure. Works in linux, untested under Windows.
    '''
    current_path = os.getcwd()
    files['ordered'] = [current_path + "/" + x for x in files['ordered']]
    files['noise'] = [current_path + "/" + x for x in files['noise']]
    files['vna'] = [current_path + "/" + x for x in files['vna']]
    return files



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--folder', '-fn', help='Name of the folder in which the data will be stored. Move to this folder before everything', type=str, default = "data")
    parser.add_argument('--subfolder', '-sb', help='Name of the sub-folder where to save all the data. Default will be the last swipe_<N>', type=str)
    parser.add_argument('--attenuation', '-att', help='Total attenuation before the DUT. Default will not consider attenuation', type=int, default = None)
    parser.add_argument('--title', '-tilte', help='Title of the pager. Default is general info about the the swipe', type=str, default = None)
    parser.add_argument('--ncores', '-nc', help='Number of cores used for processing', type=int, default = 4)
    parser.add_argument('--frontend', '-fr', help='front-end character: A or B', type=str, default="A")

    args = parser.parse_args()

    if args.frontend == 'A':
        ant = "A_RX2"
    elif args.frontend == 'B':
        ant = "B_RX2"
    else:
        err_msg = "Frontend %s unknown" % args.frontend
        u.print_warning(err_msg)
        ant = None

    os.chdir(args.folder)

    # Pick subfolder
    if args.subfolder is None:
        try:
            max_subfolder = "swipe_"+str(max([int(x.split("_")[1]) for x in glob.glob("swipe*")]))
            u.print_debug("Accessing folder: " + args.folder + "/" + max_subfolder + " ..." )
            os.chdir(max_subfolder)
        except:
            err_msg = "Cannot find any swipe_<N> folder. Please specify the name of the subfolder"
            print_error(err_msg)
            raise(ValueError(err_msg))
    else:
        max_subfolder = args.subfolder
        u.print_debug("Accessing folder: " + args.folder + "/" + max_subfolder + " ..." )
        os.chdir(max_subfolder)

    processing = Pool(args.ncores)

    ###############################
    # BUILDING THE FILE STRUCTURE #
    ###############################

    u.print_debug("Building file structure...")

    delay_file = glob.glob("USRP_Delay*.h5")[0]

    noise_files = glob.glob("USRP_Noise*.h5")

    VNA_files = glob.glob("USRP_VNA*.h5")

    files = {
        "noise" : sorted(list(set([u.format_filename(x) for x in noise_files])), key=intfromdate),
        "vna" : sorted(list(set([u.format_filename(x) for x in VNA_files])), key=intfromdate),
        "delay" : [u.format_filename(delay_file)],
        "ordered" : [],
        "gain" : []
    }
    files["ordered"] = sorted(list(files["noise"] + files["vna"]), key=intfromdate)

    for i in range(len(files["ordered"])):
        files["gain"].append(
            # ASSUMING EVENLY DISTRIBUTED POWER!!
            u.get_readout_power(files["ordered"][i], channel = 0, front_end=None, usrp_number=0)
        )

    # Clip spurs VNAs:
    for i in range(files['ordered'].index(files['noise'][0]) - 1):
        files['vna'].remove(files['vna'][0])
        files['ordered'].remove(files['ordered'][0])

    # From here we assume each noise file has an associated resonator froup already in it.
    filedict_pp(files)

    # in order to use multiprocessing we'll specify the full path of every file
    files = append_full_path(files)

    # Make some debug plot.
    try:
        os.mkdir(PLOT_FOLDER)
    except:
        pass


    processing.map(
        partial(
            u.diagnostic_VNA_noise, points = None, VNA_file = None, ant = ant, backend = 'matplotlib', add_name = PLOT_FOLDER
        ),
        files['noise']
    )
    processing.map(
        partial(
            u.diagnostic_VNA_noise, points = 1000, VNA_file = None, ant = ant, backend = 'plotly', add_name = PLOT_FOLDER, auto_open = False,
        ),
        files['noise']
    )
