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
from yattag import Doc
import argparse

# Varaibles definitions
PLOT_FOLDER = "plots2"
LINK_SUBPATH = ""#"http://resowiki.caltech.edu/twiki/pub/Resonators/20200102_K0_VNA_take2/" # Used in overview_plots to hack TWiki link redirection

# Activate or deactivate analysis stages
ANALYZE_DIAGNOSTIC = True
POINTS = 10000 # Number of points displayed in the plotly backend of the plots

ANALYZE_NOISE = False # This step requires a lot of RAM.
ANALYZE_NOISE_QF = False # Analyze the noise in terms of Quality factor and frequency shift.
WELCH = 1
DBC = True # The imag/real noise can be plotted in dBc instead of dBm

PLOT_NOISE = False
PLOT_NOISE_GENERAL = False
PLOT_QF_SPEC = False
PLOT_QF_SPEC_GENERAL = False
MAX_FREQ = 1000 # maximum frequency in Hz to use for plotting

PLOT_VNA = False
PLOT_VNA_GENERAL = False

PLOT_FITS = False
PLOT_FITS_GENERAL = False

PLOT_QF_TIMESTREAMS = False
DECIMATION = 1000
PLOT_TIME = 100 # Chunk of the timestream to plot in seconds. None plots maximum

PLOT_STAT = False # Statistics of the fits in function of power

OVERVIEW_FILE = True # Plot an html file for quick viewing the plots
OVERVIEW_FILENAME = "Plots_Overview_2"

# Style for the overview file
scrollCSS = '''
table {
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    z-index: -1;
}
td, th, .row, .col, .ff-fix {
    cursor: pointer;
    padding: 10px;
    position: relative;
}

td:hover::before,
.row:hover::before,
.ff-fix:hover::before {
    background-color: #ffa;
    content: '\00a0';
    height: 100%;
    left: -5000px;
    position: absolute;
    top: 0;
    width: 100%;
    z-index: -1;
}
td:hover::after,
.col:hover::after,
.ff-fix:hover::after {
    background-color: #ffa;
    content: '\00a0';
    height: 10000px;
    left: 0;
    position: absolute;
    top: -5000px;
    width: 100%;
    z-index: -1;
}
.zoom {
  transition: transform .2s;
  background-color: white;
  width: 100%;
  height: 100%;
  margin: 0 auto;
  z-index: 2;
  position:relative;
}

.zoom:hover {
  -ms-transform: scale(2.0); /* IE 9 */
  -webkit-transform: scale(2.0); /* Safari 3-8 */
  transform: scale(2.0);
  z-index: 3;
  position:relative;
}
'''

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

def gather_plots_filenames(filedict):
    '''
    Recover plot filenames from files.
    This function is havely affected by the predictive nomenclature hardoded in the Python library that generates the plots!!!
    '''
    print("Recovering plots filenames...")
    plots = {
        "VNA":[],
        "Diagnostic":[],
        "Diagnostic_link":[],
        "Noise":[],
        "Timestream":[],
        "Fits":[],
        "QFNoise":[]
    }
    for i in range(len(filedict['ordered'])):
        if filedict['ordered'][i] in filedict['vna']:
            vna_plot_filename = LINK_SUBPATH + PLOT_FOLDER + "/" + "VNA_"+(filedict['ordered'][i].split("/")[-1]).split(".")[0]
            plots['VNA'].append(vna_plot_filename)
            fit_plot_name = LINK_SUBPATH + PLOT_FOLDER + "/" + "Resonators_"+(filedict['ordered'][i].split("/")[-1]).split(".")[0]
            plots['Fits'].append(fit_plot_name)
        elif filedict['ordered'][i] in filedict['noise']:
            noise_plot_filename = LINK_SUBPATH + PLOT_FOLDER + "/" + "Noise_"+(filedict['ordered'][i].split("/")[-1]).split(".")[0]
            plots['Noise'].append(noise_plot_filename)
            noise_plot_filename = LINK_SUBPATH + PLOT_FOLDER + "/" + "QFNoise_"+(filedict['ordered'][i].split("/")[-1]).split(".")[0]
            plots['QFNoise'].append(noise_plot_filename)
            diagnostic_filename = LINK_SUBPATH + PLOT_FOLDER + "/" + "Diagnostic_"+(filedict['ordered'][i].split("/")[-1]).split(".")[0] + "/diagnostic_channel_0_IQ"
            plots['Diagnostic'].append(diagnostic_filename)
            diagnostic_filename = LINK_SUBPATH + PLOT_FOLDER + "/" + "diagnostic_"+(filedict['ordered'][i].split("/")[-1]).split(".")[0]
            plots['Diagnostic_link'].append(diagnostic_filename)
            noise_plot_filename = LINK_SUBPATH + PLOT_FOLDER + "/" + "USRP_freq_timestream_"+(filedict['ordered'][i].split("/")[-1]).split(".")[0]
            plots['Timestream'].append(noise_plot_filename)

    return plots



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

def per_channels_noise_plots(channel, backend, attenuation, files):
    '''
    Alias function for easy pickling
    '''
    filename = u.plot_noise_spec(filenames = files, max_frequency=MAX_FREQ, title_info=None, backend=backend,
                    cryostat_attenuation=attenuation, auto_open=False, output_filename=None, add_info = None,
                    subfolder = PLOT_FOLDER, channel_list=[channel]
    )
    return filename

def per_channels_QF_noise_plots(channel, backend, attenuation, files):
    '''
    Alias function for easy pickling
    '''
    filename = u.plot_NEF_spectra(filenames = files, max_frequency=MAX_FREQ, title_info=None, backend=backend,
                    cryostat_attenuation=attenuation, auto_open=False, output_filename=None, add_info = None,
                    subfolder = PLOT_FOLDER, channel_list=[channel]
    )
    return filename

def per_channel_fit_plotls(frequency, backend, attenuation, files):
    '''
    Alias function for easy pickling
    '''
    filename = u.plot_resonators(files, reso_freq = frequency, backend = backend, title_info = None, verbose = False, output_filename = None, auto_open = False,
    attenuation = attenuation,single_plots = False, subfolder = PLOT_FOLDER)

    return filename


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

    if ANALYZE_DIAGNOSTIC:
        matplotlib_results = processing.map_async(
            partial(
                u.diagnostic_VNA_noise, points = None, VNA_file = None, ant = ant, backend = 'matplotlib', add_name = PLOT_FOLDER
            ),
            files['noise']
        )
        processing.map(
            partial(
                u.diagnostic_VNA_noise, points = POINTS, VNA_file = None, ant = ant, backend = 'plotly', add_name = PLOT_FOLDER, auto_open = False,
            ),
            files['noise']
        )
        matplotlib_results.wait()

    if ANALYZE_NOISE:
        print("Analyzyng noise files")
        processing.map(
            partial(
            u.calculate_noise, verbose = False, welch = max(WELCH,1), dbc = DBC, clip = 0.1,
            ),
            files['noise']
        )

    if PLOT_NOISE:
        print("Analyzyng noise files")
        matplotlib_results = processing.map_async(
            partial(
            u.plot_noise_spec, channel_list=None, max_frequency=MAX_FREQ, title_info=None, backend='matplotlib',
                            cryostat_attenuation=args.attenuation, auto_open=False, output_filename=None, add_info = None,
                            subfolder = PLOT_FOLDER
            ),
            files['noise']
        )
        processing.map(
            partial(
            u.plot_noise_spec, channel_list=None, max_frequency=MAX_FREQ, title_info=None, backend='plotly',
                            cryostat_attenuation=args.attenuation, auto_open=False, output_filename=None, add_info = None,
                            subfolder = PLOT_FOLDER
            ),
            files['noise']
        )
        matplotlib_results.wait()
    if PLOT_NOISE_GENERAL:
        # For now the general plotting is just per-channel
        # This functions require aliases for parallelization as the variable is not the first element
        matplotlib_results = processing.map_async(
            partial(
            per_channels_noise_plots, backend = 'matplotlib', attenuation = args.attenuation, files = files['noise']
            ),
            [ch_i for ch_i in range(len(u.get_fit_param(files['noise'][0])))] # Assuming same number of channels in all files
        )
        processing.map(
            partial(
            per_channels_noise_plots, backend = 'plotly', attenuation = args.attenuation, files = files['noise']
            ),
            [ch_i for ch_i in range(len(u.get_fit_param(files['noise'][0])))] # Assuming same number of channels in all files
        )
        matplotlib_results.wait()
    if PLOT_VNA:
        matplotlib_results = processing.map_async(
            partial(
            u.plot_VNA, backend = 'matplotlib', plot_decim = None, unwrap_phase = True, att = args.attenuation,
            subfolder = PLOT_FOLDER,
            ),
            files['vna']
        )
        processing.map(
            partial(
            u.plot_VNA, backend = 'plotly', plot_decim = None, unwrap_phase = True, att = args.attenuation, auto_open = False,
            subfolder = PLOT_FOLDER,
            ),
            files['vna']
        )
        matplotlib_results.wait()
    if PLOT_VNA_GENERAL:
        # For now it's just a general VNA plot
        u.plot_VNA(files['vna'], backend = 'plotly', plot_decim = None, unwrap_phase = True, att = args.attenuation, auto_open = False,
        subfolder = PLOT_FOLDER)
        u.plot_VNA(files['vna'], backend = 'matplotlib', plot_decim = None, unwrap_phase = True, att = args.attenuation, auto_open = False,
        subfolder = PLOT_FOLDER)

    if PLOT_FITS:

        matplotlib_results = processing.map_async(
            partial(
            u.plot_resonators, reso_freq = None, backend = 'matplotlib', title_info = None, verbose = False, output_filename = None, auto_open = False,
            attenuation = args.attenuation,single_plots = False, subfolder = PLOT_FOLDER
            ),
            files['vna']
        )
        processing.map(
            partial(
            u.plot_resonators, reso_freq = None, backend = 'plotly', title_info = None, verbose = False, output_filename = None, auto_open = False,
            attenuation = args.attenuation,single_plots = False, subfolder = PLOT_FOLDER
            ),
            files['vna']
        )
        matplotlib_results.wait()
    if PLOT_FITS_GENERAL:
        processing.map(
            partial(
            per_channel_fit_plotls, backend = "matplotlib", attenuation = args.attenuation, files = files['vna']
            ),
            [ch_i['f0'] for ch_i in u.get_fit_param(files['noise'][0])]
        )
    if PLOT_QF_TIMESTREAMS:
        matplotlib_results = processing.map_async(
            partial(
            u.plot_frequency_timestreams, decimation=DECIMATION, displayed_samples=None, low_pass=None, backend='matplotlib', output_filename=None,
                              channel_list=None, start_time=0.1, end_time=0.1+PLOT_TIME, auto_open=False, subfolder = PLOT_FOLDER,
                  ),
            files['noise']
        )
        processing.map(
            partial(
            u.plot_frequency_timestreams, decimation=DECIMATION, displayed_samples=None, low_pass=None, backend='plotly', output_filename=None,
                              channel_list=None, start_time=0.1, end_time=0.1+PLOT_TIME, auto_open=False, subfolder = PLOT_FOLDER,
                  ),
            files['noise']
        )
        matplotlib_results.wait()

    if ANALYZE_NOISE_QF:
        processing.map(
            partial(
            u.calculate_NEF_spectra, welch = WELCH, clip = 0.1, verbose = True
                  ),
            files['noise']
        )
    if PLOT_QF_SPEC:
        print("Analyzyng noise files")
        matplotlib_results = processing.map_async(
            partial(
            u.plot_NEF_spectra, channel_list=None, max_frequency=MAX_FREQ, title_info=None, backend='matplotlib',
                            cryostat_attenuation=args.attenuation, auto_open=False, output_filename=None, add_info = None,
                            subfolder = PLOT_FOLDER
            ),
            files['noise']
        )
        processing.map(
            partial(
            u.plot_NEF_spectra, channel_list=None, max_frequency=MAX_FREQ, title_info=None, backend='plotly',
                            cryostat_attenuation=args.attenuation, auto_open=False, output_filename=None, add_info = None,
                            subfolder = PLOT_FOLDER
            ),
            files['noise']
        )
        matplotlib_results.wait()
    if PLOT_QF_SPEC_GENERAL:
        # For now the general plotting is just per-channel
        # This functions require aliases for parallelization as the variable is not the first element
        matplotlib_results = processing.map_async(
            partial(
            per_channels_QF_noise_plots, backend = 'matplotlib', attenuation = args.attenuation, files = files['noise']
            ),
            [ch_i for ch_i in range(len(u.get_fit_param(files['noise'][0])))] # Assuming same number of channels in all files
        )
        processing.map(
            partial(
            per_channels_QF_noise_plots, backend = 'plotly', attenuation = args.attenuation, files = files['noise']
            ),
            [ch_i for ch_i in range(len(u.get_fit_param(files['noise'][0])))] # Assuming same number of channels in all files
        )
        matplotlib_results.wait()
    if PLOT_STAT:
        # We still have to impelment that plotting function.
        pass

    ###########################
    # WRITE THE FILE OVERVIEW #
    ###########################
    def format_image_style(plot_filename, link_name = None):
        '''
        Format filenames for html table.
        '''
        if link_name is None:
            link_name = plot_filename+".html"
        else:
            link_name = str(link_name)+".html"
        return '<a href="%s" target="_blank"> <img src="%s" alt="404" style="max-height:100%%; max-width:100%%">'%(link_name,plot_filename+".png")

    if OVERVIEW_FILE:
        plot_kinds = ["Power", "VNA", "Noise", "NEF", "Diagnostic", "Timestream", "Fits"] # Used in the html overview
        plot_names = gather_plots_filenames(files)
        if args.attenuation is None:
            attenuation = 0
            pw_units = " dB"
        else:
            attenuation = args.attenuation
            pw_units = " dBm"
        doc, tag, text = Doc().tagtext()
        doc.asis('<!DOCTYPE html>')
        with tag('html'):
        	with tag('head'):

        		doc.asis('<meta charset="UTF-8">')
        		doc.asis('<meta name="description" content="USRP %s">'%max_subfolder)
        		doc.asis('<meta name="author" content="Caltech/JPL">')
        		doc.asis('''
        			<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
        			 integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
        		 ''')
        		with tag("style"):
        			doc.asis(scrollCSS)

        with tag('body'):
            with tag('table', klass = 'table'):
                for plot_kind in plot_kinds:
                    with tag('tr'):
                        with tag("th"):
                            text(plot_kind)
                        # Write the header
                        j = 0 # Counter for each measure
                        for i in range(len(files["ordered"])):
                            if files['ordered'][i] in files['vna']:
                                if plot_kind == "Power":
                                    with tag("th"):
                                        text("%.1f"%(files["gain"][i] - attenuation) + pw_units)
                                elif plot_kind == "VNA":
                                    with tag('td'):
                                        with tag("div", klass = 'zoom'):
                                            doc.asis(format_image_style(plot_names['VNA'][j]))
                                elif plot_kind == "Noise":
                                    with tag('td'):
                                        with tag("div", klass = 'zoom'):
                                            doc.asis(format_image_style(plot_names['Noise'][j]))
                                elif plot_kind == "NEF":
                                    with tag('td'):
                                        with tag("div", klass = 'zoom'):
                                            doc.asis(format_image_style(plot_names['QFNoise'][j]))
                                elif plot_kind == "Diagnostic":
                                    with tag('td'):
                                        with tag("div", klass = 'zoom'):
                                            doc.asis(format_image_style(plot_names['Diagnostic'][j],plot_names['Diagnostic_link'][j]))
                                elif plot_kind == "Timestream":
                                    with tag('td'):
                                        with tag("div", klass = 'zoom'):
                                            doc.asis(format_image_style(plot_names['Timestream'][j]))
                                elif plot_kind == "Fits":
                                    with tag('td'):
                                        with tag("div", klass = 'zoom'):
                                            doc.asis(format_image_style(plot_names['Fits'][j]))
                                j += 1


        f = open(OVERVIEW_FILENAME+".html", "w")
        f.write(doc.getvalue())
        print (OVERVIEW_FILENAME+".html has been generated!")
