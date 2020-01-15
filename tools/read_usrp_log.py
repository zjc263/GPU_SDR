'''
Interpreter for the USRP server log files. Produces an html version of the log file and integrates a search function.
To best visualize the result, an internet connection is required as the html file will source some code from CDNs.
'''

import glob,sys,os
import argparse
import numpy
import plotly.figure_factory as ff
import plotly
from yattag import Doc

def severity_color(severity_string):
	if severity_string == "info":
		return ""
	elif severity_string == "debug":
		return "table-primary"
	elif severity_string == "error":
		return "table-danger"
	elif severity_string == "trace":
		return "table-active"
	elif severity_string == "warning":
		return "table-warning"
	elif severity_string == "fatal":
		return "bg-danger"
	else:
		return "bg-success"

searchJS = '''
<script>
$(document).ready(function(){
  $("#myInput").on("keyup", function() {
    var value = $(this).val().toLowerCase();
    $("#logTable tr").filter(function() {
      $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
    });
  });
});
</script>
'''

scrollCSS='''

.fixme_to {
  height: 100px;
  padding: 0 15px;
  width: 100%;
  position: fixed;
  top: 0;
  z-index: 1;
  background-color: white;
}
.table-area {
  position: relative;
  z-index: 0;
  margin-top: 160px;
}
table.table-hover {
  display: table;
  table-layout: fixed;
  width: 100%;
  height: 100%;
}

table.table-hover thead {
  position: fixed;
  top: 100px;
  left: 0;
  right: 0;
  width: 100%;
  height: 50px;
  line-height: 20px;
  #background: #eee;
  table-layout: fixed;
  display: table;
}


'''
def reduce_opacity(rgb_color_string, opacity):
	'''
	Function to reduce the opacity for overlapping tasks.
	Written to compensate plotly lack of features.
	'''
	try:
		new_color = 'rgba('+(rgb_color_string.split(')'))[0].split('(')[1]+', ' + str(opacity) + ')'
	except AttributeError:
		new_color = rgb_color_string
	return new_color


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--foldername', '-F',
						help='Name of the folder containing the logs.', type=str, default = "../logs")
	parser.add_argument('--filename', '-f',
						help='Name of the logfile.', type=str)


	args = parser.parse_args()

	try:
		os.chdir(args.foldername)
	except OSError:
		print "Cannot find \'%s\' folder."%args.foldername
		exit()

	if args.filename is None:

		all_logs = glob.glob("*.log")
		target_file = max(all_logs, key=os.path.getctime)

	else:
		target_file = args.filename

	raw_log = open(target_file).read()

	full_table = [line.split(";") for line in raw_log.split("\n")]
	full_table = full_table[:-1]

	threads = list(zip(*full_table)[1])
	threads_unique = set(threads)
	time = list(zip(*full_table)[0])
	levels = list(zip(*full_table)[2])

	events = list(zip(*full_table)[3])
	events_type = [ev.split(':')[0] for ev in events]
	events_id = [ev.split(':')[1] for ev in events]

	messages = list(zip(*full_table)[4])

	# Build the Gantt chart
	# Not the most efficient way to do it, I know...
	print("Building Gantt chart...")
	gantt_data = []
	for i in range(len(time)):
		if events_type[i] == "EVENT_START":
			start_time = time[i]
			job_id = threads[i]
			resource = levels[i]
			task = messages[i]
			# Look for the end
			end_time = max(time)
			for j in range(len(time)):
				if (events_id[j] == events_id[i]) and (i!=j):
					end_time = time[j]

			gantt_data.append(
				{
					'Task':job_id,
					'Start':start_time,
					'Finish':end_time,
					'Resource':resource,
				}
			)
	print gantt_data
	colors = dict(
		debug = 'rgb(46, 137, 205)',
		info = 'rgb(114, 44, 121)',
		warning = 'rgb(198, 47, 105)',
		error = 'rgb(58, 149, 136)'
	)
	#basically, you have to build your own chart... this is sooo bugged
	fig = ff.create_gantt(gantt_data, colors=colors, group_tasks=True, index_col='Resource', title='Server log',show_colorbar=True)#, bar_width=0.8, showgrid_x=True, showgrid_y=True)

	for s in fig.data:
		print s
		s.fillcolor = reduce_opacity(s.fillcolor, 0.1)

	plotly.offline.plot(fig, filename="_plot_" + target_file+".html", auto_open=False)

	# Write the nice html list
	log_data = dict(
		Time = time,
		Levels = levels
	)

	for thread in threads_unique:
		log_data[thread] = [ msg if t == thread else '' for msg, t in zip(messages, threads)]


	doc, tag, text = Doc().tagtext()
	doc.asis('<!DOCTYPE html>')
	with tag('html'):
		with tag('head'):

			doc.asis('<meta charset="UTF-8">')
			doc.asis('<meta name="description" content="log USRP Server">')
			doc.asis('<meta name="author" content="Caltech/JPL">')
			doc.asis('''
				<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
				 integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
			 ''')
			with tag("style"):
				doc.asis(scrollCSS)

		with tag('body'):
			with tag('div', klass="fixme_to"):
				with tag('h2'):
					with tag('b'):
						text("USRP server log ")
					text("From file \'%s\'"%target_file)
				doc.asis('''<input class="form-control" id="myInput" type="text" placeholder="Search..">''')

			with tag('section', klass = "content-area"):
				with tag('div', klass = "table-area"):

					with tag('table', klass = 'table table-condensed'):
						with tag('thead', klass="thead-dark"):
							with tag('tr'):
								with tag("th"):
									text("Time")
								with tag("th"):
									text("Level")
								for thread in threads_unique:
									with tag("th"):
										text(thread)
						with tag('tbody', **{'id': 'logTable'}):
							for i in range(len(time)):
								row_class = severity_color(log_data["Levels"][i])
								with tag("tr", klass = row_class):
									with tag('td'):
										text(str(log_data["Time"][i]))
									with tag('td'):
										text(str(log_data["Levels"][i]))
									for thread in threads_unique:
										with tag('td'):
											text(str(log_data[thread][i]))

			doc.asis('''
				<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"
				 integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
				<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"
				 integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
				<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"
				 integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
			 ''')
			doc.asis(searchJS)
	f = open(target_file+".html", "w")
	f.write(doc.getvalue())
	print (target_file+".html has been generated!")
