import matplotlib.pyplot as plt


def makeGraph(data, x_label, y_label, title, legend):

	x_vals = data["x_vals"];
	y_vals = data["y_vals"];

	plt.figure(figsize=(10, 6))

	for label, data in y_vals.items():
	    plt.plot(x_vals, data, label=label, marker='o', linestyle='-', markersize=5)


	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)

	plt.grid(True)
	plt.legend(title=legend)
	plt.tight_layout()

	plt.show()


# relativeTo will tell relative to what the speedup was calculated.
# If it's a string, it will calculate the speedup like speedup[method] = y_vals[method] / y_vals[relativeTo]
# If it's a number, it will calculate the speedup like speedup[method] = y_vals[method][relativeTo] / y_vals[method]
def makeSpeedupGraph(data, x_label, y_label, title, legend, relativeTo):

	x_vals = data["x_vals"];
	y_vals = data["y_vals"];

	speedup = {}
	if (isinstance(relativeTo, str)):
		for label, data in y_vals.items():
			speedup[label] = [y_vals[relativeTo][idx] / y_vals[label][idx] for idx in range(len(x_vals))];
	elif (isinstance(relativeTo, int)):
		for label, data in y_vals.items():
			speedup[label] = [y_vals[label][relativeTo] / val for val in y_vals[label]];



	plt.figure(figsize=(10, 6))

	for label, data in speedup.items():
	    plt.plot(x_vals, data, label=label, marker='o', linestyle='-', markersize=5)


	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)

	plt.grid(True)
	plt.legend(title=legend)
	plt.tight_layout()

	plt.show()



results_num_threads = {
	"x_vals": [1, 2, 3, 4], # number of threads used
	"y_vals": {
		"OMP": [16.604175, 8.87628, 6.03486, 4.70602],
		"OMP + SIMD": [6.193465, 3.34056, 2.3584275, 1.9104375],
		"Scikit-Learn": [19.45247889, 11.09330535, 7.96846867, 6.79661989],
	}
}


results_K = {
	"x_vals": [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256], # number of clusters
	"y_vals": {
		"Basic": [13.842975, 16.512825, 18.402375, 23.492075, 28.859325, 38.6017, 49.268275, 68.21045,  89.08465, 170.961, 329.4995, 646.918],
		"SIMD": [13.76595, 14.21425, 13.475175, 13.4063, 13.21385, 17.63375, 17.848625, 21.57505, 24.774525, 37.2634, 62.544525, 112.92275],
		"OMP": [5.4087875, 5.3340125, 5.7453625, 8.1338, 8.701205, 11.103325, 14.499525, 18.809225, 23.79645, 44.837175, 86.10775, 168.1855],
		"OMP + SIMD": [5.12325, 4.72181, 4.5060975, 4.2786075, 4.242735, 5.22269, 5.24332, 6.122445, 6.8563175, 10.2356, 17.0769, 30.22315],
		"Scikit-Learn": [10.5790471, 10.15588726, 9.29654837, 11.24959946, 13.51514578, 13.67138743, 17.18689322, 21.48360848, 25.62408686, 42.34225273, 70.51656485, 129.60475802],
	}
}

results_K_64 = {
	"x_vals": [2, 3, 4, 6, 8, 12, 16, 24, 32, 64], # number of clusters
	"y_vals": {
		"Basic": [13.842975, 16.512825, 18.402375, 23.492075, 28.859325, 38.6017, 49.268275, 68.21045,  89.08465, 170.961],
		"SIMD": [13.76595, 14.21425, 13.475175, 13.4063, 13.21385, 17.63375, 17.848625, 21.57505, 24.774525, 37.2634],
		"OMP": [5.4087875, 5.3340125, 5.7453625, 8.1338, 8.701205, 11.103325, 14.499525, 18.809225, 23.79645, 44.837175],
		"OMP + SIMD": [5.12325, 4.72181, 4.5060975, 4.2786075, 4.242735, 5.22269, 5.24332, 6.122445, 6.8563175, 10.2356],
		"Scikit-Learn": [10.5790471, 10.15588726, 9.29654837, 11.24959946, 13.51514578, 13.67138743, 17.18689322, 21.48360848, 25.62408686, 42.34225273],
	}
}


makeGraph(results_K, "K", "Time Per Iteration (ms)", "Performance Comparison for Different Methods", "Methods");
makeSpeedupGraph(results_K, "K", "Speedup", "Speedup Comparison for Different Methods (Relative to Basic)", "Methods - Speedup", "Basic");

# ignore last 2 so we can see the start of the graph better
makeGraph(results_K_64, "K", "Time Per Iteration (ms)", "Performance Comparison for Different Methods", "Methods");
makeSpeedupGraph(results_K_64, "K", "Speedup", "Speedup Comparison for Different Methods (Relative to Basic)", "Methods - Speedup", "Basic");

makeGraph(results_num_threads, "Number of Threads", "Time Per Iteration (ms)", "Performance Comparison for Different Methods", "Methods");
makeSpeedupGraph(results_num_threads, "Number of Threads", "Speedup", "Speedup Comparison for Different Methods (Relative to 1 Thread)", "Methods - Speedup", 0);
