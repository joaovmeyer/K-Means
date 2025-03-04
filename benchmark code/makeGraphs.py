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
		"OMP": [19.5266, 10.5626, 6.84994, 5.49541],
		"OMP + SIMD": [7.06722, 3.86341, 2.67853, 2.04488],
		"Scikit-Learn": [23.436622619628906, 11.249675750732422, 8.124642372131348, 6.874728202819824]
	}
}


results_K = {
	"x_vals": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256], # number of clusters
	"y_vals": {
		"Basic": [25.357, 14.6224, 17.4151, 19.8269, 25.1448, 31.0203, 41.9868, 52.529, 74.2456, 99.6279, 187.44, 360.688, 707.986],
		"SIMD": [30.3082, 17.975, 15.5481, 15.3502, 16.2187, 16.224, 19.9145, 20.087, 23.7944, 27.8933, 43.4651, 74.929, 133.656],
		"OMP": [6.69565, 4.86993, 5.90309, 6.12528, 8.01051, 9.26377, 12.2898, 15.4191, 21.8297, 29.1716, 54.199, 104.17, 204.346],
		"OMP + SIMD": [10.8475, 5.50218, 4.61407, 4.75894, 4.6377, 4.59503, 6.15736, 5.98534, 6.79445, 8.06411, 12.1757, 20.7889, 37.4668],
		"Scikit-Learn": [54.68547344207764, 11.718342701594034, 10.71389743259975, 9.687161445617676, 11.562085151672363, 14.06200885772705, 14.999456405639648, 18.124361038208008, 22.186708450317383, 25.936579704284668, 40.93606948852539, 71.55993461608887, 130.30787467956543],
	}
}


makeGraph(results_K, "K", "Time Per Iteration (ms)", "Performance Comparison for Different Methods", "Methods");
makeSpeedupGraph(results_K, "K", "Speedup", "Speedup Comparison for Different Methods (Relative to Basic)", "Methods - Speedup", "Basic");

makeGraph(results_num_threads, "Number of Threads", "Time Per Iteration (ms)", "Performance Comparison for Different Methods", "Methods");
makeSpeedupGraph(results_num_threads, "Number of Threads", "Speedup", "Speedup Comparison for Different Methods (Relative to 1 Thread)", "Methods - Speedup", 0);