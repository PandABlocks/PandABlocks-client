import sys

import h5py
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with h5py.File(sys.argv[1], "r") as hdf:
        # Print the dataset names in the file
        datasets = list(hdf)
        print(datasets)
        # Plot the counter mean values for the 3 counters we know are captured
        for i in range(1, 4):
            plt.plot(hdf[f"COUNTER{i}.OUT.Mean"], label=f"Counter {i}")
        # Add a legend and show the plot
        plt.legend()
        plt.show()
