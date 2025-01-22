for i in range(1, 4):
    plt.plot(np.arange(1, 10000001) * i, label=f"Counter {i}")
plt.legend()
plt.show()