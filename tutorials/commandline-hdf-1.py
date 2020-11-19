for i in range(1, 4):
    plt.plot(np.arange(1, 1001) * i, label=f"Counter {i}")
plt.legend()
plt.show()