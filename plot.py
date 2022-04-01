import matplotlib.pyplot as plt

f = open("./weights/speed/log.txt")
lines = f.readlines()

x = []
y = []
for idx, line in enumerate(lines):
    x.append(idx)
    y.append(float(line))
    if idx>150:
        break

plt.title("loss figure")
plt.plot(x, y, color = 'blue')
plt.savefig('plot_obbloss.png')
