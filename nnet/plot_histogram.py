import matplotlib.pyplot as plt
import numpy as np

p0 = [1.802, 3.518, 1.665, 2.577, 2.418]
p5 = [8.793, 7.396, 9.521, 7.620, 8.873]
p10 = [9.861, 9.846, 8.958, 9.971, 6.912]
p20 = [9.570, 10.664, 11.013, 10.493, 10.694]
p30 = [11.544, 11.733, 11.431, 11.338, 11.132]
p40 = [11.995, 12.178, 12.100, 12.054, 12.019]
p50 = [12.389, 12.543, 12.522, 12.392, 12.417]
p60 = [12.175, 12.619, 12.477, 12.578, 12.442]
p70 = [12.746, 12.944, 12.550, 12.871, 12.865]
p80 = [13.006, 13.061, 12.877, 12.688, 12.997]
p90 = [12.841, 12.836, 12.983, 12.973, 12.840]
p100 = [12.810, 12.334, 12.774, 12.914, 12.778]

percentage = ["0%", "5%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
x_pos = np.arange(len(percentage))
x_pos = [2*i for i in x_pos]
MEANs = [np.mean(p0), np.mean(p5), np.mean(p10), np.mean(p20), np.mean(p30), np.mean(p40), np.mean(p50), np.mean(p60), np.mean(p70), np.mean(p80), np.mean(p90), np.mean(p100)]
STDs = [np.std(p0), np.std(p5), np.std(p10), np.std(p20), np.std(p30), np.std(p40), np.std(p50), np.std(p60), np.std(p70), np.std(p80), np.std(p90), np.std(p100)]

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x_pos, MEANs, yerr=STDs, align='center', width = 0.8,  ecolor='black', capsize=10, color='orange')
ax.set_ylabel('SI-SNR (dB)')

ax.set_xticks(x_pos)
ax.set_xticklabels(percentage)
ax.set_xlabel('Percent of supervised mixtures (%)')
#ax.set_title('')
ax.yaxis.grid(True)

rects = ax.patches
# Make some labels.
for rect, label in zip(rects, MEANs):
    height = rect.get_height()
    ax.text(rect.get_x(), height-0.5, "{:.2f}".format(label),
            ha='right', va='bottom')
'''
for rect, label in zip(rects, STDs):
    height = rect.get_height()
    ax.text(rect.get_x()+rect.get_width(), height, "Std: {:.2f}".format(label),
            ha='center', va='top')'''

# Save the figure and show
plt.tight_layout()
plt.savefig('resultsHistogram.png')
plt.show()