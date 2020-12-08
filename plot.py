import matplotlib.pyplot as plt

name_list = [ '600','1000']

loss_baseline= [1.489081608, 1.722715688, 1.762347062, 1.912985248]
loss_ours = [1.677180467, 1.831286921, 1.956305596, 1.995346726]

Time_total_baseline = [ 1636.869875, 2866.929965]
Time_total_ours = [ 2005.664433,  3323.672714]

rounds_baseline = [5,
9,
10,
15
]
rounds_ours = [8,
12,
17,
19
]

Time_Aver_baseline = [
181.8744305,

191.1286643,
]
Time_Aver_ours = [
167.1387027,

174.9301428
]



x = list(range(2))
total_width, n = 0.8, 2
width = total_width / n

plt.bar(x, Time_total_baseline, width=width, label='Baseline', fc='lightgreen')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, Time_total_ours, width=width, label='Ours', tick_label=name_list, fc='red')
plt.ylabel("Total Train Time")
plt.xlabel("Budget_totall")
plt.legend()
plt.show()

