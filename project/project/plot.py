#import matplotlib.pyplot as plt
#
## Data
#alphas = [0.06, 0.05, 0.04, 0.03, 0.02]
#throughputs = [
#    [39.75655, 62.558899, 168.111145, 277.541595, 320.001648],
#    [45.16534, 79.416977, 177.942337, 358.124298, 488.919891],
#    [46.187061, 89.214195, 182.003677, 362.848572, 483.550293],
#    [46.764053, 84.779472, 181.547012, 356.401581, 488.147491],
#    [48.961208, 89.658112, 186.647125, 343.59024, 488.667328]
#]
#transposed_throughput = list(zip(*throughputs))
#
#codewords = [100, 1000, 10000, 100000, 1000000]
#
## Plotting
#plt.figure(figsize=(5, 3))
#for i in range(len(alphas)):
#    plt.plot(codewords, transposed_throughput[i], label='Alpha = {}'.format(alphas[i]))
#
#plt.title('Throughput vs. Codeword for Different Alpha Values')
#plt.xlabel('Codeword (G)')
#plt.ylabel('Throughput')
#plt.xscale('log')
#plt.grid(True)
#plt.legend()
#plt.tight_layout()
#plt.show()

import matplotlib.pyplot as plt

# Data
block_dims = [32, 64, 128, 256, 512]
exec_times = [
    [2211.876953, 3299.05127, 3007.460938, 3001.13623, 3025.567871],
    [3676.326416, 3385.557617, 3566.663818, 3167.567139, 2895.596924],
    [5784.127441, 5586.415039, 5149.962891, 4907.630859, 3148.133301],
    [13333.08008, 13661.75195, 13195.73731, 12572.79785, 2838.246338],
    [45432.64844, 42091.59766, 47676.4375, 41707.23438, 2669.506104]
]
alphas = [0.06, 0.05, 0.04, 0.03, 0.02]

# Plotting
plt.figure(figsize=(7, 4))
for i in range(len(alphas)):
    plt.plot(block_dims, exec_times[i], label='Alpha = {}'.format(alphas[i]))

plt.title('Alpha vs Block Dimension based on ExecTime')
plt.xlabel('Block Dimension')
plt.ylabel('ExecTime')
plt.grid(True)
plt.legend(title='Alpha')
# Set x-axis ticks to powers of two
plt.xscale('log', base=2)
plt.xticks(block_dims, [str(x) for x in block_dims])
plt.tight_layout()
plt.show()
