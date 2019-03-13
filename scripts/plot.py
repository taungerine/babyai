import csv
import matplotlib.pyplot as plt
import numpy as np

success_rates = []
for file in ["nocomm/no_shared_reward/log0A.csv",
             "nocomm/no_shared_reward/log1A.csv",
             "nocomm/shared_reward/log0A.csv",
             "nocomm/shared_reward/log0B.csv",
             "comm/2^16/shared_reward/log0E.csv",
             "comm/2^16/shared_reward/log0.csv",
             "swobs/2^16/shared_reward/log0F.csv",
             "swobs/2^16/shared_reward/log0G.csv",
             "swobs/2^16/switched_reward/log0A.csv",
             "swobs/2^16/switched_reward/log0B.csv",
             "swobs/2^16/no_shared_reward/log0.csv",
             "swobs/2^16/no_shared_reward/log1.csv",
             "comm/2^16/no_shared_reward/log0.csv",
             "comm/2^16/no_shared_reward/log1.csv",
             "swobs/16^4/shared_reward/log0A.csv",
             "swobs/16^4/shared_reward/log0B.csv",
             "swobs/16^4/no_shared_reward/log0A.csv",
             "swobs/16^4/no_shared_reward/log0B.csv",
             "swobs/1^0/shared_reward/log0A.csv",
             "swobs/1^0/shared_reward/log0B.csv",
             "swobs/1^0/no_shared_reward/log0A.csv",
             "swobs/1^0/no_shared_reward/log0B.csv",
             "swobs/16^8/tau=0.1/shared_reward/log0A.csv",
             "swobs/16^8/tau=0.1/shared_reward/log0B.csv",
             "swobs/16^4/tau=0.1/shared_reward/log0A.csv",
             "swobs/16^4/tau=0.1/shared_reward/log0B.csv",
             "swobs/16^4/shared_reward/corr/log0A.csv",
             "swobs/16^4/shared_reward/corr/log0B.csv"]:
    with open("logs/plots/" + file, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        success_rate = []
        i = 0
        for row in csvreader:
            if i != 0:
                success_rate.append(float(row[9]))
            i += 1
        success_rates.append(success_rate)

#plt.plot(success_rates[0], color='b', label="no communication, no shared reward (2)")
#plt.plot(success_rates[1], color='b')
#plt.plot(success_rates[2], color='k', label="no communication, shared reward (2)")
#plt.plot(success_rates[3], color='k')
#plt.plot(success_rates[4], color='g', label="communication, shared reward (2)")
#plt.plot(success_rates[5], color='g')
#plt.plot(success_rates[10], color='y', label="communication, switched observations, no shared reward (2)")
#plt.plot(success_rates[11], color='y')
#plt.plot(success_rates[6], color='r', label="communication, switched observations, shared reward (2)")
#plt.plot(success_rates[7], color='r')
#plt.plot(success_rates[8], color='y', label="switched observations, switched reward (2)")
#plt.plot(success_rates[9], color='y')

sr = np.asarray(success_rates, dtype=np.float32)
#plt.plot((sr[0] +sr[1]) /2, color='y', label=r"no comms, no shared $r$")
#plt.plot(1 - (1 - (sr[0] +sr[1])/2)**2, color='b', label=r"no comms, shared $r$ (expected)", alpha=0.25)
#plt.plot((sr[2] +sr[3]) /2, color='b', label=r"no comms, shared $r$")

#plt.title(r"$|M| = 2^{16}$")
#plt.plot((sr[12] +sr[13]) /2, color='m', label=r"comms, no shared $r$")
#plt.plot(1 - (1 - (sr[12] +sr[13])/2)**2, color='g', label=r"comms, no shared $r$ (expected)", alpha=0.25)
#plt.plot((sr[4] +sr[5]) /2, color='g', label=r"comms, shared $r$")

#plt.title(r"$|M| = 2^{16}$")
#plt.plot((sr[10]+sr[11])/2, color='c', label=r"comms, switched obs, no shared $r$")
#plt.plot(1 - (1 - (sr[10]+sr[11])/2)**2, color='r', label=r"comms, switched obs, shared $r$ (expected)", alpha=0.25)
#plt.plot((sr[6] +sr[7]) /2, color='r', label=r"comms, switched obs, shared $r$")

#plt.title(r"$|M| = 16^{4}$")
#plt.plot((sr[16]+sr[17])/2, color='c', label=r"comms, switched obs, no shared $r$")
#plt.plot(1 - (1 - (sr[16]+sr[17])/2)**2, color='r', label=r"comms, switched obs, shared $r$ (expected)", alpha=0.25)
#plt.plot((sr[14] +sr[15]) /2, color='r', label=r"comms, switched obs, shared $r$")

#plt.plot((sr[20]+sr[21])/2, color='c', label=r"no comms, switched obs, no shared $r$")
#plt.plot(1 - (1 - (sr[20]+sr[21])/2)**2, color='r', label=r"no comms, switched obs, shared $r$ (expected)", alpha=0.25)
#plt.plot((sr[18] +sr[19]) /2, color='r', label=r"no comms, switched obs, shared $r$")

#plt.title(r"$|M| = 16^{8}$, $\tau = 0.1$")

#plt.plot((sr[22] +sr[23]) /2, color='r', label=r"comms, switched obs, shared $r$")

#plt.title(r"$|M| = 16^{4}$, $\tau = 0.1$")
#plt.plot((sr[24] +sr[25]) /2, color='r', label=r"comms, switched obs, shared $r$")

plt.title(r"$|M| = 16^{4}$ (corrected)")
plt.plot((sr[26] +sr[27]) /2, color='r', label=r"comms, switched obs, shared $r$")

plt.ylim(-0.1, 1.1)

plt.legend()
plt.xlabel("epochs")
plt.ylabel(r"success ($r > 0$) rate")
plt.show()
