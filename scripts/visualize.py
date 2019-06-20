import collections
import matplotlib.pyplot as plt
import numpy as np



### SETTINGS ###################################################################

env = "GoToObj"

n = 1

num_symbols = 8
max_len_msg = 8

f = 100000

show = 0

sort_by = "frequency"

# only count kth frame after message emission, 0 means all frames
k = 1

top = 20



### INITIALIZE #################################################################

# load data
data = np.load("messages/%s/data_%s_n%d_%d^%d_f%d.npy" % (env, env, n, num_symbols, max_len_msg, f))

# dictionary message to action
d_m2a = collections.defaultdict(lambda: np.zeros((2, 7)))

# dictionary message to heatmap
d_m2h_goal = collections.defaultdict(lambda: np.zeros((2, 13, 13)))
d_m2h      = collections.defaultdict(lambda: np.zeros((2, 13, 13, 3, 6)))



### CHECK ######################################################################

assert sort_by in ["message",
                   "frequency", "frequency_none", "frequency_wall",
                   "action0",   "action0_none",   "action0_wall",
                   "action1",   "action1_none",   "action1_wall",
                   "action2",   "action2_none",   "action2_wall",
                   "action3",   "action3_none",   "action3_wall",
                   "action4",   "action4_none",   "action4_wall",
                   "action5",   "action5_none",   "action5_wall",
                   "action6",   "action6_none",   "action6_wall"]

assert 0 <= k and k <= n



### COLLECT ####################################################################

j = 0
for i in range(len(data)):
    # reset if new episode
    j *= 1 - data[i, 13]
    
    if k == 0 or j % n == (k - 1):
        agent_x = data[i, 0]
        agent_y = data[i, 1]
        
        if agent_y == 1:
            # agent is facing a wall
            wall = 1
        else:
            # agent is not facing wall
            wall = 0
        
        goal_type  = data[i, 2]
        goal_color = data[i, 3]
        
        message = tuple(data[i, 4:12])
        
        action  = data[i, 12]

        objects_x     = []
        objects_y     = []
        objects_type  = []
        objects_color = []
        g = 15
        while g < len(data[0]):
            object_x     = data[i, g  ]
            object_y     = data[i, g+1]
            object_type  = data[i, g+2]
            object_color = data[i, g+3]
            
            if object_x != 0:
                objects_x.append(object_x)
                objects_y.append(object_y)
                objects_type.append(object_type)
                objects_color.append(object_color)
            
            g += 4
        
        for b in range(len(objects_x)):
            x     = 6 + objects_x[b] - agent_x
            y     = 6 + objects_y[b] - agent_y
            type  = objects_type[b]
            color = objects_color[b]
            
            d_m2a[message][wall][action] += 1
            
            d_m2h[message][wall][x][y][type-5][color] += 1

            if type == goal_type and color == goal_color:
                d_m2h_goal[message][wall][x][y] += 1

    j += 1

# sum total
S = 0
for message in d_m2a:
    S += d_m2a[message].sum()



### SORT #######################################################################

M = np.zeros((len(d_m2a), max_len_msg + 1))
for i, message in enumerate(d_m2a):
    M[i, :-1] = message

for i in reversed(range(max_len_msg)):
    M = M[M[:, i].argsort(kind="stable")]

def sort_by_action(action):
    for i, message in enumerate(M[:, :-1]):
        if sort_by.endswith("_none"):
            if 0 < d_m2a[tuple(message)][0].sum():
                M[i, -1] = d_m2a[tuple(message)][0][action] / d_m2a[tuple(message)][0].sum()
        elif sort_by.endswith("_wall"):
            if 0 < d_m2a[tuple(message)][1].sum():
                M[i, -1] = d_m2a[tuple(message)][1][action] / d_m2a[tuple(message)][1].sum()
        else:
            M[i, -1] = d_m2a[tuple(message)][:, action].sum(0) / d_m2a[tuple(message)].sum()

if sort_by != "message":
    if sort_by.startswith("frequency"):
        for i, message in enumerate(M[:, :-1]):
            if sort_by.endswith("_none"):
                M[i, -1] = d_m2a[tuple(message)][0].sum()
            elif sort_by.endswith("_wall"):
                M[i, -1] = d_m2a[tuple(message)][1].sum()
            else:
                M[i, -1] = d_m2a[tuple(message)].sum()
    elif sort_by.startswith("action"):
        sort_by_action(int(sort_by[6]))

    M = M[M[:, -1].argsort(kind="stable")]
    M = np.flip(M, axis=0)

M = M[:, :-1]



### PRINT ######################################################################

print()
print("                                   actions")
print("    message    rgt    lft    fwd    pkp      4      5      6      freq         %")
for i, m in enumerate(M[:top]):
    print("%2d " % i, end="")
    for s in m:
        print(chr(97+int(s+0.5)), end="")
    print("   ", end="")
    actions  = d_m2a[tuple(m)].sum(0)
    sum      = actions.sum()
    actions /= sum
    for action in actions:
        print("%03.2f   " % action, end="")
    print("%7d" % sum, end="")
    print("   %7.4f" % (sum/S*100))

    print("              ", end="")
    actions  = d_m2a[tuple(m)][0]
    sum      = actions.sum()
    if 0 < sum:
        actions /= sum
    for action in actions:
        print("%03.2f   " % action, end="")
    print("%7d" % sum, end="")
    print("   %7.4f" % (sum/S*100))

    print("              ", end="")
    actions  = d_m2a[tuple(m)][1]
    sum      = actions.sum()
    if 0 < sum:
        actions /= sum
    for action in actions:
        print("%03.2f   " % action, end="")
    print("%7d" % sum, end="")
    print("   %7.4f" % (sum/S*100))



### PLOT #######################################################################

#for message in d_m2h:
#    d_m2h[message] = d_m2h[message][:, :, :, 0, :].sum(-1)
d_m2h = d_m2h_goal

m = M[show]

h  = d_m2h[tuple(m)].sum(0)
h /= h.sum()

h_0  = d_m2h[tuple(m)][0]
if 0 < h_0.sum():
    h_0 /= h_0.sum()

h_1  = d_m2h[tuple(m)][1]
if 0 < h_1.sum():
    h_1 /= h_1.sum()

#h   = np.flip(h.transpose(),   axis=1)
#h_0 = np.flip(h_0.transpose(), axis=1)
#h_1 = np.flip(h_1.transpose(), axis=1)

t = ""
for s in m:
    t += chr(97+int(s+0.5))

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

fig.suptitle(r"$m$ = %s" % t)

ax0.set_title(r"$P(g | m)$")
ax0.imshow(h,   cmap="inferno", vmin=0.0, vmax=h.max(),   interpolation="bicubic")

ax1.set_title(r"$P(g | m, wall=False)$")
ax1.imshow(h_0, cmap="inferno", vmin=0.0, vmax=h_0.max(), interpolation="bicubic")

ax2.set_title(r"$P(g | m, wall=True)$")
ax2.imshow(h_1, cmap="inferno", vmin=0.0, vmax=h_1.max(), interpolation="bicubic")

plt.show()
