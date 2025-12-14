import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


data = pd.read_csv("simulation_input.csv")

interarrivals = data["interarrival"].values
service_times = data["total_service_time"].values

# Limit workload size
MAX_REQUESTS = 1000
interarrivals = interarrivals[:MAX_REQUESTS]
service_times = service_times[:MAX_REQUESTS]

SIM_TIME = 3000


# fixed server 
class FixedCloud:
    def __init__(self, env, servers=3):
        self.env = env
        self.servers = simpy.Container(env, capacity=servers, init=servers)
        self.active_servers = servers

        self.response_times = []
        self.queue_lengths = []
        self.busy_time = 0.0
        self.last_event = 0.0

        env.process(self.monitor())

    def process_request(self, service_time):
        arrival = self.env.now
        yield self.servers.get(1)

        start = self.env.now
        self._update_busy(start)
        yield self.env.timeout(service_time)
        self._update_busy(self.env.now)

        self.servers.put(1)
        self.response_times.append(self.env.now - arrival)

    def _update_busy(self, now):
        busy = self.active_servers - self.servers.level
        self.busy_time += busy * (now - self.last_event)
        self.last_event = now

    def monitor(self):
        while True:
            self.queue_lengths.append(len(self.servers.get_queue))
            yield self.env.timeout(5)


#  auto scaling system
class AutoScalingCloud:
    def __init__(self, env, init_servers=3, max_servers=10,
                 scale_up_threshold=5, scale_down_threshold=0.3):

        self.env = env
        self.active_servers = init_servers
        self.max_servers = max_servers

        self.servers = simpy.Container(
            env, capacity=max_servers, init=init_servers
        )

        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

        self.response_times = []
        self.queue_lengths = []
        self.busy_time = 0.0
        self.last_event = 0.0

        env.process(self.monitor())

    def process_request(self, service_time):
        arrival = self.env.now
        yield self.servers.get(1)

        start = self.env.now
        self._update_busy(start)
        yield self.env.timeout(service_time)
        self._update_busy(self.env.now)

        self.servers.put(1)
        self.response_times.append(self.env.now - arrival)

    def _update_busy(self, now):
        busy = self.active_servers - self.servers.level
        self.busy_time += busy * (now - self.last_event)
        self.last_event = now

    def monitor(self):
        while True:
            q_len = len(self.servers.get_queue)
            self.queue_lengths.append(q_len)

            utilization = (
                (self.active_servers - self.servers.level) / self.active_servers
            )

            if q_len > self.scale_up_threshold and self.active_servers < self.max_servers:
                self.servers.put(1)
                self.active_servers += 1

            elif utilization < self.scale_down_threshold and self.active_servers > 1:
                yield self.servers.get(1)
                self.active_servers -= 1

            yield self.env.timeout(5)


# arrival process
def arrivals(env, system, interarrivals, service_times):
    for i in range(len(interarrivals)):
        yield env.timeout(interarrivals[i])
        env.process(system.process_request(service_times[i]))


# run scenario function
def run_scenario(system_class, interarrivals, label):
    env = simpy.Environment()
    system = system_class(env)
    env.process(arrivals(env, system, interarrivals, service_times))
    env.run(until=SIM_TIME)

    avg_w = np.mean(system.response_times)
    avg_l = np.mean(system.queue_lengths)
    util = system.busy_time / (env.now * system.active_servers)
    throughput = len(system.response_times) / env.now

    return {
        "label": label,
        "response_times": system.response_times,
        "queue_lengths": system.queue_lengths,
        "W": avg_w,
        "L": avg_l,
        "rho": util,
        "throughput": throughput
    }


random.seed(42)
np.random.seed(42)

results = []

# fixed servers
results.append(run_scenario(
    lambda env: FixedCloud(env, servers=3),
    interarrivals,
    "Fixed (3 servers)"
))

# auto scaling
results.append(run_scenario(
    lambda env: AutoScalingCloud(env),
    interarrivals,
    "Auto-scaling"
))

# high load
interarrivals_high = interarrivals / 2
results.append(run_scenario(
    lambda env: AutoScalingCloud(env),
    interarrivals_high,
    "High load (2λ)"
))



for r in results:
    print(f"\n{r['label']}")
    print(f"Avg Response Time (W): {r['W']:.3f} s")
    print(f"Avg Queue Length (L): {r['L']:.2f}")
    print(f"Utilization (ρ): {r['rho']:.2f}")
    print(f"Throughput: {r['throughput']:.2f} req/s")


# Visualization
plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
for r in results:
    plt.hist(r["response_times"], bins=30, alpha=0.6, label=r["label"])
plt.title("Response Time Distribution")
plt.xlabel("Time (s)")
plt.ylabel("Frequency")
plt.legend()


plt.subplot(1,2,2)
for r in results:
    plt.plot(r["queue_lengths"], label=r["label"])
plt.title("Queue Length Over Time")
plt.xlabel("Monitoring Interval")
plt.ylabel("Queue Length")
plt.legend()

plt.tight_layout()
plt.show()
