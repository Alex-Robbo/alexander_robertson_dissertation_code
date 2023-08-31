import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#%% Solar Profile Generation
t = np.linspace(0,24,241)
#Solar Supply profile

sigma = 2
mu = 12
mu_2 = -12
t_size = t

sol = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (t - mu)**2 / (2 * sigma**2) ) 
sol_2 = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (t - mu_2)**2 / (2 * sigma**2) ) 
sol = sol + sol_2
sol_total = sum(sol) #sum of normal distribution
sub = sol[60]
sol = sol - sub
count = 0
while count < 240:
    if count < 240/4:
        sol[count] = 0
        
    if count > 241*(3/4):
        sol[count] = 0
    count += 1

#Integral Calculator
solar_panel_surface_area = 10
incident_energy_per_m2 = 2.4 #kWh per day
incident_irradience = incident_energy_per_m2 * solar_panel_surface_area #kWh per day
area_under_curve = np.trapz(sol, t) #kWh
sol_normalised = incident_irradience*sol/area_under_curve #in kW

#Solar Panel Conversion
Solar_eff = 0.2
solar_power_production = Solar_eff * sol_normalised #kW
solar_energy = solar_power_production * 0.1
print(np.sum(solar_energy), "kWh generated from Solar Panels")

#Demand Profiles

#DataEnergy = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\Sustainable_Building_Excel.csv",header = 0, usecols=[5])
#energy_demand = (DataEnergy.to_numpy())


#DataEnergytime = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\Sustainable_Building_Excel.csv", usecols=[9])
#energytime = DataEnergytime.to_numpy()
#energy_demand = (energy_demand - min(energy_demand))/1000
number_cells = 3
cells = np.zeros((1,number_cells))
cells_ones = np.ones((1,number_cells))
eff = 0.96
power_supply = solar_power_production * eff  #kW
energy_supply = solar_energy *eff

in_power = power_supply  
in_energy = energy_supply
timestamp = 0
t_size = 496

cell_capacity = 3.6#ah
voltage = 4.2 #V
cell_capacity_kwh = cell_capacity * voltage / 1000 
combined_cell_capacity = cell_capacity_kwh * number_cells #kWh

optimum_charge = 0.9
max_charge = cell_capacity_kwh * optimum_charge
cells_full = np.sum(cells_ones * max_charge)
cells = cells_ones * max_charge
max_dis_c_rate = 0.5
max_ch_c_rate = 0.3
max_dis_current_cell = cell_capacity * max_dis_c_rate #A
max_ch_current_cell = cell_capacity * max_ch_c_rate

#importing datasets
power_data_161 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\161GSOCK.csv", header = 0, usecols = [4])
power_data_161 = power_data_161.to_numpy()/1000
energy_data_161 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\161GSOCK.csv", header = 0, usecols = [5])
energy_data_161 = energy_data_161.to_numpy()/1000
print("tot energy =", max(energy_data_161) - min(energy_data_161), "kWh")
timestamps_161 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\161GSOCK.csv", header = 0, usecols = [7])
timestamps_161 = np.around(timestamps_161.to_numpy(), decimals = 1)

power_data_157 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\157- G LIGHT.csv", header = 0, usecols = [4])
power_data_157 = power_data_157.to_numpy()/1000
energy_data_157 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\157- G LIGHT.csv", header = 0, usecols = [5])
energy_data_157 = energy_data_157.to_numpy()/1000
print("tot energy =", max(energy_data_157) - min(energy_data_157), "kWh")
timestamps_157 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\157- G LIGHT.csv", header = 0, usecols = [7])
timestamps_157 = np.around(timestamps_157.to_numpy(), decimals = 1)

power_data_156 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\156- K SOCK.csv", header = 0, usecols = [4])
power_data_156 = power_data_156.to_numpy()/1000
energy_data_156 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\156- K SOCK.csv", header = 0, usecols = [5])
energy_data_156 = energy_data_156.to_numpy()/1000
print("tot energy =", max(energy_data_156) - min(energy_data_156), "kWh")
timestamps_156 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\156- K SOCK.csv", header = 0, usecols = [7])
timestamps_156 = np.around(timestamps_156.to_numpy(), decimals = 1)

power_data_154 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\154- HOB.csv", header = 0, usecols = [4])
power_data_154 = power_data_154.to_numpy()/1000
energy_data_154 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\154- HOB.csv", header = 0, usecols = [5])
energy_data_154 = energy_data_154.to_numpy()/1000
print("tot energy =", max(energy_data_154) - min(energy_data_154), "kWh")
timestamps_154 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\154- HOB.csv", header = 0, usecols = [7])
timestamps_154 = np.around(timestamps_154.to_numpy(), decimals = 1)

power_data_153 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\153- UP SOCK.csv", header = 0, usecols = [4])
power_data_153 = power_data_153.to_numpy()/1000
energy_data_153 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\153- UP SOCK.csv", header = 0, usecols = [5])
energy_data_153 = energy_data_153.to_numpy()/1000
print("tot energy =", max(energy_data_153) - min(energy_data_153), "kWh")
timestamps_153 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\153- UP SOCK.csv", header = 0, usecols = [7])
timestamps_153 = np.around(timestamps_153.to_numpy(), decimals = 1)
plt.figure(14)
plt.plot(timestamps_153,(energy_data_153- min(energy_data_153)))

power_data_151 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\151- UP LIGHT.csv", header = 0, usecols = [4])
power_data_151 = power_data_151.to_numpy()/1000
energy_data_151 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\151- UP LIGHT.csv", header = 0, usecols = [5])
energy_data_151 = energy_data_151.to_numpy()/1000
print("tot energy =", max(energy_data_151) - min(energy_data_151), "kWh")
timestamps_151 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\151- UP LIGHT.csv", header = 0, usecols = [7])
timestamps_151 = np.around(timestamps_151.to_numpy(), decimals = 1)

power_data_150 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\150- OVEN.csv", header = 0, usecols = [4])
power_data_150 = power_data_150.to_numpy()/1000
energy_data_150 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\150- OVEN.csv", header = 0, usecols = [5])
energy_data_150 = energy_data_150.to_numpy()/1000
print("tot energy =", max(energy_data_150) - min(energy_data_150), "kWh")
timestamps_150 = pd.read_csv("C:\\Users\\asbro\\OneDrive\\Desktop\\DISS_DATASETS\\150- OVEN.csv", header = 0, usecols = [7])
timestamps_150 = np.around(timestamps_150.to_numpy(), decimals = 1)

step_total_161 = len(timestamps_161)
step_total_157 = len(timestamps_157)
step_total_156 = len(timestamps_156)
step_total_154 = len(timestamps_154)
step_total_153 = len(timestamps_153)
step_total_151 = len(timestamps_151)
step_total_150 = len(timestamps_150)
total_timestamp = np.linspace(0, 240, 241)

circuit_power = np.zeros((3,241))
circuit_energy = np.zeros((3,241))
k = 0

while k < step_total_161:
    
    t = timestamps_161[k] 
    index = int(10 * t)
    circuit_power[0,index] += power_data_161[k,0]
    if energy_data_161[k,0] > energy_data_161[k-1,0] and k > 0:
        circuit_energy[0,index] += (energy_data_161[k,0] - energy_data_161[k-1,0])
    k+=1

k = 0
while k < step_total_157:
    
    t = timestamps_157[k] 
    index = int(10 * t)
    circuit_power[1,index] += power_data_157[k,0]
    if energy_data_157[k,0] > energy_data_157[k-1,0] and k > 0:
        circuit_energy[1,index] += (energy_data_157[k,0] - energy_data_157[k-1,0])
    k+=1
k = 0
while k < step_total_156:
    
    t = timestamps_156[k] 
    index = int(10 * t)
    circuit_power[2,index] += power_data_156[k,0]
    if energy_data_156[k,0] > energy_data_156[k-1,0] and k > 0:
        circuit_energy[2,index] += (energy_data_156[k,0] - energy_data_156[k-1,0])
    k+=1
k = 0
while k < step_total_154:
    
    t = timestamps_154[k] 
    index = int(10 * t)
    circuit_power[0,index] += power_data_154[k,0]
    if energy_data_154[k,0] > energy_data_154[k-1,0] and k > 0:
        circuit_energy[0,index] += (energy_data_154[k,0] - energy_data_154[k-1,0])
    k+=1
k = 0
while k < step_total_153:
    
    t = timestamps_153[k] 
    index = int(10 * t)
    circuit_power[1,index] += power_data_153[k,0]
    if energy_data_153[k,0] > energy_data_153[k-1,0] and k > 0:
        circuit_energy[1,index] += (energy_data_153[k,0] - energy_data_153[k-1,0])
    k+=1
k = 0
while k < step_total_151:
    
    t = timestamps_151[k] 
    index = int(10 * t)
    circuit_power[1,index] += power_data_151[k,0]
    if energy_data_151[k,0] > energy_data_151[k-1,0] and k > 0:
        circuit_energy[1,index] += (energy_data_151[k,0] - energy_data_151[k-1,0])
    k+=1
k = 0
while k < step_total_150:
    
    t = timestamps_150[k] 
    index = int(10 * t)
    circuit_power[0,index] += power_data_150[k,0]
    if energy_data_150[k,0] > energy_data_150[k-1,0] and k > 0:
        circuit_energy[0,index] += (energy_data_150[k,0] - energy_data_150[k-1,0])
    k+=1
    
number_of_circuits = 3
t_size = 241
timestamp = 0 

time_at_zero = np.zeros([t_size, number_cells])
cell_level = np.zeros([t_size, number_cells])
time_below_twenty = np.zeros([t_size, number_cells])
dis_current_array = np.zeros([t_size, number_cells])
ch_current_array = np.zeros([t_size, number_cells])

while timestamp < t_size :
    x = 0
    while x < number_of_circuits:
        total_current_out = circuit_power[x,timestamp]/voltage
        no_dis_cells = math.ceil(total_current_out/max_dis_current_cell)
        if no_dis_cells == 0:
            loss_per_cell = 0
        else:
            loss_per_cell = circuit_energy[x, timestamp]/no_dis_cells
            indices_dis = cells.argsort()[-no_dis_cells:][::-1]
        n = 0
        while n < no_dis_cells:
            discharging_cell = indices_dis[0,n]
            dis_current_array[timestamp, discharging_cell] = (circuit_power[x,timestamp]/voltage) / no_dis_cells
            if cells[0,discharging_cell] > loss_per_cell:
                circuit_energy[x,timestamp] -= loss_per_cell
                cells[0,discharging_cell] -= loss_per_cell
                n += 1
            else :
                circuit_energy[x,timestamp] -= cells[0,discharging_cell]
                cells[0,discharging_cell] = 0
                n += 1
        x += 1
    count = 0
    while count < number_cells:
        min_charge = 0.2 * cell_capacity_kwh
        if cells[0,count] <= min_charge:
            time_below_twenty[timestamp,count] = 1
        cell_level[timestamp,count] = cells[0,count]
        count += 1
                
        x += 1
    while in_energy[timestamp] > 0 and np.sum(cells) != cells_full:
        total_current_in = in_power[timestamp]/voltage
        if in_power[timestamp] == 0:
            no_ch_cells = 0
            gain_per_cell = 0
        else:
            no_ch_cells = math.ceil(total_current_in /max_ch_current_cell)
            gain_per_cell = in_energy[timestamp]/no_ch_cells
        
        indices_ch = np.argpartition(cells,no_ch_cells)[:no_ch_cells]
        n = 0
        while n < no_ch_cells:
            charging_cell = indices_ch[0,n]
            ch_current_array[timestamp, charging_cell] = (in_power[timestamp]/voltage) / no_ch_cells
            remainder = max_charge - cells[0,charging_cell]
            if gain_per_cell >= remainder:
                in_energy[timestamp] -= remainder
                cells[0,charging_cell] = max_charge
                n += 1
            else: 
                in_energy[timestamp] -= gain_per_cell
                cells[0,charging_cell] += gain_per_cell
                n += 1                
    timestamp += 1
count = 0
x = 0
while count < number_cells:
    plt.figure(x)
    plt.plot(total_timestamp, cell_level[:,count])
    plt.xlabel("Time of Day")
    plt.ylabel("Charge in Cell (kWh)")
    plt.title("A plot of cell charge over the course of a day")
    count += 1
    x += 1
cells_fractional = cell_level / max_charge
x = 0
count = 1
while count < t_size:
    
    if cells_fractional[count, 0] != cells_fractional[(count-1), 0]:
        difference = abs(cells_fractional[count, 0] - cells_fractional[(count-1), 0])
        x += difference /2
    count +=1


plt.plot(total_timestamp/10, dis_current_array[:,0])
plt.plot(total_timestamp/10, dis_current_array[:,1])
plt.plot(total_timestamp/10, dis_current_array[:,2])
max_dis_array = np.ones(len(total_timestamp)) * max_dis_current_cell
plt.plot(total_timestamp/10, max_dis_array)
plt.title("A plot of discharge current vs time for the smart simulation")
plt.xlabel("Time (Hours)")
plt.ylabel("Current (A)")

plt.figure()
plt.plot(total_timestamp/10, ch_current_array[:,0])
plt.plot(total_timestamp/10, ch_current_array[:,1])
plt.plot(total_timestamp/10, ch_current_array[:,2])
max_ch_array = np.ones(len(total_timestamp)) 
plt.plot(total_timestamp/10, max_ch_array)
plt.title("A plot of charging current vs time for the smart simulation")
plt.xlabel("Time (Hours)")
plt.ylabel("Current (A)")

nocells = np.array([1,2,3])
count = 0
while count < number_cells:
    time = np.sum(time_below_twenty[:,count])
    print(time/10)
    count += 1
    plt.figure(4)
    plt.bar(nocells[count - 1], (time))
    plt.xlabel("Cell Number")
    plt.ylabel("Number of Time intervals below minimum charge")
    plt.xticks([1,2,3])
    plt.title("The number of time intervals spent by each cell below minimum charge")

#%% Dumb Battery System

t = np.linspace(0,23.9,24)
number_cells = 3
cells = np.zeros((1,number_cells))
cells_ones = np.ones((1,number_cells))
cells_full = np.sum(cells_ones * max_charge)
cells = cells_ones*max_charge
timestamp = 0
t_size = 241

time_at_zero = np.zeros([t_size, number_cells])
cell_level = np.zeros([t_size, number_cells])
time_below_twenty = np.zeros([t_size, number_cells])

sigma = 2
mu = 12
mu_2 = -12

sol = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (t - mu)**2 / (2 * sigma**2) ) 
sol_2 = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (t - mu_2)**2 / (2 * sigma**2) ) 
sol = sol + sol_2
sol_total = sum(sol) #sum of normal distribution

#Integral Calculator

solar_panel_surface_area = 10
incident_energy_per_m2 = 2.4 #kWh per day
incident_irradience = incident_energy_per_m2 * solar_panel_surface_area #kWh per day
area_under_curve = np.trapz(sol, t) #kWh
sol_normalised = incident_irradience*sol/area_under_curve #in kW
Solar_eff = 0.2
solar_power_production = Solar_eff * sol_normalised #kW
solar_energy = solar_power_production * 0.1
print(np.sum(solar_energy), "kWh generated from Solar Panels")
single_power= np.zeros([241])
single_energy = np.zeros([241])

k = 0

while k < step_total_161:
    
    t = timestamps_161[k] 
    index = int(10 * t)
    single_power[index] += power_data_161[k,0]
    if energy_data_161[k,0] > energy_data_161[k-1,0] and k > 0:
        single_energy[index] += (energy_data_161[k,0] - energy_data_161[k-1,0])
    k+=1

k = 0
while k < step_total_157:
    
    t = timestamps_157[k] 
    index = int(10 * t)
    single_power[index] += power_data_157[k,0]
    if energy_data_157[k,0] > energy_data_157[k-1,0] and k > 0:
        single_energy[index] += (energy_data_157[k,0] - energy_data_157[k-1,0])
    k+=1
k = 0
while k < step_total_156:
    
    t = timestamps_156[k] 
    index = int(10 * t)
    single_power[index] += power_data_156[k,0]
    if energy_data_156[k,0] > energy_data_156[k-1,0] and k > 0:
        single_energy[index] += (energy_data_156[k,0] - energy_data_156[k-1,0])
    k+=1
k = 0
while k < step_total_154:
    
    t = timestamps_154[k] 
    index = int(10 * t)
    single_power[index] += power_data_154[k,0]
    if energy_data_154[k,0] > energy_data_154[k-1,0] and k > 0:
        single_energy[index] += (energy_data_154[k,0] - energy_data_154[k-1,0])
    k+=1
k = 0
while k < step_total_153:
    
    t = timestamps_153[k] 
    index = int(10 * t)
    single_power[index] += power_data_153[k,0]
    if energy_data_153[k,0] > energy_data_153[k-1,0] and k > 0:
        single_energy[index] += (energy_data_153[k,0] - energy_data_153[k-1,0])
    k+=1
k = 0
while k < step_total_151:
    
    t = timestamps_151[k] 
    index = int(10 * t)
    single_power[index] += power_data_151[k,0]
    if energy_data_151[k,0] > energy_data_151[k-1,0] and k > 0:
        single_energy[index] += (energy_data_151[k,0] - energy_data_151[k-1,0])
    k+=1
k = 0
while k < step_total_150:
    
    t = timestamps_150[k] 
    index = int(10 * t)
    single_power[index] += power_data_150[k,0]
    if energy_data_150[k,0] > energy_data_150[k-1,0] and k > 0:
        single_energy[index] += (energy_data_150[k,0] - energy_data_150[k-1,0])
    k+=1

dis_current_array = np.zeros([t_size, number_cells])
ch_current_array = np.zeros([t_size, number_cells])


while timestamp < t_size:
    
    demand = single_energy[timestamp]
    current = single_power[timestamp] / (voltage)
    
    while demand > 0 and np.sum(cells) != 0:
        discharging_cell = np.argmax(cells)
        dis_current_array[timestamp, discharging_cell] = current
        
        if demand > cells[0,discharging_cell]:
            single_energy[timestamp] -= cells[0,discharging_cell] 
            cells[0,discharging_cell] = 0
        else:
            cells[0,discharging_cell] -= demand
            single_energy[timestamp] = 0
    count = 0
    while count < number_cells:
        min_charge = 0.2 * cell_capacity_kwh
        if cells[0,count] <= min_charge:
            time_below_twenty[timestamp,count] = 1
        cell_level[timestamp,count] = cells[0,count]
        count += 1
                
    while in_energy[timestamp] > 0 and np.sum(cells) != cells_full :
        current = in_power[timestamp]  / (voltage)
        ch_current_array[timestamp, charging_cell] = current
        charging_cell = np.argmin(cells)
        remainder = max_charge - cells[0,charging_cell]
        if remainder > in_energy[timestamp]:
            cells[0,charging_cell] += in_energy[timestamp]
            in_energy[timestamp] = 0
        else:
            cells[0,charging_cell] = max_charge
            in_energy[timestamp] -= remainder
    timestamp += 1
count = 0
x = 0
while count < number_cells:
    plt.figure(x+5)
    plt.plot(total_timestamp, cell_level[:,count], color = "r")
    plt.xlabel("Time of Day")
    plt.ylabel("Charge in Cell (kWh)")
    plt.title("A plot of cell charge over the course of a day")
    count += 1
    x += 1

nocells = np.array([1,2,3])
count = 0
cells_fractional = cell_level / max_charge
x = 0
count = 1
while count < t_size:
    
    if cells_fractional[count, 0] != cells_fractional[(count-1), 0]:
        difference = abs(cells_fractional[count, 0] - cells_fractional[(count-1), 0])
        x += difference /2
    count +=1
print(x)
count = 0

plt.plot(total_timestamp/10, dis_current_array[:,0],color = "b")
plt.plot(total_timestamp/10, dis_current_array[:,1],color = "orange")
plt.plot(total_timestamp/10, dis_current_array[:,2],color = "g")
max_dis_array = np.ones(len(total_timestamp)) * max_dis_current_cell
plt.plot(total_timestamp/10, max_dis_array, color ="r")
plt.title("A plot of discharge current vs time for the dumb simulation")
plt.xlabel("Time (Hours)")
plt.ylabel("Current (A)")

plt.figure()
plt.plot(total_timestamp/10, ch_current_array[:,0])
plt.plot(total_timestamp/10, ch_current_array[:,1])
plt.plot(total_timestamp/10, ch_current_array[:,2])
max_ch_array = np.ones(len(total_timestamp))
plt.plot(total_timestamp/10, max_ch_array)
plt.title("A plot of charging current vs time for the dumb simulation")
plt.xlabel("Time (Hours)")
plt.ylabel("Current (A)")

while count < number_cells:
    time = np.sum(time_below_twenty[:,count])
    print(time/10)
    count += 1
    plt.figure(4)
    plt.bar(nocells[count - 1], (time/10))
    plt.xlabel("Cell Number")
    plt.ylabel("Number of Time intervals below minimum charge")
    plt.xticks([1,2,3])
    plt.title("The number of time intervals spent by each cell below minimum charge")


    
