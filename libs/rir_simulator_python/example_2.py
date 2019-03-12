import roomsimove_single
#import ipdb
import soundfile as sf
import olafilt
import numpy as np

sim_rir = roomsimove_single.RoomSim.init_from_config_file('room_sensor_config.txt')
source_pos = [1, 1, 1]
rir = sim_rir.create_rir(source_pos)


[data, fs] = sf.read('data.wav',always_2d=True)
data =  data[:,0]
data_rev_ch1 = olafilt.olafilt(rir[:,0], data)
data_rev_ch2 = olafilt.olafilt(rir[:,1], data)
data_rev = np.array([data_rev_ch1, data_rev_ch2])
sf.write('data_rev.wav', data_rev.T, fs)


import matplotlib.pyplot as plt
plt.subplot(3,1,1),plt.plot(data)
plt.subplot(3,1,2),plt.plot(data_rev_ch1)
plt.subplot(3,1,3),plt.plot(data_rev_ch2)
plt.show()

import sounddevice as sd
sd.play(data_rev_ch2, fs)

