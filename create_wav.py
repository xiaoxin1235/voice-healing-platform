import numpy as np
import soundfile as sf

sr = 16000
t = np.linspace(0, 5, sr*5)
wave = 0.3*np.sin(2*np.pi*180*t)
sf.write("test_standard.wav", wave, sr)
print("音频生成完成")