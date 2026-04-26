import numpy as np
import soundfile as sf

# 生成5秒的标准测试音频（PCM 16bit，单声道，16000Hz，librosa完美兼容）
sr = 16000
duration = 5  # 时长5秒
t = np.linspace(0, duration, int(sr * duration))
# 生成带噪声的正弦波，模拟人声
audio = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.randn(len(t))
# 保存为标准WAV格式
sf.write("test_standard.wav", audio, sr, subtype='PCM_16')
print("✅ 标准测试音频生成完成！文件名：test_standard.wav")