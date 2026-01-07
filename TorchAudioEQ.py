import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from typing import Optional, Tuple

class TorchAudioEQProcessor:
    def __init__(self, sample_rate: int = 44100, device: str = 'cpu', 
                 dtype: torch.dtype = torch.float32):
        """
        初始化TorchAudio EQ处理器
        
        参数:
            sample_rate: 采样率 (默认44100)
            device: 计算设备 (默认'cpu')
            dtype: 数据类型 (默认torch.float32)
        """
        self.sample_rate = sample_rate
        self.device = device
        self.dtype = dtype
        
    def _design_eq_filter(self, center_freq, Q, gain_db):
        """
        手动计算EQ滤波器频率响应,二阶峰值滤波器
        
        参数:
            center_freq: 中心频率 (Hz)
            Q: Q值 (带宽)
            gain_db: 增益 (dB)
            
        返回:
            h: 频域滤波器响应
        """
        device = self.device
        dtype = self.dtype
        
        center_freq_t = torch.tensor(center_freq, device=device, dtype=dtype)
        Q_t = torch.tensor(Q, device=device, dtype=dtype)
        gain_db_t = torch.tensor(gain_db, device=device, dtype=dtype)
        
        w0 = 2 * torch.pi * center_freq_t / self.sample_rate
        
        #带宽参数
        alpha = torch.sin(w0) / (2 * Q_t)
        
        # 线性增益
        A = torch.pow(10.0, gain_db_t / 40.0) 
        
        b0 = 1 + alpha * A
        b1 = -2 * torch.cos(w0)
        b2 = 1 - alpha * A

        a0 = 1 + alpha / A
        a1 = -2 * torch.cos(w0)
        a2 = 1 - alpha / A

        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        
        return b0, b1, b2, a1, a2
    
    def apply_eq(self, audio: torch.Tensor, center_freq: float, Q: float, gain_db: float) -> torch.Tensor:
        """
        应用EQ到音频
        
        参数:
            audio: 输入音频张量 [batch, samples]
            center_freq: 中心频率 (Hz)
            Q: Q值
            gain_db: 增益 (dB)
            
        返回:
            processed_audio: 处理后的音频
        """
        b0, b1, b2, a1, a2 = self._design_eq_filter(center_freq, Q, gain_db)
        
        #a0之前归一化没了
        a0 = torch.tensor(1.0, device=self.device, dtype=self.dtype)

        b_coeffs = torch.stack([b0, b1, b2])
        a_coeffs = torch.stack([a0, a1, a2])
        
        # 使用torchaudio的lfilter
        output = F.lfilter(audio, a_coeffs, b_coeffs, clamp=False)
        return output
    
if __name__ == "__main__":
    eq_processor = TorchAudioEQProcessor(sample_rate=44100)

    input_path = "input_audio.wav"
    output_path = "output_audioTA.wav"

    center_freq = 3000
    Q = 4
    gain_db = -17
    
    print(f"应用EQ: 中心频率={center_freq}Hz, Q={Q},增益={gain_db}dB")

    #此处禁用梯度计算，如果用于可微数据增强请启用
    with torch.no_grad(): 
        audio, sr = torchaudio.load(input_path)
        if sr != eq_processor.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, eq_processor.sample_rate)
            audio = resampler(audio)
        
        processed_audio = eq_processor.apply_eq(audio, center_freq, Q, gain_db)
    torchaudio.save(output_path, processed_audio, eq_processor.sample_rate)
    
    
    print("音频处理完成,已保存到output_audio.wav")