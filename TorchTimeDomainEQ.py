import torch
from typing import Tuple

import torchaudio

class TimeDomainEQProcessor:
    def __init__(self, sample_rate: int = 44100, device: str = 'cpu', 
                 dtype: torch.dtype = torch.float32):
        """
        初始化时域EQ处理器
        
        参数:
            sample_rate: 采样率 (默认44100)
            device: 计算设备 (默认'cpu')
            dtype: 数据类型 (默认torch.float32)
        """
        self.sample_rate = sample_rate
        self.device = device
        self.dtype = dtype
        
    @torch.jit.export
    def _design_eq_filter(self, center_freq: float, Q: float, gain_db: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算双二阶滤波器系数（峰值滤波器）
        
        参数:
            center_freq: 中心频率 (Hz)
            Q: Q值 (带宽)
            gain_db: 增益 (dB)
            
        返回:
            b0, b1, b2, a0, a1, a2: 滤波器系数
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

    
    @torch.jit.export
    def apply_eq(self, audio: torch.Tensor, 
                       center_freq: float, Q: float, gain_db: float) -> torch.Tensor:
        """
        应用单段EQ到音频
        
        参数:
            audio: 输入音频张量 [batch, samples]
            center_freq: 中心频率 (Hz)
            Q: Q值
            gain_db: 增益 (dB)
            
        返回:
            processed_audio: 处理后的音频
        """
        # 计算滤波器系数
        b0, b1, b2, a1, a2 = self._design_eq_filter(center_freq, Q, gain_db)

        batch_size, num_samples = audio.shape
        output = torch.zeros_like(audio)
        
        #滤波器状态
        w1 = torch.zeros(batch_size, dtype=self.dtype, device=self.device)  # 延迟状态1
        w2 = torch.zeros(batch_size, dtype=self.dtype, device=self.device)  # 延迟状态2
        
        # 直接II型实现
        # 差分方程: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        # 使用中间变量w[n] = x[n] - a1*w[n-1] - a2*w[n-2]
        # 输出y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
        
        for n in range(num_samples):
            x_n = audio[:, n]
            
            w_n = x_n - a1 * w1 - a2 * w2
            y_n = b0 * w_n + b1 * w1 + b2 * w2
            
            output[:, n] = y_n
            
            # 更新延迟状态
            w2 = w1.clone()
            w1 = w_n.clone()


        return output



# 使用示例
if __name__ == "__main__":

    eq_instance = TimeDomainEQProcessor(sample_rate=44100, device='cpu')
    eq_processor = torch.jit.script(eq_instance)
    
    input_path = "input_audio.wav"
    output_path = "output_audioD.wav"

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