import torch
import torchaudio

class STFTBaseAudioEQProcessor:
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512, device='cpu', dtype=torch.float32):
        """
        初始化EQ处理器
        
        参数:
            sample_rate: 采样率 (默认44100)
            n_fft: STFT的FFT大小 (默认2048)
            hop_length: STFT的hop长度 (默认512)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
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
        
        gain_linear = 10 ** (gain_db_t / 20.0)
        
        w0 = 2 * torch.pi * center_freq_t / self.sample_rate
        
        #带宽参数
        alpha = torch.sin(w0) / (2 * Q_t)
        
        A = torch.sqrt(gain_linear)
        
        b0 = 1 + alpha * A
        b1 = -2 * torch.cos(w0)
        b2 = 1 - alpha * A

        a0 = 1 + alpha / A
        a1 = -2 * torch.cos(w0)
        a2 = 1 - alpha / A

        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        
        freq_bins = self.n_fft // 2 + 1
        freqs = torch.linspace(0, self.sample_rate / 2, freq_bins, 
                                    device=device, dtype=dtype)
        
        w = 2 * torch.pi * freqs / self.sample_rate

        e_jw = torch.exp(-1j * w)
        e_jw2 = torch.exp(-2j * w)
        
        #H(w) = (b0 + b1*e^{-jw} + b2*e^{-2jw}) / (1 + a1*e^{-jw} + a2*e^{-2jw})
        n = b0 + b1 * e_jw + b2 * e_jw2
        d = 1 + a1 * e_jw + a2 * e_jw2
        h = n / d
        
        return h
    
    def apply_eq(self, audio, center_freq, Q, gain_db):
        """
        应用EQ到音频批量处理版本
        
        参数:
            audio: 输入音频张量 [batch, samples]
            center_freq: 中心频率 (Hz)
            Q: Q值
            gain_db: 增益 (dB)
            
        返回:
            processed_audio: 处理后的音频
        """
        batch_size, num_samples = audio.shape
        
        filter_response = self._design_eq_filter(center_freq, Q, gain_db)
        filter_response = filter_response.unsqueeze(0).unsqueeze(-1)
        
        stft_result = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        

        # filter_response: [1, freq_bins, 1]
        # stft_result: [batch, freq_bins, time_frames]
        stft_result_eq = stft_result * filter_response
        
        processed_audio = torch.istft(
            stft_result_eq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=num_samples
        )
        
        return processed_audio
    






if __name__ == "__main__":
    eq_processor = STFTBaseAudioEQProcessor(sample_rate=44100)

    input_path = "input_audio.wav"
    output_path = "output_audio.wav"

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
    
