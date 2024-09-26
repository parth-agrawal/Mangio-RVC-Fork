import os, sys
import wave
import soundfile as sf
import numpy as np
import torch
import time
import librosa
import sounddevice as sd
import noisereduce as nr
import torch.nn.functional as F
import torchaudio.transforms as tat
from multiprocessing import Queue, cpu_count
from rvc_for_realtime import RVC

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)
import multiprocessing

class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np, pyworld
        while 1:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)

class AudioProcessor:
    def __init__(self):
        self.config = self.setup_config()
        self.flag_vc = False
        self.output_audio_data = []
        self.output_wav_file = None

    def setup_config(self):
        config = type('Config', (), {})()
        config.pth_path = "weights/BarackObama.pth"
        config.index_path = "logs/obama.index"
        config.f0method = "crepe"
        config.pitch = 0
        config.index_rate = 0.5
        config.block_time = 1.0
        config.crossfade_time = 0.04
        config.extra_time = 1.0
        config.n_cpu = min(cpu_count(), 8)
        config.samplerate = 40000
        config.threhold = -30
        config.I_noise_reduce = False
        config.O_noise_reduce = False
        return config

    def start_vc(self, input_file):
        torch.cuda.empty_cache()
        self.flag_vc = True
        self.rvc = RVC(
            self.config.pitch,
            self.config.pth_path,
            self.config.index_path,
            self.config.index_rate,
            self.config.n_cpu,
            inp_q,
            opt_q,
            device,
        )
        self.config.samplerate = self.rvc.tgt_sr
        self.config.crossfade_time = min(
            self.config.crossfade_time, self.config.block_time
        )
        self.block_frame = int(self.config.block_time * self.config.samplerate)
        self.crossfade_frame = int(
            self.config.crossfade_time * self.config.samplerate
        )
        self.sola_search_frame = int(0.01 * self.config.samplerate)
        self.extra_frame = int(self.config.extra_time * self.config.samplerate)
        self.zc = self.rvc.tgt_sr // 100
        self.input_wav = np.zeros(
            int(
                np.ceil(
                    (
                        self.extra_frame
                        + self.crossfade_frame
                        + self.sola_search_frame
                        + self.block_frame
                    )
                    / self.zc
                )
                * self.zc
            ),
            dtype="float32",
        )
        self.output_wav_cache = torch.zeros(
            int(
                np.ceil(
                    (
                        self.extra_frame
                        + self.crossfade_frame
                        + self.sola_search_frame
                        + self.block_frame
                    )
                    / self.zc
                )
                * self.zc
            ),
            device=device,
            dtype=torch.float32,
        )
        self.pitch = np.zeros(
            self.input_wav.shape[0] // self.zc,
            dtype="int32",
        )
        self.pitchf = np.zeros(
            self.input_wav.shape[0] // self.zc,
            dtype="float64",
        )
        self.output_wav = torch.zeros(
            self.block_frame, device=device, dtype=torch.float32
        )
        self.sola_buffer = torch.zeros(
            self.crossfade_frame, device=device, dtype=torch.float32
        )
        self.fade_in_window = torch.linspace(
            0.0, 1.0, steps=self.crossfade_frame, device=device, dtype=torch.float32
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.config.samplerate, new_freq=16000, dtype=torch.float32
        ).to(device)
        
        self.output_wav_file = wave.open("output2.wav", "wb")
        self.output_wav_file.setnchannels(2)
        self.output_wav_file.setsampwidth(2)
        self.output_wav_file.setframerate(self.config.samplerate)
        
        audio_generator = self.stream_from_file(input_file)
        self.process_audio(audio_generator)

    def stream_from_file(self, file_path):
        with sf.SoundFile(file_path) as sound_file:
            print(f"Input file: {file_path}, duration: {sound_file.frames / sound_file.samplerate:.2f} seconds")
            if sound_file.samplerate != self.config.samplerate:
                resampler = tat.Resample(sound_file.samplerate, self.config.samplerate)
            
            while True:
                audio_block = sound_file.read(self.block_frame)
                if len(audio_block) == 0:
                    print("Reached end of input file")
                    break
                print(f"Read audio block of length {len(audio_block)}")

                if sound_file.samplerate != self.config.samplerate:
                    audio_block = resampler(torch.from_numpy(audio_block).float()).numpy()
                
                if sound_file.channels > 1:
                    audio_block = np.mean(audio_block, axis=1)
                
                if len(audio_block) < self.block_frame:
                    audio_block = np.pad(audio_block, (0, self.block_frame - len(audio_block)))
                elif len(audio_block) > self.block_frame:
                    audio_block = audio_block[:self.block_frame]
                
                audio_block = np.column_stack((audio_block, audio_block))
                
                yield audio_block.astype(np.float64)

    def process_audio(self, audio_generator):
        while self.flag_vc:
            try:
                indata = next(audio_generator)
                outdata = np.zeros((self.block_frame, 2), dtype=np.float32)
                self.audio_callback(indata, outdata, self.block_frame, None, None)
                time.sleep(self.config.block_time)
                print("Audio block processed.")
            except StopIteration:
                print("Reached end of input file")
                break
        print("Ending VC")
        self.stop_vc()

    def audio_callback(self, indata, outdata, frames, times, status):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.config.I_noise_reduce:
            indata[:] = nr.reduce_noise(y=indata, sr=self.config.samplerate)
        
        frame_length = 2048
        hop_length = 1024
        rms = librosa.feature.rms(
            y=indata, frame_length=frame_length, hop_length=hop_length
        )
        if self.config.threhold > -60:
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * hop_length : (i + 1) * hop_length] = 0
        self.input_wav[:] = np.append(self.input_wav[self.block_frame :], indata)
        inp = torch.from_numpy(self.input_wav).to(device)
        res1 = self.resampler(inp)
        rate1 = self.block_frame / (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
        )
        rate2 = (
            self.crossfade_frame + self.sola_search_frame + self.block_frame
        ) / (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
        )
        res2 = self.rvc.infer(
            res1,
            res1[-self.block_frame :].cpu().numpy(),
            rate1,
            rate2,
            self.pitch,
            self.pitchf,
            self.config.f0method,
        )
        self.output_wav_cache[-res2.shape[0] :] = res2
        infer_wav = self.output_wav_cache[
            -self.crossfade_frame - self.sola_search_frame - self.block_frame :
        ]
        cor_nom = F.conv1d(
            infer_wav[None, None, : self.crossfade_frame + self.sola_search_frame],
            self.sola_buffer[None, None, :],
        )
        cor_den = torch.sqrt(
            F.conv1d(
                infer_wav[
                    None, None, : self.crossfade_frame + self.sola_search_frame
                ]
                ** 2,
                torch.ones(1, 1, self.crossfade_frame, device=device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            cor_nom = cor_nom.cpu()
            cor_den = cor_den.cpu()
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        print("sola offset: " + str(int(sola_offset)))
        self.output_wav[:] = infer_wav[sola_offset : sola_offset + self.block_frame]
        self.output_wav[: self.crossfade_frame] *= self.fade_in_window
        self.output_wav[: self.crossfade_frame] += self.sola_buffer[:]
        if sola_offset < self.sola_search_frame:
            self.sola_buffer[:] = (
                infer_wav[
                    -self.sola_search_frame
                    - self.crossfade_frame
                    + sola_offset : -self.sola_search_frame
                    + sola_offset
                ]
                * self.fade_out_window
            )
        else:
            self.sola_buffer[:] = (
                infer_wav[-self.crossfade_frame :] * self.fade_out_window
            )
        if self.config.O_noise_reduce:
            if sys.platform == "darwin":
                noise_reduced_signal = nr.reduce_noise(
                    y=self.output_wav[:].cpu().numpy(), sr=self.config.samplerate
                )
                outdata[:] = noise_reduced_signal[:, np.newaxis]
            else:
                outdata[:] = np.tile(
                    nr.reduce_noise(
                        y=self.output_wav[:].cpu().numpy(),
                        sr=self.config.samplerate,
                    ),
                    (2, 1),
                ).T
        else:
            if sys.platform == "darwin":
                outdata[:] = self.output_wav[:].cpu().numpy()[:, np.newaxis]
            else:
                outdata[:] = self.output_wav[:].repeat(2, 1).t().cpu().numpy()
        
        self.output_audio_data.extend(outdata.flatten())
        print(f"Writing audio block of length {len(outdata)}")

        self.output_wav_file.writeframes((outdata * 32767).astype(np.int16).tobytes())

        total_time = time.perf_counter() - start_time
        print("infer time:" + str(total_time))

    def stop_vc(self):
        self.flag_vc = False
        self.close_output_file()

    def close_output_file(self):
        if self.output_wav_file:
            self.output_wav_file.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_q = Queue()
    opt_q = Queue()
    n_cpu = min(cpu_count(), 8)
    
    for _ in range(n_cpu):
        Harvest(inp_q, opt_q).start()

    processor = AudioProcessor()
    input_file = "/home/paperspace/Mangio-RVC-Fork/audios/gettysburg.wav"
    print("Using CUDA:" + str(torch.cuda.is_available()))
    processor.start_vc(input_file)