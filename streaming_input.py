import os, sys
import wave
import soundfile as sf

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


if __name__ == "__main__":
    from multiprocessing import Queue
    from queue import Empty
    import numpy as np
    import multiprocessing
    import traceback, re
    import json
    import PySimpleGUI as sg
    import sounddevice as sd
    import noisereduce as nr
    from multiprocessing import cpu_count
    import librosa, torch, time, threading
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    from i18n import I18nAuto

    i18n = I18nAuto()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    current_dir = os.getcwd()
    inp_q = Queue()
    opt_q = Queue()
    n_cpu = min(cpu_count(), 8)
    for _ in range(n_cpu):
        Harvest(inp_q, opt_q).start()
    from rvc_for_realtime import RVC

    class GUIConfig:
        def __init__(self) -> None:
            self.pth_path: str = ""
            self.index_path: str = ""
            self.pitch: int = 12
            self.samplerate: int = 40000
            self.block_time: float = 1.0  # s
            self.buffer_num: int = 1
            self.threhold: int = -30
            self.crossfade_time: float = 0.08
            self.extra_time: float = 0.04
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.index_rate = 0.3
            self.n_cpu = min(n_cpu, 8)
            self.f0method = "harvest"
            

    class GUI:
        def __init__(self) -> None:
            self.config = GUIConfig()
            self.flag_vc = False
            self.output_audio_data = []
            self.output_wav_file = None

            self.launcher()

        def load(self):
            input_devices, output_devices, _, _ = self.get_devices()
            try:
                with open("values1.json", "r") as j:
                    data = json.load(j)
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
            except:
                with open("values1.json", "w") as j:
                    data = {
                        "pth_path": " ",
                        "index_path": " ",
                        "sg_input_device": input_devices[sd.default.device[0]],
                        "sg_output_device": output_devices[sd.default.device[1]],
                        "threhold": "-45",
                        "pitch": "0",
                        "index_rate": "0",
                        "block_time": "1",
                        "crossfade_length": "0.04",
                        "extra_time": "1",
                        "f0method": "rmvpe",
                    }
            return data

        def launcher(self):
            data = self.load()
            sg.theme("LightBlue3")
            input_devices, output_devices, _, _ = self.get_devices()
            layout = [
                [
                    sg.Frame(
                        title=i18n("加载模型"),
                        layout=[
                            [
                                sg.Input(
                                    default_text=data.get("pth_path", ""),
                                    key="pth_path",
                                ),
                                sg.FileBrowse(
                                    i18n("选择.pth文件"),
                                    initial_folder=os.path.join(os.getcwd(), "weights"),
                                    file_types=((". pth"),),
                                ),
                            ],
                            [
                                sg.Input(
                                    default_text=data.get("index_path", ""),
                                    key="index_path",
                                ),
                                sg.FileBrowse(
                                    i18n("选择.index文件"),
                                    initial_folder=os.path.join(os.getcwd(), "logs"),
                                    file_types=((". index"),),
                                ),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("输入设备")),
                                sg.Combo(
                                    input_devices,
                                    key="sg_input_device",
                                    default_value=data.get("sg_input_device", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("输出设备")),
                                sg.Combo(
                                    output_devices,
                                    key="sg_output_device",
                                    default_value=data.get("sg_output_device", ""),
                                ),
                            ],
                        ],
                        title=i18n("音频设备(请使用同种类驱动)"),
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("响应阈值")),
                                sg.Slider(
                                    range=(-60, 0),
                                    key="threhold",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("threhold", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("音调设置")),
                                sg.Slider(
                                    range=(-24, 24),
                                    key="pitch",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("pitch", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("Index Rate")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="index_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("index_rate", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("音高算法")),
                                sg.Radio(
                                    "pm",
                                    "f0method",
                                    key="pm",
                                    default=data.get("pm", "") == True,
                                ),
                                sg.Radio(
                                    "harvest",
                                    "f0method",
                                    key="harvest",
                                    default=data.get("harvest", "") == True,
                                ),
                                sg.Radio(
                                    "crepe",
                                    "f0method",
                                    key="crepe",
                                    default=data.get("crepe", "") == True,
                                ),
                                sg.Radio(
                                    "rmvpe",
                                    "f0method",
                                    key="rmvpe",
                                    default=data.get("rmvpe", "") == True,
                                ),
                            ],
                        ],
                        title=i18n("常规设置"),
                    ),
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("采样长度")),
                                sg.Slider(
                                    range=(0.12, 2.4),
                                    key="block_time",
                                    resolution=0.03,
                                    orientation="h",
                                    default_value=data.get("block_time", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("harvest进程数")),
                                sg.Slider(
                                    range=(1, n_cpu),
                                    key="n_cpu",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get(
                                        "n_cpu", min(self.config.n_cpu, n_cpu)
                                    ),
                                ),
                            ],
                            [
                                sg.Text(i18n("淡入淡出长度")),
                                sg.Slider(
                                    range=(0.01, 0.15),
                                    key="crossfade_length",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("crossfade_length", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("额外推理时长")),
                                sg.Slider(
                                    range=(0.05, 3.00),
                                    key="extra_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("extra_time", ""),
                                ),
                            ],
                            [
                                sg.Checkbox(i18n("输入降噪"), key="I_noise_reduce"),
                                sg.Checkbox(i18n("输出降噪"), key="O_noise_reduce"),
                            ],
                        ],
                        title=i18n("性能设置"),
                    ),
                ],
                [
                    sg.Button(i18n("开始音频转换"), key="start_vc"),
                    sg.Button(i18n("停止音频转换"), key="stop_vc"),
                    sg.Text(i18n("推理时间(ms):")),
                    sg.Text("0", key="infer_time"),
                ],
            ]
            self.window = sg.Window("RVC - GUI", layout=layout)
            self.event_handler()

        def setup_config(self):
            self.config.pth_path = "weights/BarackObama.pth"
            self.config.index_path = "logs/obama.index"
            self.config.f0method = "crepe"  # or "pm", "harvest", "crepe"
            self.config.pitch = 0  # adjust as needed
            self.config.index_rate = 0.5  # adjust as needed
            self.config.block_time = 1.0  # adjust as needed
            self.config.crossfade_time = 0.04  # adjust as needed
            self.config.extra_time = 1.0  # adjust as needed
            self.config.n_cpu = 8  # adjust as needed
        def event_handler(self):

            # Automatically start VC with the specified file
            self.setup_config()
            input_file = "/home/paperspace/Mangio-RVC-Fork/audios/gettysburg.wav"
            print("using_cuda:" + str(torch.cuda.is_available()))
            self.start_vc(input_file)
            while self.flag_vc:
                time.sleep(1)  # Check every second if processing is done
            self.stop_vc()



           
        def start_vc(self, input_source=None):
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
            self.input_wav: np.ndarray = np.zeros(
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
            self.output_wav_cache: torch.Tensor = torch.zeros(
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
            self.pitch: np.ndarray = np.zeros(
                self.input_wav.shape[0] // self.zc,
                dtype="int32",
            )
            self.pitchf: np.ndarray = np.zeros(
                self.input_wav.shape[0] // self.zc,
                dtype="float64",
            )
            self.output_wav: torch.Tensor = torch.zeros(
                self.block_frame, device=device, dtype=torch.float32
            )
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.crossfade_frame, device=device, dtype=torch.float32
            )
            self.fade_in_window: torch.Tensor = torch.linspace(
                0.0, 1.0, steps=self.crossfade_frame, device=device, dtype=torch.float32
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.config.samplerate, new_freq=16000, dtype=torch.float32
            ).to(device)
           
            
            
            self.output_wav_file = wave.open("output2.wav", "wb")
            self.output_wav_file.setnchannels(2)
            self.output_wav_file.setsampwidth(2)  # 16-bit audio
            self.output_wav_file.setframerate(self.config.samplerate)
            
            if isinstance(input_source, str) and os.path.isfile(input_source):
                # If input_source is a file path, create a generator
                audio_generator = self.stream_from_file(input_source)
                input_device = None
            else:
                # If it's not a file, it's a device ID or None (for default device)
                audio_generator = None
                input_device = input_source
    
            thread_vc = threading.Thread(target=self.soundinput, args=(audio_generator, input_device))
            thread_vc.start()
        def stream_from_file(self, file_path):
            """
            Create a generator to stream audio from a file
            """
            with sf.SoundFile(file_path) as sound_file:
                # Resample if necessary
                if sound_file.samplerate != self.config.samplerate:
                    resampler = tat.Resample(sound_file.samplerate, self.config.samplerate)
                
                while True:
                    audio_block = sound_file.read(self.block_frame)
                    if len(audio_block) == 0:
                        break
                    
                    # Resample if necessary
                    if sound_file.samplerate != self.config.samplerate:
                        audio_block = resampler(torch.from_numpy(audio_block).float()).numpy()
                    
                    # Convert to mono if stereo
                    if sound_file.channels > 1:
                        audio_block = np.mean(audio_block, axis=1)
                    
                    # Pad or truncate the block to match self.block_frame
                    if len(audio_block) < self.block_frame:
                        audio_block = np.pad(audio_block, (0, self.block_frame - len(audio_block)))
                    elif len(audio_block) > self.block_frame:
                        audio_block = audio_block[:self.block_frame]
                    
                    # Convert mono to stereo
                    audio_block = np.column_stack((audio_block, audio_block))
                    
                    yield audio_block.astype(np.float64)  # Change this to float64
        def stop_vc(self):
            self.flag_vc = False
            self.close_output_file()

        

        def soundinput(self, audio_generator=None, input_device=None):
            """
            接受音频输入
            """
            channels = 1 if sys.platform == "darwin" else 2

            def callback(indata, outdata, frames, time, status):
                if status:
                    print(status)
                if audio_generator is not None:
                    try:
                        indata[:] = next(audio_generator)
                    except StopIteration:
                        raise sd.CallbackStop()
                self.audio_callback(indata, outdata, frames, time, status)


            with sd.Stream(
                channels=channels,
                blocksize=self.block_frame,
                samplerate=self.config.samplerate,
                dtype="float32",
                device=input_device,
                callback=callback,
            ):
                while self.flag_vc:
                    time.sleep(self.config.block_time)
                    print("Audio block passed.")
            print("ENDing VC")

        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            音频处理
            """
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            if self.config.I_noise_reduce:
                indata[:] = nr.reduce_noise(y=indata, sr=self.config.samplerate)
            """noise gate"""
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
            # infer
            inp = torch.from_numpy(self.input_wav).to(device)
            ##0
            res1 = self.resampler(inp)
            ###55%
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
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
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
            # crossfade
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
            
            # Intercept the output audio
            self.output_audio_data.extend(outdata.flatten())
            
            # Write the output to the WAV file
            self.output_wav_file.writeframes((outdata * 32767).astype(np.int16).tobytes())


            total_time = time.perf_counter() - start_time
            self.window["infer_time"].update(int(total_time * 1000))
            print("infer time:" + str(total_time))

        def close_output_file(self):
            if self.output_wav_file:
                self.output_wav_file.close()
        
        def get_devices(self, update: bool = True):
            """获取设备列表"""
            if update:
                sd._terminate()
                sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            input_devices = [
                f"{d['name']} ({d['hostapi_name']})"
                for d in devices
                if d["max_input_channels"] > 0
            ]
            output_devices = [
                f"{d['name']} ({d['hostapi_name']})"
                for d in devices
                if d["max_output_channels"] > 0
            ]
            input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0
            ]
            output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0
            ]
            return (
                input_devices,
                output_devices,
                input_devices_indices,
                output_devices_indices,
            )

        def set_devices(self, input_device, output_device):
            """设置输出设备"""
            (
                input_devices,
                output_devices,
                input_device_indices,
                output_device_indices,
            ) = self.get_devices()
            sd.default.device[0] = input_device_indices[
                input_devices.index(input_device)
            ]
            sd.default.device[1] = output_device_indices[
                output_devices.index(output_device)
            ]
            print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
            print(
                "output device:" + str(sd.default.device[1]) + ":" + str(output_device)
            )

    gui = GUI()
    gui.event_handler()
