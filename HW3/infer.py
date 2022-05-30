import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa
import sounddevice as sd
import time
import validators
from torchvision.transforms import ToTensor
from einops import rearrange


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--wav-file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="https://github.com/justineopuls/ObjectDetectionAssignment/releases/download/v1.1.0/kws-best-acc.pt")
    parser.add_argument("--gui", default=False, action="store_true")
    parser.add_argument("--rpi", default=False, action="store_true")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    return args


# main routine
if __name__ == "__main__":
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    args = get_args()

    if validators.url(args.checkpoint):
        checkpoint = args.checkpoint.rsplit('/', 1)[-1]
        # check if checkpoint file exists
        if not os.path.isfile(checkpoint):
            torch.hub.download_url_to_file(args.checkpoint, checkpoint)
    else:
        checkpoint = args.checkpoint

    print("Loading model checkpoint: ", checkpoint)
    scripted_module = torch.jit.load(checkpoint)

    import PySimpleGUI as sg
    sample_rate = 16000
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    sg.theme('DarkAmber')
    
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=args.n_fft,
                                                     win_length=args.win_length,
                                                     hop_length=args.hop_length,
                                                     n_mels=args.n_mels,
                                                     power=2.0)

    layout = [ 
        [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140), key='-OUTPUT-'),],
        [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'),],
        [sg.Text('Speed', expand_x=True, font=("Helvetica", 28), key='-TIME-')],
    ]

    window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True).Finalize()
    window.Maximize()
    window.BringToFront()

    total_runtime = 0
    n_loops = 0
    while True:
        event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        
        waveform = sd.rec(sample_rate).squeeze()
        
        sd.wait()
        if waveform.max() > 1.0:
            continue
        start_time = time.time()

        waveform = torch.from_numpy(waveform).unsqueeze(0)
        mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = rearrange(mel, 'c h (p w) -> p (c h w)', p=16)
        mel = mel.unsqueeze(0)
        pred = scripted_module(mel)
        pred = torch.functional.F.softmax(pred, dim=1)
        max_prob =  pred.max()
        elapsed_time = time.time() - start_time
        total_runtime += elapsed_time
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        if max_prob > args.threshold:
            pred = torch.argmax(pred, dim=1)
            human_label = f"{idx_to_class[pred.item()]}"
            window['-OUTPUT-'].update(human_label)
            window['-OUTPUT-'].update(human_label)
            if human_label == "stop":
                window['-STATUS-'].update("Goodbye!")
                # refresh window
                window.refresh()
                time.sleep(1)
                break
                
        else:
            window['-OUTPUT-'].update("...")
        
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")


    window.close()

            

