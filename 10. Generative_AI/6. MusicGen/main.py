import torchaudio
from transformers import get_scheduler,SchedulerType,AutoConfig, MusicgenForConditionalGeneration,MusicgenProcessor
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
from torch.utils.data import Dataset
import os
import glob
from tqdm import trange
from accelerate import Accelerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train','--train',action='store_true')
parser.add_argument('-datadir', '--datadir')
parser.add_argument('-testdatapath','--testdatapath') 
parser.add_argument('-ckptdir','--ckptdir')
parser.add_argument('-audioprompt','--audioprompt',action='store_true')
args = parser.parse_args()

data_dir = args.datadir
test_data_path = args.testdatapath
ckpt_dir = args.ckptdir
model_id = "facebook/musicgen-small"
lr = 1e-5
epochs =  1
gradient_accumulation_steps = 2
weight_decay = 1e-5
warmup_steps = 10
batch_size = 2
duration = 30
model_config = AutoConfig.from_pretrained(model_id)
audio_encoder_config = model_config.audio_encoder
decoder_config = model_config.decoder
sample_rate = audio_encoder_config.sampling_rate
codebook_size = audio_encoder_config.codebook_size

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = glob.glob(f'{self.data_dir}/*.wav')
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

## Audio Preprocessing
# Preprocess the audio file by 
# 1. resampling the audio file to match the model's sample rate
# 2. convert the audio to monophonic
# 3. random sampling of the audio file to extract different sections of the music
# 4. compress the model to output vectors based on the number of codebooks used and its associated cardinality
def preprocess_audio(audio_path, audio_encoder, sample_rate,duration):
    wav, sr = torchaudio.load(audio_path)
    # resample wav to model's sample rate
    wav = torchaudio.functional.resample(wav, sr, sample_rate)
    # convert to monophonic audio
    wav = wav.mean(dim=0, keepdim=True)
    if wav.shape[1] < sample_rate * duration:
        return None
    # end index for sampling
    end_sample = int(sample_rate * duration)
    # randomize start index for sampling
    start_sample = random.randrange(0, max(wav.shape[1] - end_sample, 1))
    wav = wav[:, start_sample : start_sample + end_sample]

    assert wav.shape[0] == 1 # ensure monophonic audio

    wav = wav.cuda()
    wav = wav.unsqueeze(1)

    with torch.no_grad():
        gen_audio = audio_encoder(wav) # vector quantization (1,num_codebooks,codebook cardinality)

    audio_codes = gen_audio[0]
    audio_values = gen_audio[1]

    return audio_codes

# One hot encode the audio based on the cardinality of the codebooks
def one_hot_encode(tensor, num_classes):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], shape[2],num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                index = tensor[i, j,k].item()
                one_hot[i, j, k,index] = 1

    return one_hot

if args.train:
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    dataset = AudioDataset(data_dir)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    audio_encoder = model.audio_encoder
    decoder = model.decoder
    audio_encoder = audio_encoder.to('cuda') # audio encoder is used for vector quantization of audio and is not trained in this example
    decoder = decoder

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    audio_encoder.eval()
    decoder.train()

    optimizer = AdamW(
        decoder.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )

    scheduler = get_scheduler(
        SchedulerType.COSINE,
        optimizer,
        warmup_steps,
        int(epochs * len(dataloader) / gradient_accumulation_steps),
    )

    # Accelerate handles the distribution of tensors across GPUs
    decoder, optimizer, dataloader, scheduler = accelerator.prepare(decoder, optimizer, dataloader, scheduler)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, audio in enumerate(dataloader):
            with accelerator.accumulate(decoder):
                codes_list = []
                with torch.no_grad():
                    for inner_audio in audio:
                        codes = preprocess_audio(inner_audio, audio_encoder,sample_rate,duration)  # returns tensor
                        if codes is None:
                            continue
                        codes_list.append(codes)
                    if len(codes_list) == 0:
                        continue
                    codes = torch.cat(codes_list, dim=0)
                    
                decoder_output = decoder(codes)
                codes = codes.squeeze()
                logits = decoder_output.logits
                codes = one_hot_encode(codes, codebook_size)
                codes = codes.cuda()
                logits = logits.cuda()
                loss = criterion(logits.view(-1,codebook_size), codes.view(-1,codebook_size))
                    
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(dataloader)}, Loss: {loss.item()}")
    accelerator.wait_for_everyone()
    accelerator.save_model(decoder, ckpt_dir)
else:    
    num_samples = 1
    output_audio_path = 'output.wav'
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)
    ckpt = torch.load(f'{ckpt_dir}/pytorch_model.bin')
    model.decoder.load_state_dict(ckpt)
    model.cuda()
    model.eval()
    
    if args.audioprompt: 
        wav,sr = torchaudio.load(test_data_path)
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.mean(dim=0, keepdim=True).squeeze()
        sample = wav[:len(wav)//4].numpy()
        processor = MusicgenProcessor.from_pretrained(model_id)
        inputs = processor(
            audio=[sample],
            sampling_rate=sample_rate,
            padding=True,
            return_tensors="pt",
        )
        inputs['padding_mask'] = inputs['padding_mask'].cuda()
        inputs['input_values'] = inputs['input_values'].cuda()
        audio_values = model.generate(**inputs,do_sample=True,guidance_scale=3,max_new_tokens=1500)
        audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)
        torchaudio.save(output_audio_path,torch.tensor(audio_values[0]).float(),sample_rate)
    else:
        unconditional_inputs = model.get_unconditional_inputs(num_samples=num_samples,max_new_tokens=1500)
        audio_values = model.generate(**unconditional_inputs,do_sample=True)
        audio = audio_values.cpu()
        torchaudio.save(output_audio_path,audio[0],sample_rate)