import math, io, struct

import torch
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm

from encodec.model import EncodecModel
from encodec import binary


CODE_OFFSET = 0
SCALE_OFFSET = CODE_OFFSET + 1024
SPECIAL_OFFSET = SCALE_OFFSET + 100
RESIDUAL_PAD = SPECIAL_OFFSET + 0
SCALE_PAD = SPECIAL_OFFSET + 1
SEPARATOR = SPECIAL_OFFSET + 2


def read_ecdc(ecdc_file, model, max_frames=None):
    fo = io.BytesIO(ecdc_file.read_bytes())
    metadata = binary.read_ecdc_header(fo)
    model_name = metadata['m']
    audio_length = metadata['al']
    num_codebooks = metadata['nc']
    use_lm = metadata['lm']
    assert isinstance(audio_length, int)
    assert isinstance(num_codebooks, int)
    assert model_name == 'encodec_48khz'

    frames, scales = [], []
    segment_length = model.segment_length or audio_length
    segment_stride = model.segment_stride or audio_length
    count = 0
    for offset in range(0, audio_length, segment_stride):
        this_segment_length = min(audio_length - offset, segment_length)
        frame_length = int(math.ceil(this_segment_length * model.frame_rate / model.sample_rate))
        if frame_length == 1:
            break # stub at the end of a song
        if model.normalize:
            scale_f, = struct.unpack('!f', binary._read_exactly(fo, struct.calcsize('!f')))
            scale = round(100*scale_f)
        else:
            scale = None
        unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)
        frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long)
        for t in range(frame_length):
            code_list: tp.List[int] = []
            for k in range(num_codebooks):
                code = unpacker.pull()
                if code is None:
                    raise EOFError("The stream ended sooner than expected.")
                code_list.append(code)
            codes = torch.tensor(code_list, dtype=torch.long)
            frame[0, :, t] = codes

        frames.append(frame)
        scales.append(scale)

        count += 1
        if max_frames and count > max_frames:
            break

    return audio_length/model.sample_rate, frames, scales


def tokenize(frames, scales, residuals=4):
    assert residuals > 0

    # truncate unused residuals and add a token region offset
    frames = [frame[0,:residuals] + CODE_OFFSET for frame in frames]

    # represent scales with dummy residuals so that the model can treat everything homogeneously
    scales = [torch.tensor([s + SCALE_OFFSET] + (residuals-1)*[SCALE_PAD]).view(residuals,1) for s in scales]

    # tack the scales onto the front of each (1-second) block of audio codes
    chunks = torch.cat([v for pair in zip(scales, frames) for v in pair], axis=1)

    # MusicGen-style interleaving
    codes = F.pad(chunks, (0,residuals-1), mode='constant', value=RESIDUAL_PAD)
    codes = torch.stack([torch.roll(codes[i], i) for i in range(residuals)])

    # flatten the codes into a sequence
    return codes.T.flatten().tolist()


def detokenize(codes, residuals=4):
    assert residuals > 0

    # unroll the MusicGen interleaving
    codes = torch.tensor(codes).reshape(-1, residuals).T
    codes = torch.stack([torch.roll(codes[i], -i) for i in range(residuals)])[:,:-(residuals-1)]

    # split up the codes into (1-second) blocks
    chunks = [codes[:, i:i+151].unsqueeze(0) for i in range(0, codes.shape[1], 151)]

    # split up the scales and frames
    scales = [int(chunk[0,0,0]) - SCALE_OFFSET for chunk in chunks]
    frames = [chunk[:,:,1:] - CODE_OFFSET for chunk in chunks]

    return frames, scales


def pack_tokens(ecdcs, output, idx):
    model = EncodecModel.encodec_model_48khz()

    files = bad_files = seqcount = 0
    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for ecdc in tqdm(ecdcs, desc=f'#{idx}', position=idx+1, leave=True):
            try:
                audio_length, frames, scales = read_ecdc(Path(ecdc), model)
            except Exception as e:
                bad_files += 1
                continue

            tokens = tokenize(frames, scales) 
            tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR, SEPARATOR]
            concatenated_tokens.extend(tokens)    
            files += 1

            # write out full sequences to file
            while len(concatenated_tokens) >= 8192:
                seq = concatenated_tokens[0:8192]
                concatenated_tokens = concatenated_tokens[8192:]

                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    return (files, bad_files, seqcount)
