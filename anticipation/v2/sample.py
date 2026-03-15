import torch
import torch.nn.functional as F

import mido

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.types import Token
from anticipation.v2.convert import compound_to_midi
from anticipation.v2 import ops as v2_ops


def safe_logits(
    logits: torch.Tensor, idx: int, settings: AnticipationV2Settings
) -> torch.Tensor:
    v = settings.vocab
    logits[v.CONTROL_OFFSET : v.SPECIAL_OFFSET] = -float(
        "inf"
    )  # don't generate controls
    logits[v.SPECIAL_OFFSET :] = -float("inf")  # don't generate special tokens

    # forbid the tick token
    logits[v.TICK] = -float("inf")

    # don't generate stuff in the wrong time slot
    if idx % 3 == 0:
        # ensure offset generated
        logits[v.DUR_OFFSET : v.DUR_OFFSET + settings.max_dur] = -float("inf")
        logits[v.NOTE_OFFSET : v.NOTE_OFFSET + settings.max_note] = -float("inf")
    elif idx % 3 == 1:
        # ensure duration generated
        logits[
            v.TIME_OFFSET : v.TIME_OFFSET + settings.tick_token_frequency_in_midi_ticks
        ] = -float("inf")
        logits[v.NOTE_OFFSET : v.NOTE_OFFSET + settings.max_note] = -float("inf")
    elif idx % 3 == 2:
        # ensure note generated
        logits[
            v.TIME_OFFSET : v.TIME_OFFSET + settings.tick_token_frequency_in_midi_ticks
        ] = -float("inf")
        logits[v.DUR_OFFSET : v.DUR_OFFSET + settings.max_dur] = -float("inf")

    return logits


def nucleus(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    # from HF implementation
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("inf")

    return logits


def future_logits(
    logits: torch.Tensor, curtime: int, settings: AnticipationV2Settings
) -> torch.Tensor:
    """don't sample events in the past"""
    if curtime > 0:
        # curtime is the absolute time since inference started in ticks
        rel_position = curtime % settings.tick_token_frequency_in_midi_ticks

        # prevent generating events that should have come before... we sort by time
        # so this scenario won't appear in the training data
        logits[
            settings.vocab.TIME_OFFSET : settings.vocab.TIME_OFFSET + rel_position
        ] = -float("inf")

    return logits


def instr_logits(
    logits: torch.Tensor,
    full_history: list[tuple[Token, ...]],
    settings: AnticipationV2Settings,
) -> torch.Tensor:
    """don't sample more than 16 instruments"""
    # need to remove the ticks
    v1_style_seq = [x for x in full_history if len(x) == 3]
    # then flatten to triples, we can use the v1 logit
    v1_style_seq_unwrapped = [x for b in v1_style_seq for x in b]
    instrs = v2_ops.get_instruments(v1_style_seq_unwrapped, settings)

    if len(instrs) < 15:  # 16 - 1 to account for the reserved drum track
        return logits

    v = settings.vocab
    for instr in range(settings.max_midi_instrument):
        if instr not in instrs:
            logits[
                v.NOTE_OFFSET + instr * settings.max_midi_pitch : v.NOTE_OFFSET
                + (instr + 1) * settings.max_midi_pitch
            ] = -float("inf")

    return logits


def check_probs(probs):
    # just a spot check for debugging
    if not torch.isfinite(probs).all():
        bad = (~torch.isfinite(probs)).nonzero()
        raise RuntimeError(f"Non-finite probs")

    if (probs < 0).any():
        bad = (probs < 0).nonzero()
        raise RuntimeError(f"Negative probs at")

    row_sums = probs.sum(dim=-1)
    if (row_sums == 0).any():
        raise RuntimeError(f"Zero-sum prob row at step")


def add_token(
    model,
    z: list[Token],
    tokens: list[tuple[Token, ...]],
    top_p: float,
    current_time: int,
    settings: AnticipationV2Settings,
) -> tuple[Token, ...]:
    tmp = tokens.copy()
    unwrapped_tokens = [x for b in tmp for x in b]

    # not sure why this is 1017, but keeping it the same as v1
    look_back_n_tokens = 1017
    history = unwrapped_tokens[-1 * look_back_n_tokens :]  # Markov window

    with torch.no_grad():
        input_tokens = torch.tensor(z + history).unsqueeze(0).to(model.device)
        logits = model(input_tokens).logits[0, -1]
        logits = nucleus(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = int(torch.multinomial(probs, 1))
        if token == settings.vocab.TICK:
            # model decided to place a tick
            return (token,)
        else:
            new_token = []
            # model did not place a tick, go forward and generate a triple
            for i in range(3):
                input_tokens = (
                    torch.tensor(z + history + new_token).unsqueeze(0).to(model.device)
                )
                logits = model(input_tokens).logits[0, -1]

                # uses idx % 3 to determine what to mask
                idx = input_tokens.shape[1] - 1
                logits = safe_logits(logits, idx, settings)

                if i == 0:
                    future_logits(logits, current_time, settings)
                elif i == 2:
                    # restrict the total number of instruments
                    logits = instr_logits(logits, tokens, settings)

                logits = nucleus(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                check_probs(probs)
                token = torch.multinomial(probs, 1)
                new_token.append(int(token))

    return tuple(new_token)


def tick_tokens_to_abs_time(
    generated_tokens: list[tuple[Token, ...]], settings: AnticipationV2Settings
) -> list[Token]:
    ticks_seen = 0
    cur_time = 0

    abs_triples = []
    for event in generated_tokens:
        if len(event) == 1:
            # this is a tick, exclude them from the returned sequence
            cur_time = ticks_seen * settings.tick_token_frequency_in_midi_ticks
            ticks_seen += 1
        else:
            # this is a triple
            rel_time, dur, note_instr = event
            rel_time -= settings.vocab.TIME_OFFSET
            cur_time += rel_time
            dur -= settings.vocab.DUR_OFFSET
            note_instr -= settings.vocab.NOTE_OFFSET
            abs_triples.extend([cur_time, dur, note_instr])

    return abs_triples


def generate_ar_simple(
    model,
    settings: AnticipationV2Settings,
    num_events_to_generate: int,
    top_p: float = 1.0,
) -> mido.MidiFile:
    """
    This is the simplest possible inference you can do with the model - supports AR only.

    For now, this is a function we are using just to verify that inference works during
    training / so we can eyeball some samples.
    """
    # conditioning / controls
    z = [settings.vocab.AUTOREGRESS]

    # interleave the controls with the events
    tokens = []
    current_time = 0
    ticks_seen = 0
    for i in range(num_events_to_generate):
        new_event: tuple[Token, ...] = add_token(
            model, z, tokens, top_p, current_time, settings
        )
        if len(new_event) == 1:
            # this is a tick
            current_time = ticks_seen * settings.tick_token_frequency_in_midi_ticks
            ticks_seen += 1
        else:
            # this is a triple
            new_onset, new_dur, new_note_isntr = new_event
            new_token_rel_time = new_onset - settings.vocab.TIME_OFFSET
            current_time += new_token_rel_time
            tokens.append(new_event)

    # need to un-relativize
    tokens = tick_tokens_to_abs_time(tokens, settings)

    # convert from tokens back to 5-tuple
    # TODO: would be better to reorg this function
    assert len(tokens) % 3 == 0
    out = 5 * (len(tokens) // 3) * [0]
    out[0::5] = tokens[0::3]
    out[1::5] = tokens[1::3]
    out[2::5] = [tok - (2**7) * (tok // 2**7) for tok in tokens[2::3]]
    out[3::5] = [tok // 2**7 for tok in tokens[2::3]]
    out[4::5] = (len(tokens) // 3) * [72]  # default velocity
    assert max(out[1::5]) < settings.max_dur
    assert max(out[2::5]) < settings.max_midi_pitch
    assert max(out[3::5]) < settings.max_midi_instrument
    assert all(tok >= 0 for tok in out)

    return compound_to_midi(out, settings)
