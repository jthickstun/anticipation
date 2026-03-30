from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from anticipation.v2.config import AnticipationV2Settings
from train.v2.dataset_utils import PreTokenizedDataset
from train.v2.hf_gpt2_rewrite import GPT2LMHeadModelLite


def log_loss(
    model: GPT2LMHeadModelLite,
    dataset: PreTokenizedDataset,
    settings: AnticipationV2Settings,
    subsample_ratio: int = 10,
):
    total_samples = len(dataset)
    ce = torch.empty(0).cpu()
    ce_ticks = torch.empty(0).cpu()
    ce_events = torch.empty(0).cpu()

    for sample_idx in tqdm(range(total_samples)):
        if sample_idx % subsample_ratio != 0:
            continue

        tokens = dataset[sample_idx]["input_ids"]
        tokens = tokens.unsqueeze(0).cuda()
        with torch.no_grad():
            # (s, vocab)
            logits = model(tokens).logits[0]

            # per token nll
            targets = tokens[0, 1:]
            curr_ce = F.cross_entropy(logits[:-1], targets, reduction="none")

            # do not measure for controls, those are given by us
            controls = targets >= settings.vocab.SPECIAL_OFFSET

            # isolate the tick cross entropy
            ticks = targets == settings.vocab.TICK
            tick_ce = curr_ce[ticks]

            # anything that is not a tick and is not a control is a triple
            event_ce = curr_ce[(~ticks) & (~controls)]
            if event_ce.shape[0] % 3 != 0:
                # truncate incomplete remaining triple if needed
                event_ce = event_ce[: -1 * (event_ce.shape[0] % 3)]

            assert event_ce.shape[0] % 3 == 0

            ce = torch.cat([ce, curr_ce.cpu()])  # everything
            ce_events = torch.cat([ce_events, event_ce.cpu()])  # triples
            ce_ticks = torch.cat([ce_ticks, tick_ce.cpu()])  # ticks

    d_rounding = 4
    res = {}
    res["loss"] = np.round(ce.mean().item(), d_rounding)

    # PPL(e)
    res["event_ppl"] = np.round(np.exp(3 * ce_events.mean().item()), d_rounding)

    # consider the total log loss of onsets and ticks, average over number
    # of events
    num_events = ce_events.shape[0] // 3
    # PPL(t)
    res["onset_ppl"] = np.round(
        np.exp((ce_events[0::3].sum() + ce_ticks.sum()).item() / num_events), d_rounding
    )
    res["onset_ppl_no_ticks"] = np.round(
        np.exp((ce_events[0::3].sum()).item() / num_events), d_rounding
    )
    # PPL(d)
    res["dur_ppl"] = np.round(
        np.exp((ce_events[1::3].sum() + ce_ticks.sum()).item() / num_events), d_rounding
    )
    res["dur_ppl_no_ticks"] = np.round(
        np.exp((ce_events[1::3].sum()).item() / num_events), d_rounding
    )
    # PPL(n)
    res["note_ppl"] = np.round(
        np.exp((ce_events[2::3].sum() + ce_ticks.sum()).item() / num_events), d_rounding
    )
    res["note_ppl_no_ticks"] = np.round(
        np.exp((ce_events[2::3].sum()).item() / num_events), d_rounding
    )

    # ticks only
    res["tick_ppl"] = np.round(np.exp(ce_ticks.mean().item()), d_rounding)

    # WARNING: hard-coded lakh dataset test split number of hours
    # test split is f
    res["bps"] = (
        subsample_ratio
        * ce_events.mean().item()
        * np.log2(np.e)
        * (len(ce) / (560.98 * 3600))
    )
    return res


def main() -> None:
    data_dir_giga = "data/tokenized_datasets/giga_midi/6fb2094dfa7c0d16278dfaa4a401e3b8"
    data_dir_lmd = "data/tokenized_datasets/lmd_full/6fb2094dfa7c0d16278dfaa4a401e3b8"
    pretrained_checkpoint_path = "output/slurm_logs/791545/checkpoints/step-40000"
    dataset = PreTokenizedDataset(Path(data_dir_lmd) / "test.npy")
    model = GPT2LMHeadModelLite.from_pretrained(
        pretrained_checkpoint_path,
    ).cuda()
    settings = AnticipationV2Settings.load_from_disk(
        Path(data_dir_lmd) / "settings_6fb2094dfa7c0d16278dfaa4a401e3b8.json"
    )
    res = log_loss(model, dataset, settings)
    pprint(res)


if __name__ == "__main__":
    """

        PYTHONPATH=. python eval/v2/eval_loss.py

    """
    main()
