from anticipation.v2.config import Vocab
from anticipation.v2.ops import (
    get_truncated_token_groups_from_truncated_flat_token_sequence,
)


def test_get_truncated_token_groups_from_truncated_flat_token_sequence() -> None:
    token_groups = [(0, 100, 1600), (1, 120, 1600)]
    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        [0, 120], token_groups
    )
    assert to_add_next_seq == [token_groups[-1]]

    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        [120], token_groups
    )
    assert to_add_next_seq == [token_groups[-1]]

    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        [1600], token_groups
    )
    assert to_add_next_seq == [token_groups[-1]]
    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        [120, 1600], token_groups
    )
    assert to_add_next_seq == [token_groups[-1]]

    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        [1, 120, 1600], token_groups
    )
    assert to_add_next_seq == [token_groups[-1]]

    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        [1600, 1, 120, 1600], token_groups
    )
    assert to_add_next_seq == token_groups

    # if no truncation, nothing to add
    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        [], token_groups
    )
    assert to_add_next_seq == []


def test_get_truncated_token_groups_many_ticks(local_midi_vocab: Vocab) -> None:
    # many ticks + control, this actually happens a lot
    truncated_end = [
        local_midi_vocab.ANOTE_OFFSET,
        local_midi_vocab.TICK,
        local_midi_vocab.TICK,
        local_midi_vocab.TICK,
    ]
    token_groups = [
        (
            local_midi_vocab.ATIME_OFFSET,
            local_midi_vocab.ADUR_OFFSET + 1,
            local_midi_vocab.ANOTE_OFFSET,
        ),
        # --- truncation boundary ---
        (
            local_midi_vocab.ATIME_OFFSET + 10,
            local_midi_vocab.ADUR_OFFSET + 1,
            local_midi_vocab.ANOTE_OFFSET,
        ),
        (local_midi_vocab.TICK,),
        (local_midi_vocab.TICK,),
        (local_midi_vocab.TICK,),
    ]
    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        truncated_end, token_groups
    )
    assert to_add_next_seq == token_groups[1:]


def test_get_truncated_token_group_real_example() -> None:
    truncated_end = [9003, 17612]
    token_groups = [
        (87, 119, 10515),
        (17612,),
        (13, 125, 10516),
        (37, 198, 9003),
        (17612,),
    ]
    to_add_next_seq = get_truncated_token_groups_from_truncated_flat_token_sequence(
        truncated_end, token_groups
    )
    assert to_add_next_seq == token_groups[-2:]
