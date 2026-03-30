"""
These are operations that use the settings object instead of using global config.
Their functionality is the same as v1 ops unless stated otherwise.
"""

from collections import defaultdict
from itertools import chain
from typing import Optional, Union, Iterator, Iterable, TypeVar

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.types import Token


T = TypeVar("T")


def min_time(
    tokens: list[Token],
    settings: AnticipationV2Settings,
    seconds: bool = True,
    instr: Optional[int] = None,
) -> Union[int, float]:
    mt = None
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # stop calculating at sequence separator
        if note == settings.vocab.SEPARATOR:
            break

        if note < settings.vocab.CONTROL_OFFSET:
            time -= settings.vocab.TIME_OFFSET
            note -= settings.vocab.NOTE_OFFSET
        else:
            time -= settings.vocab.ATIME_OFFSET
            note -= settings.vocab.ANOTE_OFFSET

        # min time of a particular instrument
        if instr is not None and instr != note // 2**7:
            continue

        mt = time if mt is None else min(mt, time)

    if mt is None:
        mt = 0
    return mt / float(settings.time_resolution) if seconds else mt


def max_time(
    tokens: list[Token],
    settings: AnticipationV2Settings,
    seconds: bool = True,
    instr: Optional[int] = None,
) -> Union[int, float]:
    mt = 0
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # keep checking for max_time, even if it appears after a separator
        # (this is important because we use this check for vocab overflow in tokenization)
        if note == settings.vocab.SEPARATOR:
            continue

        if note < settings.vocab.CONTROL_OFFSET:
            time -= settings.vocab.TIME_OFFSET
            note -= settings.vocab.NOTE_OFFSET
        else:
            time -= settings.vocab.ATIME_OFFSET
            note -= settings.vocab.ANOTE_OFFSET

        # max time of a particular instrument
        if instr is not None and instr != note // 2**7:
            continue

        mt = max(mt, time)

    return mt / float(settings.time_resolution) if seconds else mt


def get_instruments(
    tokens: list[Token], settings: AnticipationV2Settings
) -> dict[int, int]:
    instruments = defaultdict(int)
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note >= settings.vocab.SPECIAL_OFFSET:
            continue

        if note < settings.vocab.CONTROL_OFFSET:
            note -= settings.vocab.NOTE_OFFSET
        else:
            note -= settings.vocab.ANOTE_OFFSET

        instr = note // 2**7
        instruments[instr] += 1

    # (instrument type, num instances)
    # new: cast to regular dictionary just to be safe
    return dict(instruments)


def translate(
    tokens: list[Token],
    dt: Union[int, float],
    settings: AnticipationV2Settings,
    seconds: bool = False,
) -> list[Token]:
    assert len(tokens) % 3 == 0, "bad length"

    if seconds:
        dt = int(settings.time_resolution * dt)

    # new_tokens = []
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # stop translating after EOT
        if note == settings.vocab.SEPARATOR:
            yield (time, dur, note)
            # new_tokens.extend([time, dur, note])
            dt = 0
            continue

        if note < settings.vocab.CONTROL_OFFSET:
            this_time = time - settings.vocab.TIME_OFFSET
        else:
            this_time = time - settings.vocab.ATIME_OFFSET

        assert 0 <= this_time + dt
        yield (time + dt, dur, note)
        # new_tokens.extend([time + dt, dur, note])

    # return new_tokens


def add_rests(
    tokens: list[Token],
    settings: AnticipationV2Settings,
    start_time_in_ticks: Optional[int] = None,
    end_time_in_ticks: Optional[int] = None,
) -> list[Token]:
    """
    Equivalent to `ops.pad` in v1. We rename this 'add rests' because there is now a
    padding token, which serves a different purpose.
    """
    density = settings.time_resolution
    end_time = settings.vocab.TIME_OFFSET + (
        end_time_in_ticks
        if end_time_in_ticks is not None
        else max_time(tokens, settings, seconds=False)
    )
    previous_time = settings.vocab.TIME_OFFSET + (
        start_time_in_ticks if start_time_in_ticks is not None else 0
    )

    new_tokens = []
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        # must pad before separation, anticipation
        assert note < settings.vocab.CONTROL_OFFSET

        # insert pad tokens to ensure the desired density
        while time > previous_time + density:
            new_tokens.extend(
                [
                    previous_time + density,
                    settings.vocab.DUR_OFFSET,
                    settings.vocab.TICK,
                ]
            )
            previous_time += density

        new_tokens.extend([time, dur, note])
        previous_time = time

    while end_time > previous_time + density:
        new_tokens.extend(
            [
                previous_time + density,
                settings.vocab.DUR_OFFSET,
                settings.vocab.TICK,
            ]
        )
        previous_time += density

    return new_tokens


# --- NEW TO V2 BELOW THIS LINE ---


def streaming_add_ticks(
    events: list[Token], settings: AnticipationV2Settings
) -> Iterator[tuple[Token, ...]]:
    assert len(events) % 3 == 0, "bad length"

    add_every = settings.tick_token_every_n_ticks
    recent_tick = 0

    # original logic: https://github.com/jthickstun/anticipation/blob/6927699c5243fd91d1d252211c29885377d9dda5/train/tokenize-new.py#L33
    for i in range(0, len(events), 3):
        time, duration, note = events[i : i + 3]

        if add_every > 0:
            while time >= round(recent_tick * add_every):
                # tick - a tick is only a single token!
                yield (settings.vocab.TICK,)
                recent_tick += 1

        yield (
            time,
            duration,
            note,
        )


def streaming_prefix(stream: Iterator[T], prefix: Iterable[T]) -> Iterator[T]:
    return chain(prefix, stream)


def is_triple(
    logical_group: tuple[Token, ...], settings: AnticipationV2Settings
) -> bool:
    if len(logical_group) != 3:
        return False

    # the triple might not be relativized yet, there are some compositions
    # that are very long, so the absolute time might be larger than
    # the maximum token before relativization
    time, dur, note_instr = logical_group
    return (
        dur < settings.vocab.SPECIAL_OFFSET
        and note_instr < settings.vocab.SPECIAL_OFFSET
    )


def is_control_triple(
    logical_group: tuple[Token, ...], settings: AnticipationV2Settings
) -> bool:
    return is_triple(logical_group, settings) and (
        logical_group[0] >= settings.vocab.ATIME_OFFSET
        and logical_group[1] >= settings.vocab.ADUR_OFFSET
        and logical_group[2] >= settings.vocab.ANOTE_OFFSET
    )


def streaming_relativize_to_tick(
    token_stream_iterator: Iterator[tuple[Token, ...]],
    settings: AnticipationV2Settings,
    start_from_tick: int = -1,
) -> Iterator[tuple[Token, ...]]:
    if isinstance(token_stream_iterator, list):
        raise TypeError(
            "Streaming relativize must take iterators, not lists. The token_stream_iterator argument is a list."
        )

    add_every = settings.tick_token_every_n_ticks
    delta_in_ticks = settings.delta * settings.time_resolution

    recent_tick = start_from_tick
    for next_element in token_stream_iterator:
        if next_element == (settings.vocab.TICK,):
            # this is a tick
            recent_tick += 1
            to_add = next_element
        elif is_triple(next_element, settings):
            relativize = max(round(recent_tick * add_every), 0)
            is_control = is_control_triple(next_element, settings)
            if is_control:
                control_abs_time = next_element[0] - settings.vocab.CONTROL_OFFSET
                if control_abs_time < delta_in_ticks:
                    # can't move backwards
                    continue
                else:
                    relativize += delta_in_ticks

            to_add = (
                next_element[0] - relativize,
                next_element[1],
                next_element[2],
            )

            if is_control:
                # don't let the time be over-subtracted
                assert to_add[0] >= settings.vocab.CONTROL_OFFSET, (
                    f"!({to_add[0]} >= {settings.vocab.CONTROL_OFFSET})"
                )
                # don't let time be under-subtracted
                assert to_add[0] <= settings.vocab.ADUR_OFFSET, (
                    f"!({to_add[0]} <= {settings.vocab.ADUR_OFFSET})"
                )
            else:
                # don't let the time be over-subtracted
                assert to_add[0] >= settings.vocab.TIME_OFFSET, (
                    f"!({to_add[0]} >= {settings.vocab.TIME_OFFSET})"
                )
                # don't let time be under-subtracted
                assert to_add[0] <= settings.vocab.DUR_OFFSET, (
                    f"!({to_add[0]} <= {settings.vocab.DUR_OFFSET})"
                )
        else:
            to_add = next_element

        yield to_add


def extract_instruments(
    all_events: list[Token], instruments: list[int], settings: AnticipationV2Settings
) -> tuple[list[Token], list[Token]]:
    # assert len(all_events) % 3 == 0, "bad length"

    events = []
    controls = []
    # for time, dur, note in zip(all_events[0::3], all_events[1::3], all_events[2::3]):
    for x in all_events:
        time, dur, note = x
        assert note < settings.vocab.CONTROL_OFFSET  # shouldn't be in the sequence yet
        instr = (note - settings.vocab.NOTE_OFFSET) // 2**7
        if instr in instruments:
            # mark this event as a control
            controls.extend(
                [
                    settings.vocab.CONTROL_OFFSET + time,
                    settings.vocab.CONTROL_OFFSET + dur,
                    settings.vocab.CONTROL_OFFSET + note,
                ]
            )
        else:
            events.extend([time, dur, note])

    return events, controls


### ---- BLOCK ANTICIPATION ----
def block_anticipation(
    events_and_ticks: Iterable[tuple[Token, ...]],
    controls: list[Token],
    settings: AnticipationV2Settings,
    start_at_ticks_seen: int = 0,
) -> Iterator[tuple[Token, ...]]:
    # need to run anticipation where the controls always appear directly after the tick
    # and those controls condition the sequence for t + delta.

    # events always have ticks within them
    add_every = settings.tick_token_every_n_ticks
    assert add_every > 0

    if controls:
        control_time = controls[0] - settings.vocab.ATIME_OFFSET
    else:
        control_time = float("inf")

    delta = settings.delta * settings.time_resolution
    ticks_seen = start_at_ticks_seen
    for e in events_and_ticks:
        if e == (settings.vocab.TICK,):
            # special behavior when we encounter a tick
            yield e
            tick_time = settings.tick_token_every_n_ticks * ticks_seen
            next_tick_time = tick_time + settings.tick_token_every_n_ticks

            while next_tick_time > control_time - delta:
                next_token_group = tuple(controls[0:3])
                yield next_token_group

                # consume this control and setup for next one
                controls = controls[3:]
                control_time = (
                    controls[0] - settings.vocab.ATIME_OFFSET
                    if len(controls) > 0
                    else float("inf")
                )

            ticks_seen += 1
            continue
        else:
            yield e


def get_truncated_token_groups_from_truncated_flat_token_sequence(
    truncated_end: list[Token], token_groups: list[tuple[Token, ...]]
) -> list[tuple[Token, ...]]:
    truncated_token_groups = []
    truncated_part = list(truncated_end)
    for t in reversed(token_groups):
        if not truncated_part:
            break

        if len(truncated_part) < len(t):
            truncated_token_groups.insert(0, t)
            break

        if truncated_part[-len(t) :] == list(t):
            truncated_part = truncated_part[: -len(t)]
            truncated_token_groups.insert(0, t)

    return truncated_token_groups
