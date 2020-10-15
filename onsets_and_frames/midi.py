import multiprocessing

from pretty_midi.pretty_midi import PrettyMIDI
from onsets_and_frames.constants import SAMPLE_RATE
import sys
from pathlib import Path
import soundfile as sf

import mido
import pretty_midi
import numpy as np
import collections
from joblib import Parallel, delayed
from mir_eval.util import hz_to_midi
from tqdm import tqdm


def extract_instruments(mid: PrettyMIDI, instruments = None):
    INSTRUMENTS_TO_RANGE = {
        'bass': range(32, 40),
    }

    if instruments == 'drums':
        return [inst for inst in mid.instruments if inst.is_drum]
    elif instruments == 'all':
        return [inst for inst in mid.instruments if not inst.is_drum]
    elif instruments in INSTRUMENTS_TO_RANGE:
        return [inst for inst in mid.instruments if inst.program in INSTRUMENTS_TO_RANGE[instruments]]
    else:
        raise RuntimeError(f"Unsupported instument {instruments}")


def parse_midi(path, instruments = 'all'):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    mid = pretty_midi.PrettyMIDI(path)
    mid.instruments = extract_instruments(mid, instruments)

    data = []
    notes_out_of_range = 0
    for instrument in mid.instruments:
        for note in instrument.notes:
            if int(note.pitch) in range(21, 109):
                data.append((note.start, note.end, int(note.pitch), int(note.velocity)))
            else:
                notes_out_of_range += 1
    if notes_out_of_range > 0:
        print(f"{notes_out_of_range} notes were out of piano range for file {path}")
    
    data.sort(key=lambda x: x[0])
    return np.array(data)


def synthesize_midi(midi_path: Path, save_path: Path, instruments = None, sample_rate = SAMPLE_RATE, soundfont_path = None):
    mid = pretty_midi.PrettyMIDI(str(midi_path))
    prev_instruments = list(mid.instruments)
    mid.instruments = extract_instruments(mid, instruments)

    if len(mid.instruments) == 0:
        print(f"All instuments in this track are removed! (There were {len(prev_instruments)} instruments)")
         
        # If track is empty the flac file will be corrupt
        # This hack synthesizes the whole midi file and converts the data to a zeros (silent)
        mid.instruments = prev_instruments
        synthesized_np = np.zeros(mid.synthesize(fs=sample_rate).shape)
    else:
        synthesized_np = mid.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(save_path, synthesized_np, sample_rate)


def save_midi(path, pitches, intervals, velocities, midi_program=0, is_drum=False):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=midi_program, is_drum=is_drum)

    # Remove overlapping intervals (end time should be smaller of equal start time of next note on the same pitch)
    intervals_dict = collections.defaultdict(list)
    for i in range(len(pitches)):
        pitch = int(round(hz_to_midi(pitches[i])))
        intervals_dict[pitch].append((intervals[i], i))
    for pitch in intervals_dict:
        interval_list = intervals_dict[pitch]
        interval_list.sort(key=lambda x: x[0][0])
        for i in range(len(interval_list) - 1):
            # assert interval_list[i][1] <= interval_list[i+1][0], f'End time should be smaller of equal start time of next note on the same pitch. It was {interval_list[i][1]}, {interval_list[i+1][0]} for pitch {key}'
            interval_list[i][0][1] = min(interval_list[i][0][1], interval_list[i+1][0][0])

    for pitch in intervals_dict:
        interval_list = intervals_dict[pitch]
        for interval,i in interval_list:
            pitch = int(round(hz_to_midi(pitches[i])))
            velocity = int(127*min(velocities[i], 1))
            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=interval[0], end=interval[1])
            instrument.notes.append(note)

    file.instruments.append(instrument)
    file.write(path)


if __name__ == '__main__':

    def process(input_file, output_file):
        midi_data = parse_midi(input_file)
        np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


    def files():
        for input_file in tqdm(sys.argv[1:]):
            if input_file.endswith('.mid'):
                output_file = input_file[:-4] + '.tsv'
            elif input_file.endswith('.midi'):
                output_file = input_file[:-5] + '.tsv'
            else:
                print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
                continue

            yield (input_file, output_file)

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(in_file, out_file) for in_file, out_file in files())
