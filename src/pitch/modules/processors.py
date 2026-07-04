import os
import numpy as np
import pandas as pd
import yaml

from pitch import save_data, smooth_pitch_curve, interpolate

def varnam_svaras(smoothing_factor, interpolation_gap, logger):
    with open(os.path.join("data", "varnam_tonics.yaml"), "r") as f:
        tonics = yaml.safe_load(f)

    for raga in os.listdir(os.path.join('data', 'varnam_annotations')):
        logger(f"{raga.upper()} (svaras)")
        prec, curr, succ, svaras = [], [], [], []
        labels = ['S', 'R', 'G', 'M', 'D'] if raga=='abhogi' else ['S', 'R', 'G', 'P', 'D'] if raga=='mohanam' else ['S', 'R', 'G', 'M', 'P', 'D', 'N']

        for performer in os.listdir(os.path.join('data', 'varnam_annotations', raga)):
            logger(f"Performer: {performer.replace(f'_{raga}.tsv', '').capitalize()}")

            annotations_path = os.path.join('data', 'varnam_annotations', raga, performer)
            annotations = pd.read_csv(annotations_path, delimiter='\t')

            pitch_track_path = os.path.join('data', 'varnam_pitch_tracks', raga, performer.replace(f'_{raga}', ''))
            pitch_track = pd.read_csv(pitch_track_path, delimiter='\t', names=['time', 'frequency'], header=None)
            time = pitch_track['time'].values
            pitch = pitch_track['frequency'].values
            pitch[pitch == 0] = np.nan

            tonic = float(tonics[performer.split("_")[0]])
            pitch = interpolate(pitch, np.nan, interpolation_gap)
            pitch = smooth_pitch_curve(time, pitch, smoothing_factor=smoothing_factor, min_points=4)
            pitch = 1200 * np.log2(pitch / tonic)

            for i, row in annotations.iterrows():
                logger.pbar(i + 1, len(annotations))

                start_time = float(row["Begin time"].split(":")[-2]) * 60 + float(row["Begin time"].split(":")[-1])
                end_time = float(row["End time"].split(":")[-2]) * 60 + float(row["End time"].split(":")[-1])
                annotation = row["Annotation"][0]

                prec_pitch = pitch[np.where((time > start_time - 0.5) & (time < start_time))[0]]
                curr_pitch = pitch[np.where((time > start_time) & (time < end_time))[0]]
                succ_pitch = pitch[np.where((time > end_time) & (time < end_time + 0.5))[0]]
                prec.append(prec_pitch)
                curr.append(curr_pitch)
                succ.append(succ_pitch)
                svaras.append(labels.index(annotation))

        classes = len(set(svaras))
        class_counts = [svaras.count(i) for i in range(classes)]
        class_distribution = {labels[i]: count for i, count in enumerate(class_counts)}
        logger(f"\tNumer of artists: {len(os.listdir(os.path.join('data', 'varnam_annotations', raga)))}\tNumber of samples: {len(svaras)}\n\tNumber of classes: {classes}\n\tClass distribution:{class_distribution}\n")

        save_data(prec, os.path.join('dataset', raga, 'prec.pkl'))
        save_data(curr, os.path.join('dataset', raga, 'curr.pkl'))
        save_data(succ, os.path.join('dataset', raga, 'succ.pkl'))
        save_data(svaras, os.path.join('dataset', raga, 'svaras.pkl'))
        save_data(labels, os.path.join('dataset', raga, 'labels.pkl'))

def varnam_svara_forms(smoothing_factor, interpolation_gap, logger):
    annotations = pd.read_csv(os.path.join('data', 'varnam_svara_forms.csv'))

    with open(os.path.join("data", "varnam_tonics.yaml"), "r") as f:
        tonics = yaml.safe_load(f)

    prec, curr, succ, svaras, clusters = [], [], [], [], []
    for raga in os.listdir(os.path.join('data', 'varnam_annotations')):
        logger(f"{raga.upper()} (svara-forms)")
        labels = ['S', 'R', 'G', 'M', 'P', 'D', 'N']

        for performer in os.listdir(os.path.join('data', 'varnam_annotations', raga)):
            logger(f"Performer: {performer.replace(f'_{raga}.tsv', '').capitalize()}")

            pitch_track_path = os.path.join('data', 'varnam_pitch_tracks', raga, performer.replace(f'_{raga}', ''))
            pitch_track = pd.read_csv(pitch_track_path, delimiter='\t', names=['time', 'frequency'], header=None)
            time = pitch_track['time'].values
            pitch = pitch_track['frequency'].values
            pitch[pitch == 0] = np.nan

            tonic = float(tonics[performer.split("_")[0]])
            pitch = interpolate(pitch, np.nan, interpolation_gap)
            pitch = smooth_pitch_curve(time, pitch, smoothing_factor=smoothing_factor, min_points=4)
            pitch = 1200 * np.log2(pitch / tonic)

            performance_annotations = annotations[(annotations['raga'] == raga) & (annotations['performer'] == performer.replace(f'_{raga}.tsv', ''))]

            for i, row in performance_annotations.iterrows():
                start_time = float(row["start"])
                end_time = float(row["end"])
                svara = row["svara"][0]
                cluster = row["cluster"]

                prec_pitch = pitch[np.where((time > start_time - 0.5) & (time < start_time))[0]]
                curr_pitch = pitch[np.where((time > start_time) & (time < end_time))[0]]
                succ_pitch = pitch[np.where((time > end_time) & (time < end_time + 0.5))[0]]

                prec.append(prec_pitch)
                curr.append(curr_pitch)
                succ.append(succ_pitch)
                svaras.append(labels.index(svara))
                clusters.append(cluster)

    logger(f"\tNumer of artists: {len(os.listdir(os.path.join('data', 'varnam_annotations', raga)))}\tNumber of samples: {len(svaras)}\n")

    save_data(prec, os.path.join('dataset', 'forms', 'prec.pkl'))
    save_data(curr, os.path.join('dataset', 'forms', 'curr.pkl'))
    save_data(succ, os.path.join('dataset', 'forms', 'succ.pkl'))
    save_data(svaras, os.path.join('dataset', 'forms', 'svaras.pkl'))
    save_data(clusters, os.path.join('dataset', 'forms', 'clusters.pkl'))

def iam_svaras(smoothing_factor, interpolation_gap, logger):
    with open(os.path.join("data", "iam_tonics.yaml"), "r") as f:
        tonics = yaml.safe_load(f)

    sequences, labels = [], []
    excluded = {39, 40, 42, 48, 67, 68, 76, 83, 118, 143, 146, 148, 150}
    idx = 0

    for file in os.listdir(os.path.join('data', 'iam_annotations')):
        performer = file.replace('.tsv', '')
        logger(f"{performer}")

        tonic = float(tonics[performer])
        if tonic is None or np.isnan(tonic):
            logger(f"\tSkipping {performer} (no tonic)\n")
            continue

        annotations_path = os.path.join('data', 'iam_annotations', file)
        annotations = pd.read_csv(annotations_path, delimiter='\t')

        pitch_track_path = os.path.join('data', 'iam_pitch_tracks', file)
        pitch_track = pd.read_csv(pitch_track_path, delimiter='\t', names=['time', 'frequency'], header=None)
        time = pitch_track['time'].values
        pitch = pitch_track['frequency'].values
        pitch[pitch == 0] = np.nan

        pitch = interpolate(pitch, np.nan, interpolation_gap)
        pitch = smooth_pitch_curve(time, pitch, smoothing_factor=smoothing_factor, min_points=4)
        pitch = 1200 * np.log2(pitch / tonic)

        for i, row in annotations.iterrows():
            logger.pbar(i + 1, len(annotations))

            parts = row["Begin time"].split(":")
            start_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            parts = row["End time"].split(":")
            end_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            annotation = int(row["Annotation"])

            segment = pitch[np.where((time >= start_time) & (time <= end_time))[0]]
            segment = segment[~np.isnan(segment)]
            if len(segment) < 10:
                continue
            if np.ptp(segment) < 50:
                continue
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            residual = segment - np.polyval(coeffs, x)
            if np.std(residual) < 10:
                continue
            if idx in excluded:
                idx += 1
                continue
            sequences.append(segment)
            labels.append(annotation)
            idx += 1

    logger(f"\tNumber of files: {len(os.listdir(os.path.join('data', 'iam_annotations')))}\tNumber of samples: {len(sequences)}\n")

    save_data(sequences, os.path.join('dataset', 'segments.pkl'))
    save_data(labels, os.path.join('dataset', 'ids.pkl'))

def cmmr_plausible_svaras(smoothing_factor, interpolation_gap, logger):
    tonics = pd.read_csv(os.path.join('data', "cmr_tonics.tsv"), delimiter="\t")
    note_lengths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    plausible_svaras = []

    for pitch_track_file in os.listdir(os.path.join('data', "cmr_pitch_tracks")):
        uid = int(pitch_track_file.split("_")[0])
        tonic = tonics[tonics["UID"] == uid]["tonic"].values[0]
        if tonic is None or np.isnan(tonic):
            continue

        beats_path = os.path.join('data', "cmr_beats", pitch_track_file.replace(".tsv", ".beats"))
        if not os.path.exists(beats_path):
            continue
        beats = pd.read_csv(beats_path, header=None, names=["time", "beat"])
        beats_time = beats["time"].values

        pitch_track_path = os.path.join('data', "cmr_pitch_tracks", pitch_track_file)
        pitch_track = pd.read_csv(pitch_track_path, header=None, names=["time", "frequency"], delimiter="\t")
        pitch = pitch_track["frequency"].values
        pitch_time = pitch_track["time"].values
        pitch = np.where(pitch == 0, np.nan, pitch)

        pitch = interpolate(pitch, np.nan, interpolation_gap)
        pitch = smooth_pitch_curve(pitch_time, pitch, smoothing_factor=smoothing_factor, min_points=4)
        pitch = 1200 * np.log2(pitch / tonic)

        logger(f"{pitch_track_file}")

        for i in range(len(beats_time) - 1):
            logger.pbar(i + 1, len(beats_time) - 1)

            note_length = np.random.choice(note_lengths)
            start_time = beats_time[i]
            beat_length = beats_time[i + 1] - beats_time[i]
            end_time = start_time + beat_length * note_length
            if end_time > beats_time[-1]:
                logger.pbar(len(beats_time) - 1, len(beats_time) - 1)
                break

            segment = pitch[(pitch_time >= start_time) & (pitch_time <= end_time)]
            plausible_svaras.append(segment)

    logger(f"\tNumber of samples: {len(plausible_svaras)}\n")

    save_data(plausible_svaras, os.path.join("dataset", "cmr.pkl"))
