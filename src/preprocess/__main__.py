import argparse
import os
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from utils.audio import save_data
from utils.pitch import smooth_pitch_curve, interpolate

console = Console()
CACHE = ".cache"
DATA = "data"


def main():
    args = parse_args()
    varnam_svaras(args["smoothing_factor"], args["interpolation_gap"])
    varnam_svara_forms(args["smoothing_factor"], args["interpolation_gap"])
    cmmr_plausible_svaras(args["smoothing_factor"], args["interpolation_gap"])
    iam_svaras(args["smoothing_factor"], args["interpolation_gap"])


def parse_args():
    parser = argparse.ArgumentParser()
    defaults = {"smoothing_factor": 0.5, "interpolation_gap": 0.02}
    for k, v in defaults.items():
        parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    try:
        cfg = yaml.safe_load(open("config.yaml")).get("pitch", {})
        for k in defaults:
            if k in cfg:
                setattr(args, k, cfg[k])
    except FileNotFoundError:
        pass
    return vars(args)


def varnam_svaras(smoothing_factor, interpolation_gap):
    with open(os.path.join(DATA, "Varnam", "tonics.yaml"), "r") as f:
        tonics = yaml.safe_load(f)

    for raga in os.listdir(os.path.join(DATA, "Varnam", "annotations")):
        console.print(f"[bold]{raga.upper()}[/bold]")
        prec, curr, succ, svaras = [], [], [], []
        labels = (
            ["S", "R", "G", "M", "D"]
            if raga == "abhogi"
            else ["S", "R", "G", "P", "D"]
            if raga == "mohanam"
            else ["S", "R", "G", "M", "P", "D", "N"]
        )

        for performer in os.listdir(os.path.join(DATA, "Varnam", "annotations", raga)):
            annotations_path = os.path.join(DATA, "Varnam", "annotations", raga, performer)
            annotations = pd.read_csv(annotations_path, delimiter="\t")

            pitch_track_path = os.path.join(
                DATA, "Varnam", "pitch_tracks", raga, performer.replace(f"_{raga}", "")
            )
            pitch_track = pd.read_csv(pitch_track_path, delimiter="\t", names=["time", "frequency"], header=None)
            time = pitch_track["time"].values
            pitch = pitch_track["frequency"].values
            pitch[pitch == 0] = np.nan

            tonic = float(tonics[performer.split("_")[0]])
            pitch = interpolate(pitch, np.nan, interpolation_gap)
            pitch = smooth_pitch_curve(time, pitch, smoothing_factor=smoothing_factor, min_points=4)
            pitch = 1200 * np.log2(pitch / tonic)

            for _, row in annotations.iterrows():
                start_time = float(row["Begin time"].split(":")[-2]) * 60 + float(row["Begin time"].split(":")[-1])
                end_time = float(row["End time"].split(":")[-2]) * 60 + float(row["End time"].split(":")[-1])
                annotation = row["Annotation"][0]

                prec.append(pitch[np.where((time > start_time - 0.5) & (time < start_time))[0]])
                curr.append(pitch[np.where((time > start_time) & (time < end_time))[0]])
                succ.append(pitch[np.where((time > end_time) & (time < end_time + 0.5))[0]])
                svaras.append(labels.index(annotation))

        console.print(f"\tSamples: {len(svaras)}, Classes: {len(set(svaras))}\n")

        save_data(prec, os.path.join(CACHE, raga, "prec.pkl"))
        save_data(curr, os.path.join(CACHE, raga, "curr.pkl"))
        save_data(succ, os.path.join(CACHE, raga, "succ.pkl"))
        save_data(svaras, os.path.join(CACHE, raga, "svaras.pkl"))
        save_data(labels, os.path.join(CACHE, raga, "labels.pkl"))


def varnam_svara_forms(smoothing_factor, interpolation_gap):
    annotations = pd.read_csv(os.path.join(DATA, "Varnam", "svara_forms.csv"))
    with open(os.path.join(DATA, "Varnam", "tonics.yaml"), "r") as f:
        tonics = yaml.safe_load(f)

    prec, curr, succ, svaras, clusters = [], [], [], [], []
    labels = ["S", "R", "G", "M", "P", "D", "N"]

    for raga in os.listdir(os.path.join(DATA, "Varnam", "annotations")):
        console.print(f"[bold]{raga.upper()}[/bold]")

        for performer in os.listdir(os.path.join(DATA, "Varnam", "annotations", raga)):
            pitch_track_path = os.path.join(
                DATA, "Varnam", "pitch_tracks", raga, performer.replace(f"_{raga}", "")
            )
            pitch_track = pd.read_csv(pitch_track_path, delimiter="\t", names=["time", "frequency"], header=None)
            time = pitch_track["time"].values
            pitch = pitch_track["frequency"].values
            pitch[pitch == 0] = np.nan

            tonic = float(tonics[performer.split("_")[0]])
            pitch = interpolate(pitch, np.nan, interpolation_gap)
            pitch = smooth_pitch_curve(time, pitch, smoothing_factor=smoothing_factor, min_points=4)
            pitch = 1200 * np.log2(pitch / tonic)

            perf_annotations = annotations[
                (annotations["raga"] == raga)
                & (annotations["performer"] == performer.replace(f"_{raga}.tsv", ""))
            ]

            for _, row in perf_annotations.iterrows():
                start_time = float(row["start"])
                end_time = float(row["end"])

                prec.append(pitch[np.where((time > start_time - 0.5) & (time < start_time))[0]])
                curr.append(pitch[np.where((time > start_time) & (time < end_time))[0]])
                succ.append(pitch[np.where((time > end_time) & (time < end_time + 0.5))[0]])
                svaras.append(labels.index(row["svara"][0]))
                clusters.append(row["cluster"])

    console.print(f"\tSamples: {len(svaras)}\n")

    save_data(prec, os.path.join(CACHE, "forms", "prec.pkl"))
    save_data(curr, os.path.join(CACHE, "forms", "curr.pkl"))
    save_data(succ, os.path.join(CACHE, "forms", "succ.pkl"))
    save_data(svaras, os.path.join(CACHE, "forms", "svaras.pkl"))
    save_data(clusters, os.path.join(CACHE, "forms", "clusters.pkl"))


def iam_svaras(smoothing_factor, interpolation_gap):
    with open(os.path.join(DATA, "IAMMS", "tonics.yaml"), "r") as f:
        tonics = yaml.safe_load(f)

    sequences, labels = [], []
    excluded = {39, 40, 42, 48, 67, 68, 76, 83, 118, 143, 146, 148, 150}
    idx = 0

    for file in os.listdir(os.path.join(DATA, "IAMMS", "annotations")):
        performer = file.replace(".tsv", "")
        console.print(f"{performer}")

        tonic = float(tonics[performer])
        if tonic is None or np.isnan(tonic):
            continue

        annotations_path = os.path.join(DATA, "IAMMS", "annotations", file)
        annotations = pd.read_csv(annotations_path, delimiter="\t")

        pitch_track_path = os.path.join(DATA, "IAMMS", "pitch_tracks", file)
        pitch_track = pd.read_csv(pitch_track_path, delimiter="\t", names=["time", "frequency"], header=None)
        time = pitch_track["time"].values
        pitch = pitch_track["frequency"].values
        pitch[pitch == 0] = np.nan

        pitch = interpolate(pitch, np.nan, interpolation_gap)
        pitch = smooth_pitch_curve(time, pitch, smoothing_factor=smoothing_factor, min_points=4)
        pitch = 1200 * np.log2(pitch / tonic)

        for _, row in annotations.iterrows():
            parts = row["Begin time"].split(":")
            start_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            parts = row["End time"].split(":")
            end_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

            segment = pitch[np.where((time >= start_time) & (time <= end_time))[0]]
            segment = segment[~np.isnan(segment)]
            if len(segment) < 10 or np.ptp(segment) < 50:
                continue
            x = np.arange(len(segment))
            residual = segment - np.polyval(np.polyfit(x, segment, 1), x)
            if np.std(residual) < 10:
                continue
            if idx in excluded:
                idx += 1
                continue
            sequences.append(segment)
            labels.append(int(row["Annotation"]))
            idx += 1

    console.print(f"\tSamples: {len(sequences)}\n")

    save_data(sequences, os.path.join(CACHE, "segments.pkl"))
    save_data(labels, os.path.join(CACHE, "ids.pkl"))


def cmmr_plausible_svaras(smoothing_factor, interpolation_gap):
    tonics = pd.read_csv(os.path.join(DATA, "CMR", "tonics.tsv"), delimiter="\t")
    note_lengths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    plausible_svaras = []

    for pitch_track_file in os.listdir(os.path.join(DATA, "CMR", "pitch_tracks")):
        uid = int(pitch_track_file.replace(".tsv", ""))
        tonic = tonics[tonics["UID"] == uid]["tonic"].values[0]
        if tonic is None or np.isnan(tonic):
            continue

        beats_path = os.path.join(DATA, "CMR", "beats", pitch_track_file)
        if not os.path.exists(beats_path):
            continue
        beats = pd.read_csv(beats_path, header=None, names=["time", "beat"])
        beats_time = beats["time"].values

        pitch_track_path = os.path.join(DATA, "CMR", "pitch_tracks", pitch_track_file)
        pitch_track = pd.read_csv(pitch_track_path, header=None, names=["time", "frequency"], delimiter="\t")
        pitch = pitch_track["frequency"].values
        pitch_time = pitch_track["time"].values
        pitch = np.where(pitch == 0, np.nan, pitch)

        pitch = interpolate(pitch, np.nan, interpolation_gap)
        pitch = smooth_pitch_curve(pitch_time, pitch, smoothing_factor=smoothing_factor, min_points=4)
        pitch = 1200 * np.log2(pitch / tonic)

        console.print(f"{pitch_track_file}")

        for i in range(len(beats_time) - 1):
            note_length = np.random.choice(note_lengths)
            start_time = beats_time[i]
            beat_length = beats_time[i + 1] - beats_time[i]
            end_time = start_time + beat_length * note_length
            if end_time > beats_time[-1]:
                break
            segment = pitch[(pitch_time >= start_time) & (pitch_time <= end_time)]
            plausible_svaras.append(segment)

    console.print(f"\tSamples: {len(plausible_svaras)}\n")
    save_data(plausible_svaras, os.path.join(CACHE, "cmr.pkl"))


if __name__ == "__main__":
    main()
