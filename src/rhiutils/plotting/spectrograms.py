from gwpy.timeseries import TimeSeries

from typing import Tuple


def plot_spectrogram(
    timeseries: TimeSeries,
    title: str,
    frequency_range: Tuple[float, float] = (16, 100),
    spectrogram_color_maximum: float = 25,
    q_value: int = 20,
    figsize = (16, 8)
):
    # Make a spectrogram
    qspecgram = timeseries.q_transform(
        qrange=[
            q_value,
            q_value,
        ],
        frange=frequency_range,
    )

    fig = qspecgram.plot(figsize=figsize, edgecolors="none", rasterized=True)
    ax = fig.gca()
    ax.set_yscale("log")
    ax.set_xlabel("Time [seconds]")
    ax.set_title(title)
    ax.colorbar(
        cmap="viridis",
        label="Normalized energy",
        vmin=0,
        vmax=spectrogram_color_maximum,
    )
    ax.grid(axis="y", which="both")
    return fig, ax
