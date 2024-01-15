from typing import Callable, Dict, List, Optional, Tuple

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

from utils import compute_correlations

# Latex font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = "20"

COLORS = [
    ["#6d6875", "#e5989b", "#ffcdb2"],
    ["#354f52", "#84a98c", "#cad2c5"],
    ["#f72585", "#3a0ca3", "#4cc9f0"],
    ["#b7094c", "#5c4d7d", "#0091ad"],
    ["#355764", "#5A8F7B", "#81CACF"],
    ["#76549A", "#DF7861", "#94B49F"],
]

DIVERGING_COLORS = [
    ["#1c53ae", "#96242c"]
]

ORDER = ["USA", "Germany", "China"]
MEDIA = ["audio", "image", "text"]

COUNTRY_COLORS = {
    "USA": COLORS[-1][0],
    "Germany": COLORS[-1][1],
    "China": COLORS[-1][2],
}


def plot_forest(
    fixed_data: List[pd.DataFrame],
    titles: List[str],
    random_data: Optional[List[pd.DataFrame]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    grid: bool = False,
    colors: Optional[List["str"]] = None,
):
    """Plot predictor variables as forest plot.
    """
    if random_data is not None:
        if len(random_data) != len(fixed_data):
            raise ValueError("Random and fixed data list must be same size!")
        if colors is None:
            raise ValueError("Must supply colors with random data")

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = GridSpec(1, len(fixed_data), figure=fig)

    # calculate x-axis limit
    limit = 0
    for data in fixed_data + (random_data or []):
        limit_cur = max(abs(data["low"].min()), abs(data["high"].max())) + .1
        limit = max(limit, limit_cur)

    for i, data in enumerate(fixed_data):
        ax = fig.add_subplot(gs[0, i])
        data_sorted = data.sort_values("name")

        # print effect#326c80
        y = 0
        pos_to_name = {}
        for _, row in data_sorted.iterrows():
            predictor_name = row["name"]

            # plot predictor mean + hdi
            ax.plot(row["mean"], y, "o", color="k")
            ax.hlines(y, row["low"], row["high"], color="k", linewidth=1)

            # save position + name for later
            pos_to_name[y] = predictor_name
            y -= 1.5  # offset predictor variables

            # handle additional random variable
            if random_data:
                assert colors is not None

                rand_data = random_data[i]
                pred_rand = rand_data[rand_data.name.str.contains(
                    predictor_name)]

                y -= .75  # offset random vars
                for j, (_, rand_row) in enumerate(pred_rand.sort_values("name", ascending=False).iterrows()):
                    ax.plot(rand_row["mean"], y, "o", color=colors[j])
                    ax.hlines(y, rand_row["low"], rand_row["high"],
                              color=colors[j], linewidth=1)

                    pos_to_name[y] = rand_row["name"].replace(
                        predictor_name, "").strip()
                    y -= 1.5

                y -= 2.5  # offset group

        # remove spines and ticks
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('none')

        # fix labels
        if i == 0:
            ticks, ticklabels = list(zip(*pos_to_name.items()))
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
        else:
            ax.set_yticklabels([])

        ax.axvline(0, color="k", linestyle="--")

        ax.set_xlim(-limit, limit)
        ax.set_title(titles[i])
        ax.set_xlabel("Influence of Predictor Variable")

        if grid:
            ax.yaxis.grid(True)

    # set shared x limit

    fig.tight_layout()


def plot_category(
    data: pd.DataFrame,
    x: str,
    title: str,
    order: List[str],
    colors: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    kind: str = "count",
    show_legend: bool = False,
    plot_kwargs: Optional[Dict] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    vline: Optional[List[float]] = None,
    hline: Optional[List[float]] = None,
    regplot: bool = False,
    log_scale: bool = False,
    hue: str = "country",
    y_label: bool = True,
):
    """Plot category as a grid layout.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True,
                     facecolor="white")
    gs = GridSpec(3, 4, figure=fig)

    plot_args = plot_kwargs or {}

    # main plot all together
    main = fig.add_subplot(gs[0:, 0:3])
    main_kwargs = {
        "x": x,
        "palette": colors,
        "data": data,
        "hue": hue,
        "ax": main,
        "hue_order": order,
        "alpha": 0.7,
        # "s": 2.5,
    }
    main_kwargs.update(plot_args)

    # format axis
    if ylim:
        main.set_ylim(*ylim)

    if xlim:
        main.set_xlim(*xlim)

    if vline:
        for level in vline:
            main.axvline(x=level, c="k", linewidth=1, linestyle="dotted")

    if hline:
        for level in hline:
            main.axhline(y=level, c="k", linewidth=1, linestyle="dotted")

    # plot
    if kind == "count":
        sn.countplot(**main_kwargs)
    elif kind == "stripplot":
        sn.stripplot(**main_kwargs)
    elif kind == "hist":
        sn.histplot(multiple="stack", legend=True, **main_kwargs)
    elif kind == "point":
        sn.pointplot(**main_kwargs)
    elif kind == "scatter":
        sn.scatterplot(**main_kwargs)

        if regplot:
            sn.regplot(
                color="k",
                scatter=False,
                x=main_kwargs["x"],
                y=main_kwargs["y"],
                data=main_kwargs["data"],
                ax=main_kwargs["ax"],
            )
    else:
        raise ValueError("Unsupported plot type.")

    # set labels etc
    main.set_ylabel(main.get_ylabel().title())
    main.set_xlabel("")

    # remove legend
    if not show_legend and main.get_legend():
        main.get_legend().remove()

    if log_scale:
        main.set_yscale("log")

    if not y_label:
        main.set_ylabel("")

    def plot_single_ax(ax: plt.Axes, single_name: str, color: str, sharex: Optional[plt.Axes] = None, ax_selector: str = "country"):
        single_kwargs = {
            "x": x,
            "color": color,
            "data": data[data[ax_selector] == single_name],
            "ax": ax,
        }
        single_kwargs.update(plot_args)

        # format axis
        if ylim:
            ax.set_ylim(ylim)

        if xlim:
            ax.set_xlim(xlim)

        if vline:
            for level in vline:
                ax.axvline(x=level, c="k", linewidth=1)

        if hline:
            for level in hline:
                ax.axhline(y=level, c="k", linewidth=1)

        if sharex:
            ax.sharex(sharex)

        # plot
        if kind == "count":
            sn.countplot(**single_kwargs)
        elif kind == "stripplot":
            sn.stripplot(**single_kwargs)
        elif kind == "hist":
            sn.histplot(multiple="stack", **single_kwargs)
        elif kind == "point":
            sn.pointplot(**single_kwargs)
        elif kind == "scatter":
            sn.scatterplot(**single_kwargs)

            if regplot:
                sn.regplot(
                    color="k",
                    scatter=False,
                    x=single_kwargs["x"],
                    y=single_kwargs["y"],
                    data=single_kwargs["data"],
                    ax=single_kwargs["ax"],
                )
        else:
            raise ValueError("Unsupported plot type.")

        # set labels
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(single_name.title())

        if log_scale:
            ax.set_yscale("log")

    # seperate regions
    prev = None
    for i, country in enumerate(order):
        ax = fig.add_subplot(gs[i, 3])

        plot_single_ax(ax, country, colors[i], prev, ax_selector=hue)
        prev = ax

    fig.suptitle(title)


def plot_category2(
    data: pd.DataFrame,
    x: str,
    title: str,
    order: List[str],
    colors: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    kind: str = "count",
    show_legend: bool = False,
    plot_kwargs: Optional[Dict] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    vline: Optional[List[float]] = None,
    hline: Optional[List[float]] = None,
    regplot: bool = False,
    log_scale: bool = False,
    hue: str = "media_type",
    y_label: bool = True,
):
    """Plot category as a grid layout.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True,
                     facecolor="white")
    gs = GridSpec(3, 4, figure=fig)

    plot_args = plot_kwargs or {}

    # main plot all together
    main = fig.add_subplot(gs[0:, 0:3])

    # sn.set(font_scale=2)
    #sn.set(font_scale=2, rc={'text.usetex' : True})
    # sn.set_theme(style='white')

    # plt.grid(False)

    #plt.rcParams['font.size'] = '20'

    # format axis
    if ylim:
        main.set_ylim(*ylim)

    if xlim:
        main.set_xlim(*xlim)

    for i, media_type in enumerate(["audio", "image", "text"]):

        main_kwargs = {
            "x": x,
            "palette": colors,
            "data": data[data["media_type"] == media_type],
            "hue": hue,
            "ax": main,
            "hue_order": order,
            "alpha": 0.7,
            "s": 8,
        }
        main_kwargs.update(plot_args)

        sn.stripplot(**main_kwargs)

    if vline:
        for level in vline:
            main.axvline(x=level, c="k", linewidth=1, linestyle="dotted")

    if hline:
        for level in hline:
            main.axhline(y=level, c="k", linewidth=1, linestyle="dotted")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch

    legend_elements = [Line2D([0], [0], linestyle='none', color=colors[0], marker='o', label='Image'),
                       Line2D([0], [0], linestyle='none',
                              color=colors[1], marker='o', label='Audio'),
                       Line2D([0], [0], linestyle='none', color=colors[2], marker='o', label='Text')]

    # main.legend(['Audio', 'Image', 'Text'], loc='upper left', ncol=3, bbox_to_anchor=(0.01, 1.1))
    main.legend(handles=legend_elements, loc='upper left',
                ncol=3, bbox_to_anchor=(0.01, 1.1))

    main.spines['top'].set_visible(False)

    # set labels etc
    main.set_ylabel(main.get_ylabel().title())
    main.set_xlabel("")

    # # remove legend
    # if not show_legend and main.get_legend():
    #     main.get_legend().remove()

    if log_scale:
        main.set_yscale("log")

    if not y_label:
        main.set_ylabel("")

    plt.savefig('accuracy.pdf', bbox_inches='tight')


def plot_cat_vs_acc(data, cat: str, title: str, corr_func: Callable = pearsonr, corr_label: str = "Pearson's r"):
    """Plot accuracy vs. the given category.
    """
    plot_category(
        data=data,
        x=cat,
        title=f"{title} (Per Media Type)",
        order=["audio", "image", "text"],
        colors=COLORS[-3],
        kind="scatter",
        plot_kwargs={
            "y": "Acc",
        },
        hline=[.1, .5, .9],
        ylim=(-.1, 1.1),
        regplot=True,
        hue="media_type",
    )

    plot_category(
        data=data,
        x=cat,
        title=f"{title} (Per Country)",
        order=ORDER,
        colors=COLORS[-1],
        kind="scatter",
        plot_kwargs={
            "y": "Acc",
        },
        hline=[.1, .5, .9],
        ylim=(-.1, 1.1),
        regplot=True,
    )

    for media in MEDIA:
        plot_category(
            data=data[data.media_type == media],
            x=cat,
            title=f"{title} ({media.title()})",
            order=ORDER,
            colors=COLORS[-1],
            kind="scatter",
            plot_kwargs={
                "y": "Acc",
            },
            hline=[.1, .5, .9],
            ylim=(-.1, 1.1),
            regplot=True,
        )

    return compute_correlations(data, y="Acc", x=cat, corr_func=corr_func, corr_label=corr_label)


def plot_prediagnostics(prior_pred):
    """Plot Predicted amount of succesess, log-odds, ods
    """
    fig = plt.figure(figsize=(14, 3))

    gs = GridSpec(1, 3, figure=fig)
    ax = fig.add_subplot(gs[0, 0])

    succ = prior_pred.prior_predictive.y[0, :, 0:50].values
    sn.histplot(data=succ, ax=ax, alpha=.2, bins=25, multiple="stack")

    ax.get_legend().remove()
    ax.set_ylabel("Predicted amount of succeses.")

    ax = fig.add_subplot(gs[0, 1])
    for i in range(50):
        # first chain, first observation, prior anyway
        pi_log = prior_pred.prior.pi_log[0, :, i]
        sn.kdeplot(data=pi_log, ax=ax, alpha=.2, color="k")

    ax.set_xlabel("Log-Odds")
    ax.axvline(-4, color="k")
    ax.axvline(4, color="k")

    ax = fig.add_subplot(gs[0, 2])
    for i in range(50):
        # first chain, first observation, prior anyway
        pi_log = prior_pred.prior.pi[0, :, i]
        sn.kdeplot(data=pi_log, ax=ax, alpha=.2, color="k")

    ax.axvline(0, color="k")
    ax.axvline(1, color="k")

    ax.set_xlabel("Odds for success")

    fig.tight_layout()


def plot_diagnostics(trace):
    """Plot different diagnostics for a fitted trace.
    """
    summary = az.summary(trace)
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig)

    # r_hat
    r_hat_ax = fig.add_subplot(gs[0, 0])
    r_hat = summary.r_hat
    failed = r_hat > 1.05
    df = pd.DataFrame({
        "r_hat": r_hat,
        "failed": failed,
    })

    sn.histplot(x="r_hat", data=df, ax=r_hat_ax, hue="failed", bins=50)
    r_hat_ax.get_legend().remove()
    r_hat_ax.axvline(1.05, color="k")
    r_hat_ax.set_xlabel("Rhat statistic")

    # monte carlo se / posterior sd > 10%
    mcse_ax = fig.add_subplot(gs[0, 1])
    mcse_post = summary.mcse_mean / summary.sd
    failed = mcse_post > .1

    df = pd.DataFrame({
        "mcse_post": mcse_post,
        "failed": failed,
    })

    sn.histplot(x="mcse_post", data=df, ax=mcse_ax, hue="failed", bins=50)
    mcse_ax.get_legend().remove()
    mcse_ax.axvline(.1, color="k")
    mcse_ax.set_xlabel("Monte Carlo SE / Posterior SD")

    # effective sample size / iterations bigger 10% of data
    eff_ax = fig.add_subplot(gs[0, 2])

    n = trace.posterior.dims['chain'] * trace.posterior.dims['draw']
    eff = summary.ess_bulk / n
    failed = eff < 1.

    df = pd.DataFrame({
        "eff": eff,
        "failed": failed,
    })

    sn.histplot(x="eff", data=df, ax=eff_ax, hue="failed", bins=50)
    eff_ax.get_legend().remove()
    eff_ax.axvline(1., color="k")
    eff_ax.set_xlabel("Effective Sample Size / Iterations", labelpad=15)
