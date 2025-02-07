import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats


# Constants for font size and line width
FONT_SIZE = 10
LABELS_FONT_SIZE_FRACTION = 0.8
TICKS_FONT_SIZE_FRACTION = 0.9
LEGEND_FRACTION = 0.5
TITLE_FRACTION = 0.95

LINEWIDTH = 1
TH_FRACTION = 0.75
GRID_FRACTION = 0.25
BORDER_FRACTION = 0.25
GRID_ALPHA = 0.4
SHADE_ALPHA = 0.15


# Set style
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
sns.set_theme(context='paper', style="whitegrid", palette='colorblind', font="Times New Roman", rc={"grid.color": "gray", "grid.alpha": GRID_ALPHA})
sns.set_context("paper", rc={"lines.line_width":LINEWIDTH})
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": False,
    "font.family":"Times New Roman",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": FONT_SIZE * TITLE_FRACTION,
    "axes.labelsize": FONT_SIZE * LABELS_FONT_SIZE_FRACTION,
    "font.size": FONT_SIZE,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": FONT_SIZE * LEGEND_FRACTION,
    "xtick.labelsize": FONT_SIZE * LABELS_FONT_SIZE_FRACTION,
    "ytick.labelsize": FONT_SIZE * LABELS_FONT_SIZE_FRACTION,
    # reduce padding of x/y label ticks with plots
    "xtick.major.pad":0,
    "ytick.major.pad":0,
    # set figure dpi
    'figure.dpi': 300,
}
plt.rcParams.update(tex_fonts)


def plot_transformer(losses,
                     legends,
                     title,
                     colors=None,
                     ci_widths=None,
                     x_label="# in-context examples",
                     y_label="Mean squared error",
                     y_ticks_max=None,
                     y_ticks_interval=2.5,):
    plt.rcParams["figure.figsize"] = (5, 3)
    
    for i, (loss, legend) in enumerate(zip(losses, legends)):
        color = colors[i] if colors is not None else None
        line = plt.plot(loss, label=legend, color=color, linewidth=LINEWIDTH)
        shade_color = color if colors is not None else line[0].get_color()

        # Adding the confidence interval shade
        if ci_widths is not None:
            lower_bound = np.array(loss) - ci_widths[i]
            upper_bound = np.array(loss) + ci_widths[i]
            plt.fill_between(range(len(loss)), lower_bound, upper_bound, color=shade_color, alpha=SHADE_ALPHA)

    if y_ticks_max is None:
        y_ticks_max = 1.5 * max(loss.max() for loss in losses)

    length = len(losses[0])

    tick_positions = [0, length * 0.25, length * 0.5, length * 0.75, length - 1]
    tick_labels = ['0', '50', '100', '150', '200'] if length > 100 else ['0', '25', '50', '75', '100']

    plt.xticks(tick_positions, tick_labels)
    plt.yticks(np.arange(0, y_ticks_max + y_ticks_interval, y_ticks_interval))

    plt.xlabel(x_label, fontsize=FONT_SIZE * LABELS_FONT_SIZE_FRACTION)
    plt.ylabel(y_label, fontsize=FONT_SIZE * LABELS_FONT_SIZE_FRACTION)

    x_position = 40 if length < 102 else 100

    plt.axvline(x=x_position, linestyle='--', linewidth=LINEWIDTH * TH_FRACTION, color='black', label="max pretraining length")

    plt.title(title, fontsize=FONT_SIZE * TITLE_FRACTION)
    plt.legend(fontsize=FONT_SIZE * LEGEND_FRACTION, loc='upper center', ncol=3)

    plt.grid(linewidth=LINEWIDTH * GRID_FRACTION)

    plt.gca().spines['top'].set_linewidth(LINEWIDTH * BORDER_FRACTION)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(LINEWIDTH * BORDER_FRACTION)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_linewidth(LINEWIDTH * BORDER_FRACTION)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['right'].set_linewidth(LINEWIDTH * BORDER_FRACTION)
    plt.gca().spines['right'].set_color('black')

    plt.tight_layout()
    plt.show()
    plt.clf()


def compute_confidence_interval(losses, confidence=0.95):
    mean_rewards = np.mean(losses, axis=1)
    std_dev = np.std(mean_rewards, ddof=1)

    sem = std_dev / np.sqrt(len(mean_rewards))
    df = len(mean_rewards) - 1

    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    ci_width = t_critical * sem

    return ci_width
