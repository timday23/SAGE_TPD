import os
import sys
import re
import math
import itertools
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy.stats import gmean, gstd
from PIL import Image
from tqdm.notebook import tqdm
from pathlib import Path
from scipy.stats import linregress


REF_DATASET = "Final_Testv1_norm"  # default reference dataset

PP_UNIT = 'nm'
PP_BIN_SIZE = 1.25
ERROR_MARGIN_DEFAULT = 0.25
MIN_DP_VALUE = 0

DEFAULT_FIGSIZE = (12, 8)
DEFAULT_DPI = 100

# Global plot defaults
PLOT_DEFAULTS = {
    # -----------------------
    # General figure options
    # -----------------------
    "figsize": DEFAULT_FIGSIZE,      # Figure size
    "dpi": DEFAULT_DPI,              # Figure DPI
    "palette": sns.color_palette("tab10"),  # Default color palette
    "show_grid": True,               # Show grid in plots
    "ref_dataset": REF_DATASET,      # Reference GT dataset for comparisons

    # -----------------------
    # PSD / histogram plotting
    # -----------------------
    "plot_type": "hist",             # 'hist', 'kde', or 'bar'
    "normalize": False,              # plot density instead of raw counts
    "log_scale": False,              # y-axis log scale
    "bin_size": PP_BIN_SIZE,         # Bin width for histograms
    "unit": PP_UNIT,                 # Measurement unit for diameters ('nm' or 'pix')
    "overlay": True,                 # Overlay multiple methods
    "filter_small": False,           # Filter particles smaller than min threshold

    # -------------------------
    # Metric agreement plotting
    # -------------------------
    "gt_method": REF_DATASET,
    "autoscale": True,
    "pad_frac": 0.1,
    "xlim": None,
    "ylim": None,
    "legend_loc": None, 

    "mode": "single", 
    "layout": "row",
    "legend_loc": None,
    "subplot_size": 8,


    # --------------------------
    # Parity plot stuff
    # --------------------------
    "error_margin": ERROR_MARGIN_DEFAULT,  # Margin for metric calculations
    "buffer_ratio": 0.05,            # Buffer for plotting metrics
    

    # Method-specific plotting
    "linewidth": 3,                  # Line width for PSD lines
    "linestyle": "-",                # Default line style (can be overridden per method)
    "marker": "o",                   # Default marker for scatter/KDE
    "marker_size": 100,              # Marker size for scatter plots
    "marker_edgecolor": "black",     # Marker edge color
    "alpha": 0.8,                    # Transparency for lines/markers

    # Diameter style mapping (for multiple diameters: equiv, feret, min_feret)
    "diameter_styles": {
        "dp": {"linestyle": "-", "marker": "o", "label": "Area Equiv. Dia."},
        "feret": {"linestyle": "--", "marker": "s", "label": "Max Feret Dia."},
        "min_feret": {"linestyle": ":", "marker": "^", "label": "Min Feret Dia."},
    },

    # -----------------------
    # Metrics / statistics
    # -----------------------
    "show_metrics": True,            # Show geometric mean ± std in legend
    

    # -----------------------
    # Error / special plots
    # -----------------------
    "error_style": {                 # Only color used by default
        "color": None
    },

    # -----------------------
    # Text / labels
    # -----------------------
    "title": None,                   # Plot title
    "xlabel": None,                  # X-axis label (auto-generated if None)
    "ylabel": None,                  # Y-axis label (auto-generated if None)
    "fontsize_title": 24,
    "fontsize_axes": 20,
    "fontsize_ticks": 16,
    "fontsize_legend": 14,

    # -----------------------
    # Miscellaneous / advanced
    # -----------------------
    "show_legend": True,             # Show legend
    "legend_loc": "best",            # Legend location
}

METHOD_REGISTRY = {}

DEFAULT_COLORS = sns.color_palette("tab10") + sns.color_palette("Set2")
DEFAULT_MARKERS = ['o','s','^','D','v','X','*','P','H']
DEFAULT_LINESTYLES = ['-', '--', '-.', ':']
# -----------------------------
# Helper: update defaults globally (optional)
# -----------------------------
def set_defaults(ref_dataset=None, unit=None, bin_size=None, **kwargs):
    global PLOT_DEFAULTS
    if ref_dataset is not None:
        PLOT_DEFAULTS["ref_dataset"] = ref_dataset
    if unit is not None:
        PLOT_DEFAULTS["unit"] = unit
    if bin_size is not None:
        PLOT_DEFAULTS["bin_size"] = bin_size
    # update any other keys passed via kwargs
    for k,v in kwargs.items():
        PLOT_DEFAULTS[k] = v



def register_methods(method_dict):
    """
    Add/update curated methods in the module-level registry.
    """
    global METHOD_REGISTRY
    METHOD_REGISTRY.update(method_dict)

def build_method_styles(
    methods, 
    plot_type='psd', 
    method_registry=None, 
    default_colors=None, 
    default_markers=None, 
    default_linestyles=None
):
    """
    Returns a dict of styles for the given methods.
    Uses the module-level registry by default.
    """
    method_registry = method_registry or METHOD_REGISTRY
    default_colors = default_colors or DEFAULT_COLORS
    default_markers = default_markers or DEFAULT_MARKERS
    default_linestyles = default_linestyles or DEFAULT_LINESTYLES

    styles = {}
    color_cycle = itertools.cycle(default_colors)
    marker_cycle = itertools.cycle(default_markers)
    linestyle_cycle = itertools.cycle(default_linestyles)

    for method in methods:
        style={}

        #top level color
        if method in method_registry:
            reg_entry = method_registry[method]

            top_color = reg_entry.get("color", next(color_cycle))
            style["color"] = top_color

            #label
            style["label"] = reg_entry.get("label", method)

            plot_style_key = f"{plot_type}_style"
            plot_style = reg_entry.get(plot_style_key, {})
            for k, v in plot_style.items():
                style[k] = v
             # Fill missing defaults
            if "color" not in style:
                style["color"] = top_color

            if plot_type == "psd":
                style.setdefault("linestyle", next(linestyle_cycle))
                style.setdefault("marker", "")
            elif plot_type == "scatter":
                style.setdefault("marker", next(marker_cycle))
                style.setdefault("size", 150)
                style.setdefault("edgecolor", "black")
            elif plot_type == "error":
                # only color is needed, already handled by top-level color
                pass
            else:
                # fallback generic
                style.setdefault("linestyle", "-")
                style.setdefault("marker", "")

        else:
            # Unknown method → auto-generate
            style["color"] = next(color_cycle)
            style["label"] = method
            if plot_type == "psd":
                style["linestyle"] = next(linestyle_cycle)
                style["marker"] = ""
            elif plot_type == "scatter":
                style["marker"] = next(marker_cycle)
                style["size"] = 150
                style["edgecolor"] = "black"
            elif plot_type == "error":
                # only color
                pass

        styles[method] = style

    return styles


    #     else: #unknown method, autogenerate color and label
    #         style["color"] = next(color_cycle)
    #         style["label"] = method
    #     #fill in plot-specific defaults

    #     if method in method_registry:
    #         style = method_registry[method].get(f"{plot_type}_style", {}).copy()
    #         style.setdefault("label", method_registry[method].get("label", method))
    #         styles[method] = style
    #     else:
    #         # Auto-generate style for unknown methods
    #         style = {"label": method, "color": next(color_cycle)}
    #         if plot_type == "psd":
    #             style["linestyle"] = next(linestyle_cycle)
    #             style["marker"] = ""
    #         elif plot_type == "scatter":
    #             style["marker"] = next(marker_cycle)
    #             style["size"] = 150
    #             style["edgecolor"] = "black"
    #         styles[method] = style

    # return styles

#Loading data/utilites




def load_PP_info(pp_info_folder, shorten=False, conf_filter=None):

    #print(f"Available Methods:\n")
    all_data = []
    for file in os.listdir(pp_info_folder):
        if file.endswith("_pp_info.csv"):
            file_path = os.path.join(pp_info_folder, file)

            #extract method name
            if shorten:
                method=re.sub(r"(_part(_\d+(\.\d+)?)?|(_\d+(\.\d+)?))?_pp_info\.csv$", "", file)
                #print(method)
            else:
                method=re.sub(r"(_\d+(\.\d+)?)?_pp_info\.csv$", "", file)

            #Try to extract confidence threshold from filename
            conf_match = re.search(r"_([0-9.]+)_pp_info\.csv$", file)
            if conf_match:
                conf_threshold = float(conf_match.group(1))
            else:
                conf_threshold = float('nan')

            if conf_filter is not None and not pd.isna(conf_threshold):
                if conf_threshold != conf_filter:
                    continue

            #load csv and attach method column
            df = pd.read_csv(file_path)
            df.insert(1, "method", method)
            df.insert(2, "conf_threshold", conf_threshold)
            #print(df.head(5))

            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    #print(combined_df.columns)
    #print(combined_df.head())
    return combined_df

def load_collab_csv(PSD_file_folder, method_name="Dreier_Sizes", unit = "nm"):
    all_data = []
    for file in os.listdir(PSD_file_folder):
        if file.endswith("Results.csv"):
            file_path = os.path.join(PSD_file_folder, file)
            print(f"Loading: {file_path}")

            image_name = file.split("_Results")[0]

            df = pd.read_csv(file_path)
           
            #normalize columns names (lowercase and strip spaces)
            df.columns = [c.strip().lower() for c in df.columns]
           
            #reanme first column to PP #
            df = df.rename(columns={df.columns[0]: "PP #"})

            #rename columns to match my convention
            rename_map = {}
            if 'feret' in df.columns:
                rename_map['feret'] = f'feret ({unit})'
            
            
            if 'feretx' in df.columns:
                rename_map['feretx'] = f'feret_x ({unit})'

         
            if 'ferety' in df.columns:
                rename_map['ferety'] = f'feret_y ({unit})'

            if 'feretangle' in df.columns:
                rename_map['feretangle'] = f'feret_angle (degrees)'

           
            if 'minferet' in df.columns:
                rename_map['minferet'] = f'min_feret ({unit})'

            df = df.rename(columns=rename_map)

            df.insert(0, "image", image_name)
            df.insert(1, 'method', method_name)
            all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    print(combined_df.columns)
    return combined_df

# Fractal/Rg Conversions

def convert_to_3D_fractal_dim(df, col_2D = 'fractal_dim'):
    """
    Converts 2D fractal dimension to 3D using linear regression formula
    Df = 1.391 + 0.1exp(2.164*Df_2D)
    """
    new_col_name = '3D Fractal Dimension'

    df[new_col_name] = 1.391 + 0.01 * np.exp(2.164*df[col_2D])

    return df

def convert_3D_Rg(df, col_2D = None, unit='nm'):
        #df['3D Rg [nm]'] = 1.023*df[col_2D]# (np.sqrt((2*df[col_3D_Df]) / (df[col_3D_Df] +2)) * df[col_2D]) * 1.023
    if col_2D is None:
        col_2D = f'Rg [{unit}]'
        if col_2D not in df.columns:
            raise ValueError(f"Column {col_2D} not found in dataframe")

        new_col = f'3D Rg [{unit}]'
        df[new_col] = 1.023 * df[col_2D]
   
    return df



    
#Method utilities


def inspect_methods(pp_df, registry=None, excluded=None):
    print("="*60)
    df = pp_df
    df_methods = set()
    reg_methods = set()
    
    #methods present in dataframe
    if df is not None and hasattr(df, "columns") and "method" in df.columns:
        df_methods = set(df["method"].unique())
        print("Methods in loaded DataFrame")
        for m in sorted(df_methods):
            print(f" - {m}")

    else:
        print("No valid datafram or 'method' column found")

    print("-"*60)

    #registered methods
    if registry is not None:
        reg_methods = set(registry.keys())
        print("Methods in METHOD_REGISTRY:")
        for m in sorted(reg_methods):
            print(f"  - {m}")
    else:
        reg_methods = []

    print("-"*50)

    if excluded:
        print("Excluded methods:")
        for m in excluded:
            print(f"  - {m}")

    # ---- Cross-checks ----
    if df_methods and reg_methods:

        missing_from_registry = df_methods - reg_methods
        missing_from_df = reg_methods - df_methods
        matched = df_methods & reg_methods

        print("Registry ↔ DF Check")

        if matched:
            print("\nMethods in BOTH DF and registry:")
            for m in sorted(matched):
                print(f"  - {m}")

        if missing_from_registry:
            print("\nMethods in DF but NOT in registry:")
            for m in sorted(missing_from_registry):
                print(f"  - {m}")

        if missing_from_df:
            print("\nMethods in registry but NOT present in DF:")
            for m in sorted(missing_from_df):
                print(f"  - {m}")

    print("=" * 60)





# -----------------------------
# Size statistics
# -----------------------------
def compute_size_stats(df, dp_col, group_col="method"):
    arith_stats = df.groupby(group_col)[dp_col].agg(["mean","std"]).reset_index().rename(
        columns={"mean":f"{dp_col} Arithmetic Mean","std":"Arithmetic Mean STD"}
    )

    def safe_geo_mean(x):
        x_pos = x[x>0]
        return gmean(x_pos) if len(x_pos)>0 else np.nan

    def safe_geo_std(x):
        x_pos = x[x>0]
        return gstd(x_pos) if len(x_pos)>0 else np.nan

    geo_stats = df.groupby(group_col)[dp_col].agg(geo_mean=safe_geo_mean, geo_std=safe_geo_std).reset_index().rename(
        columns={"geo_mean":f"{dp_col} Geo Mean","geo_std":"Geo Mean STD"}
    )

    return pd.merge(arith_stats, geo_stats, on=group_col, how="outer")


# -----------------------------
# Filtering
# -----------------------------
def filter_by_feret_ratio(df, feret_ratio_thresh=1.2, verbose=True):
    if 'feret (nm)' not in df.columns or 'min_feret (nm)' not in df.columns:
        raise ValueError("DataFrame must have 'feret (nm)' and 'min_feret (nm)' columns")
    df = df.copy()
    df['feret_ratio'] = df['feret (nm)']/df['min_feret (nm)']
    if verbose:
        print(df.groupby('method')['feret_ratio'].describe())
    return df[df['feret_ratio'] <= feret_ratio_thresh]


# -----------------------------
#PSD Plotting


def plot_psd_wrapper(df,
             methods=None,
             dia_type='equiv',
             #unit = 'nm',
             #bin_size= None,
             #plot_type = 'hist',
             #filter_small=False,
             #show_metrics = True,
             #log_scale = False,
             plot_kwargs=None, 
             **kwargs
             ):
    
    #Merge defaults with local overrides

    # Merge defaults with kwargs

    plot_kwargs = {**PLOT_DEFAULTS, **(plot_kwargs or {}),**kwargs}

    # # Figure kwargs
    # fig_kwargs = plot_kwargs.get("fig_kwargs", {"figsize": plot_kwargs.get("figsize", (12,8)),
    #                                             "dpi": plot_kwargs.get("dpi", 100)})
    
    # Prepare method styles
    methods_to_plot = methods or df["method"].unique()

    method_styles = kwargs.get("method_styles")

    if method_styles is None:
        method_styles = build_method_styles(
            methods_to_plot,
            plot_type="psd",
            method_registry=METHOD_REGISTRY
        )

    fig, ax =psd_plot_from_df(df, dia_type=dia_type,
                               methods=methods_to_plot, 
                               plot_kwargs=plot_kwargs, 
                               method_styles=method_styles)
    return fig, ax
    



def psd_plot_from_df(df,
                     dia_type='equiv',
                     methods=None,
                     method_styles=None,
                     plot_kwargs=None):
    """
    Core PSD plotting function with full kwargs control.
    """

    plot_kwargs = plot_kwargs or {}
    methods_to_plot = methods or df["method"].unique()

    # -------------------
    # General plot settings
    # -------------------
    figsize = plot_kwargs.get("figsize", DEFAULT_FIGSIZE)
    dpi = plot_kwargs.get("dpi", DEFAULT_DPI)
    title = plot_kwargs.get("title", None)
    show_legend = plot_kwargs.get("show_legend", True)
    log_scale = plot_kwargs.get("log_scale", False)

    fontsize_title = plot_kwargs.get("fontsize_title", 24)
    fontsize_axes = plot_kwargs.get("fontsize_axes", 20)
    fontsize_ticks = plot_kwargs.get("fontsize_ticks", 16)
    fontsize_legend = plot_kwargs.get("fontsize_legend", 14)

    # -------------------
    # PSD / histogram specific
    # -------------------
    plot_type = plot_kwargs.get("plot_type", "hist")  # 'hist', 'kde', 'bar'
    normalize = plot_kwargs.get("normalize", False)
    filter_small = plot_kwargs.get("filter_small", False)
    bin_size = plot_kwargs.get("bin_size", PP_BIN_SIZE)
    unit = plot_kwargs.get("unit", PP_UNIT)
    alpha = plot_kwargs.get("alpha", 0.8)
    linewidth = plot_kwargs.get("linewidth", 3)
    marker_size = plot_kwargs.get("marker_size", 100)
    marker_edgecolor = plot_kwargs.get("marker_edgecolor", "black")
    ylabel = 'Count' if plot_type in ['hist','bar'] and not normalize else 'Density (1/nm)'
    show_metrics = plot_kwargs.get("show_metrics", True)

    diameter_styles_default = plot_kwargs.get("diameter_styles", {
        f'dp ({unit})': {'linestyle':'-', 'marker':'o', 'label':'Area Equiv. Dia.'},
        f'feret ({unit})': {'linestyle':'--', 'marker':'s', 'label':'Max Feret Dia.'},
        f'min_feret ({unit})': {'linestyle':':', 'marker':'^', 'label':'Min Feret Dia.'},
    })

    # -------------------
    # Map methods -> diameter columns
    # -------------------
    if isinstance(dia_type, dict):
        method_diameter_cols = {
            method: [
                f"{col if col not in ['equiv','dp'] else 'dp'} ({unit})"
                if col.lower() in ['equiv','dp','feret','min_feret'] else f"{col} ({unit})"
                for col in cols
            ] for method, cols in dia_type.items()
        }
        x_label = f"Particle Diameter ({unit})"
    else:
        col_map = {'equiv':'dp', 'feret':'feret', 'min_feret':'min_feret'}
        dp_col_base = col_map.get(dia_type, 'dp')
        dp_col = f"{dp_col_base} ({unit})"
        method_diameter_cols = {m: [dp_col] for m in df['method'].unique()}

        x_label_map = {
            'equiv': f"Particle Area Equivalent Diameter ({unit})",
            'feret': f"Particle Feret Diameter ({unit})",
            'min_feret': f"Particle Min. Feret Diameter ({unit})"
        }
        x_label = x_label_map.get(dia_type, f"Particle Diameter ({unit})")

    multiple_diameter_types = any(len(cols) > 1 for cols in method_diameter_cols.values())

    # -------------------
    # Build long-format DF for plotting
    # -------------------
    plot_df_list = []
    for method, cols in method_diameter_cols.items():
        for col in cols:
            if col not in df.columns:
                continue
            temp = df[df['method'] == method].copy()
            temp = temp.rename(columns={col: 'dp_plot'})
            temp['method_plot'] = f"{method} ({col})" if len(cols) > 1 else method
            temp['_dp_orig_col'] = col
            plot_df_list.append(temp)
    plot_df = pd.concat(plot_df_list, ignore_index=True)

    # -------------------
    # Bin calculation
    # -------------------
    all_dp = plot_df['dp_plot'].dropna()
    min_dp, max_dp = all_dp.min(), all_dp.max()
    if bin_size is None:
        q75, q25 = np.percentile(all_dp, [75,25])
        iqr = q75 - q25
        n = len(all_dp)
        bin_size = 2*iqr / (n **(1/3)) if iqr != 0 else 1
    bins = np.arange(min_dp, max_dp + bin_size, bin_size)
    bin_text = f"Bin size: {bin_size:.2f} {unit}"

    # -------------------
    # Compute stats
    # -------------------
    stats_df = compute_size_stats(plot_df, 'dp_plot', group_col='method_plot')
    stats_df = stats_df.set_index('method_plot')

    # -------------------
    # Start plotting
    # -------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for method in methods_to_plot:
        method_rows = plot_df[plot_df['method'] == method]
        if method_rows.empty:
            continue

        for mplot in method_rows['method_plot'].unique():
            rows = method_rows[method_rows['method_plot'] == mplot]
            data = rows['dp_plot'].dropna()
            orig_col = rows['_dp_orig_col'].iloc[0]

            # small particle filter
            if filter_small:
                min_size = 0
                data = data[data >= min_size]

            # select style
            style = method_styles.get(method, {})
            color = style.get('color', 'black')
            label_name = style.get('label', method)

            if multiple_diameter_types:
                dstyle = diameter_styles_default.get(orig_col, {'linestyle':'-', 'marker':'o', 'label':orig_col})
                linestyle = dstyle.get('linestyle', '-')
                marker = dstyle.get('marker', 'o')
                dia_label = dstyle.get('label', orig_col)
            else:
                linestyle = style.get('linestyle', '-')
                marker = style.get('marker', 'o')
                dia_label = ''

            # legend
            gm = stats_df.loc[mplot, 'dp_plot Geo Mean'] if show_metrics else None
            gs = stats_df.loc[mplot, 'Geo Mean STD'] if show_metrics else None
            legend_label = f"{label_name} {dia_label}" + (f" ({gm:.2f} ± {gs:.2f})" if show_metrics else "")

            # -------------------
            # Plot type selection
            # -------------------
            if plot_type == 'hist':
                sns.histplot(
                    data,
                    bins=bins,
                    kde=False,
                    color=color,
                    label=legend_label,
                    element='poly',
                    stat='density' if normalize else 'count',
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker=marker,
                    fill=False,
                    alpha=alpha,
                    common_norm=False,
                    markeredgecolor=marker_edgecolor,
                    kde_kws={"bw_adjust":0.325}
                )
            elif plot_type == 'kde':
                sns.kdeplot(
                    data,
                    color=color,
                    label=legend_label,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    bw_method=0.2
                )
            elif plot_type == 'bar':
                sns.histplot(
                    data,
                    bins=bins,
                    kde=False,
                    color=color,
                    label=legend_label,
                    element='bars',
                    stat='density' if normalize else 'count',
                    linewidth=1,
                    edgecolor='k',
                    fill=True,
                    alpha=alpha,
                    common_norm=False
                )

    # -------------------
    # Final labels / legend
    # -------------------
    if title:
        ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel(x_label, fontsize=fontsize_axes)
    print(ylabel)
    ax.set_ylabel(ylabel, fontsize=fontsize_axes)

    if show_legend:
        ax.legend(
            loc=plot_kwargs.get("legend_loc", 'best'),
            fontsize=fontsize_legend,
            ncol=1,
            handlelength=3.0,
            handletextpad=0.8,
            borderaxespad=0.5,
            columnspacing=1.5,
            frameon=False
        )

    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks, direction='inout')
    if log_scale:
        ax.set_yscale('log')


    #axis limits

    xlim = plot_kwargs.get("xlim", None)
    ylim = plot_kwargs.get("ylim", None)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # add bin text
    ax.text(
        0.02, 0.95, bin_text,
        transform=ax.transAxes,
        fontsize=16,
        ha='left',
        va='top',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3)
    )

    plt.show()
    return fig, ax


def plot_metric_dist(aggs, 
                    included_methods=None,
                    metric="3D Rg [nm]" ,
                    #gt_method = None,
                    ax=None,
                    method_styles=None, 
                    
                    #show_legend=True,
                    #autoscale=True,
                    #pad_frac=0.1,
                    #xlim=None,ylim=None,legend_loc=False,
                    plot_kwargs=None):
    
    # Merge defaults and overrides
    plot_kwargs = plot_kwargs or {}
    gt_method = plot_kwargs.get("gt_method", REF_DATASET)
    autoscale = plot_kwargs.get("autoscale", True)
    pad_frac = plot_kwargs.get("pad_frac", 0.1)
    xlim = plot_kwargs.get("xlim", None)
    ylim = plot_kwargs.get("ylim", None)
    show_legend = plot_kwargs.get("show_legend", True)
    legend_loc = plot_kwargs.get("legend_loc", None)

    figsize = plot_kwargs.get("figsize", DEFAULT_FIGSIZE)
    dpi = plot_kwargs.get("dpi", DEFAULT_DPI)
    title = plot_kwargs.get("title", None)
    
    fontsize_title = plot_kwargs.get("fontsize_title", 24)
    fontsize_axes = plot_kwargs.get("fontsize_axes", 20)
    fontsize_ticks = plot_kwargs.get("fontsize_ticks", 16)
    fontsize_legend = plot_kwargs.get("fontsize_legend", 14)

    linewidth = plot_kwargs.get("linewidth", 3)
    marker_size = plot_kwargs.get("marker_size", 100)



    metric_labels = {
        "3D Rg [nm]": r'R$_g^{3D}$ [nm]',
        '3D Fractal Dimension': r'D$_f^{3D}$',
        'Rg [nm]':r'R$_g^{2D}$ [nm]',
        'fractal_dim':  r'D$_f^{2D}$',
        "dp [nm] geo mean": r'd$_p$ (geo mean)',
        'coverage_score': 'Coverage Score [%]'
    }

    #images = aggs["image"].unique()

    #handle dict mapping of methods to target class
    if isinstance(included_methods, dict):
        method_class_map = included_methods
        methods_to_plot = [m for m in included_methods.keys() if m != gt_method]
    else: 
        method_class_map = {m: None for m in (included_methods or aggs["method"].unique())}
        methods_to_plot = [m for m in method_class_map.keys() if m != gt_method]

    #get GT vals
    gt_df = aggs[aggs["method"] == gt_method][["image", metric]].drop_duplicates(subset="image").set_index("image")

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize, dpi=dpi)

    else: 
        fig =ax.figure

    


    #methods_to_plot = [m for m in (included_methods or aggs["method"].unique()) if m != gt_method]
    #print(methods_to_plot)

    all_vals = []

    for method in methods_to_plot:
        target_class = method_class_map.get(method, None)
        if target_class:
            method_df = aggs[(aggs["method"]==method) & (aggs["target_class"] == target_class)][["image", metric]]
        else:
            method_df = aggs[aggs["method"] == method][["image", metric]]

        if method_df.empty:
            print(f"Method: {method} has no data")
            continue

        #get rid of duplicates
        method_df = method_df.drop_duplicates(subset=["image"])
        
        #get predicted metric
        #method_df = aggs[aggs["method"]==method][["image", metric]]
        #merge GT and predicrted df on image index
        #merged = method_df.set_index("image").join(gt_df, lsuffix="_pred", rsuffix="_gt")

        merged = method_df.set_index("image").join(gt_df, lsuffix="_pred", rsuffix="_gt").dropna()

        if merged.empty:
            print(f"No data to plot for method: {method}")
            continue
        
        all_vals.extend(merged[f"{metric}_gt"].tolist())
        all_vals.extend(merged[f"{metric}_pred"].tolist())

        #remocve rows with missing vals
        #merged.dropna(inplace=True)

        # print(f"Method: {method}")
        # print(merged)

        style = method_styles.get(method, {}) if method_styles else {}

        #z = style.get("zorder",3)

        ax.scatter(merged[f"{metric}_gt"],
                merged[f"{metric}_pred"],
                label=style.get("label", f"{method} ({target_class or 'unspecified'})"),
                color=style.get("color", None),
                marker=style.get("marker", 'o'),
              
                s=style.get("size", marker_size),
                edgecolor=style.get("edgecolor",style.get("edgecolor", plot_kwargs.get("marker_edgecolor", "k"))),
                alpha=style.get("alpha",style.get("alpha", plot_kwargs.get("alpha", 0.8))),
                zorder=style.get("zorder",3)
        )
        
    
    
    if not all_vals:
        return fig, ax
    

    if autoscale:
        # autoscale with padding
        min_val = min(all_vals)
        max_val = max(all_vals)
        pad = (max_val - min_val) * pad_frac
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_ylim(min_val - pad, max_val + pad)

    else:
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
    # else:
    #     # fallback to some reasonable manual ranges
    #     if metric == "3D Rg [nm]":
    #         ax.set_xlim(0, 50)
    #         ax.set_ylim(0, 50)
    #     elif metric == "3D Fractal Dimension":
    #         ax.set_xlim(1.60, 1.85)
    #         ax.set_ylim(1.60, 1.85)
    #     elif metric == "dp [nm] geo mean":
    #         ax.set_xlim(0, 55)
    #         ax.set_ylim(0, 55)


    #perfect match line
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--', label="Perfect Match", linewidth=linewidth, zorder=1)

    pretty_metric = metric_labels.get(metric, metric)
    ax.set_xlabel(f"Ground Truth {pretty_metric}", fontsize=fontsize_axes)
    ax.set_ylabel(f"Predicted {pretty_metric}", fontsize=fontsize_axes)
    ax.tick_params(axis='both',labelsize=fontsize_ticks)
    ax.set_aspect('equal', adjustable='box')

    # if show_legend:
    #     ax.legend(
    #         fontsize=opts["fontsize_legend"],
    #         loc='upper left',
    #         bbox_to_anchor=(1.02, 1.0),
    #         frameon=False
    #     )

    # if show_legend:
   
    #     if legend_loc:
    #         ax.legend(fontsize=opts["fontsize_legend"], loc=legend_loc, frameon=False)
    #     else:
    #         ax.legend(fontsize=opts["fontsize_legend"], loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    # Axes-level legend (optional, usually False when using wrapper)
    if show_legend:
        if legend_loc:
            ax.legend(fontsize=fontsize_legend, loc=legend_loc, frameon=False)
        else:
            ax.legend(fontsize=fontsize_legend, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)

    ax.grid(True)
    return fig, ax


def plot_metric_agreement(aggs, metrics, 
                          included_methods=None, 
                       
                         # gt_method=REF_DATASET,
                            #method_styles=None, 
                          
                         # mode = "single", #or subplots to plot all in one figure,
                          #layout = None, # None | "square" | "row" | "column" | (rows, cols)
                          #subplot_size=8,
                          #show_legend=True,
                          #autoscale=True,
                          #xlim=None,
                          #ylim=None,
                          #legend_loc=None,
                          plot_kwargs=None,
                          **kwargs):
    
   # opts =  {**PLOT_DEFAULTS, **kwargs}
    plot_kwargs = {**PLOT_DEFAULTS, **(plot_kwargs or {}),**kwargs}

    mode = plot_kwargs.get("mode", "single")
    layout = plot_kwargs.get('layout', 'row')

    subplot_size = plot_kwargs.get("subplot_size",8)
    gt_method=plot_kwargs.get('gt_method', REF_DATASET)
    
    show_legend= plot_kwargs.get('show_legend', True)
    legend_loc = plot_kwargs.get('legend_loc', None)

    #print(included_methods)
    # ---- SAFE METHOD INFERENCE ----
    included_methods = included_methods or aggs["method"].unique()
    #print(included_methods)
    
    
    method_styles = kwargs.get("method_styles")
    if method_styles is None:
        method_styles = build_method_styles(
            included_methods,
            plot_type="scatter",
            method_registry=METHOD_REGISTRY
        )

    #--------------------------------
    # Mode 1: One figure per metric
    #------------------------------
    if mode == "single":
        figs = []
        axes_list = []
        #one figure per metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(subplot_size, subplot_size))

            plot_metric_dist(
                aggs=aggs,
                metric=metric,
                ax=ax,
                method_styles=method_styles,
                included_methods=included_methods,
                plot_kwargs=plot_kwargs
            )

            # if show_legend:
            #     handles, labels = ax.get_legend_handles_labels()
            #     # Figure legends cannot use 'best'
            #     loc = legend_loc if legend_loc else "center left"
            #     if loc == "best":
            #         loc = "center left"
            #     fig.legend(handles, labels,
            #                loc=loc,
            #                bbox_to_anchor=(1.02, 0.5) if legend_loc is None else None,
            #                frameon=False)
                
            plt.tight_layout()
            plt.show()

            figs.append(fig)
            axes_list.append(ax)

        return figs, axes_list

    #-----------------------------
    # Mode 2 - grid of subplots
    # ---------------------------
    elif mode == "subplots":

        #grid 
        num_metrics = len(metrics)

        #layout logic
        if layout is None or layout == "square":

            cols = math.ceil(math.sqrt(num_metrics))
            rows = math.ceil(num_metrics / cols)

        elif layout == "row":
            rows = 1
            cols = num_metrics

        elif layout == "column":
            rows = num_metrics
            cols = 1
        
        elif isinstance(layout, tuple):
            rows, cols = layout

        else: raise ValueError("Invalid layout option")



        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(cols * subplot_size, rows * subplot_size)
        )

        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, metric in enumerate(metrics):
            plot_metric_dist(
                aggs=aggs,
                metric=metric,
                #gt_method=gt_method,
                ax=axes[i],
                method_styles=method_styles,
                included_methods=included_methods,
                #show_legend=False,
                # autoscale=autoscale,
                #xlim=xlim,
                #ylim=ylim,
                #legend_loc=legend_loc,
                #**kwargs,
                plot_kwargs={**plot_kwargs, "show_legend": False}
                )

        #legend placement
        if show_legend:
            handles, labels = axes[0].get_legend_handles_labels()

            # Determine orientation based on layout
            if layout == "column" or (isinstance(layout, tuple) and layout[1] == 1):
                # column layout → put legend below
                fig.legend(
                    handles, labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=2,
                    frameon=False
                )
                fig.subplots_adjust(bottom=0.15)
            else:
                # square or row → put legend to the right
                fig.legend(
                    handles, labels,
                    loc="center left",
                    fontsize=plot_kwargs.get("fontsize_lenged", 14),
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False
                )
                fig.subplots_adjust(right=0.8)

        #hide unused axes (except legend slot)
        for j in range(num_metrics, len(axes) -1 ):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

        return fig, axes

    else:
        raise ValueError("mode must be 'subplots' or 'single'")
    
    



def plot_parity_metric(df,method_x, method_y, 
                       metric_col = 'dp (nm)',
                        error_margin=None,
                        min_val=None, max_val=None,
                        buffer_ratio = 0.05,
                        x_label=None, y_label=None,
                        method_registry=None,
                        plot_kwargs=None,
                        **kwargs):
    
    opts =  {**PLOT_DEFAULTS, **(plot_kwargs or {}),**kwargs}

    #Use default error margin if not provied
    error_margin = error_margin if error_margin is not None else opts["error_margin"]
    buffer_ratio = buffer_ratio if buffer_ratio is not None else opts["buffer_ratio"]
    
    index_cols = ["image", "PP #"] if "PP #" in df.columns else ["image"]

    df_subset = df[df['method'].isin([method_x, method_y])][index_cols+["method",metric_col]]
    #print(df_subset)

    df_pivot = df_subset.pivot(index=index_cols,columns="method", values = metric_col).dropna()
    #print(df_pivot)
    
    x_vals = df_pivot[method_x]
    y_vals = df_pivot[method_y]
    if min_val == None and max_val == None:
        all_vals = np.concatenate([x_vals.values, y_vals.values])
        data_min, data_max = all_vals.min(), all_vals.max()
        data_range = data_max-data_min
        buffer = data_range*buffer_ratio

        min_val = data_min - buffer if min_val is None else min_val
        max_val = data_max + buffer if max_val is None else max_val

    # Determine colors from registry or defaults
    method_registry = method_registry or {}
    color_x = method_registry.get(method_x, {}).get("color", opts.get("color_x", "tab:blue"))
    color_y = method_registry.get(method_y, {}).get("color", opts.get("color_y", "tab:orange"))
    marker_size = opts.get("marker_size", 50)
    

    fig, ax = plt.subplots(**opts.get("fig_kwargs", {"figsize": opts["figsize"], "dpi": opts["dpi"]}))
    #plot
    #plt.figure(figsize=(8,8))
    ax.scatter(x_vals, y_vals, s=marker_size, color=color_x, edgecolor=None, label=f"{method_x} vs {method_y}")
    
    #1:1 line

    
    line_range = np.linspace(min_val,max_val,100)
    ax.plot(line_range, line_range, 'k--', label = '1:1 Line')

    #error regions
    if error_margin:
        ax.fill_between(
            line_range,
            line_range * (1 - error_margin),
            line_range * (1 + error_margin),
            color='gray',
            alpha=0.3,
            label=f'±{int(error_margin * 100)}% Error Region'
        )



    # #plt.plot(line_range, line_range * (1+error_margin), 'r--', label=f'+{int(error_margin*100)}% Error')
    # #plt.plot(line_range, line_range * (1-error_margin), 'b--', label=f'-{int(error_margin*100)}% Error')
    # #ax.set_xlabel(x_label if x_label else f"{method_x} {metric_col}", fontsize = 25*1.3)
    # #ax.set_ylabel(y_label if y_label else f"{method_y} {metric_col}", fontsize = 25*1.3)
    # #plt.xlabel(f"Projected Mask $d_p$ [nm]",fontsize = 25*1.3)
    # #plt.ylabel(f" EDM $d_p$ [nm]",fontsize = 25*1.3)

    # ax.set_xlabel(x_label or f"{method_x} {metric_col}", fontsize=opts["fontsize_axes"])
    # ax.set_ylabel(y_label or f"{method_y} {metric_col}", fontsize=opts["fontsize_axes"])
    # ax.tick_params(axis='both', labelsize=opts["fontsize_ticks"])
    # ax.set_xlim(min_val, max_val)
    # ax.set_ylim(min_val, max_val)
    # ax.legend(fontsize=opts["fontsize_legend"])
    # ax.grid(opts["show_grid"])
    # ax.set_aspect('equal', adjustable='box')
    # plt.tight_layout()
    # plt.show()

    # Labels, limits, legend, grid
    x_label = kwargs.get("x_label") or f"{method_x} {metric_col}"
    y_label = kwargs.get("y_label") or f"{method_y} {metric_col}"
    ax.set_xlabel(x_label, fontsize=opts.get("fontsize_axes", 14))
    ax.set_ylabel(y_label, fontsize=opts.get("fontsize_axes", 14))
    ax.tick_params(axis='both', labelsize=opts.get("fontsize_ticks", 12))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(fontsize=opts.get("fontsize_legend", 12), loc=opts.get("legend_loc", "best"))
    ax.grid(opts.get("show_grid", True))
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


    return fig, ax






def add_percent_diff_per_image(df, metrics, gt_method='PROCI_Test', included_methods=None,debug=False):
    """
    Compute percent difference of metrics **for each image** relative to GT.
    
    This version is for **metric agreement plots**, where each image is a point.
    """
  


    df = df.copy()

    # Filter for included methods if given
    if included_methods is not None:
        df = df[df['method'].isin(included_methods)]
    gt_df = df[df['method'] == gt_method].set_index("image")

    for metric in metrics:
        error_col = f"{metric} error"
        errors = []

        for idx, row in df.iterrows():
            img = row["image"]
            method = row["method"]
            if img not in gt_df.index:
                if debug:
                    print(f"Image {img}, Method {method}: not in GT index, skipping")
                errors.append(None)
                continue
            true_val = gt_df.loc[img, metric] if metric in gt_df.columns else None
            pred_val = row[metric] if metric in df.columns else None

            if true_val is None or pd.isna(true_val) or true_val == 0 or pd.isna(pred_val):
                errors.append(None)
                if debug:
                    print(f"Skipping Image {img}, Method {method}: GT={true_val}, Pred={pred_val}")
            else:
                err = np.abs(pred_val - true_val) / true_val * 100
                errors.append(err)
                if debug:
                    print(f"Image {img}, Method {method}: GT={true_val}, Pred={pred_val}, %Error={err:.2f}")

        df[error_col] = errors

    return df


def get_means(df, metrics, use_pp_means=False, pp_df=None, gt_method='Finalv1_Test'):
    summary = []

    for method in df["method"].unique():
        print(method)
        method_df = df[df["method"]==method]

        row = {"Method": method}

        for metric in metrics:
            #logic for gathering mean dp data from pp_df

            if use_pp_means and pp_df is not None and metric in ["dp [nm]", "dp [nm] geo mean"]:
            # if metric == "dp [nm]" and use_pp_means and pp_df is not None:
                
                col_name = metric if metric == "dp [nm]" else "dp [nm] geo mean"
                print(f"Using pp_df column: {col_name}")
                print("Available methods in pp_df:", pp_df["method"].unique())
                
                pp_vals = pp_df[pp_df["method"]==method][col_name]
                row[f"Mean {metric}"] = pp_vals.mean()

                if method != gt_method:
                    gt_vals = pp_df[pp_df["method"]==gt_method]["dp (nm)"]
                    gt_mean = gt_vals.mean()
                    if gt_mean !=0:
                        errors = abs(pp_vals.mean()-gt_mean)/gt_mean
                        row[f"{metric} Error Mean"] = errors
                        #row[f"{metric} Error Std"] = pp_vals.std()
                        #row[f"{metric} Error Median"] = errors.median()

            else:
                row[f"Mean {metric}"] = method_df[metric].mean()
                error_col = f"{metric} error"
                #print(error_col)
                if error_col in method_df.columns:
                    row[f"{metric} Error Mean"] = method_df[error_col].mean()
                    row[f"{metric} Error Std"] = method_df[error_col].std()
                    row[f"{metric} Error Median"] = method_df[error_col].median()
                else:
                    print(f"Missing column {error_col}")

           

        summary.append(row)


        result_df = pd.DataFrame(summary)
        #reorder columns
        column_order = []
        for metric in metrics:
            column_order.append(f"Mean {metric}")
            if f"{metric} Error Mean" in result_df.columns:
                column_order.append(f"{metric} Error Mean")
            if f"{metric} Error Std" in result_df.columns:
                column_order.append(f"{metric} Error Std")
            if f"{metric} Error Median" in result_df.columns:
                column_order.append(f"{metric} Error Median")

        result_df = result_df[["Method"]+column_order]
    return result_df








def _plot_single_err_boxplot(df, metric,
                            method_styles, 
                            opts=None,
                            gt_method=None,
                            ax =None,
                            title=None,
                            hardcoded_order=None,
                            included_methods=None):
    opts = opts or PLOT_DEFAULTS
    
    df.copy()

    if included_methods is not None:
        df = df[df['method'].isin(included_methods)]
    if gt_method:
        df = df[df['method'] != gt_method]

    metric_labels = {
        "3D Rg [nm] error": r'R$_g^{3D}$ % Error',
        "3D Fractal Dimension error": r'D$_f^{3D}$ % Error',
        'dp [nm] geo mean error': r'geometric mean d$_p$ Error',
        'coverage_score error': 'Coverage % Error'
    }

    pretty_metric = metric_labels.get(metric, metric)

    # Prepare DataFrame for plotting
    temp_df = df[['method', metric]].dropna().copy()
    temp_df = temp_df.rename(columns={metric: 'Error'})
    temp_df['Method Label'] = temp_df['method'].map(lambda x: method_styles.get(x, {}).get('label', x))


    palette = {v['label']: v.get('color', None) for v in method_styles.values()}

     # Determine x_order
    if hardcoded_order:
        x_order = [label for label in hardcoded_order if label in temp_df['Method Label'].unique()]
    else:
        x_order = temp_df['Method Label'].unique()
    
    # Create figure if no ax given
    if ax is None:
        fig, ax = plt.subplots(figsize=opts.get("figsize", (12, 8)), dpi=opts.get("dpi", 100))
    else:
        fig = ax.figure

    #fig, ax = plt.subplots(figsize=opts.get("figsize", (12, 8)), dpi=opts.get("dpi", 100))

    ax = sns.boxplot(
        data=temp_df,
        x='Method Label',
        y='Error',
        hue='Method Label',
        palette=palette,
        showfliers=True,
        width=0.7,
        linewidth=opts.get("linewidth", 2),
        flierprops={
            'marker':  opts.get("flier_marker", 'o'),
            'markersize': opts.get("marker_size", 5),
            'markerfacecolor': 'black',
            'linestyle': 'none',
            'linewidth': 1.5
        },
        order=x_order,
        ax=ax
    )

   

    
    # Axis formatting
    ax.tick_params(axis='x', rotation=30, labelsize=opts.get('fontsize_ticks', 20))
    ax.tick_params(axis='y', labelsize=opts.get('fontsize_ticks', 20))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.set_xlabel('', fontsize=opts.get('fontsize_axes', 24))
    ax.set_ylabel(pretty_metric, fontsize=opts.get('fontsize_axes', 24))
    if title:
        ax.set_title(title, fontsize=opts.get('fontsize_title', 28))

    # Remove legend (labels are below boxes)
    ax.legend_.remove() if ax.get_legend() else None

    return fig, ax


def plot_error_boxplots(df, metrics, included_methods=None,
                        hardcoded_order=None,  
                        debug=False,
                        plot_kwargs=None,
                        **kwargs
                        ):
    
    plot_kwargs = {**PLOT_DEFAULTS, **(plot_kwargs or {}),**kwargs}
    mode = plot_kwargs.get("mode", "single")
    gt_method = plot_kwargs.get("gt_method", REF_DATASET)
    title=plot_kwargs.get("title", None)

    layout = plot_kwargs.get('layout', 'rows')

    subplot_size = plot_kwargs.get("subplot_size", 8)
    df=df.copy()


    included_methods = included_methods or df["method"].unique()

    method_styles = kwargs.get("method_styles")
    if method_styles is None:
        method_styles = build_method_styles(
            included_methods,
            plot_type="error",
            method_registry=METHOD_REGISTRY
        )

        
    
    metrics_error_cols = [f"{m} error" for m in metrics]
    missing_errors = any(col not in df.columns for col in metrics_error_cols)
    if missing_errors:
        df = add_percent_diff_per_image(df, metrics, gt_method, included_methods, debug=debug)

    # Prepare metrics for plotting
    metrics = [f"{m} error" for m in metrics]

    if mode=='single':

        figs, axes_list = [],[]
        for metric in metrics:
            fig, ax = _plot_single_err_boxplot(
                df,
                metric,
                method_styles,
                opts=plot_kwargs,
                title=title,
                gt_method=gt_method,
                hardcoded_order=hardcoded_order,
                included_methods=included_methods
            )
            plt.tight_layout()
            plt.show()
            figs.append(fig)
            axes_list.append(ax)
        return df, figs, axes_list
    
    # Subplots
    elif mode == "subplots":
        num_metrics = len(metrics)
        if layout is None or layout == 'square':
            cols = math.ceil(math.sqrt(num_metrics))
            rows = math.ceil(num_metrics / cols)
        elif layout == 'row':
            rows = 1
            cols = num_metrics
        elif layout == 'column':
            rows = num_metrics
            cols = 1
        elif isinstance(layout, tuple):
            rows, cols = layout
        else:
            raise ValueError("Invalid layout option")

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*subplot_size, rows*subplot_size))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, metric in enumerate(metrics):
            _plot_single_err_boxplot(
                df,
                metric,
                method_styles,
                opts=plot_kwargs,
                gt_method=gt_method,
                title=None,
                hardcoded_order=hardcoded_order,
                included_methods=included_methods,
                ax=axes[i]
            )

        # Hide unused axes
        for j in range(num_metrics, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()
        return df, fig, axes

    else:
        raise ValueError("mode must be 'single' or 'subplots'")


def filter_aggs(aggs, methods, gt_method=None, auto_exclude_gt=True):
    """
    Standardized filtering for aggregation dataframe
    """

    df = aggs.copy()
    
    #infer GT if not provided
    if gt_method is None:
        gt_method = REF_DATASET #global default

    df = df[df["method"].isin(methods)]


    if auto_exclude_gt:
        df = df[df["method"] != gt_method]

    return df 

def create_method_legend_figure(methods_to_plot, figsize=(16,1), fontsize=32, exclude_methods=None ):
    
    legend_elements = []

    #Optional exclusion
    if exclude_methods is not None:
        methods_to_plot = [
            m for m in methods_to_plot
            if m not in exclude_methods
        ]

    #build legend entire
    for method_name in methods_to_plot:
        if method_name not in METHOD_REGISTRY:
            print(f"[WARN] {method_name} not found in METHOD_REGISTRY, using defaults")
            method_info= {}
            scatter_style = {}
        else:
            method_info = METHOD_REGISTRY[method_name]
            scatter_style = method_info.get("scatter_style", {})
        
        #styling extraction
        marker = scatter_style.get("marker", "o")
        color = scatter_style.get("color", method_info.get("color", "gray"))
        marker_size = scatter_style.get("size", 100)
        size = math.sqrt(marker_size)
        edgecolor = scatter_style.get("edgecolor", "black")
        alpha = scatter_style.get("alpha", 0.8)
        label = method_info.get("label", method_name)

        legend_elements.append(Line2D(
            [0],[0],
            marker=marker,
            color=color,
            label=label,
            markersize=size,
            markeredgecolor=edgecolor,
            linestyle="None",
            alpha=alpha
        ))

    #Add perfect match line
    legend_elements.append(Line2D(
        [0],[0],
        color = 'black',
        linestyle = '--',
        linewidth = 2,
        label = 'Perfect Match'
    ))

    #Render
    fig, ax = plt.subplots(figsize=figsize)

    ax.legend(
        handles = legend_elements,
        loc = 'center',
        ncol = len(legend_elements),
        fontsize = fontsize,
        frameon=False
    )

    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return fig, ax


def add_conf_to_method(df, conf_col='conf_threshold', new_col='method'):
    df = df.copy()

    df['method_orig'] = df['method']

    df[new_col] = df.apply(
        lambda r: f"{r['method']}_{r[conf_col]:.1f}"
        if conf_col in r and not pd.isna(r[conf_col])
        else r['method'],
        axis=1
    )

    return df

def extract_conf_label(method_name, base_model=None):
    last = method_name.split("_")[-1]

    try:
        conf = float(last)
        return f"{conf:.1f}"
    except ValueError:
        return method_name

def get_conf_methods(df, model_name, thresholds=None, include_ref=False, ref_method=REF_DATASET):

    """
    Return method names corresponding to confidence thresholds for a given model.
    """    

    rows = df[df["method_orig"] == model_name]

    if thresholds is not None:
        rows = rows[rows["conf_thresholds"].isin(thresholds)]

    methods = sorted(rows["method"].unique())

    if include_ref:
        methods = [ref_method] + methods

    return methods


def list_available_sweeps(root, verbose=True):

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Sweep root does not exist: {root}")
    
    models = []

    for subfolder in root.iterdir():
        if subfolder.is_dir():
            csv_path = subfolder / "mAP_sweep.csv"

            if csv_path.exists():
                models.append(subfolder.name)

    models.sort()

    if verbose:
        print("Available mAP sweeps:")
        for m in models:
            print(f"  - {m}")


    return models
   
def load_mAP_sweep(model_name, root):
    root = Path(root)

    csv_path = root / model_name / "mAP_sweep.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No sweep file found at: {csv_path}")

    return pd.read_csv(csv_path)



def plot_map_sweep(
        models,
        root,
        datasets=None,
        stages=None,
        metric ="AP_50",
        aggregate=False,
        ax=None,
        stage_brackets=True,
        stage_linestyles=False 
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Linestyle map for stages
    linestyle_map = {
        "heads": "-",
        "5plus": "--",
        "4plus": ":",
        "full": "-."
    }

    # To offset brackets for multiple runs
    run_offset_step = 0.03
    

    for run_idx, model in enumerate(models):
        df = load_mAP_sweep(model, root)

        #filter datasets and stages
        if datasets is not None:
            df = df[df["Dataset"].isin(datasets)]
        if stages is not None:
            df = df[df['Stage'].isin(stages)]

        if df.empty:
            print(f"No data to plot for model {model} with the selected filters.")
            continue

        df = df.sort_values("Epoch")

        #Determine y-axis limits early for bracket heigh
        ymin, ymax =df[metric].min(), df[metric].max()

        y_range=ymax-ymin

        bracket_low = ymin - 0.05 * y_range +run_idx * run_offset_step * y_range
        bracket_high = ymin + 0.05 * y_range + run_idx * run_offset_step * y_range

                # Aggregate or per-dataset plotting
        if aggregate:
            grouped = df.groupby("Epoch")[metric].mean()
            ax.plot(
                grouped.index,
                grouped.values,
                label=f"{model} (avg)",
                linewidth=2
            )
        else: #per dataset

            for dataset in df["Dataset"].unique():

                sub = df[df["Dataset"] == dataset]

                if stage_linestyles:
                    #plot stage by stage with linestyles
                    for stage, group in sub.groupby("Stage"):
                        ax.plot(
                            group["Epoch"],
                            group[metric],
                            linestyle = linestyle_map.get(stage,"-"),
                            label = f"{model} - {dataset} - {stage}"
                        )

                else:
                    ax.plot(
                        sub["Epoch"],
                        sub[metric],
                        linestyle="-",
                        label=f"{model} - {dataset}"
                    )

                if stage_brackets:
                    #draw stage brackets

                    for stage, group in sub.groupby("Stage"):
                        start_epoch = group["Epoch"].min()
                        end_epoch=group["Epoch"].max()
                        mid_epoch = (start_epoch+end_epoch)/2


                        # Stage-specific min/max
                        stage_ymin = group[metric].min()
                        stage_ymax = group[metric].max()
                        y_span = stage_ymax - stage_ymin

                        # Small padding for the bracket
                        pad = 0.05 * y_span if y_span > 0 else 0.02 *stage_ymax

                        bracket_bottom = stage_ymin - pad
                        bracket_top = stage_ymax + pad

                        #Vertocal lines for stage start/end
                        ax.vlines(start_epoch, bracket_bottom, bracket_top, color = 'grey',alpha=0.5)
                        ax.vlines(end_epoch, bracket_bottom, bracket_top, color='grey', alpha=0.5)

                        # Stage label above bracket
                        ax.text(
                            mid_epoch,
                            (bracket_bottom + bracket_top) / 2,  # center of bracket
                            stage,
                            ha='center',
                            va='center',  # center vertically
                            fontsize=9,
                            color='black',
                    
                        )


               
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Sweep Comparison")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return ax

def print_best_epochs(models, root, dataset_name, per_stage=True):
    """
    For a given dataset, prints the best epoch for each model:
      - Best AP_50 epoch and its AP_range
      - Best AP_range epoch and its AP_50
    """
    for model in models:
        df = load_mAP_sweep(model, root)

        # filter to dataset
        df = df[df["Dataset"] == dataset_name]

        if df.empty:
            print(f"No data for model {model} on dataset {dataset_name}")
            continue

        print(f"\nModel: {model} | Dataset: {dataset_name}")

        # Best AP_50 epoch
        best_ap50_row = df.loc[df["AP_50"].idxmax()]
        print(f"  Best AP_50: Epoch {int(best_ap50_row['Epoch'])} → "
              f"AP_50: {best_ap50_row['AP_50']:.4f}, "
              f"AP_range: {best_ap50_row['AP_range']:.4f}")

        # Best AP_range epoch
        best_aprange_row = df.loc[df["AP_range"].idxmax()]
        print(f"  Best AP_range: Epoch {int(best_aprange_row['Epoch'])} → "
              f"AP_50: {best_aprange_row['AP_50']:.4f}, "
              f"AP_range: {best_aprange_row['AP_range']:.4f}")


        if per_stage:
            print("  Per-stage bests:")
            for stage, group in df.groupby("Stage"):
                stage_best_ap50 = group.loc[group["AP_50"].idxmax()]
                stage_best_aprange = group.loc[group["AP_range"].idxmax()]

                print(f"    Stage: {stage}")
                print(f"      Best AP_50: Epoch {int(stage_best_ap50['Epoch'])} → "
                      f"AP_50: {stage_best_ap50['AP_50']:.4f}, "
                      f"AP_range: {stage_best_ap50['AP_range']:.4f}")
                print(f"      Best AP_range: Epoch {int(stage_best_aprange['Epoch'])} → "
                      f"AP_50: {stage_best_aprange['AP_50']:.4f}, "
                      f"AP_range: {stage_best_aprange['AP_range']:.4f}")