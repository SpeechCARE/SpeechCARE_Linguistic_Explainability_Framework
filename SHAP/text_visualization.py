
import json
import random
import string
import warnings
import numpy as np
from scipy import linalg
from warnings import warn
from matplotlib.colors import LinearSegmentedColormap


try:
    from IPython.display import HTML
    from IPython.display import display as ipython_display

    have_ipython = True
except ImportError:
    have_ipython = False


def _prepare_colorarray(arr, *, channel_axis=-1):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.shape[channel_axis] != 3:
        msg = f"the input array must have size 3 along `channel_axis`, got {arr.shape}"
        raise ValueError(msg)

    return arr.astype(np.float64)


xyz_from_rgb = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)

rgb_from_xyz = linalg.inv(xyz_from_rgb)


def _convert(matrix, arr):
    """Do the color space conversion.

    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : (..., C=3, ...) array_like
        The input array. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The converted array. Same dimensions as input.
    """
    arr = _prepare_colorarray(arr)

    return arr @ matrix.T.astype(arr.dtype)


def xyz2rgb(xyz):
    """XYZ to RGB color space conversion.

    Parameters
    ----------
    xyz : (..., C=3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr


def _lab2xyz(lab):
    """Convert CIE-LAB to XYZ color space.

    Internal function for :func:`~.lab2xyz` and others. In addition to the
    converted image, return the number of invalid pixels in the Z channel for
    correct warning propagation.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in XYZ format. Same dimensions as input.
    n_invalid : int
        Number of invalid pixels in the Z channel after conversion.
    """
    arr = _prepare_colorarray(lab, channel_axis=-1).copy()

    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    invalid = np.atleast_1d(z < 0).nonzero()
    n_invalid = invalid[0].size
    if n_invalid != 0:
        # Warning should be emitted by caller
        if z.ndim > 0:
            z[invalid] = 0
        else:
            z = 0

    out = np.stack([x, y, z], axis=-1)

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = np.array([0.95047, 1.0, 1.08883])
    out *= xyz_ref_white
    return out, n_invalid


def lab2rgb(lab):
    """Convert image in CIE-LAB to sRGB color space.

    Parameters
    ----------
    lab : (..., C=3, ...) array_like
        The input image in CIE-LAB color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in sRGB color space, of same shape as input.

    Raises
    ------
    ValueError
        If `lab` is not at least 2-D with shape (..., C=3, ...).
    """
    xyz, n_invalid = _lab2xyz(lab)
    if n_invalid != 0:
        warn(
            "Conversion from CIE-LAB, via XYZ to sRGB color space resulted in "
            f"{n_invalid} negative Z values that have been clipped to zero",
            stacklevel=3,
        )
    return xyz2rgb(xyz)


def lch2lab(lch):
    """Convert image in CIE-LCh to CIE-LAB color space.

    CIE-LCh is the cylindrical representation of the CIE-LAB (Cartesian) color
    space.

    Parameters
    ----------
    lch : (..., C=3, ...) array_like
        The input image in CIE-LCh color space.
        Unless `channel_axis` is set, the final dimension denotes the CIE-LAB
        channels.
        The L* values range from 0 to 100;
        the C values range from 0 to 100;
        the h values range from 0 to ``2*pi``.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in CIE-LAB format, of same shape as input.
    """
    lch = _prepare_lab_array(lch)

    c, h = lch[..., 1], lch[..., 2]
    lch[..., 1], lch[..., 2] = c * np.cos(h), c * np.sin(h)
    return lch


def _prepare_lab_array(arr):
    """Ensure input for lab2lch and lch2lab is well-formed.

    Input array must be in floating point and have at least 3 elements in the
    last dimension. Returns a new array by default.
    """
    arr = np.asarray(arr, dtype=np.float64)
    shape = arr.shape
    if shape[-1] < 3:
        raise ValueError("Input image has less than 3 channels.")
    return arr

#######################################################################################################


def lch2rgb(x: list[float]) -> np.ndarray:
    return lab2rgb(lch2lab([[x]]))[0][0]

def update_colors(red_lch, l_mid, blue_lch, light_blue, light_red, transparent_colors):
    """
    Update global color variables based on the given input parameters.

    Parameters:
    red_lch : list
        LCh values for the red color.
    l_mid : float
        Middle lightness value for the color scale.
    blue_lch : list
        LCh values for the blue color.
    light_blue : list
        RGB values for the light blue (as white alternative).
    light_red : list
        RGB values for the light red (as green alternative).
    transparent_colors : dict
        Dictionary containing transparent color values.
    """
    global red_rgb, white_rgb, blue_rgb, gray_rgb, light_blue_rgb, light_red_rgb
    global red_blue_transparent, red_blue_circle, red_transparent_blue, transparent_blue, transparent_red

    # Convert LCh colors to RGB
    gray_lch = [55.0, 0.0, 0.0]
    red_rgb = lch2rgb(red_lch)
    white_rgb = np.array([1.0, 1.0, 1.0])
    blue_rgb = lch2rgb(blue_lch)
    gray_rgb = lch2rgb(gray_lch)

    # Update light colors
    light_blue_rgb = np.array(light_blue) / 255
    light_red_rgb = np.array(light_red) / 255

    # Define red-blue color map with transparency
    nsteps = 100
    l_vals = list(np.linspace(blue_lch[0], l_mid, nsteps // 2)) + list(np.linspace(l_mid, red_lch[0], nsteps // 2))
    c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
    h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)

    reds, greens, blues, alphas = [], [], [], []
    for pos, l, c, h in zip(np.linspace(0, 1, nsteps), l_vals, c_vals, h_vals):
        lch = [l, c, h]
        rgb = lch2rgb(lch)
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))

    red_blue_transparent = LinearSegmentedColormap(
        "red_blue_transparent",
        {"red": reds, "green": greens, "blue": blues, "alpha": [(a[0], 0.5, 0.5) for a in alphas]},
    )

    # Circular color map for categorical coloring
    reds, greens, blues, alphas = [], [], [], []
    for pos, c, h in zip(np.linspace(0, 0.5, nsteps), c_vals, h_vals):
        lch = [blue_lch[0], c, h]
        rgb = lch2rgb(lch)
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))
    for pos, c, h in zip(np.linspace(0.5, 1, nsteps), c_vals[::-1], h_vals[::-1]):
        lch = [blue_lch[0], c, h]
        rgb = lch2rgb(lch)
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))

    red_blue_circle = LinearSegmentedColormap(
        "red_blue_circle", {"red": reds, "green": greens, "blue": blues, "alpha": alphas}
    )

    # Define transparent color maps
    colors = [
        (transparent_colors["white"][0] / 255, transparent_colors["white"][1] / 255, transparent_colors["white"][2] / 255, j)
        for j in np.linspace(0, 1, 100)
    ]
    transparent_blue = LinearSegmentedColormap.from_list("transparent_blue", colors)

    colors = [
        (transparent_colors["red"][0] / 255, transparent_colors["red"][1] / 255, transparent_colors["red"][2] / 255, j)
        for j in np.linspace(0, 1, 100)
    ]
    transparent_red = LinearSegmentedColormap.from_list("transparent_red", colors)

    # Define red-transparent-blue gradient
    colors = [
        (transparent_colors["white"][0] / 255, transparent_colors["white"][1] / 255, transparent_colors["white"][2] / 255, j)
        for j in np.linspace(1, 0, 100)
    ] + [
        (transparent_colors["green"][0] / 255, transparent_colors["green"][1] / 255, transparent_colors["green"][2] / 255, j)
        for j in np.linspace(0, 1, 100)
    ]
    red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

#######################################################################################################

# TODO: we should support text output explanations (from models that output text not numbers), this would require the force
# the force plot and the coloring to update based on mouseovers (or clicks to make it fixed) of the output text
def text(
    shap_values,
    num_starting_labels=0,
    grouping_threshold=0.01,
    separator="",
    xmin=None,
    xmax=None,
    cmax=None,
    display=True,
    target_label=None,
):

    def values_min_max(values, base_values):
        """Used to pick our axis limits."""
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    # loop when we get multi-row inputs
    if len(shap_values.shape) == 2 and (shap_values.output_names is None or isinstance(shap_values.output_names, str)):
        print("In len 2")
        print(shap_values.output_names)
        if shap_values.output_names == "control":
            transparent_colors = {
                                    "white": [230, 242, 244],
                                    "red": [50, 181, 65],# green [0, 150, 60], #green

                                    "green": [50, 181, 65]# green [0, 150, 60], #green
                                }

            update_colors(
                red_lch=[64.9, 74.9, 140.5], #green [54.0, 90.0, 2.0943951], #green
                l_mid=40.0,
                blue_lch=[94.7, 4.3, 213.2], #white
                light_blue=[230, 242, 244], #white
                light_red=[111, 211, 110], #green [144.0, 238, 144], #green
                transparent_colors=transparent_colors
            )
        elif shap_values.output_names == "mci":
            # print("setting to blue")
            transparent_colors = {
                                    "white": [230, 242, 244],
                                    "red": [255, 195, 0], # [255, 216, 107], #yellow [0, 128, 169], #blue
                                    "green": [255, 195, 0], # [255, 216, 107] #yellow [0, 128, 169], #blue
                                }

            update_colors(
                red_lch=[82.0, 84.1, 84.3], #[87.7, 70.8, 88.5], #yellow [80.2, 49.1, 133.0],
                l_mid=40.0,
                blue_lch=[94.7, 4.3, 213.2], #white
                light_blue=[230, 242, 244], #white
                light_red=[255, 212, 122], #[255, 232, 147], #light yellow [127.0, 196, 252], #light blue
                transparent_colors=transparent_colors
            )
        else:
            # print("setting to red")
            transparent_colors = {
                                "white": [230, 242, 244],
                                "red": [171, 105, 212], #purple [255, 13, 87], #red
                                "green": [171, 105, 212] #purple [255, 13, 87], #red
                            }

            update_colors(
                red_lch=[55.7, 63.8, 315.1], #purple [54.0, 90.0, 0.35470565 + 2 * np.pi], #red
                l_mid=40.0,
                blue_lch=[94.7, 4.3, 213.2], #white
                light_blue=[230, 242, 244], #white
                light_red=[189, 144, 214], #ligh purple [255.0, 127, 167], #light red
                transparent_colors=transparent_colors
            )
        xmin = 0
        xmax = 0
        cmax = 0

        for i, v in enumerate(shap_values):
            values, clustering = unpack_shap_explanation_contents(v)
            tokens, values, group_sizes = process_shap_values(v.data, values, grouping_threshold, separator, clustering)

            if i == 0:
                xmin, xmax, cmax = values_min_max(values, v.base_values)
                continue

            xmin_i, xmax_i, cmax_i = values_min_max(values, v.base_values)
            if xmin_i < xmin:
                xmin = xmin_i
            if xmax_i > xmax:
                xmax = xmax_i
            if cmax_i > cmax:
                cmax = cmax_i
        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
    <br>
    <hr style="height: 1px; background-color: #e6f2f4; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
    <div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #e6f2f4; padding: 5px; color: #999; font-family: monospace"></div>
    </div>
     <div align='center' style='background: rgba(230.0, 242.0, 244.0, 1.0); padding-bottom:25px; font-family: Arial'>
                """
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
        out += '</div>'
        if display:
            print('displaying 85')
            _ipython_display_html(out)
            return
        else:
            return out

    if len(shap_values.shape) == 2 and shap_values.output_names is not None:
        print("in len 2 and names not none")
        # print(output_names)
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            values, clustering = unpack_shap_explanation_contents(shap_values[:, i])
            tokens, values, group_sizes = process_shap_values(
                shap_values[:, i].data, values, grouping_threshold, separator, clustering
            )

            # if i == 0:
            #     xmin, xmax, cmax = values_min_max(values, shap_values[:,i].base_values)
            #     continue

            xmin_i, xmax_i, cmax_i = values_min_max(values, shap_values[:, i].base_values)
            if xmin_computed is None or xmin_i < xmin_computed:
                xmin_computed = xmin_i
            if xmax_computed is None or xmax_i > xmax_computed:
                xmax_computed = xmax_i
            if cmax_computed is None or cmax_i > cmax_computed:
                cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = f"""<div align='center' style='background: rgba(230.0, 242.0, 244.0, 1.0); padding-bottom:25px; font-family: Arial'>
<script>
    document._hover_{uuid} = '_tp_{uuid}_output_0';
    document._zoom_{uuid} = undefined;
    function _output_onclick_{uuid}(i) {{
        var next_id = undefined;

        if (document._zoom_{uuid} !== undefined) {{
            document.getElementById(document._zoom_{uuid}+ '_zoom').style.display = 'none';

            if (document._zoom_{uuid} === '_tp_{uuid}_output_' + i) {{
                document.getElementById(document._zoom_{uuid}).style.display = 'block';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = '3px solid #000000';
            }} else {{
                document.getElementById(document._zoom_{uuid}).style.display = 'none';
                document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = 'none';
            }}
        }}
        if (document._zoom_{uuid} !== '_tp_{uuid}_output_' + i) {{
            next_id = '_tp_{uuid}_output_' + i;
            document.getElementById(next_id).style.display = 'none';
            document.getElementById(next_id + '_zoom').style.display = 'block';
            document.getElementById(next_id+'_name').style.borderBottom = '3px solid #000000';
        }}
        document._zoom_{uuid} = next_id;
    }}
    function _output_onmouseover_{uuid}(i, el) {{
        if (document._zoom_{uuid} !== undefined) {{ return; }}
        if (document._hover_{uuid} !== undefined) {{
            document.getElementById(document._hover_{uuid} + '_name').style.borderBottom = 'none';
            document.getElementById(document._hover_{uuid}).style.display = 'none';
        }}
        document.getElementById('_tp_{uuid}_output_' + i).style.display = 'block';
        el.style.borderBottom = '3px solid #000000';
        document._hover_{uuid} = '_tp_{uuid}_output_' + i;
    }}
</script>
<div style=\"color: rgb(120,120,120); font-size: 12px;\">outputs</div>"""
        output_values = shap_values.values.sum(0) + shap_values.base_values
        output_max = np.max(np.abs(output_values))
        ############################*******************************############################*******************************
        for i, name in enumerate(shap_values.output_names):
            if target_label is not None and name != target_label:
                continue  # SKIP processing for irrelevant labels
            if name == "control":
                transparent_colors = {
                                        "white": [230, 242, 244],
                                        "red": [50, 181, 65],# green [0, 150, 60], #green

                                        "green": [50, 181, 65]# green [0, 150, 60], #green
                                    }

                update_colors(
                    red_lch=[64.9, 74.9, 140.5], #green [54.0, 90.0, 2.0943951], #green
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[111, 211, 110], #green [144.0, 238, 144], #green
                    transparent_colors=transparent_colors
                )
            elif name == "mci":
                # print("setting to blue")
                transparent_colors = {
                                        "white": [230, 242, 244],
                                        "red": [255, 195, 0], # [255, 216, 107], #yellow [0, 128, 169], #blue
                                        "green": [255, 195, 0], # [255, 216, 107] #yellow [0, 128, 169], #blue
                                    }

                update_colors(
                    red_lch=[82.0, 84.1, 84.3], #[87.7, 70.8, 88.5], #yellow [80.2, 49.1, 133.0],
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[255, 212, 122], #[255, 232, 147], #light yellow [127.0, 196, 252], #light blue
                    transparent_colors=transparent_colors
                )
            else:
                # print("setting to red")
                transparent_colors = {
                                    "white": [230, 242, 244],
                                    "red": [171, 105, 212], #purple [255, 13, 87], #red
                                    "green": [171, 105, 212] #purple [255, 13, 87], #red
                                }

                update_colors(
                    red_lch=[55.7, 63.8, 315.1], #purple [54.0, 90.0, 0.35470565 + 2 * np.pi], #red
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[189, 144, 214], #ligh purple [255.0, 127, 167], #light red
                    transparent_colors=transparent_colors
                )
            scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
            color = red_transparent_blue(scaled_value)
            # color = green_white_transparent(scaled_value)
            # ********************************************************************************
            # color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
            color = (float(color[0] * 255), float(color[1] * 255), float(color[2] * 255), float(color[3]))
            # '#dddddd' if i == 0 else '#ffffff' border-bottom: {'3px solid #000000' if i == 0 else 'none'};
            out += f"""
<div style="display: inline; border-bottom: {'3px solid #000000' if i == 0 else 'none'}; background: rgba{color}; border-radius: 3px; padding: 0px" id="_tp_{uuid}_output_{i}_name"
    onclick="_output_onclick_{uuid}({i})"
    onmouseover="_output_onmouseover_{uuid}({i}, this);">{name}</div>"""
        out += "<br><br>"
        for i, name in enumerate(shap_values.output_names):
            if target_label is not None and name != target_label:
                continue  # SKIP processing for irrelevant labels
            ###############################*******************************###############################*******************************
            if name == "control":
                transparent_colors = {
                                        "white": [230, 242, 244],
                                        "red": [50, 181, 65],# green [0, 150, 60], #green

                                        "green": [50, 181, 65]# green [0, 150, 60], #green
                                    }

                update_colors(
                    red_lch=[64.9, 74.9, 140.5], #green [54.0, 90.0, 2.0943951], #green
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[111, 211, 110], #green [144.0, 238, 144], #green
                    transparent_colors=transparent_colors
                )
            elif name == "mci":
                # print("setting to blue")
                transparent_colors = {
                                        "white": [230, 242, 244],
                                        "red": [255, 195, 0], # [255, 216, 107], #yellow [0, 128, 169], #blue
                                        "green": [255, 195, 0], # [255, 216, 107] #yellow [0, 128, 169], #blue
                                    }

                update_colors(
                    red_lch=[82.0, 84.1, 84.3], #[87.7, 70.8, 88.5], #yellow [80.2, 49.1, 133.0],
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[255, 212, 122], #[255, 232, 147], #light yellow [127.0, 196, 252], #light blue
                    transparent_colors=transparent_colors
                )
            else:
                # print("setting to red")
                transparent_colors = {
                                    "white": [230, 242, 244],
                                    "red": [171, 105, 212], #purple [255, 13, 87], #red
                                    "green": [171, 105, 212] #purple [255, 13, 87], #red
                                }

                update_colors(
                    red_lch=[55.7, 63.8, 315.1], #purple [54.0, 90.0, 0.35470565 + 2 * np.pi], #red
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[189, 144, 214], #ligh purple [255.0, 127, 167], #light red
                    transparent_colors=transparent_colors
                )

            out += f"<div id='_tp_{uuid}_output_{i}' style='display: {'block' if i == 0 else 'none'}';>"
            out += text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
            out += "</div>"
            out += f"<div id='_tp_{uuid}_output_{i}_zoom' style='display: none;'>"
            ###############################*******************************###############################*******************************
            if name == "control":
                transparent_colors = {
                                        "white": [230, 242, 244],
                                        "red": [50, 181, 65],# green [0, 150, 60], #green

                                        "green": [50, 181, 65]# green [0, 150, 60], #green
                                    }

                update_colors(
                    red_lch=[64.9, 74.9, 140.5], #green [54.0, 90.0, 2.0943951], #green
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[111, 211, 110], #green [144.0, 238, 144], #green
                    transparent_colors=transparent_colors
                )
            elif name == "mci":
                # print("setting to blue")
                transparent_colors = {
                                        "white": [230, 242, 244],
                                        "red": [255, 195, 0], # [255, 216, 107], #yellow [0, 128, 169], #blue
                                        "green": [255, 195, 0], # [255, 216, 107] #yellow [0, 128, 169], #blue
                                    }

                update_colors(
                    red_lch=[82.0, 84.1, 84.3], #[87.7, 70.8, 88.5], #yellow [80.2, 49.1, 133.0],
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[255, 212, 122], #[255, 232, 147], #light yellow [127.0, 196, 252], #light blue
                    transparent_colors=transparent_colors
                )
            else:
                # print("setting to red")
                transparent_colors = {
                                    "white": [230, 242, 244],
                                    "red": [171, 105, 212], #purple [255, 13, 87], #red
                                    "green": [171, 105, 212] #purple [255, 13, 87], #red
                                }

                update_colors(
                    red_lch=[55.7, 63.8, 315.1], #purple [54.0, 90.0, 0.35470565 + 2 * np.pi], #red
                    l_mid=40.0,
                    blue_lch=[94.7, 4.3, 213.2], #white
                    light_blue=[230, 242, 244], #white
                    light_red=[189, 144, 214], #ligh purple [255.0, 127, 167], #light red
                    transparent_colors=transparent_colors
                )

            out += text(
                shap_values[:, i],
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                display=False,
            )
            out += "</div>"
        out += "</div>"
        if display:
            print('displaying 346')
            _ipython_display_html(out)
            return out
        else:
            return out
        # text_to_text(shap_values)
        # return

    if len(shap_values.shape) == 3:
        print("In len 3")
        xmin_computed = None
        xmax_computed = None
        cmax_computed = None

        for i in range(shap_values.shape[-1]):
            for j in range(shap_values.shape[0]):
                values, clustering = unpack_shap_explanation_contents(shap_values[j, :, i])
                tokens, values, group_sizes = process_shap_values(
                    shap_values[j, :, i].data, values, grouping_threshold, separator, clustering
                )

                xmin_i, xmax_i, cmax_i = values_min_max(values, shap_values[j, :, i].base_values)
                if xmin_computed is None or xmin_i < xmin_computed:
                    xmin_computed = xmin_i
                if xmax_computed is None or xmax_i > xmax_computed:
                    xmax_computed = xmax_i
                if cmax_computed is None or cmax_i > cmax_computed:
                    cmax_computed = cmax_i

        if xmin is None:
            xmin = xmin_computed
        if xmax is None:
            xmax = xmax_computed
        if cmax is None:
            cmax = cmax_computed

        out = ""
        for i, v in enumerate(shap_values):
            out += f"""
<br>
<hr style="height: 1px; background-color: #e6f2f4; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
<div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #e6f2f4; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
</div>
            """
            out += text(
                v,
                num_starting_labels=num_starting_labels,
                grouping_threshold=grouping_threshold,
                separator=separator,
                xmin=xmin,
                xmax=xmax,
                cmax=cmax,
                display=False,
            )
        if display:
            print('displaying 400')
            _ipython_display_html(out)
            return out
        else:
            return out
    # print("no len. continue", len(shap_values.values))
    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(shap_values.values, shap_values.base_values)
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new

    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(
        shap_values.data, values, grouping_threshold, separator, clustering
    )

    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())

    # uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [t.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "") for t in tokens]
    output_name = shap_values.output_names if isinstance(shap_values.output_names, str) else ""
    out += svg_force_plot(
        values,
        shap_values.base_values,
        shap_values.base_values + values.sum(),
        encoded_tokens,
        uuid,
        xmin,
        xmax,
        output_name,
    )
    out += (
        "<div align='center'><div style=\"color: rgb(120,120,120); font-size: 12px; margin-bottom:25px;margin-top: -15px;\">transcript text</div>"
    )
    for i, token in enumerate(tokens):
        scaled_value = 0.5 + 0.5 * values[i] / (cmax + 1e-8)
        if token in {'.', ',', '?', '!', ':', ';'}:
            scaled_value = 0.0
        color = red_transparent_blue(scaled_value)
        # color = green_white_transparent(scaled_value)
        # ************************************************************************************************
        # color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
        color = (float(color[0] * 255), float(color[1] * 255), float(color[2] * 255), float(color[3]))
        
        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"

        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])

        # the HTML for this token
        # print('color: 474, ', color)
        ##############################################################################################################################
        out += f"""<div style='display: {wrapper_display}; text-align: center;'
    ><div style='display: {label_display}; color: #999; padding-top: 0px; font-size: 12px;'>{value_label}</div
        ><div id='_tp_{uuid}_ind_{i}'
            style='display: inline; background: rgba{color}; border-radius: 3px; padding: 0px'
            onclick="
            if (this.previousSibling.style.display == 'none') {{
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            }} else {{
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }}"
            onmouseover="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;"
        >{token.replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '')}</div></div>"""
    out += "</div>"

    if display:
        print('displaying')
        _ipython_display_html(out)
        return out
    else:
        return out


def process_shap_values(tokens, values, grouping_threshold, separator, clustering=None, return_meta_data=False):
    # See if we got hierarchical input data. If we did then we need to reprocess the
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:
        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError(
                "The length of the attribution values must match the number of "
                "tokens if shap_values.clustering is None! When passing hierarchical "
                "attributions the clustering is also required."
            )

        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i, 0])
            ri = int(clustering[i, 1])
            groups.append(groups[li] + groups[ri])
            lower_values[M + i] = lower_values[li] + lower_values[ri] + values[M + i]
            max_values[i + M] = max(abs(values[M + i]) / len(groups[M + i]), max_values[li], max_values[ri])

        # compute the upper_values
        upper_values = np.zeros(len(values))

        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i - M, 0])
            ri = int(clustering[i - M, 1])
            upper_values[i] = value
            value += values[i]
            #             lower_credit(upper_values, clustering, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
            #             lower_credit(upper_values, clustering, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, clustering, li, value * 0.5)
            lower_credit(upper_values, clustering, ri, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)

        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_values = []
        group_sizes = []

        # meta data
        token_id_to_node_id_mapping = np.zeros((M,))
        collapsed_node_ids = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):
            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)

                # meta data
                collapsed_node_ids.append(i)
                token_id_to_node_id_mapping[i] = i

            else:
                # compute the dividend at internal nodes
                li = int(clustering[i - M, 0])
                ri = int(clustering[i - M, 1])
                dv = abs(values[i]) / len(groups[i])

                # if the interaction level is too high then just treat this whole group as one token
                if max(max_values[li], max_values[ri]) < dv * grouping_threshold:
                    new_tokens.append(
                        separator.join([tokens[g] for g in groups[li]])
                        + separator
                        + separator.join([tokens[g] for g in groups[ri]])
                    )
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))

                    # setting collapsed node ids and token id to current node id mapping metadata

                    collapsed_node_ids.append(i)
                    for g in groups[li]:
                        token_id_to_node_id_mapping[g] = i

                    for g in groups[ri]:
                        token_id_to_node_id_mapping[g] = i

                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)

        merge_tokens(new_tokens, new_values, group_sizes, len(group_values) - 1)

        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)

        # meta data
        token_id_to_node_id_mapping = np.array(token_id_to_node_id_mapping)
        collapsed_node_ids = np.array(collapsed_node_ids)

        M = len(tokens)
    else:
        group_sizes = np.ones(M)
        token_id_to_node_id_mapping = np.arange(M)
        collapsed_node_ids = np.arange(M)

    for i, token in enumerate(tokens):
        if token in {'. ', ', ', '? ', '! ', ': ', '; '}:
            values[i] = 0.0

    if return_meta_data:
        return tokens, values, group_sizes, token_id_to_node_id_mapping, collapsed_node_ids
    else:
        return tokens, values, group_sizes



def svg_force_plot(values, base_values, fx, tokens, uuid, xmin, xmax, output_name):
    return ''


def unpack_shap_explanation_contents(shap_values):
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)

    return np.array(values), clustering

def _ipython_display_html(data):
    """Check IPython is installed, then display HTML"""
    if not have_ipython:
        msg = "IPython is required for this function but is not installed. Fix this with `pip install ipython`."
        raise ImportError(msg)
    return ipython_display(HTML(data))
