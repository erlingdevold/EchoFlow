from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
import scipy.ndimage as ndi
import cv2 as cv
import xarray as xr


def preprocess_bd2(da, depth0=50, backstep=2, return_idx=True):
    # tag the data with 'bottom detection v1' and parameters
    da.attrs["tag"] = f"bd2-d%i-bs%i" % (depth0, backstep)

    # drop the top of the water column ('forward step')
    da = da.where(da.depth >= depth0, drop=True)

    # compute first approx. of the bottom as the max sv depth
    max = da.idxmax("depth").compute()
    bottom_median = max.median()

    # crop out all depths below 1.8 * bottom_median
    da = da.where(da.depth <= 1.6 * bottom_median, drop=True)

    # recompute the max
    max = da.idxmax("depth").compute()
    total_max = max.max()

    da = xr.where(da.depth <= max - backstep, da, np.nan, keep_attrs=True)

    return da.where(da.depth <= total_max, drop=True)  # crop


def preprocess_da(da, n=7):
    da = 10 * np.log10(da)
    # da = xr.apply_ufunc(medfilt2d, da, kwargs={"kernel_size": n})
    da = preprocess_bd2(da, depth0=25, backstep=2)

    da = da.dropna("depth", how="all")
    return da


def create_simrad_cmap():
    simrad_color_table = [
        (1, 1, 1),
        (0.6235, 0.6235, 0.6235),
        (0.3725, 0.3725, 0.3725),
        (0, 0, 1),
        (0, 0, 0.5),
        (0, 0.7490, 0),
        (0, 0.5, 0),
        (1, 1, 0),
        (1, 0.5, 0),
        (1, 0, 0.7490),
        (1, 0, 0),
        (0.6509, 0.3255, 0.2353),
        (0.4705, 0.2353, 0.1568),
    ]

    simrad_cmap = LinearSegmentedColormap.from_list("Simrad", simrad_color_table)
    simrad_cmap.set_bad(color="grey")

    return simrad_cmap


def to_colors(da, cmap, vmin=-70, vmax=-30):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    norm_data = norm(da)
    rgb = cmap(norm_data)
    return rgb


def normalize_sv(sv):

    sv = (sv - sv.min()) / (sv.max() - sv.min()) * 255  # normalize to 0-255
    sv = sv.astype(np.uint8)

    # apply gaussian
    sv = cv.GaussianBlur(sv, (13, 13), 0)  # 13x13 gaussian kernel
    sv = cv.GaussianBlur(
        sv, (1, 51), 0
    )  # 1x51 gaussian kernel, this is to smooth horizontally
    return sv


def process_file(fn: str):

    da = xr.open_dataarray(fn)
    # da['depth'] = da['range']

    da = 10 * np.log10(da)

    da = preprocess_bd2(da, depth0=25, backstep=5)
    da = da.dropna("depth", how="all")
    # da.plot.imshow(x='ping_time',y='depth',cmap=create_simrad_cmap(),vmin=-70,vmax=-30)
    # plt.gca().invert_yaxis()
    # plt.savefig(f'figs/{fn.stem}_simrad.png')
    # plt.show()

    return da


def get_all_hauls(files: list[str]):

    svs = [process_file(fn) for fn in files]
    max_height = max([sv.data.shape[0] for sv in svs])
    new = []
    print(svs[0])
    for sv in svs:
        sv = np.pad(
            sv.data,
            ((0, max_height - sv.data.shape[0]), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )
        new.append(sv)

    return new


def find_roi(sv):
    sv, boundaries = find_boundaries(sv)  # otsu thresholding

    contours, _ = cv.findContours(boundaries, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        # cv.contourArea(c) calculates area of contour
        x, y, w, h = cv.boundingRect(c)
        if w >= sv.shape[1]:
            continue

        bboxes.append((x, y, w, h))

    return contours, bboxes


def plot_rois(sv, contour, bboxes):
    plt.figure(figsize=(20, 10))

    plt.subplot(121)
    plt.imshow(sv, aspect="auto", cmap=create_simrad_cmap())
    for c in contour:
        plt.plot(c[:, :, 0].flatten(), c[:, :, 1].flatten())

    plt.subplot(122)
    plt.imshow(sv, aspect="auto", cmap=create_simrad_cmap())
    for x, y, w, h in bboxes:
        plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], "r")
    if __name__ == "__main__":
        plt.show()


def normalize_sv(sv):

    sv = (sv - sv.min()) / (sv.max() - sv.min()) * 255  # normalize to 0-255
    sv = sv.astype(np.uint8)

    # apply gaussian
    sv = cv.GaussianBlur(sv, (13, 13), 0)  # 13x13 gaussian kernel
    sv = cv.GaussianBlur(
        sv, (1, 51), 0
    )  # 1x51 gaussian kernel, this is to smooth horizontally
    return sv


def threshold_sv(sv, vmin=-30, vmax=-70):
    sv[np.isnan(sv)] = np.nanmean(sv)
    sv = np.clip(sv, -70, -30)
    return sv


def find_boundaries(sv):
    ## otsu thresholding
    thr, sv_t = cv.threshold(sv, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if plot:
        plt.figure(figsize=(20, 10))
        plt.imshow(sv_t, aspect="auto", cmap=create_simrad_cmap())
        if __name__ == "__main__":
            plt.show()

    return sv, sv_t
