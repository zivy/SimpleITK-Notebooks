#! /usr/bin/env python

import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import sys
import shutil
import subprocess
import platform
import matplotlib.pyplot as plt

# We use the multiprocess package instead of the official
# multiprocessing as it currently has several issues as discussed
# on the software carpentry page: https://hpc-carpentry.github.io/hpc-python/06-parallel/
import multiprocess as mp
from functools import partial
import argparse

import hashlib
import tempfile

# Maximal number of parallel processes we run.
MAX_PROCESSES = 15

"""
This script inspects/characterizes images in a given directory structure. It
recursively traverses the directories and either inspects the files one by one
or if in DICOM series inspection mode, inspects the data on a per series basis
(all 2D series files combined into a single 3D image).

To run the script one needs to specify:
    1. Root of the data directory.
    2. Output file name.
    3. The analysis type to perform per_file or per_series. The latter indicates
       we are only interested in DICOM files. When run using per_file empty lines
       in the results file are due to:
           a. The file is not an image or is a corrupt image file.
           b. SimpleITK was unable to read the image file (contact us with an example).
    4. Optional SimpleITK imageIO to use. The default value is
       the empty string, indicating that all file types should be read.
       To see the set of ImageIO types supported by your version of SimpleITK,
       call ImageFileReader::GetRegisteredImageIOs() or simply print an
       ImageFileReader object.
    5. Optional external applications to run. Their return value (zero or
       non zero) is used to log success or failure. A nice example is the
       dciodvfy program from David Clunie (https://www.dclunie.com/dicom3tools.html)
       which validates compliance with the DICOM standard.
    6. When the external applications are provided corresponding column headings
       are also required. These are used in the output csv file.
    7. Optional metadata keys. These are image specific keys such as DICOM tags
       or other metadata tags that may be found in the image. The content of the
       tags is written to the result file.
    8. When the metadata tags are provided corresponding column headings
       are also required. These are used in the output csv file.

Examples:
Run a generic file analysis:
python characterize_data.py ../Data/ Output/generic_image_data_report.csv per_file \
--imageIO "" --external_applications ./dciodvfy --external_applications_headings "DICOM Compliant" \
--metadata_keys "0008|0060" "0018|5101" --metadata_keys_headings "modality" "radiographic view"


Run a DICOM series based analysis:
python characterize_data.py ../Data/ Output/DICOM_image_data_report.csv per_series \
--metadata_keys "0008|0060" "0018|5101" --metadata_keys_headings "modality" "radiographic view"

Run a generic file analysis, omit problematic files from csv (--ignore_problems) and create a summary
image which is comprised of thumbnails from all the images.
python characterize_data.py ../../Data/ Output/generic_image_data_report.csv per_file \
--external_applications ./dciodvfy --external_applications_headings "DICOM compliant" \
--metadata_keys "0008|0060" --metadata_keys_headings modality --ignore_problems --create_summary_image

Output:
The raw information is written to the specified output file (e.g. output.csv).
Additionally, minimal analysis of the raw information is performed:
1. If there are duplicate images these are reported in output_duplicates.csv.
2. Two figures: output_image_size_distribution.pdf and output_min_max_intensity_distribution.pdf

NOTE: For the same directory structure, the order of the rows in the output csv file will vary
across operating systems (order of files in the "files" column also varies). This is a consequence
of using os.walk to traverse the file system (internally os.walk uses os.scandir and that method's 
documentation says "The entries are yielded in arbitrary order.").

Convert from x, y, z (zero based) indexes from the "summary image" to information from "summary csv" file.

import pandas as pd
import SimpleITK as sitk

def xyz_to_index(x, y, z):
    tile_size=[20, 20]
    thumbnail_size=[64, 64]
    # add 2 to the index because the csv starts counting lines at 1 and the first
    # line is the table header
    return (z * tile_size[0] * tile_size[1]
            + int(y / thumbnail_size[1]) * tile_size[0]
            + int(x / thumbnail_size[0])
            )

csv_file_name = "Output/generic_image_data_report.csv"
df = pd.read_csv(csv_file_name)

file_names = eval(df["files"].iloc[xyz_to_index(x=xval, y=yval, z=zval)])
print(file_names)
sitk.Show(sitk.ReadImage(file_names))
"""


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a directory path or directory does not exist."
        )


def inspect_image(sitk_image, image_info, meta_data_info, create_summary_image):
    """
    Inspect a SimpleITK image, and update the image_info dictionary with the values associated with the
    contents of the meta_data_info.

    Parameters
    ----------
    sitk_image (SimpleITK.Image): Input image for inspection.
    image_info (dict): Image information is written to this dictionary (e.g. image_info["image size"] = "(512,512)").
                           The image_info dict is filled with the following values:
                           "MD5 intensity hash" - Enable identification of duplicate images in terms of intensity.
                                                  This is different from SimpleITK image equality where the
                                                  same intensities with different image spacing/origin/direction cosine
                                                  are considered different images as they occupy a different spatial
                                                  region.
                           "image size" - number of pixels in each dimension.
                           "pixel type" - type of pixels (scalar - gray, vector - gray or color).
                           "min intensity"/"max intensity" - if a scalar image, min and max values.
                           metadata values for the values listed in the meta_data_info dictionary (e.g. "radiographic view" : "AP").
    meta_data_info(dict(str:str)): The meta-data information whose values will be reported.
                                   Dictionary structure is description:meta_data_tag
                                   (e.g. {"radiographic view" : "0018|5101", "modality" : "0008|0060"}).
    """
    image_info["MD5 intensity hash"] = hashlib.md5(
        sitk.GetArrayViewFromImage(sitk_image)
    ).hexdigest()
    image_info["image size"] = sitk_image.GetSize()
    image_info["image spacing"] = sitk_image.GetSpacing()
    image_info["image origin"] = sitk_image.GetOrigin()
    image_info["axis direction"] = sitk_image.GetDirection()
    if (
        sitk_image.GetNumberOfComponentsPerPixel() == 1
    ):  # greyscale image, get the min/max pixel values
        image_info["pixel type"] = sitk_image.GetPixelIDTypeAsString() + " gray"
        mmfilter = sitk.MinimumMaximumImageFilter()
        mmfilter.Execute(sitk_image)
        image_info["min intensity"] = mmfilter.GetMinimum()
        image_info["max intensity"] = mmfilter.GetMaximum()
    else:  # either a color image or a greyscale image masquerading as a color one
        pixel_type = sitk_image.GetPixelIDTypeAsString()
        channels = [
            sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(sitk_image, i))
            for i in range(sitk_image.GetNumberOfComponentsPerPixel())
        ]
        if np.array_equal(channels[0], channels[1]) and np.array_equal(
            channels[0], channels[2]
        ):
            pixel_type = (
                pixel_type
                + f" {sitk_image.GetNumberOfComponentsPerPixel()} channels gray"
            )
        else:
            pixel_type = (
                pixel_type
                + f" {sitk_image.GetNumberOfComponentsPerPixel()} channels color"
            )
        image_info["pixel type"] = pixel_type
    img_keys = sitk_image.GetMetaDataKeys()
    for k, v in meta_data_info.items():
        if v in img_keys:
            image_info[k] = sitk_image.GetMetaData(v)
    if create_summary_image:
        image_info["thumbnail"] = image_to_thumbnail(sitk_image)


def inspect_single_file(
    file_name,
    imageIO="",
    meta_data_info={},
    external_programs_info={},
    create_summary_image=False,
):
    """
    Inspect a file using the specified imageIO, returning a dictionary with the relevant information.

    Parameters
    ----------
    file_name (str): Image file name.
    imageIO (str): Name of image IO to use. To see the list of registered image IOs use the
                   ImageFileReader::GetRegisteredImageIOs() or print an ImageFileReader.
                   The empty string indicates to read all file formats supported by SimpleITK.
    meta_data_info(dict(str:str)): The meta-data information whose values will be reported.
                                   Dictionary structure is description:meta_data_tag
                                   (e.g. {"radiographic view" : "0018|5101", "modality" : "0008|0060"}).
    external_programs_info(dict(str:str)): A dictionary of programs that are run with the file_name as input
                                  the return value 'succeeded' or 'failed' is recorded. This
                                  is useful for example if you need to validate conformance
                                  to a standard such as DICOM. Dictionary format is description:program (e.g.
                                  {"DICOM compliant" : "path_to_dicom3tools/dciodvfy"}).
    Returns
    -------
     dict with the following entries: file name, MD5 intensity hash,
                                       image size, image spacing, image origin, axis direction,
                                       pixel type, min intensity, max intensity,
                                       meta data_1...meta_data_n,
                                       external_program_res_1...external_program_res_m
    If the given file is not readable by SimpleITK, the only entry in the dictionary
    will be the file name.
    """
    # Using a list so that returned csv is consistent with the series based analysis (an
    # image is defined by multiple files).
    file_info = {}
    file_info["files"] = [file_name]
    try:
        reader = sitk.ImageFileReader()
        reader.SetImageIO(imageIO)
        reader.SetFileName(file_name)
        img = reader.Execute()
        inspect_image(img, file_info, meta_data_info, create_summary_image)
        for k, p in external_programs_info.items():
            try:
                # run the external programs, check the return value, and capture all output so it
                # doesn't appear on screen. The CalledProcessError exception is raised if the
                # external program fails (returns non zero value).
                subprocess.run([p, file_name], check=True, capture_output=True)
                file_info[k] = "succeeded"
            except Exception:
                file_info[k] = "failed"
    except Exception:
        pass
    return file_info


def inspect_files(
    root_dir,
    imageIO="",
    meta_data_info={},
    external_programs_info={},
    create_summary_image=False,
):
    """
    Iterate over a directory structure and return a pandas dataframe with the relevant information for the
    image files. This also includes non image files. The resulting dataframe will only include the file name
    if that file wasn't successfully read by SimpleITK. The two reasons for failure are: (1) the user specified
    imageIO isn't compatible with the file format (e.g. user is only interested in reading jpg and the file
    format is mha) or (2) the file could not be read by the SimpleITK IO (corrupt file or unexpected limitation of
    SimpleITK).

    Parameters
    ----------
    root_dir (str): Path to the root of the data directory. Traverse the directory structure
                    and inspect every file (also report non image files, in which
                    case the only valid entry will be the file name).
    imageIO (str): Name of image IO to use. To see the list of registered image IOs use the
                   ImageFileReader::GetRegisteredImageIOs() or print an ImageFileReader.
                   The empty string indicates to read all file formats supported by SimpleITK.
    meta_data_info(dict(str:str)): The meta-data information whose values will be reported.
                                   Dictionary structure is description:meta_data_tag
                                   (e.g. {"radiographic view" : "0018|5101", "modality" : "0008|0060"}).
    external_programs_info(dict(str:str)): A dictionary of programs that are run with the file_name as input
                                  the return value 'succeeded' or 'failed' is recorded. This
                                  is useful for example if you need to validate conformance
                                  to a standard such as DICOM. Dictionary format is description:program (e.g.
                                  {"DICOM compliant" : "path_to_dicom3tools/dciodvfy"}).
    Returns
    -------
    pandas DataFrame: Each row in the data frame corresponds to a single file.

    """
    all_file_names = []
    for dir_name, subdir_names, file_names in os.walk(root_dir):
        all_file_names += [
            os.path.join(os.path.abspath(dir_name), fname) for fname in file_names
        ]
    # Get list of dictionaries describing the results and then combine into a dataframe, faster
    # than appending to the dataframe one by one. Use parallel processing to speed things up.
    if platform.system() == "Windows":
        res = map(
            partial(
                inspect_single_file,
                imageIO=imageIO,
                meta_data_info=meta_data_info,
                external_programs_info=external_programs_info,
                create_summary_image=create_summary_image,
            ),
            all_file_names,
        )
    else:
        with mp.Pool(processes=MAX_PROCESSES) as pool:
            res = pool.map(
                partial(
                    inspect_single_file,
                    imageIO=imageIO,
                    meta_data_info=meta_data_info,
                    external_programs_info=external_programs_info,
                    create_summary_image=create_summary_image,
                ),
                all_file_names,
            )
    return pd.DataFrame.from_dict(res)


def inspect_single_series(series_data, meta_data_info={}, create_summary_image=False):
    """
    Inspect a single DICOM series (DICOM hierarchy of patient-study-series-image).
    This can be a single file, or multiple files such as a CT or
    MR volume.

    Parameters
    ----------
    series_data (two entry tuple): First entry is study:series, second entry is the list of
                                   files comprising this series.
    meta_data_info(dict(str:str)): The meta-data information whose values will be reported.
                                   Dictionary structure is description:meta_data_tag
                                   (e.g. {"radiographic view" : "0018|5101", "modality" : "0008|0060"}).
    Returns
    -------
     dictionary containing all of the information about the series.
    """
    series_info = {}
    series_info["files"] = series_data[1]
    try:
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        _, sid = series_data[0].split(":")
        file_names = series_info["files"]
        # As the files comprising a series with multiple files can reside in
        # separate directories and SimpleITK expects them to be in a single directory
        # we use a tempdir and symbolic links to enable SimpleITK to read the series as
        # a single image. Additionally the files are renamed as they may have resided in
        # separate directories with the same file name. Finally, unfortunately on Windows
        # we copy the files to the tempdir as the os.symlink documentation says that
        # "On newer versions of Windows 10, unprivileged accounts can create symlinks
        # if Developer Mode is enabled. When Developer Mode is not available/enabled,
        # the SeCreateSymbolicLinkPrivilege privilege is required, or the process must be
        # run as an administrator."
        with tempfile.TemporaryDirectory() as tmpdirname:
            if platform.system() == "Windows":
                for i, fname in enumerate(file_names):
                    shutil.copy(
                        os.path.abspath(fname), os.path.join(tmpdirname, str(i))
                    )
            else:
                for i, fname in enumerate(file_names):
                    os.symlink(os.path.abspath(fname), os.path.join(tmpdirname, str(i)))
            reader.SetFileNames(
                sitk.ImageSeriesReader_GetGDCMSeriesFileNames(tmpdirname, sid)
            )
            img = reader.Execute()
            for k in meta_data_info.values():
                if reader.HasMetaDataKey(0, k):
                    img.SetMetaData(k, reader.GetMetaData(0, k))
            inspect_image(img, series_info, meta_data_info, create_summary_image)
    except Exception:
        pass
    return series_info


def inspect_series(root_dir, meta_data_info={}, create_summary_image=False):
    """
    Inspect all series found in the directory structure. A series does not have to
    be in a single directory (the files are located in the subtree and combined
    into a single image).

    Parameters
    ----------
    root_dir (str): Path to the root of the data directory. Traverse the directory structure
                    and inspect every series. If the series is comprised of multiple image files
                    they do not have to be in the same directory. The only expectation is that all
                    images from the series are under the root_dir.
    meta_data_info(dict(str:str)): The meta-data information whose values will be reported.
                                   Dictionary structure is description:meta_data_tag
                                   (e.g. {"radiographic view" : "0018|5101", "modality" : "0008|0060"}).
    Returns
    -------
    pandas DataFrame: Each row in the data frame corresponds to a single file.
    """
    all_series_files = {}
    reader = sitk.ImageFileReader()
    # collect the file names of all series into a dictionary with the key being
    # study:series. This traversal is faster, O(n), than calling GetGDCMSeriesIDs on each
    # directory followed by iterating over the series and calling
    # GetGDCMSeriesFileNames with the seriesID on that directory, O(n^2).
    for dir_name, subdir_names, file_names in os.walk(root_dir):
        for file in file_names:
            try:
                fname = os.path.join(dir_name, file)
                reader.SetFileName(fname)
                reader.ReadImageInformation()
                sid = reader.GetMetaData("0020|000e")
                study = reader.GetMetaData("0020|000d")
                key = f"{study}:{sid}"
                if key in all_series_files:
                    all_series_files[key].append(fname)
                else:
                    all_series_files[key] = [fname]
            except Exception:
                pass
    # Get list of dictionaries describing the results and then combine into a dataframe, faster
    # than appending to the dataframe one by one.
    res = [
        inspect_single_series(series_data, meta_data_info, create_summary_image)
        for series_data in all_series_files.items()
    ]
    return pd.DataFrame.from_dict(res)


def image_to_thumbnail(
    img,
    thumbnail_size=[64, 64],
    interpolator=sitk.sitkNearestNeighbor,
    projection_axis=2,
):
    """
    Create a grayscale thumbnail image from the given image. If the image is 3D it is
    projected to 2D using a Maximum Intensity Projection (MIP) approach. Color images
    are converted to grayscale, and high dynamic range images are window leveled using
    a robust estimate of the relevant minimum and maximum intensity values.

    Parameters
    ----------
    img (SimpleITK.Image): A 2D or 3D grayscale or sRGB image.
    thumbnail_size (list/tuple(int)): The 2D sizes of the thumbnail.
    interpolator: Interpolator to use when resampling to a thumbnail. Nearest neighbor
                  is computationally efficient and is applicable for
                  both segmentation masks and scalar images.
    projection_axis(int in [0,2]): The axis along which we project 3D images.

    Returns
    -------
    2D SimpleITK image with sitkUInt8 pixel type.
    """
    if (
        img.GetDimension() == 3 and img.GetSize()[2] == 1
    ):  # 2D image masquerading as 3D image
        img = img[:, :, 0]
    elif img.GetDimension() == 3:  # 3D image projected along projection_axis direction
        img = sitk.MaximumProjection(img, projection_axis)
        slc = list(img.GetSize())
        slc[projection_axis] = 0
        img = sitk.Extract(img, slc)
    # convert multi-channel image to gray
    # sRGB, sRGBA or image with more than 4 channels. assume the first three channels represent
    # RGB. when there are more than 4 channels this assumption is likely incorrect, but there's
    # nothing more sensible to do. maybe select an arbitrary channel but that is problematic
    # if the 4 channel image is sRGBA and A is 255, so selecting the last channel is just a "white" image.
    if img.GetNumberOfComponentsPerPixel() >= 3:
        # Convert image to gray scale and rescale results to [0,255]
        channels = [
            sitk.VectorIndexSelectionCast(img, i, sitk.sitkFloat32) for i in range(3)
        ]
        # linear mapping
        I = (
            1
            / 255.0
            * (0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])
        )
        # nonlinear gamma correction
        I = (
            I * sitk.Cast(I <= 0.0031308, sitk.sitkFloat32) * 12.92
            + I ** (1 / 2.4) * sitk.Cast(I > 0.0031308, sitk.sitkFloat32) * 1.055
            - 0.055
        )
        img = sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8)
    else:
        if img.GetPixelID() != sitk.sitkUInt8:
            # To deal with high dynamic range images that also contain outlier intensities
            # we use window-level intensity mapping and set the window:
            # to [max(Q1 - w*IQR, min_intensity), min(Q3 + w*IQR, max_intensity)]
            # IQR = Q3-Q1
            # The bounds which should exclude outliers are defined by the parameter w,
            # where 1.5 is a standard default value (same as used in box and
            # whisker plots to define whisker lengths).
            w = 1.5
            min_val, q1_val, q3_val, max_val = np.percentile(
                sitk.GetArrayViewFromImage(img).ravel(), [0, 25, 75, 100]
            )
            min_max = [
                np.max([(1.0 + w) * q1_val - w * q3_val, min_val]),
                np.min([(1.0 + w) * q3_val - w * q1_val, max_val]),
            ]
            wl_image = sitk.IntensityWindowing(
                img,
                windowMinimum=min_max[0],
                windowMaximum=min_max[1],
                outputMinimum=0.0,
                outputMaximum=255.0,
            )
            img = sitk.Cast(wl_image, sitk.sitkUInt8)
        else:
            img = sitk.IntensityWindowing(
                img,
                windowMinimum=np.min(sitk.GetArrayViewFromImage(img)),
                windowMaximum=np.max(sitk.GetArrayViewFromImage(img)),
            )
    res = sitk.Resample(
        img,
        size=thumbnail_size,
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=img.GetOrigin(),
        outputSpacing=[
            (sz - 1) * spc / (nsz - 1)
            for nsz, sz, spc in zip(thumbnail_size, img.GetSize(), img.GetSpacing())
        ],
        outputDirection=img.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=img.GetPixelID(),
    )
    # new_spacing = [
    #     ((osz - 1) * ospc) / (nsz - 1)
    #     for ospc, osz, nsz in zip(img.GetSpacing(), img.GetSize(), thumbnail_size)
    # ]
    # new_spacing = [max(new_spacing)] * img.GetDimension()
    # center = img.TransformContinuousIndexToPhysicalPoint(
    #     [sz / 2.0 for sz in img.GetSize()]
    # )
    # new_origin = [
    #     c - c_index * nspc
    #     for c, c_index, nspc in zip(
    #         center, [sz / 2.0 for sz in thumbnail_size], new_spacing
    #     )
    # ]
    # res = sitk.Resample(
    #     img,
    #     size=thumbnail_size,
    #     transform=sitk.Transform(),
    #     interpolator=interpolator,
    #     outputOrigin=new_origin,
    #     outputSpacing=new_spacing,
    #     outputDirection=img.GetDirection(),
    #     defaultPixelValue=0,
    #     outputPixelType=img.GetPixelID(),
    # )
    res.SetOrigin([0, 0])
    res.SetSpacing([1, 1])
    res.SetDirection([1, 0, 0, 1])
    return res


def image_list_to_faux_volume(image_list, tile_size=[20, 20]):
    """
    Create a faux volume from a list of images all having the same size.

    Parameters
    ----------
    image_list (list[SimpleITK.Image]): List of images all with the same size.
    tile_size([int,int]): The number of images in x and y in each faux volume slice.

    Returns
    -------
    3D SimpleITK image combining all the images contained in the image_list.
    """
    step_size = tile_size[0] * tile_size[1]
    faux_volume_slices = [
        sitk.Tile(image_list[i : i + step_size], tile_size, 0)
        for i in range(0, len(image_list), step_size)
    ]
    # if last tile image is smaller than others, add background content to match the size
    if len(faux_volume_slices) > 1 and (
        faux_volume_slices[-1].GetHeight() != faux_volume_slices[-2].GetHeight()
        or faux_volume_slices[-1].GetWidth() != faux_volume_slices[-2].GetWidth()
    ):
        image = sitk.Image(faux_volume_slices[-2]) * 0
        faux_volume_slices[-1] = sitk.Paste(
            image,
            faux_volume_slices[-1],
            faux_volume_slices[-1].GetSize(),
            [0, 0],
            [0, 0],
        )
    return sitk.JoinSeries(faux_volume_slices)


def characterize_data(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_of_data_directory",
        type=dir_path,
        help="path to the topmost directory containing data",
    )
    parser.add_argument("output_file", help="output csv file path")
    parser.add_argument(
        "analysis_type",
        default="per_file",
        help='type of analysis, "per_file" or "per_series"',
    )
    parser.add_argument(
        "--imageIO",
        default="",
        help="SimpleITK imageIO to use for reading (e.g. BMPImageIO)",
    )
    parser.add_argument(
        "--external_applications",
        default=[],
        nargs="*",
        help="paths to external applications",
    )
    parser.add_argument(
        "--external_applications_headings",
        default=[],
        nargs="*",
        help="titles of the results columns for external applications",
    )
    parser.add_argument(
        "--metadata_keys",
        nargs="*",
        default=[],
        help="inspect values of these metadata keys (DICOM tags or other keys stored in the file)",
    )
    parser.add_argument(
        "--metadata_keys_headings",
        default=[],
        nargs="*",
        help="titles of the results columns for the metadata_keys",
    )
    parser.add_argument(
        "--ignore_problems",
        action="store_true",
        help="problematic files will not be listed if parameter is given on commandline",
    )
    parser.add_argument(
        "--create_summary_image",
        action="store_true",
        help="create a summary image, volume of thumbnails representing all images",
    )

    args = parser.parse_args(argv)
    if len(args.external_applications) != len(args.external_applications_headings):
        print("Number of external applications and their headings do not match.")
        sys.exit(1)
    if len(args.metadata_keys) != len(args.metadata_keys_headings):
        print("Number of metadata keys and their headings do not match.")
        sys.exit(1)
    if args.analysis_type not in ["per_file", "per_series"]:
        print("Unexpected analysis type.")
        sys.exit(1)

    if args.analysis_type == "per_file":
        df = inspect_files(
            args.root_of_data_directory,
            imageIO=args.imageIO,
            meta_data_info=dict(zip(args.metadata_keys_headings, args.metadata_keys)),
            external_programs_info=dict(
                zip(args.external_applications_headings, args.external_applications)
            ),
            create_summary_image=args.create_summary_image,
        )
    elif args.analysis_type == "per_series":
        df = inspect_series(
            args.root_of_data_directory,
            meta_data_info=dict(zip(args.metadata_keys_headings, args.metadata_keys)),
            create_summary_image=args.create_summary_image,
        )
    # corner case, no images
    if len(df.columns) == 1:
        print(
            f"No report created, mo successfully read images from root directory {args.root_of_data_directory}"
        )
        return 0

    # create summary image and remove the column containing the thumbnail images from the
    # dataframe.
    if args.create_summary_image:
        faux_volume = image_list_to_faux_volume(df["thumbnail"].dropna().to_list())
        sitk.WriteImage(
            faux_volume,
            f"{os.path.splitext(args.output_file)[0]}_summary_image.nrrd",
            useCompression=True,
        )
        df.drop("thumbnail", axis=1, inplace=True)

    # remove all rows associated with problematic files (non-image files or image files with problems).
    # all the valid rows contain at least 2 non-na values so use that threshold when dropping rows.
    if args.ignore_problems:
        df.dropna(inplace=True, thresh=2)
    # save the raw information, create directory structure if it doesn't exist
    dirname = os.path.dirname(args.output_file)
    if not dirname:
        dirname = "."
    os.makedirs(dirname, exist_ok=True)
    df.to_csv(args.output_file, index=False)

    # minimal analysis on the image information, detect image duplicates and plot the image size
    # distribution and distribution of min/max intensity values of scalar
    # images
    image_counts = (
        df["MD5 intensity hash"].dropna().value_counts().reset_index(name="count")
    )
    duplicates = df[
        df["MD5 intensity hash"].isin(
            image_counts[image_counts["count"] > 1]["MD5 intensity hash"]
        )
    ].sort_values(by=["MD5 intensity hash"])
    if not duplicates.empty:
        duplicates.to_csv(
            f"{os.path.splitext(args.output_file)[0]}_duplicates.csv", index=False
        )

    size_counts = (
        df["image size"]
        .astype("string")
        .dropna()
        .value_counts()
        .reset_index(name="count")
    )
    if not size_counts.empty:
        # Compute appropriate size for figure using a specific font size
        # based on stack-overflow: https://stackoverflow.com/questions/35127920/overlapping-yticklabels-is-it-possible-to-control-cell-size-of-heatmap-in-seabo
        fontsize_pt = 8
        dpi = 72.27

        # compute the matrix height in points and inches
        matrix_height_pt = fontsize_pt * len(size_counts)
        matrix_height_in = matrix_height_pt / dpi

        # compute the required figure height
        top_margin = 0.04  # in percentage of the figure height
        bottom_margin = 0.04  # in percentage of the figure height
        figure_height = matrix_height_in / (1 - top_margin - bottom_margin)

        # build the figure instance with the desired height
        fig, ax = plt.subplots(
            figsize=(6, figure_height),
            gridspec_kw=dict(top=1 - top_margin, bottom=bottom_margin),
        )

        ax.tick_params(axis="y", labelsize=fontsize_pt)
        ax.tick_params(axis="x", labelsize=fontsize_pt)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax = size_counts.plot.barh(
            x="image size",
            y="count",
            xlabel="image size",
            ylabel="# of images",
            legend=None,
            ax=ax,
        )
        ax.bar_label(
            ax.containers[0], fontsize=fontsize_pt
        )  # add the number at the top of each bar
        plt.savefig(
            f"{os.path.splitext(args.output_file)[0]}_image_size_distribution.pdf",
            bbox_inches="tight",
        )

    min_intensities = df["min intensity"].dropna()
    max_intensities = df["max intensity"].dropna()
    if not min_intensities.empty:
        fig, ax = plt.subplots()
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.hist(
            min_intensities,
            bins=256,
            alpha=0.5,
            label="min intensity",
            color="blue",
        )
        ax.hist(
            max_intensities,
            bins=256,
            alpha=0.5,
            label="max intensity",
            color="green",
        )
        plt.legend()
        plt.xlabel("intensity")
        plt.ylabel("# of images")
        plt.savefig(
            f"{os.path.splitext(args.output_file)[0]}_min_max_intensity_distribution.pdf",
            bbox_inches="tight",
        )

    return 0


if __name__ == "__main__":
    sys.exit(characterize_data())
