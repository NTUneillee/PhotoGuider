# FLMS ground truth

The index page is gone; use the authors’ direct data objects. From the repository root,
run `bash tools/download_flms_gt.sh` before FLMS benchmark reproduction.
It downloads [release_data.tar](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_gt/release_data.tar)
(40,960 bytes), verifies SHA-256
`754d5a9cead6cd658b0ff59866d6cd75ff304c88d88a5d245d4af9dfbedb4c2e`, extracts
`500_image_dataset.mat` and places it at `caches/flms_500_gt.mat`. Its SHA-256 is
`716ae0885439456fee83d92ae6940586ccc4c9d54889d1bda185b13f1f8ae6db`, byte-identical
to the annotations used by the benchmark. `FLMS_CACHE_DIR` overrides the destination.
The archive also contains the dataset readme.

Optional `--with-images` downloads [image.tar](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_image/image.tar)
(213,626,880 bytes, 501 archive entries) and verifies it against the validated digest
before placing it at `caches/flms_image.tar`.
Image archive SHA-256: `d809f4bd9a28dcaebdd4ffaa3f3d6753a914046e0ed16c3bfb78ae67b15fb9fc`.

The dataset readme states: “This dataset is only for research purpose”. Boxes use
(height_min, width_min, height_max, width_max), 0 denotes the first pixel, and negative
boxes must be ignored. Images are Flickr crawls provided by the authors on request.
Annotations and images are not redistributed; see [`DATA_TERMS.md`](../DATA_TERMS.md).
