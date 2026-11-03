# Environments and external assets

Follow the ordered [installation steps](../README.md#installation) from the repository root. It
provisions the six Python 3.12.7 environments, the mandatory weights for both modes, the
full-mode Venus consultant, the optional explanation adapter, and (for the benchmark
reproduction only) the frozen feature caches.
`setup_envs.sh --with-weights` reapplies pinned requirements on every run, runs import
checks, and verifies FGAesQ, SAM and LAION SHA-256 values, including existing files.
It checks out FGAesQ at the immutable commit in `asset_pins.json`.

| Environment | Frozen model / role |
|---|---|
| main | `openai/clip-vit-large-patch14`; timm `vit_small_patch16_dinov3.lvd1689m`; InSPyReNet via `transparent-background` |
| laion | OpenCLIP `ViT-L-14`, pretrained `openai`, with `sa_0_4_vit_l_14_linear.pth` |
| fgaesq | FG-IAA `FGAesQ_Inference`, checkpoint `FGAesQ.pt` |
| sam | Segment Anything `vit_b`, checkpoint `sam_vit_b_01ec64.pth` |
| venus | `popo28/Venus-Q-Stage2`, full mode; separate Transformers 4.37.2 environment |
| explainer | `Qwen/Qwen2.5-VL-7B-Instruct` plus release adapter, only for `--explain` |

DINOv3 access/weights are described in the [official repository](https://github.com/facebookresearch/dinov3).
Library-managed backbones download on first use. Allow network/cache space for those
assets even in fast mode. Run public commands from `main`; extractor and explanation
work is dispatched to the appropriate environment automatically.

`asset_pins.json` is the expected external-asset inventory. Verify the unmodified Venus
snapshot with `python3 envs/verify_assets.py --venus` **before** applying
`envs/venus/apply_patches.sh` or loading remote code. Patching changes the source digests;
the inventory describes the downloaded snapshot, not its patched files.

On its first full-mode run the Venus tokenizer module fetches `SimSun.ttf` (≈10 MB) from the model vendor's CDN into the weights directory; later runs reuse it. No other network access happens at inference time once the assets are installed.

The inventory and exact environment versions come from the validated cold-start build.
The FGAesQ weight download is pinned by its validated SHA-256.
The default verifier warns for assets not yet downloaded and fails for present files with
wrong hashes. Use `--require-present` after download; setup does this automatically.
HF snapshot and other backbone digests are inventoried for cache auditing; the default
check covers the nine pinned files, and `--venus` checks the unpatched snapshot.

## Custom locations

Edit `envs.toml`, or use `CROPPER_ENV_MAIN`, `CROPPER_ENV_LAION`,
`CROPPER_ENV_FGAESQ`, `CROPPER_ENV_SAM`, `CROPPER_ENV_VENUS`, `CROPPER_ENV_EXPLAINER`.
Values may name a venv directory or its Python executable. Asset keys have corresponding
`CROPPER_ASSET_<UPPERCASE_KEY>` overrides. The setup verifier checks default download
locations; verify any custom copy against the same expected digest before use.
