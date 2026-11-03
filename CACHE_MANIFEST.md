# Final cache archive inventory

The `caches.tar.zst` asset in the GitHub release tagged `release` extracts to `caches/`.
It excludes FLMS ground truth; see `tools/FLMS_GT.md`.
Regenerated with `tools/build_cache_manifest.py`.

| file | bytes | SHA-256 | purpose |
|---|---:|---|---|
| `FCDB-test.json` | 77377 | `9523e278023ffd0e6229e603a9124e8c819299c572098873bd4ecbf7b8410302` | FCDB test split (345 images) with GT crops |
| `FCDB-train.json` | 250819 | `7c4513cc3ed488c28d6ce3c233b0f0a9532d566cbb00d47061135d9f78ccf1d0` | FCDB training split (1,369 images) with GT crops |
| `aes_delta_ens.parquet` | 100406 | `8849615cb59d6a787d2e70870d57d75c47545d8184fbdbf11ee7f8b7f9d60f7a` | per-image aesthetic-delta LUT over expansion scales (aes gate driver) |
| `aes_gold_delta.parquet` | 87243 | `be7d2b64d8636de692ebb6598b5ebd439722eb143ffb5a300a3644edcbfe5f0f` | per-image golden-ratio-delta LUT over expansion scales (gold gate driver) |
| `aesthetic_integrity_features.parquet` | 412224 | `edd16d95f5b8afdcd318d431843c2b7829ad92083c83c9681debe7871022ff9f` | FCDB per-crop SAM decoder features (axis 5, thirds prompts) |
| `aspect_free_cache/aspect_free_clipmp_flms.parquet` | 383472 | `bd4b2616ba777af223232e32fc9139f4b7bbf17693502030cbf0437be658c3c2` | aspect-free per-crop CLIP-L multi-prompt similarities |
| `aspect_free_cache/aspect_free_consultant_blend_candidates_flms_core_features.parquet` | 770485 | `07b00381ce4975f7112fd965bcfcfceaaa091facfe4e2d65d23d22a1fe760377` | aspect-free twin-candidate manifest with core features |
| `aspect_free_cache/aspect_free_consultant_blend_candidates_flms_core_features_features_only.parquet` | 729124 | `173d5ad2e37799e82e73ffcea1c392919b58f5ac64e8fafcd855a659d35095a8` | aspect-free twin features (manifest companion) |
| `aspect_free_cache/aspect_free_consultant_blend_clipmp_flms.parquet` | 313544 | `c77e39b2a66c71193d10104522006bf8ab47678faedacb3e7d1e5142a809f932` | aspect-free twin CLIP-L similarities |
| `aspect_free_cache/aspect_free_consultant_blend_fgaesq_flms.parquet` | 33727 | `4d0f9ad0111373c3a5727a922dbb28ad6ef053882aa70b19c17656f6d84f0921` | aspect-free twin FGAesQ scores |
| `aspect_free_cache/aspect_free_consultant_blend_laion_flms.parquet` | 41069 | `d00045c746db91dfd0eb8158d4636a0b7b3c3841549d0de2816115b9668aa672` | aspect-free twin LAION scores |
| `aspect_free_cache/aspect_free_consultant_blend_segment_decoder_flms.parquet` | 582786 | `eb134ce318f807f1f317b39b921e223b148f561fb28fef88bb95c3174832a351` | aspect-free twin SAM decoder features |
| `aspect_free_cache/aspect_free_consultant_blend_segment_encoder_flms.parquet` | 380291 | `37454890ee28b6da52952bce0a3fd8c00dc236488dcdeeb3d5dae525046dacb0` | aspect-free twin SAM encoder features |
| `aspect_free_cache/aspect_free_fgaesq_flms.parquet` | 37306 | `ae32fb76695547c934774a7aa1bd26a7ef684e867c26a05360324561060186e6` | aspect-free per-crop FGAesQ scores |
| `aspect_free_cache/aspect_free_flms_pool.parquet` | 60201642 | `e2d84d3f0d8050955c0a71dbedf43eea42f5ab923ce79e8398ec5378b0d9d8c3` | aspect-free FLMS candidate pool (fair free-form protocol) |
| `aspect_free_cache/aspect_free_flms_selected_candidates.parquet` | 683331 | `da709bc7e4fa38d46e6955071585ac6db09b8a669c18299082606781787c4d90` | aspect-free FLMS DPP-K10 candidates |
| `aspect_free_cache/aspect_free_laion_flms.parquet` | 46339 | `afe9981a10d6eaa2bcbe1d154ce39084d8c69eb670bf09d52cd1876733c1f8ca` | aspect-free per-crop LAION scores |
| `aspect_free_cache/aspect_free_segment_decoder_flms.parquet` | 705853 | `a213f4713c13f9caa8793ddb53e0e3ecc968f448a06a72d3784c7e2587e99324` | aspect-free per-crop SAM decoder features |
| `aspect_free_cache/aspect_free_segment_encoder_flms.parquet` | 465100 | `ec7407f31711f39a69ffcec6de7fb72cb8775ae98fd3a8f8cad7a22a7a9b5e71` | aspect-free per-crop SAM encoder features |
| `aspect_free_cache/aspect_free_selection_surface.csv` | 4103 | `3695f36528ba8c4f31bc5e20fbea3b4e8f8e2ce560ec30bf9a5bb8c28a9b68d5` | free-form selection surface |
| `aspect_free_cache/dpp_q_lgbm.pkl` | 174219 | `ae6c326b328242ea112ce9b1859c04a090971b2a5145a24e08adf64e217b46cf` | canonical DPP quality model (LightGBM, frozen) |
| `clip_composition_type.pkl` | 198869 | `1e3668edbffeeac8ab2025c20f3578789397bf434af1e33186898b5a74002e8c` | per-image CLIP composition-type vectors (wide/tight/composition prompts) |
| `clip_multi_prompt_features.parquet` | 1335977 | `d60a33efe4d3e16f78ac7e2f159b787bdf5c9cefec3dd8065d3f08ee6881edcc` | FCDB per-crop CLIP-L multi-prompt similarities |
| `consultant_blend_candidates_fcdb_core_features.parquet` | 2869947 | `952a1c80f242144a4a1b49a75f754bb9d40bc6aa6a3e251bdfdffccf8b42af47` | FCDB twin-candidate manifest with core features |
| `consultant_blend_candidates_flms_core_features.parquet` | 850944 | `4d5c252fcc10bab604c3dadc37c7850ec453a2a77c409e88ee0edab74a077ac4` | FLMS twin-candidate manifest with core features |
| `consultant_blend_clipmp_fcdb.parquet` | 1115639 | `1393e1c7baf00ac22c356ce59385948ac4341077b116713159ffe689e97a31cb` | twin per-crop CLIP-L multi-prompt similarities |
| `consultant_blend_clipmp_flms.parquet` | 330912 | `553b3eff7cc39bfbebdbe28e40804f397fe264bb37771c9d96b2b380a0737bd1` | twin per-crop CLIP-L multi-prompt similarities |
| `consultant_blend_fgaesq_fcdb.parquet` | 108817 | `8038d71c7a7e06f30860f0c76caa695fadffd19fcac2451c42b31d2004caeeb3` | twin per-crop FGAesQ scores |
| `consultant_blend_fgaesq_flms.parquet` | 35039 | `6ea95918a7aced4d2efab2830dc0b3f1e18f092a0ab8c2e9d54a693a44f2f04e` | twin per-crop FGAesQ scores |
| `consultant_blend_laion_fcdb.parquet` | 134900 | `0aa989e8c7998d86050407da9b4041178605f3b63db434f0df9eadd6f4a78a0f` | twin per-crop LAION scores |
| `consultant_blend_laion_flms.parquet` | 42686 | `6fae06f95923652f8f2e6e16090c05137cf58c7ec266919a32d125592ceab237` | twin per-crop LAION scores |
| `consultant_blend_segment_decoder_fcdb.parquet` | 2130573 | `11de6e07cee45c1a55559ce8927687e26b5263596769df225edb643e4593b735` | twin per-crop SAM decoder features |
| `consultant_blend_segment_decoder_flms.parquet` | 614460 | `0455daf6e4cec99e4f08c5e912df32eeb430984f79abc3ba3e6983c0f98a6028` | twin per-crop SAM decoder features |
| `consultant_blend_segment_encoder_fcdb.parquet` | 1342237 | `e2b0213026d58732fbcdea109de7ea95757188da9610748b53bcab525a67c922` | twin per-crop SAM encoder features |
| `consultant_blend_segment_encoder_flms.parquet` | 399502 | `01941fa42eda8231aa02117ae6065bb36354fa3cf6948a4bfadc15615bb41ed4` | twin per-crop SAM encoder features |
| `fcdb_attention_saliency.pkl` | 1837411 | `4776cf26ffa1e7b4ee79d9ab5b9046bfdb9c40c58350901624b4d82996282275` | FCDB DINOv3 V1 saliency maps (historical V1 variant; single-image cached mode) |
| `fcdb_attention_saliency_variants.pkl` | 1837411 | `5ea3e2e0201d18efe26f0545869e03c5a78ac495a1ee4d109ea50d3e33913f18` | FCDB DINOv3 V1 saliency maps (variants extractor; twin featurization) |
| `fcdb_candidate_photographic_features.parquet` | 38793111 | `bc92a5879b997c1f034638ba1bff20beca87effa8a657ce04e32cfa137e014fc` | photographic per-crop features for the full candidate pool |
| `fcdb_clip_cam_heatmaps.npz` | 14527197 | `87d4b281561e011f2113599d3bacdbfed16cca5aace79e822d3847d18f943a54` | FCDB per-image CLIP-L CAM heatmaps (16x16x9) |
| `fcdb_conditioned_candidate_pool.parquet` | 33454452 | `be09903b22bda5fbeb8babe60f43634587de85d1cc7b2130782d070d13a6ad81` | FCDB full conditioned candidate pool (single-image cached-pool mode) |
| `fcdb_laion_aesthetic_scores.parquet` | 291290 | `32814962be6d4b12ad2dabe84b1efd8080f01c7068f8624e63d83dcca242a8eb` | FCDB per-crop LAION aesthetic scores |
| `fcdb_selected_candidates.parquet` | 4147454 | `197d0bce2215c4b870444dd963322b27958b3d5cdf456e1c49f22296177ab7fb` | FCDB canonical DPP-K10 candidate boxes (box<->rank ground truth) |
| `fgaesq_features.parquet` | 138237 | `9848f94db78bb3c98d03a2e58ca691431efb11d9c825187685415d77813d3604` | FCDB per-crop fine-grained aesthetic (FGAesQ) scores |
| `flms_aesthetic_integrity_features.parquet` | 120902 | `e80e5c8fd493bb56631aa8006c17a4ade94daa7e8571c0c41436cf2842a9093e` | FLMS per-crop SAM decoder features (axis 5, thirds prompts) |
| `flms_attention_saliency.pkl` | 543819 | `0ad476db730d12aba493b6cd84253d844c78f866ab79aa207ff6808c944774b4` | FLMS DINOv3 V1 saliency maps |
| `flms_clip_cam_heatmaps.npz` | 4240003 | `2a66e60c5359add51fec076e1de280da106e120ccdd00599f31faa8ff6deb9f8` | FLMS per-image CLIP-L CAM heatmaps |
| `flms_clip_multi_prompt_features.parquet` | 383589 | `7ba421182478196bc8a0e84085b00fffc15bbcfe41822854aadbbd7de979cc21` | FLMS per-crop CLIP-L multi-prompt similarities |
| `flms_fgaesq_scores.parquet` | 64591 | `95ee637118ba9a745029b87338178da73b0ab1ec195a95c2ddcdeb29e7e31ea2` | FLMS per-crop FGAesQ scores |
| `flms_handcrafted_geometry_features.parquet` | 126373 | `6068738dd8219f4de1ab9cb96f644527f88f787ad0071ac6b639f40880df7425` | FLMS per-crop SAM encoder features (hand-crafted scalars) |
| `flms_inspy_masks.pkl` | 32806247 | `54150318094e52b25039e4087fdade0f4c74d2670737e3183de1a1e9b4922396` | FLMS InSPyReNet subject masks (256x256) |
| `flms_laion_scores.parquet` | 86456 | `a88b0e9e7a7bb6ceda230d7b0de1adadd70ec0470d3c6d076b38364ed3de691b` | FLMS per-crop LAION aesthetic scores |
| `flms_saliency_combined_sum.pkl` | 543819 | `de3e82abf6f6ea125dc15f0065af424cc39a572e8013ba89d2b45fffa5671941` | FLMS DINOv3 V_MIX saliency maps |
| `flms_selected_candidates.parquet` | 695448 | `9dc786cf4f29e935fd684cdbe172fd7afbead3e007faedbf0cd06e339009c245` | FLMS canonical DPP-K10 candidate boxes (conditioned reference) |
| `flms_spatial_composition_features.parquet` | 149138 | `d74dd713d4e7289487de5309c9995e5c2d04eb287b2544e3d5605ea4eb75eb3e` | FLMS per-crop SAM encoder features (spatial patch norms) |
| `flms_subject_composition_features.parquet` | 187615 | `2ddb79fcb9c5fd5514f1fce11c4e757310cf0c5f3e6162e7b85a7cd7f2c9cc24` | FLMS per-crop SAM decoder features (axis 3, auto-mask object stats) |
| `flms_subject_composition_integrity_features.parquet` | 420198 | `ddeb1b201951d38be78d78699b5742e1f503a2517f3d9496af1cb490b4e5e278` | FLMS per-crop SAM decoder features (axes 1+2+4) |
| `flms_subject_mask_features.parquet` | 210911 | `65f09d397481055275cca45a4260ec65f4f15d17fc934562384b6383844396cc` | FLMS per-crop SAM encoder features (point-prompt mask stats) |
| `gaicd_deep60f/deep60f_00.parquet` | 508112 | `f399b9c6a241532d970d982085c7ee666410455e210e8c272591eef1b18a3b3a` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_01.parquet` | 511745 | `04bb2f79cf1d8c31a30970a7cca8d3f89e85e932ec25cf4f3a7ea9790994a96b` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_02.parquet` | 508809 | `1a81f44a884ed7b599505a4ed723c738ac0a9a5b0fa29eafe8a723f890ffb3a9` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_03.parquet` | 511178 | `7a800d97fa56c3b0c441eedb363a78dde9568c24d49d682a65129dc7893ab731` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_04.parquet` | 513638 | `180a46f2c7bc6107efe57d97d710ee0b8898aa2e46e6aa20d57b9ae5f75c363b` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_05.parquet` | 512495 | `6b788c25c420205ee1edb9ddc489041383e666babe55949b9112b081b8c5c440` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_06.parquet` | 511501 | `5304b40fbe4bfcb9785196a4a4449fa12fa510d2ee6b88ff88167ff9dbb669b6` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_07.parquet` | 515379 | `ad7d1b0d67445bdc0b8576e8565583a344cf59d051e34fd4215cb6dd04a0ad1b` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_08.parquet` | 511979 | `82f1a986aa4216bcb451125ce1e3326998823d12c20810cee46531a744771a36` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_09.parquet` | 511131 | `da1089bae0ef0f317ec2aac1f761dc70505c9e06b2f336cf4531e1626233189a` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_10.parquet` | 517258 | `472041202998bd33debeaeab2e102f75a391dcff2e3718a2f4c76ec77db7ed17` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_11.parquet` | 513283 | `8309ea654eb0b7bfb7a0b22a41c4034ecbe694c49aa539c6f316e60f22afc5d8` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_12.parquet` | 513297 | `6952fcfae7d5be6fa31d30cbabf5c16400577b79ab50f853aca76fccafb5c8a0` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_13.parquet` | 511172 | `1c2d16346c93b54225822ebe346e697540aa1863d26da1811015d7dd0a075b90` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_14.parquet` | 510086 | `7455ad3343e6565cd02beb179dffac1e00ba1b23408e08f0bcbdcd5842c55f96` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_15.parquet` | 515863 | `1751eab34721fe5f88647a99b0194f50e06367b2684da7a4393ce349e9307e83` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_16.parquet` | 512314 | `f6485a008774634f884555fd01fe89085661dab0d8704d7741af5515a20920c8` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_17.parquet` | 512128 | `3cc1b59c97fa9ffd2d3e8e26255ed8883cf1e66a1aa0b679df8587d343c43c7a` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_18.parquet` | 509552 | `0fa639d17d480a42c3558a853d8a3ee5975f2f3d701998ba352cd1eaa59c5bc1` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_19.parquet` | 512147 | `edb447ee2bb151c952e3df579e0dc7c6cefae363e9a2c2c41d05847741cac265` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_20.parquet` | 512824 | `9ef7c3f4ae3b63ae73bccd0209e670877ab81ee280f6b6ea4b519d7248fc4bba` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_21.parquet` | 510230 | `c464ae91080e940fe71c14b7f7d89aaf5588a694746923dae387292553b923d2` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_22.parquet` | 509447 | `7ac36ff16b6d15fb507e1b2c3a35d598ae4efc3d2f50cfe829186c8ac578e443` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_23.parquet` | 511071 | `218d1620b027563a3898dcbadb4ee1ad5f078401fdda5499cff65ed286829cef` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_24.parquet` | 508847 | `56e6f432418d8cbea3e2093c641a39896efb4b1384c9588ad492cdc53a877f1d` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_25.parquet` | 510639 | `056fd3a15e0b8753f02924610bdc87a24b48e085399aac3e68f9d0fcf254502e` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_deep60f/deep60f_26.parquet` | 221932 | `f246752940acab78342f099a4e50d3ef0a8e6ae25edd3cc875c3425650475d07` | GAICD per-crop 60-feature rows (features and box share one row) |
| `gaicd_selected_candidates.parquet` | 403345 | `746686fcf2ee71fe26dc277d818f624c17feca6fca7ce33e9e81d8b385434d31` | GAICD mixed-training candidate pool (box<->rank ground truth for the self-check) |
| `gaicd_subset_order.parquet` | 33101 | `c020169d38c4690a26ec4ef1f804fae4cf277f2439607935b6c097f7ab81f7aa` | seed-42 shuffle order of GAICD-train images (subset_rank < 1000 = the mixed rows) |
| `handcrafted_geometry_features.parquet` | 430704 | `79710474a1043b169431349b73d9250ca99667d9d2e6e1ede9112ce17439dc22` | FCDB per-crop SAM encoder features (hand-crafted scalars) |
| `inspy_masks_fcdb.pkl` | 112433018 | `8e9e7a95634728e64c0dc9f5f5e9ad5656fa7658141e4609f79d400e4b194eb5` | FCDB InSPyReNet subject masks (256x256) |
| `saliency_maps_dv3s.pkl` | 1837411 | `8affd958c829d4de3581166c0931cabc7e2555d696ed0bdc4cf0dad80aa9bf8e` | FCDB DINOv3 V_MIX saliency maps (single-image cached mode) |
| `spatial_composition_features.parquet` | 508045 | `1f69ae156cb9f69bb672685a4e08fcb0cd07c5c9510e8d9da5c594e8a80708be` | FCDB per-crop SAM encoder features (spatial patch norms) |
| `subject_composition_features.parquet` | 636866 | `e072ddb2c4ff92681497b455e5250fbfce0684171f228cb29844c59ea636df88` | FCDB per-crop SAM decoder features (axis 3, auto-mask object stats) |
| `subject_composition_integrity_features.parquet` | 1546955 | `13cb2c61cd834e200e866565dbf1217e1099cc77b19788738924951894e117bf` | FCDB per-crop SAM decoder features (axes 1+2+4) |
| `subject_mask_features.parquet` | 727103 | `babe3ccf0b4a2df1ec5a80fa18d4b0dce524b3103fc888527c4bdae1e15eb26a` | FCDB per-crop SAM encoder features (point-prompt mask stats) |
| `venus_boxes_fcdb.json` | 15446 | `ef53be03bd5160ad4736597ef4ae59f402fca7326f89ef30e6eba8af73da56ba` | consultant greedy boxes, FCDB test |
| `venus_boxes_flms.json` | 28038 | `6622c51ab8d29ec597b7ef129b3f934cc6c12a5e0d0842a5cc04f60ff14610c3` | consultant greedy boxes, FLMS |
| `venus_multibox_fcdb_te.json` | 59075 | `d1f3ec929044323d739654a160b2de8447d938d7004743427cdec37d33dd7312` | consultant temperature-sample boxes (5 slots), FCDB test |
| `venus_multibox_flms.json` | 97310 | `823003fb0dd66290a059a98eb79d17730947973179c735efdaa7f3f9aa78b84e` | consultant temperature-sample boxes (5 slots), FLMS |
| `venus_promptens_fcdb_te.json` | 45720 | `9d845376d56aad36ae849aabbdb36cf3f9f13b7d7cf41e2f700d201d6ee4e558` | consultant prompt-ensemble boxes (4 paraphrases), FCDB test |
| `venus_promptens_flms.json` | 75747 | `be3cf969e48fe25368fa38ff9a194e03caae26195496559c7e384e668ae7d6bc` | consultant prompt-ensemble boxes (4 paraphrases), FLMS |
