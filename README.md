# cistem2dtm：cisTEM `match_template` 的 PyTorch 重写

这是一个独立、可执行的 PyTorch 2D template matching package。科学实现目标固定为 cisTEM：

- 程序：`src/programs/match_template/match_template.cpp`
- 精确提交：`5bc5f8cd5f804d8b12771aed156dc928510b1e47`
- 短提交号：`5bc5f8c`

项目完全删除 wxWidgets、GUI socket、跨服务器 master/worker 和 cisTEM database glue；基础运行时依赖为：

- PyTorch
- NumPy
- mrcfile

Triton projector 与 correlation multiply 是 Linux/CUDA 上的可选 JIT 加速后端；未安装或首次 JIT 失败时，可按配置退回 PyTorch 路径。

项目重点不是重新设计 2DTM，而是把 cisTEM 的科学流程拆成可测试的 Python/PyTorch 模块，并保留后续加入 isSPA filter、第二遍搜索、分类和 CCF-stack 分析的接口。

> **当前状态**：这是完整、可安装、可运行的研究实现，已经通过 CPU 自动测试和多个端到端 smoke test；但尚未用编译后的精确 cisTEM 提交逐中间数组完成 numerical oracle 验证。因此，不能把当前版本描述为已经证明逐像素 1:1。最需要进一步对照验证的是复杂 DataSizer/post-resize 边界、奇偶尺寸 Nyquist 行为，以及 histogram 边缘平滑。详见 [COMPATIBILITY.md](COMPATIBILITY.md)。

0.3.6新增内容见 [RELEASE_NOTES_0.3.6.md](RELEASE_NOTES_0.3.6.md)，PDB/STAR批处理细节见 [docs/PDB_AND_STAR_BATCH.md](docs/PDB_AND_STAR_BATCH.md)；0.3.5的GisSPA与mixed-float16说明仍见 [RELEASE_NOTES_0.3.5.md](RELEASE_NOTES_0.3.5.md)，GisSPA算法归属说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 已实现功能

- 独立 `DataSizer`，分别保存 MIP/winner 与 moments 使用的 source-valid ROI、可选的 empirical-histogram ROI、threshold pixel count、binning 和输出恢复信息。
- cisTEM 风格的 outlier replacement、径向功率谱白化和第二次模板-footprint 规范化。
- 直接翻译的 cisTEM Euler 网格与旋转矩阵，不调用 SciPy、PyTorch3D 或通用 Euler 库。
- rFFT 半厄米三维 Fourier central-slice projector。
- 两个中央切片后端：
  - `triton`：可选 JIT 融合八邻点 half-Hermitian 插值；CUDA 且可用时为 `auto` 的首选后端。
  - `gather`：显式向量化八邻点插值、负 x 共轭映射和 x=0 Hermitian line 处理；严格兼容/回退后端。
  - `grid_sample`：完整 complex Fourier volume 上的 5D trilinear interpolation；source 默认缓存，用于比较与性能测试。
- cisTEM CTF、输入 whitening curve 和可扩展 extra-filter pipeline。
- 可选 `gisspa_repo` 权重：JSON读取 `kk` 与六个仓库指数系数、每个projection独立径向whitening、signed CTF、双侧cosine带通的等价合并，以及norm-type-1 scaled MIP。
- projection edge-mean subtraction、template-footprint variance normalization。
- batched FFT cross-correlation；默认float32严格路径，以及可选input-prephase、Triton融合复数乘法和mixed-float16 large-FFT路径。
- MIP、Euler/defocus/pixel-size winner maps、逐像素 sum/sum-of-squares，以及 `exact`/`sampled`/`off` 三种 histogram 模式。
- 可选保存全部 CCF stack：关闭、CPU tensor、流式 MRC。
- global Z-score、local flat-field scaled MIP、无 SciPy 理论阈值、peak picking。
- 单图多 GPU：每 GPU 一个进程，按连续 orientation shard 分配，最终按全局 task index 确定性合并。
- RELION micrograph STAR 批处理：纯文本解析 optics/CTF 字段；多 GPU 时每张卡同时处理一张完整 micrograph，单卡时逐张执行。
- PDB 直接生成模板：纯文本读取 ATOM/HETATM，不依赖 Biopython/xpdb，不要求原子序号唯一；保存原始全链质量中心，支持后续按 chain include/exclude 生成共中心 cross-validation 模板。
- 10 阶段 debug checkpoint 和两套 checkpoint 自动比较。
- 保留 cisTEM 的 `MIP=0` 初值与重复的 `psi=0/360°`。
- 修复目标源码中 histogram 边界和非-MKL threshold assignment 的明显错误，而不复制错误行为。

## 安装

### 从 wheel 安装

```bash
python -m pip install cistem2dtm-0.3.6-py3-none-any.whl
```

请先按服务器 CUDA 版本安装合适的 PyTorch。wheel 不会捆绑 CUDA 或 PyTorch 二进制。

### 从源码安装

```bash
python -m pip install -e .
```

验证命令：

```bash
cistem2dtm --version
cistem2dtm stages
cistem2dtm self-test --output-dir self_test_output
```

## 最快的运行方式

先生成一个很小的合成例子：

```bash
cistem2dtm make-example example_2dtm \
  --template-size 12 \
  --search-size 24

cistem2dtm plan example_2dtm/synthetic_config.json --device cpu
cistem2dtm run  example_2dtm/synthetic_config.json --device cpu
```

在真实数据上先生成默认配置：

```bash
cistem2dtm init-config config.json
```

修改至少这些字段：

```json
{
  "input_mrc": "micrograph_or_search_image.mrc",
  "template_mrc": "template_volume.mrc",
  "output_dir": "2dtm_output",
  "output_prefix": "match",
  "search": {
    "pixel_size_angstrom": 1.5,
    "high_resolution_limit_angstrom": 8.0,
    "angular_step_deg": 5.0,
    "in_plane_step_deg": 5.0,
    "symmetry": "C1",
    "particle_radius_angstrom": 100.0
  },
  "runtime": {
    "device": "cuda:0",
    "dtype": "float32",
    "projector_backend": "auto",
    "projection_fft_mode": "auto",
    "cache_sampled_projection_filter": true,
    "cache_orientation_tensors": true,
    "precompute_rotation_matrices": true,
    "orientation_batch_size": 32
  }
}
```

然后：

```bash
cistem2dtm plan config.json
cistem2dtm run config.json
```

`plan` 不执行 template matching，先报告：

- DataSizer 计划，包括 CCF 搜索网格、post-search 输出网格和精确 ROI；
- Euler 数量，包括 out-of-plane 数、最后索引和 psi 数；
- defocus/pixel-size 搜索数量；
- CCF 总数；
- 若保留完整 CCF stack，float32 容量估计。


## 直接使用 PDB 生成 3-D template

原有 `template_mrc` 输入完全保留。要改用 PDB，只需把配置改成：

```json
{
  "template_mrc": "",
  "template": {
    "source": "pdb",
    "pdb_path": "model.pdb",
    "box_size": 448,
    "pixel_size_angstrom": null,
    "resolution_angstrom": null,
    "center": true,
    "include_hetatm": false,
    "include_chains": [],
    "exclude_chains": [],
    "center_reference_json": null,
    "raster_backend": "vectorized",
    "atom_batch_size": 1024
  }
}
```

`pixel_size_angstrom=null` 时继承 `search.pixel_size_angstrom`；`resolution_angstrom=null` 时默认使用：

```text
resolution = 2 × template pixel size
```

PDB 使用固定列纯文本读取，每一条 `ATOM` 记录都独立加入，不检查或重排 atom serial；chain ID 保留。默认先从**未进行 chain include/exclude 的原始原子集合**计算质量中心，再将选中链按同一个中心平移，因此以后删去 A/B/C 等 chain 生成 cross-validation template 时不会重新对中。中心记录写入：

```text
<output_dir>/_template_cache/<pdb_stem>_center.json
```

自动生成的 MRC 和完整生成参数也保存在 `_template_cache`，配置未变化时会复用。可以单独生成而不跑 2DTM：

```bash
cistem2dtm prepare-template examples/pdb_template_config.json
```

若要删除 chain B，但继续使用原始完整结构中心：

```json
{
  "template": {
    "source": "pdb",
    "pdb_path": "model.pdb",
    "box_size": 448,
    "exclude_chains": ["B"],
    "center_reference_json": "2dtm_output/_template_cache/model_center.json"
  }
}
```

默认 `vectorized` rasterizer 将多个原子 patch 分块后用 `scatter_add` 写入体数据；`reference` 保留用户提供的逐原子 PyTorch 实现，主要用于数值对照。两者使用相同的 H/C/N/O/P/S 原子数与质量表、截断宽度和指数核。

参考配置：[`examples/pdb_template_config.json`](examples/pdb_template_config.json)。

## 读取 RELION `micrographs_ctf.star` 批量处理

将 `input_mrc` 留空，并指定：

```json
{
  "input_mrc": "",
  "micrographs_star": "micrographs_ctf.star",
  "batch": {
    "micrograph_root": null,
    "output_subdirectories": true,
    "continue_on_error": true,
    "write_resolved_configs": true,
    "manifest_prefix": "batch",
    "first_micrograph": 0,
    "last_micrograph": null
  },
  "runtime": {
    "devices": [0, 1, 2, 3]
  }
}
```

解析器只使用 Python 标准库，不依赖 `starfile`、pandas 或 SciPy。它合并：

- `data_optics`：`_rlnMicrographPixelSize`、电压、Cs、振幅衬度；
- `data_micrographs`：MRC 文件、optics group、DefocusU/V、DefocusAngle，以及可选 phase shift。

相对 micrograph 路径默认相对于 STAR 文件所在目录；`batch.micrograph_root` 可覆盖根目录。每张图的输出默认位于：

```text
<output_dir>/<STAR中的相对micrograph路径去掉扩展名>/
```

调度方式与单图多 GPU 不同：

- `runtime.devices=[0,1,2,3]`：建立四个持久 worker，每个 GPU 同时处理一张**完整 micrograph**，完成后领取下一张；
- 单 GPU 或 CPU：按 STAR 顺序 one-by-one；
- 不会把同一张图再次按 orientation 分片。

最终写出：

```text
<output_dir>/batch_manifest.json
<output_dir>/batch_manifest.tsv
```

其中记录每张图的 optics/CTF、GPU、运行状态、耗时、阈值、peak 数和输出文件。

参考配置：

- [`examples/micrographs_star_mrc_config.json`](examples/micrographs_star_mrc_config.json)
- [`examples/micrographs_star_pdb_config.json`](examples/micrographs_star_pdb_config.json)
- [`examples/micrographs_ctf_example.star`](examples/micrographs_ctf_example.star)

先只检查 STAR 展开结果：

```bash
cistem2dtm plan examples/micrographs_star_pdb_config.json
```

然后运行：

```bash
cistem2dtm run examples/micrographs_star_pdb_config.json
```


## 320 px、1.5 Å、5 Å 这一关键案例

目标源码会把 320×320 输入按 5 Å 限制 Fourier 降采样到 192×192，搜索像素变为 2.5 Å；CCF、MIP winner 更新和 raw sum/sumSq 都在 192×192 网格上计算。开启：

```json
"apply_result_rescaling": true
```

后，`ResizeImage_postSearch` 才把连续统计图通过 Fourier 上采样、把 Euler/defocus/pixel-size winner 图通过最近邻规则恢复为 320×320。关闭该选项时，5 Å 情况输出 192×192，这与目标提交的默认 CLI 行为一致。

同一输入若实际 high-resolution limit 是 Nyquist 3 Å，则不做 Fourier 降采样；标准 FFT 路径会把 320 padding 到 324 做 CCF，再裁回 320 输出。因此，看到 320×320 输出本身不能证明是否曾经在 192×192 搜索，必须同时检查 histogram 的 effective trial count 和 `plan`。

### 0.3.0 的 padding 与统计 ROI 兼容配置

0.2.0 在“只扩大 FFT box、不做 Fourier resampling”的 320→324 分支中硬编码了 zero padding。0.3.0 改为所有 sizing 分支都服从：

```json
{
  "padding_mode": "noise",
  "random_seed": 0
}
```

`noise` 在边缘生成可复现的 N(0,1) 噪声，适用于第一遍 whitening 已将实空间方差规范化为1的 cisTEM 流程。还可选择 `zero`、`replicate`（重复边缘的对称延拓）和旧版 `edge`（边缘均值填充）。

为匹配当前预编译参考程序的 full-Nyquist histogram，默认还使用：

```json
{
  "statistics_roi_mode": "precompiled",
  "statistics_roi_border_pixels": null,
  "threshold_pixel_mode": "full"
}
```

对320→324，该 profile 将三个范围明确分开：

```text
FFT work grid / threshold pixels       324 × 324
MIP、winner、sum/sumSq moments ROI       320 × 320
empirical histogram compatibility ROI    316 × 316
最终输出                                 320 × 320
```

316×316 是根据预编译参考程序的 histogram 计数反推的**仅 histogram 兼容行为**，不会再影响 `sum/sumSq`、`corr_average`、`corr_stddev` 或 scaled MIP。目标源码在 `ResizeImage_postSearch` 中以同一有效区域控制 moments，并在最终 post-search mask 外清零 accumulators；v0.3.2 已按这一行为修正。要同时恢复目标源码的 histogram/threshold ROI 语义，设置：

```json
{
  "statistics_roi_mode": "source",
  "threshold_pixel_mode": "source"
}
```

包内提供可直接修改的参考配置和四组A/B脚本：

```bash
cp examples/precompiled_full_nyquist_config.json config_nyquist.json
# 修改输入、模板和显微镜参数后：
bash examples/run_padding_roi_ab_test.sh config_nyquist.json
```

也可不修改JSON，直接用CLI覆盖 `--padding-mode`、`--statistics-roi-mode`、`--statistics-roi-border-pixels` 和 `--threshold-pixel-mode`。

角度网格已改为 C++ `float` 累加语义。C1、out-of-plane=3°、in-plane=3°时：

```text
out-of-plane positions = 4606
last out-of-plane index = 4605
psi positions           = 121
total orientations      = 557326
```

原先 Python float64 循环得到 4586（最后索引4585），会改变 MIP 极值统计。

## Histogram / 配置一致性诊断

```bash
cistem2dtm diagnose-histogram 2dtm_match_histogram.txt config.json
```

该命令将 histogram 首个极低 SNR bin 的理论 survival 反推出 effective independent trials，并与当前配置的：

```text
valid spatial pixels × orientations × defocus positions × pixel-size positions × independence fraction
```

逐项比较。报告还会给出“假设其他计数都相同”时推断的有效空间像素数；该项只是因子分解提示，不能替代对 defocus、pixel-size search 和 orientation subset 的核对。

## 单图的 orientation-sharded 多 GPU

以下模式只适用于 `input_mrc` 单图配置；STAR 批处理采用上一节的一图一卡调度。

单 GPU：

```bash
cistem2dtm run config.json --device cuda:0
```

同一服务器的 4 张 GPU：

```bash
cistem2dtm run config.json --devices 0,1,2,3
```

多 GPU 实现使用 `spawn`，每张卡持有自己的输入 FFT、template 和局部统计量。orientation 被连续分块；最终：

- sum、sum-of-squares、histogram 相加；
- MIP 取最大值；
- 相同正 MIP 时使用更早的全局 task index，复现单进程严格 `>` 的顺序语义；
- 全部 final scaling 和输出仅执行一次。

多 GPU 只针对一台服务器，不包含跨服务器通信。

## Storage dtype 与 range-safe mixed-float16 correlation（0.3.5）

旧配置：

```json
"runtime": {
  "dtype": "float16"
}
```

主要控制输入/template storage。whitening、448/450³ template FFT以及large correlation FFT可能因为尺寸或算子支持自动提升为float32/complex64，所以只改这个字段**不保证加速**。

真正控制热路径的是：

```json
"runtime": {
  "dtype": "float32",
  "correlation_precision": "mixed_float16",
  "center_phase_mode": "input",
  "correlation_multiply_backend": "triton",
  "half_fft_shape_mode": "pad_to_power2",
  "mixed_precision_range_scaling": true,
  "mixed_precision_l1_target": 16384.0,
  "mixed_precision_health_check_batches": 1,
  "mixed_precision_fallback_to_float32": true
}
```

mixed路径保持科学预处理、3-D template FFT、central slice、每projection whitening和small-projection normalization为float32，只将：

```text
large projection RFFT
prephased input Fourier storage
fused image × conjugate(projection)
large CCF IRFFT
```

改为float16/complex32。IRFFT后CCF立即恢复float32，因此sum/sumSq、MIP、winner、CCF-stack consumer和scaled MIP接口不变。

0.3.5修复了真实projection在half RFFT中可能出现的动态范围问题：FFT前先去除projection均值，再根据L1上界施加逐projection的二进制精确缩放；逆缩放在融合Triton复乘中以float32恢复。程序还会对真实首批projection Fourier、correlation product和CCF做finite/nonzero检查。若失败：

- `mixed_precision_fallback_to_float32=false`：立即报错，不再跑完整搜索后才失败；
- `true`：在当前已解析的FFT work grid上切换为float32 correlation，并在metadata中记录原因。

CUDA half FFT要求两个变换长度为2的幂。`half_fft_shape_mode=pad_to_power2`只改变大尺寸二维correlation work grid，例如4096×3888→4096×4096；3-D template仍保持cisTEM-standard sizing。参考 `examples/performance_mixed_float16_config.json` 与 `examples/validate_mixed_float16_safe.sh`。

`correlation_precision=float32`仍是默认值，并保持0.3.4 cisTEM权重路径的数值行为。


## 大图与 `max_search_size`

目标提交默认把搜索最长边限制为1024。对于大于1024的micrograph，即使配置的是Nyquist，DataSizer也会提高实际high-resolution limit并对图像和模板降采样。若要在3838×3710 micrograph上使用完整频率和4096×3888 FFT网格，应显式设置：

```json
"search": {
  "max_search_size": 4096
}
```

并先运行：

```bash
cistem2dtm plan config.json
```

确认 `max_search_size_applied=false`、search pixel size未改变，且template没有被缩小。

## 性能计时与瓶颈定位（0.3.1+）

完整micrograph建议使用低开销抽样模式：

```bash
cistem2dtm run config.json \
  --device cuda:0 \
  --orientation-batch-size 16 \
  --timing \
  --timing-mode sampled \
  --timing-warmup-batches 10 \
  --timing-sample-every-batches 500 \
  --timing-max-samples 200 \
  --timing-progress-every-batches 1000
```

输出：

- `match_timing_summary.txt`：按CUDA和host占比排序的摘要；
- `match_timing.json`：完整报告；
- `match_timing_wall_sections.tsv`：preprocess/search/finalize/I/O等major wall sections；
- `match_timing_sections.tsv`：抽样batch内各操作的mean/median/p90/p95/p99；
- `match_timing_batches.tsv`：抽样batch、吞吐量和显存状态。

主要细分包括central-slice gather、projection padding、full-grid projection FFT、CCF inverse FFT、moments、histogram、MIP更新和隐式GPU同步。`sampled`模式只在被抽样batch末尾同步一次；`synchronized`会逐batch同步，只适合很短的orientation subset。

配置也可以直接写：

```json
"timing": {
  "enabled": true,
  "mode": "sampled",
  "warmup_batches": 10,
  "sample_every_batches": 500,
  "max_samples": 200,
  "progress_every_batches": 1000,
  "print_summary": true,
  "save_batch_samples": true,
  "record_cuda_memory": true,
  "emit_nvtx": false
}
```

详细字段解释和报告判读见 [docs/PERFORMANCE_TIMING.md](docs/PERFORMANCE_TIMING.md)。包内另有 [examples/performance_timing_config.json](examples/performance_timing_config.json) 和 [examples/run_timing_profile.sh](examples/run_timing_profile.sh)。


## Histogram 性能模式与 scaled-MIP 边缘修正（0.3.2）

v0.3.1 的实测中，逐 batch 精确 histogram 约占 40% CUDA 时间。v0.3.2 提供：

```json
"runtime": {
  "histogram_mode": "exact",
  "histogram_backend": "histc",
  "histogram_sample_orientation_stride": 16,
  "histogram_sample_pixel_stride": 4
}
```

三种模式：

- `exact`：访问全部 histogram ROI 内的 CCC；默认使用优化后的 `histc` backend。为避免 float32 计数丢失整数精度，内部将输入拆成不超过 `2^24` 个样本的 partial histograms，再以 float64 累加。它仍遵守 cisTEM 的半开区间 `[min,max)`。
- `sampled`：按全局 task index 和固定 pixel stride 做确定性抽样，再按完整 population 重加权；仅用于诊断分布。
- `off`：完全跳过 empirical histogram 热循环和 histogram TSV 输出。MIP、scaled MIP、`corr_average`、`corr_stddev`、winner maps、理论 threshold 与 peak picking 均仍由 moments 和解析公式计算，不依赖 histogram。

完整生产搜索推荐：

```bash
cistem2dtm run config.json \
  --device cuda:0 \
  --histogram-mode off
```

保留精确 histogram：

```bash
cistem2dtm run config.json \
  --histogram-mode exact \
  --histogram-backend histc
```

用于验证旧实现：

```bash
cistem2dtm run config.json \
  --histogram-mode exact \
  --histogram-backend bincount
```

同时，v0.3.2 修正了 scaled MIP 边缘明暗环：v0.3.0/0.3.1 的 `precompiled` histogram ROI 曾被错误地同时用于 `sum/sumSq`，使 histogram ROI 与 MIP ROI 之间的一圈像素只有 global Z-score、没有 local flat-field。现在 moments 始终使用 source-valid MIP ROI，histogram ROI 只控制 histogram；post-search 后再按 cisTEM 的 binarized cosine valid mask 清零 moments、均值填充显示图。原版缩放公式仍保持不变。

包内示例：

- `examples/performance_fast_config.json`：`histogram_mode=off`；
- `examples/performance_sampled_histogram_config.json`：抽样 histogram；
- `examples/performance_timing_config.json`：精确 histogram + timing。

## Projector、orientation 与 FFT 加速（0.3.3）

v0.3.3 将此前 profile 中最重的 central-slice gather、每批 metadata 构建和大尺寸 Fourier 临时复制改为可选的加速路径。推荐生产配置：

```json
"runtime": {
  "projector_backend": "auto",
  "triton_fallback_to_gather": true,
  "cache_grid_sample_source": true,
  "cache_sampled_projection_filter": true,
  "cache_orientation_tensors": true,
  "precompute_rotation_matrices": true,
  "projection_fft_mode": "auto",
  "histogram_mode": "off"
}
```

主要变化：

- 搜索循环只更新 MIP 和 `winner_task_index`；五张参数图在搜索结束后一次性反解。
- 全部 orientation angles/index 预先变为 tensor，默认分块预计算 rotation matrices。
- 固定 defocus/pixel-size 条件下 sampled projection filter 只建立一次。
- `auto` 在 CUDA+Triton 可用时使用融合八邻点 kernel，否则自动回退到 `gather`。
- `grid_sample` 的 full-complex 双通道 source 默认缓存。
- 删除两个冗余 clone，并把 projection Fourier buffer 原地复用为 correlation product。
- `projection_fft_mode=direct_s` 使用 `rfft2(..., s=search_shape)` 加缓存 phase ramp，避免显式 padded-real tensor；`auto` 当前选择此路径。

严格数值对照建议使用：

```bash
cistem2dtm run config.json \
  --projector-backend gather \
  --projection-fft-mode explicit
```

这条路径与 v0.3.2 synthetic 端到端输出逐值一致。加速后端验证：

```bash
bash examples/validate_accelerated_backends.sh config.json
```

Stage 6 会保存 gather、grid_sample，以及 Triton 可用时的 Triton slice 和差值。Triton 是可选依赖；可尝试：

```bash
python -m pip install 'cistem2dtm[triton]'
```

很多 Linux CUDA PyTorch 环境已经带有匹配版本，无需重复安装。完整说明见 [RELEASE_NOTES_0.3.3.md](RELEASE_NOTES_0.3.3.md)。

## Correlation phase、融合乘法与 mixed precision（0.3.4–0.3.5）

### 严格float32默认

```json
"runtime": {
  "correlation_precision": "float32",
  "center_phase_mode": "auto",
  "correlation_multiply_backend": "auto"
}
```

默认仍按projection逐batch应用center phase，并使用PyTorch复数运算顺序。

### float32快速路径

```json
"runtime": {
  "correlation_precision": "float32",
  "center_phase_mode": "input",
  "correlation_multiply_backend": "triton"
}
```

center-embedding phase被一次性移到input Fourier，Triton在一次频谱遍历中完成`image * conjugate(projection)`。尺寸和dtype仍是float32/complex64。参考 `examples/performance_float32_prephase_config.json`。

### range-safe mixed-float16

```json
"runtime": {
  "correlation_precision": "mixed_float16",
  "center_phase_mode": "input",
  "correlation_multiply_backend": "triton",
  "half_fft_shape_mode": "pad_to_power2",
  "mixed_precision_range_scaling": true,
  "mixed_precision_l1_target": 16384.0,
  "mixed_precision_health_check_batches": 1,
  "mixed_precision_fallback_to_float32": true
}
```

完整验证可用：

```bash
bash examples/validate_correlation_precision.sh config.json
bash examples/validate_mixed_float16_safe.sh config.json
```

本版本**没有**融合moments/MIP/argmax/winner；完整float32 CCF batch仍在IRFFT后提供给accumulator、CCF-stack和后续函数。

## GisSPA repository weighting（0.3.5）

启用：

```json
"weighting": {
  "mode": "gisspa_repo",
  "kk": 3.0,
  "a": -9.32,
  "b": 2.65,
  "b2": 0.01908,
  "bfactor": -78.7757,
  "bfactor2": -12.9121,
  "bfactor3": 1.28732,
  "cosine_edge_width_pixels": 8.0,
  "image_high_frequency_damping": true,
  "projection_whitening_epsilon": 1e-12
}
```

该模式只移植当前工作流需要的GisSPA仓库算法：

1. 每个3-D central-slice projection用自己的径向power独立whitening；
2. 读取JSON中的`kk`和六个指数曲线参数；
3. 使用signed CTF，不对micrograph做phase flip；
4. 图像端实数权重在完整search Fourier网格上计算；projection端在自身网格上使用signed CTF权重，因此不需要对micrograph做phase flip；
5. 图像端保留低/高分辨率cosine shoulder，仓库template分支只施加高分辨率shoulder；两者在CCF的Fourier乘法中自然组合；
6. 继续使用当前一遍式MIP/sum/sumSq和scaled MIP，也就是GisSPA `norm_type=1`最佳分数图所需的统计量。

没有加入：

- `norm_type=0`；
- HDF/template-bank projector；
- soft mask；
- tiled/overlap backend；
- GisSPA输入输出格式；
- 所有orientation-specific候选输出；
- paper-only FSC/SSNR模式。

参考配置：

```text
examples/gisspa_repo_config.json
examples/gisspa_repo_mixed_float16_config.json
examples/user_v034_config_gisspa_v035.json
```

短区间检查：

```bash
bash examples/validate_gisspa_weighting.sh config.json
```

### 固定阈值覆盖

默认：

```json
"search": {
  "threshold_override": null
}
```

`null`使用cisTEM理论阈值；设置数值（例如`8.0`）则直接覆盖peak picking阈值。该选项不改变scaled MIP本身。


## 保存全部 orientation 的 CCF stack

默认关闭：

```json
"ccf_stack_mode": "none"
```

保存为 CPU tensor，搜索结束后写 `.pt`：

```bash
cistem2dtm run config.json \
  --ccf-stack-mode cpu \
  --ccf-stack-path output/all_ccf.pt
```

逐 batch 流式写 MRC stack：

```bash
cistem2dtm run config.json \
  --ccf-stack-mode mrc \
  --ccf-stack-path output/all_ccf.mrc
```

两种模式都会额外写一个 TSV metadata 文件。stack 顺序为：

```text
pixel-size offset -> defocus offset -> global orientation index
```

TSV 包含 `task_index`、orientation index、phi/theta/psi、defocus offset 和 pixel-size offset。后续 second pass 或分类应优先依据 `task_index`/TSV，而不是假设用户修改后的搜索范围仍有固定层数。

## 10 阶段调试

查看阶段：

```bash
cistem2dtm stages
```

| 阶段 | 名称 | 主要输出 |
|---:|---|---|
| 1 | input_and_sizing | 原始图像、模板、DataSizer plan、MRC pixel size |
| 2 | first_whitening | outlier mask、径向 whitening curve、第一次白化结果 |
| 3 | resize_and_second_normalization | search real image、valid mask、search Fourier data |
| 4 | euler_grid_and_rotations | Euler 列表、搜索步长、rotation matrices |
| 5 | ctf_and_projection_filter | CTF、CTF × whitening filter |
| 6 | fourier_central_slice | Fourier slice、sampled filter、filtered slice、两 projector 差值 |
| 7 | projection_realspace_normalization | edge mean、variance、normalized/padded projection |
| 8 | single_orientation_ccf | 第一批完整 CCF 与 metadata |
| 9 | raw_search_accumulators | raw MIP、winner maps、sum、sumSq、histogram |
| 10 | postresize_scaled_mip_and_peaks | MIP Z-score、scaled MIP、local mean/std、threshold、peaks |

建议先固定一个 orientation：

```bash
cistem2dtm run config.json \
  --device cuda:0 \
  --single-orientation 23 47 71 \
  --debug-dir debug_torch \
  --compare-projectors \
  --record-timing
```

在某阶段停止：

```bash
cistem2dtm run config.json \
  --single-orientation 23 47 71 \
  --debug-dir debug_stage6 \
  --stop-after-stage 6
```

与另一套 NPY checkpoint 对比：

```bash
cistem2dtm compare-debug \
  debug_cistem_reference \
  debug_torch \
  --rtol 1e-5 \
  --atol 1e-6 \
  --report-prefix comparison/report \
  --fail-on-difference
```

输出 JSON 和 TSV，逐文件报告：maximum absolute error、mean absolute error、RMSE、relative L2 和 `allclose`。

完整建议见 [docs/DEBUGGING.md](docs/DEBUGGING.md)。

## 主要输出

以 `output_prefix=match` 为例：

- `match_mip.mrc`：global-normalized MIP，即 MIP Z-score。
- `match_scaled_mip.mrc`：可选 local flat-field 后的最终 scaled MIP；v0.3.2 中 histogram compatibility ROI 不再影响其边缘。
- `match_phi.mrc`、`match_theta.mrc`、`match_psi.mrc`：获胜 Euler maps。
- `match_defocus.mrc`：获胜 defocus **offset**。
- `match_pixel_size.mrc`：获胜 pixel-size **offset**。
- `match_corr_average.mrc`、`match_corr_stddev.mrc`：逐像素 orientation background。
- `match_raw_mip.mrc`、`match_raw_correlation_sum*.mrc`：可选原始累加量。
- `match_histogram.tsv`：固定 SNR bin、raw/smoothed count、observed/expected survival；`histogram_mode=off` 时不生成。
- `match_peaks.tsv`：threshold 后的非极大值抑制结果。
- `match_orientations.tsv`：本次实际搜索的 orientation。
- `match_config.json`：最终实际配置。
- `match_metadata.json`：尺寸、数量、阈值、dtype、设备、兼容性选择和运行信息。

## 包结构

```text
src/cistem2dtm/
  config.py       参数和 JSON schema
  datasizer.py    搜索尺寸、ROI、padding、resize 和 post-resize
  fft.py          cisTEM-compatible FFT normalization/layout helpers
  image_ops.py    白化、径向曲线、pixel-size change、Image 子集
  geometry.py     EulerSearch 和直接翻译的 cisTEM rotation matrix
  ctf.py          CTF
  filters.py      CTF × whitening × extra filters；isSPA 扩展点
  projector.py    half-Hermitian gather 与 grid_sample projector
  statistics.py   MIP、moments、histogram、scaled MIP、threshold、peaks
  stack.py        可选 CCF stack 与多 GPU shard 合并
  matcher.py      端到端计算流程
  multigpu.py     单服务器多 GPU 调度
  debug.py        10 阶段 checkpoint
  timing.py       低开销wall/CUDA-event性能分析与报告
  validation.py   debug directory 数值比较
  cli.py          命令行
```

## 当前兼容性边界

当前版本有意保持：

- MIP 初值为 0；
- MIP 更新使用严格 `>`；
- psi=0° 和 360° 都进入搜索及统计；
- CTF 符号和 cisTEM rotation matrix；
- 不使用 SciPy；
- histogram 越界和非-MKL threshold assignment 的明显 bug 不复制。

尚需用精确 cisTEM oracle 进一步确认：

1. 所有非方形、odd/even 和强 resampling 组合下的 DataSizer 尺寸搜索。
2. post-search continuous/label map 的逐像素边缘映射和 valid mask。
3. Fourier resize 的 Nyquist 平面细节。
4. half-Hermitian gather 在所有偶数边界上的严格值。
5. Savitzky-Golay 最前/最后两个 histogram bin 的 cisTEM `Curve` 边界行为。
6. CUDA FP16 与 cisTEM 自定义 FastFFT/half kernel 的数值差异和吞吐量。

项目默认选择可复现、可 debug 的标准 PyTorch FFT 路径，不声称 bitwise 重现 cisTEM FastFFT。详细状态见 [COMPATIBILITY.md](COMPATIBILITY.md)。

## 运行测试

```bash
python -m compileall -q src tests
PYTHONPATH=src pytest -q
```

当前发布包的验证记录见 [TEST_REPORT.md](TEST_REPORT.md)。

## 源码对应关系与许可证

函数级对应表见 [SOURCE_MAPPING.md](SOURCE_MAPPING.md)。本项目是独立 PyTorch 重写；cisTEM 源码提交和许可证归属见 [LICENSE](LICENSE)。
