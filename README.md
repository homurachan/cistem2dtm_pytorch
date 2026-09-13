# cistem2dtm：cisTEM `match_template` 的 PyTorch 重写

这是一个独立、可执行的 PyTorch 2D template matching package。科学实现目标固定为 cisTEM：

项目完全删除 wxWidgets、GUI socket、跨服务器 master/worker 和 cisTEM database glue；基础运行时依赖为：

- PyTorch
- NumPy
- mrcfile

Triton projector 与 correlation multiply 是 Linux/CUDA 上的可选 JIT 加速后端；未安装或首次 JIT 失败时，可按配置退回 PyTorch 路径。

项目重点不是重新设计 2DTM，而是把 cisTEM 的科学流程拆成可测试的 Python/PyTorch 模块，并保留后续加入 isSPA filter、第二遍搜索、分类和 CCF-stack 分析的接口。

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

## 运行方式

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


## 许可证

本项目是独立 PyTorch 重写；cisTEM 源码提交和许可证归属见 [LICENSE](LICENSE)。
