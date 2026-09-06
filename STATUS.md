# CosmoGrid Mock Pipeline — 项目状态文档

> 记录时间:2026-09-05
> 仓库:`git@github.com:suqik/cosmogrid_mock_pipe.git`(分支 `master`)

## 1. 项目概览

基于 CosmoGrid / FastPM / Abacus 模拟套件生成模拟星系(前景)、空洞与背景 shape(剪切)目录的管线,配套 GGL / void-lensing 测量代码。

- 前景(lens):PKD / Rockstar halo catalog → HOD 填充星系 → box-to-lightcone → BOSS / 2dFLenS 巡天几何 + n(z)
- 空洞:DIVE 空洞查找器
- 背景(src):按巡天 mask + tomo n(z) 生成源位置 → 从剪切图分配 shear → 加形状噪声
- 测量:`measurements/`(GGL 计算器、SurveyData 容器、run_ggl / run_vl)

## 2. Git 状态(截至 2026-09-05)

- `master` = `origin/master` @ `993c2aa feat: organize CosmoGrid and FastPM run scripts`(已同步,无领先/落后)
- 备份分支:`wip/local-backup`(de15dfe)—— 同步远程前本地旧结构修改的完整快照,可随时回溯
- **已提交并推送(Abacus 相关工作,本会话)**:
  - `handler.py`:source clustering(`gen_gal_positions(method="density")`、`mass_maps`/`z_to_mass_label`、`PipeConfig.seed_pos`)
  - `runner.py`:`AbacusRunner` 类 + mass map 加载
  - `abacus_runs/run_mock_shape.py`、`STATUS.md`
- **仍未提交(本地保留)**:

| 文件 | 内容 |
|---|---|
| `cosmogrid_runs/run_mock_gal.py` | 前景 n(z) 路径拆分(boss_nofzs / 2dflens_nofzs) |
| `cosmogrid_runs/run_mock_shape.py` | KiDS1000-North 背景巡天切换、n(z) 路径、输出目录 kids1000_north_2tomos |
| `cosmogrid_runs/run_mock_void.py` | ngal_ref=4e-4、nofz_method=downsample、sigma_* 参数、n(z) 路径 |
| `cosmogrid_runs/run_sampling_hod.py` | 同上配置修改 |
| `measurements/calculator.py` | jackknife `return_samples` 选项、njk 参数修复 |
| `runner.py`(局部 hunk) | `CosmoGridRunner._load_shear_maps` 打印、`gen_mock_void_boostrap` 方法 |

- 未跟踪:`measurements/ngals_list.txt`、`measurements/showup/`、`notebooks/`
- 测试:90/90 通过(`PYTHONPATH=/usr/lib/python3/dist-packages .pixi/envs/default/bin/python -m pytest tests/`)

## 3. 代码结构

```
runner.py                     # CosmoGridRunner / FastPMRunner / AbacusRunner
handler.py                    # PipeConfig, CatalogLoader, HODPopulator,
                              # SurveyGenerator, VoidFinder, ShearAssigner
cosmogrid_runs/               # CosmoGrid V1 的 4 个 MPI 运行脚本
fastpm_runs/                  # FastPM 的 4 个运行脚本
abacus_runs/                  # Abacus 运行脚本(新增)
  run_mock_shape.py           # 从 Abacus shear/mass maps 生成 shape catalog
utils/                        # io_func / hod_utils / mkfore_utils / mkback_utils / match_fore_back
measurements/                 # GGL / void-lensing 测量
aux/  develop/                # 辅助脚本 / 开发草稿
tests/                        # 90 个测试(远程仓库自带)
catalogs/                     # 巡天 mask 与 n(z)(gitignored)
```

## 4. AbacusRunner(新增于 runner.py)

只做 shape catalog,不管 galaxy / void。接口:

```python
AbacusRunner.build_shape_runner(
    config, shear_map_fmt, back_mask_fnames_dict, back_nofz_fnames_dict,
    back_survey_labels_dict, back_ngals_dict, tomo_labels_dict,
    redshift_src_list, shear_ofmt,
    mass_map_fmt=None, z_to_mass_label=None,
    position_method="random", bias=1.0)
runner.gen_mock_shear(save=True)
```

- `_load_shear_maps`:读 Abacus `gamma{1,2}_rt_z{z}.fits`(HEALPix RING, NSIDE=4096,196608 行 × 1024 float32 展平),损坏文件给出明确报错
- `_load_mass_maps`:读 `shell_{:d}.fits` 的 `SIGNAL` 列(逐像素一行 float32),转 `δ = Σ/⟨Σ⟩_fullsky − 1`
- `gen_mock_shear`:背景位置 → 覆盖截断(z < 2.05,超出即剔除并打印)→ 赋 shear → 赋权重
- 另有 `CosmoGridRunner.gen_mock_void_boostrap`(bootstrap 空洞子样本)为本地新增方法

## 5. Source Clustering(handler.py 的 ShearAssigner)

`gen_gal_positions(ngal, survey_name, tomo_label, survey_label, method="random", bias=1.0, seed=None)`:

- `method="random"`:原有均匀撒点逻辑(未改动行为,可选 seed)
- `method="density"`:GLASS 式逐壳层逐像素 Poisson 采样
  - 壳层边缘与 `assign_shear_vals` 同款约定;tomo n(z) 按边缘重分桶得每壳层份额 f_s
  - 每像素期望数 `λ = ngal × A_cell × f_s × (1 + b·δ)`;λ<0 截断为 0 并打印
  - `counts = rng.poisson(λ)`;同一 rng 在像素内**拒绝采样**均匀放点(盒采样 + ang2pix 校验,20 轮兜底);壳层内 z 均匀采样
  - photo-z 误差仍用 `seed_Phz`(与随机模式一致);各壳层顺序拼接,不 shuffle
  - 种子:新增 `PipeConfig.seed_pos = 0`
- 构造:`ShearAssigner(..., mass_maps=None, z_to_mass_label=None)`(默认 None,对 CosmoGrid / FastPM runner 无影响)

**实现中修复的两个坑**:
1. healpy 1.18 `hp.boundaries` 返回布局因输入类型而异(scalar (3,4) / array (n,3,4)),统一转置处理
2. 跨赤道像素的 cos(dec) 采样区间退化 → 改用北极角 θ=90°−dec 采样(与 `gen_angle_positions_from_healpix` 同款)

## 6. Abacus 数据资产状态(/data2/suchen/Abacus/)

| 目录 | 内容 | 状态 |
|---|---|---|
| `shear_maps/` | 80 个 `gamma{1,2}_rt_z{0.05..2.00}.fits`(HEALPix RING NSIDE=4096 全天空) | ✅ 全部完好(此前 20 个云端损坏,已由上传方修复后重下,本地已全量校验) |
| `mass_maps/` | `shell_{:d}.fits`(`SIGNAL` 面密度列) | ⚠️ 目前只有 **shell_12 / shell_13**(40 壳层未齐) |
| `shape_cats/` | 生成的 shape 目录 | 见 §7 |
| `refetch.sh` / `refetch_filelist.txt` | 损坏文件重下脚本 | 备用 |
| `transfer.sh` / `newfiles` | 用户下载脚本 / 损坏文件清单 | 用户维护 |

## 7. 已生成的 shape 目录(shape_cats/)

| 文件 | 模式 | 说明 |
|---|---|---|
| `abacus_run_0_kids_north_5tomos.fits` | random | KiDS1000-North,5 tomo,9,843,285 源,完整 40 壳层剪切图 |
| `abacus_run_0_kids_north_5tomos_density.fits` | density | 同上但 source clustering,9,740,357 源,用 shell_12/13 质量图 |

density 目录验证:各 tomo 数目 ≈ random 模式(带 Poisson 涨落);n(z) 保持(平均 z_ph 一致);g1/g2 噪声 std=0.300;聚类相关(Pearson r,逐像素计数 vs δ):z<0.5 → shell_12 r≈0.85,z≥0.5 → shell_13 r≈0.90。

## 8. 当前测试期的临时对应表(注意!)

```python
z_to_mass_label = {z: (12 if z < 0.5 else 13) for z in redshift_src_list}
```

mass map 全部 40 壳层到位后,应替换为完整的两列对应文件(`shell_z_map.txt`)或完整 dict。

## 9. 待办 / 已知事项

1. **mass_maps 只下载了 shell_12/13**,density 目录仅为测试版;需补齐 40 壳层并更新 `z_to_mass_label`
2. 其余本地改动(`cosmogrid_runs/*` 配置迁移、`measurements/calculator.py` jackknife、`runner.py` 的 void bootstrap hunk)**尚未提交/推送**
3. `measurements/run_ggl.py` 的 `srcs_fmt` 仍指向 CosmoGrid 目录,未接 Abacus 目录
4. `assign_shear_vals` 存在静默截断 z 超出覆盖源的历史隐患(AbacusRunner 已在前置截断规避;CosmoGrid/FastPM 路径未动)
5. `notebooks/`、`measurements/showup/` 等未跟踪文件未整理

## 10. 常用运行命令

```bash
# 环境
cd /home/suchen/Program/CosmoGrid
.pixi/envs/default/bin/python ...           # pixi 环境解释器

# 生成 Abacus shape catalog(random / density 由脚本内 position_method 控制)
.pixi/envs/default/bin/python abacus_runs/run_mock_shape.py

# 回归测试
PYTHONPATH=/usr/lib/python3/dist-packages .pixi/envs/default/bin/python -m pytest tests/ -q
```
