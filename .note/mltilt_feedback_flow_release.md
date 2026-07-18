# Downstream Feedback: Flow Release "default" (from mltilt / planet_freq_flow)

Date: 2026-07-15
Reporter: mltilt(/rogue1_8/nunota/planet_freq_flow)での N24 再現テストから。
再現スクリプトと数値の一次記録:
`/rogue1_8/nunota/planet_freq_flow/docs/notes/2026-07-15_flowbank_v2.md`
(および同 2026-07-13_flow_test.md)。

release "default"(l∈[−5,5], b∈[−6,−2])を Nunota+24 の 22 惑星尤度グリッド
再現に組み込んだ際に見つかった問題 2 件。どちらも「カーネル flow の学習品質」の
問題ではない(カーネル条件付き分布は genulens と KS≈0.01 で一致)。

---

## 1. source_distance_grid の (DS, group) 測度が genulens の重みと系統的に不一致

### Resolution (2026-07-15)

Confirmed and fixed in the default release package. The original grid used a
physical `nMS * 1e-6 * DS^2` source factor and a `rho * DL^2` lens column, but
the unselected genulens sampler that produced the rate-removed kernel uses
`nMS * sqrt(DS / 8000) * 1e-3` (`gammaDs=0.5`) and samples the lens column from
total number density. Rebuilding the grid with that exact measure at
`(l,b)=(-1.65,-3.69)` changed the 250k independent `SMALLGAMMA=1` comparison
from KS `0.118` to `0.034`; all five source-group fractions agree within Monte
Carlo noise. The remaining difference is compatible with 0.5-degree
sightline interpolation and finite sampling.

**Downstream verification (mltilt, 2026-07-15, commit ae5528e):** 独立ラン
(genulens v2 既定設定, n=10^6, seed=21)でも KS 0.117 → **0.035**、group 割合は
4 桁一致(thin 0.0891/0.0891, thick 0.0246/0.0245, bulge 0.8828/0.8829,
halo 0.0035/0.0035)を確認。N24 尤度グリッド(OB07368, 8×10^6, quantize-dl)の
genulens 不要パス(flowbank)は中心領域残差 rms 0.118 → **0.086**(半分割ゆらぎ
0.080 と同水準)になり、golden と統計的に整合。残るノイズ水準は下記 2.
(rate 重み方式の設計制約)によるもの。

### 症状(バグ相当と考える)

release の設計上の含意は
「grid 測度 × kernel × イベント率因子 = genulens のイベント分布」のはずだが、
**grid 測度そのもの(kernel を介さない解析値)が、genulens イベントの
de-rated 重み(wtj / (DL²·θE·μrel))の (DS, group) 周辺分布と一致しない**:

- 視線 (l, b) = (−1.65, −3.69)、genulens v2(PYTHONPATH=build、既定設定)
  n_simu=10^6, seed=21 との比較:
  - de-rated DS 周辺: **KS = 0.117**、中央値 genulens 8.83 kpc vs grid 9.02 kpc
  - group 割合(de-rated): thin 0.089 vs **0.103**、thick 0.025 vs **0.039**、
    bulge 0.883 vs 0.855、NSD 0/0、halo 0.0035/0.0033
- 下流への影響: grid をそのまま使ってイベント率重み付き分布を組むと
  DS 周辺(rate 込み)で KS≈0.16、tE 周辺で KS≈0.05 ずれ、
  N24 尤度グリッドの中心領域残差が genulens バックエンド比で ~2 倍になる。

### 切り分け済みの事項

- kernel は無実: p(ML|DS), p(μ|DS) 等の条件付きは genulens と KS≈0.01。
- NSD 設定差は無関係(この視線で NSD 寄与ゼロ)。
- 疑い先は manifest の source_measure = "nMS × 1e-6 × DS² × integrated lens
  column to DS" の定義と、genulens が wtj に入れている source 重み
  (LF 全体? MS 以外の扱い? 正規化?)の差。
  thin/thick が系統的に過大・bulge が過小なのは、成分別の源数規格
  (nMS の成分別定義)のずれを示唆する。

### 未確認の交絡(こちらで排除できていない)

- 学習データは NSD=1, SMALLGAMMA=1 で生成とのこと(handover)。比較した
  genulens v2 ランは既定設定(SMALLGAMMA 相当は Python API から触れず未指定)。
  SMALLGAMMA の有無が wtj 周辺分布に効くなら、その分の差が混ざっている
  可能性は残る。grid 生成時と同一設定の genulens ランとの直接比較を推奨。

### 再現手順

```bash
cd /rogue1_8/nunota/genulens && PYTHONPATH=build python3 <<'EOF'
# genulens de-rated 周辺 vs FlowSourceDistanceGrid.at(l,b) を CDF 比較
# (完全なコードは planet_freq_flow 側ノート参照。要点のみ)
EOF
```
実体は `/rogue1_8/nunota/planet_freq_flow` の会話ログ/ノートにあり。
比較関数は `scripts/flow_vs_genulens.py` の weighted_ks と同じ。

---

## 3. 【新規・重要】イベント率因子の不整合: kernel は DL² を既に含んでいる(log_density の DL² 二重掛け疑い)

Date: 2026-07-15(ae5528e 後の検証で発見)

### 症状

handover では「学習重みから event-rate factor(= log_event_rate の DL²·θE·μ)を
外した」とされているが、**実測では kernel から外れているのは θE·μ のみで、
DL² は kernel 内に残っている**。

決定的な実験(視線 (−1.65, −3.69)、DS ∈ [8.8, 9.0] kpc・bulge スライス、
genulens v2 n=2×10^6 の wtj 加重 p(DL|DS) と、kernel サンプル×各重みの比較):

| kernel × 重み | p(DL|DS) KS | E[ln\|8160−DL·cosb\|] の差 |
|---|---|---|
| DL²·θE·μ(handover の想定) | **0.154** | −0.42 |
| **θE·μ** | **0.005** | +0.007 |
| θE のみ | 0.035 | −0.06 |
| 等重み | 0.250 | −0.45 |

原因の推定: genulens のレンズ proposal(全数密度の柱)に対する wtj は
DL² 因子を含まない形になっており、学習前処理の de-rate(wtj ÷ DL²θEμ)が
DL² を余分に割った結果、kernel が DL² を「持ち込んだ」状態で学習された、
あるいは前処理の de-rate が実は θEμ のみだった、のどちらか。

### 影響(sampling だけでなく log_density も)

- `GalaxyModel.log_density` は kernel logp に `log_event_rate`(2·logDL 込み)を
  足すため、**kernel が既に DL² を含んでいるなら DL² が二重掛け**になり、
  DL 方向に系統的に歪んだ密度を返す。下流の推論全てに効く。
- 実害の実測: N24 の 19 イベント合算尤度面で、DL²θEμ 重み構成は
  r(銀河中心距離冪)の周辺中央値を +1.3 ずらした(golden +0.18 → +1.45)。
  イベント別には +0.1 程度の小さな系統バイアスだが、合算で増幅される。
- また ae5528e の grid 再較正は「wtj/(DL²θEμ) の DS 周辺」を基準にしたはず
  なので、de-rate の定義を θEμ に正すなら grid の基準も再確認が必要
  (kernel×θEμ 再構成に整合する grid は「wtj/(θEμ) の DS 周辺」)。
  実測でも、修正版 grid + kernel×θEμ の DS 周辺は genulens wtj と KS=0.146
  でずれる(kernel×DL²θEμ 基準だと合うように較正されているため)。

### 提案

- 学習前処理(prepare_balanced 系)の de-rate 式と、`log_event_rate` /
  `GalaxyModel.log_density` / `sample()` の rate 適用式を突き合わせて、
  どちらかに統一する(kernel を再学習しない場合は、適用側を θEμ にし、
  grid を wtj/(θEμ) 基準で再生成するのが最小変更)。
- 検証テストとして「grid × kernel × rate の joint を genulens wtj と
  DS/DL/tE 周辺で KS 比較」を release CI に足すと、この種の測度不整合を
  一発で検出できる(mltilt 側の scripts/flow_vs_genulens.py が雛形)。

### mltilt 側の暫定対処

`mltilt/galactic/flow_bank.py` は重みを θE·μ に変更し、(DS, group) は
genulens 経験分布(wtj)+ DS ビン内 Ẑ 自己正規化で運用(grid バックエンドは
上記の整合が取れるまで保留)。

---

## 4. 【最終検証結果・6c7a64f 後】測度整合は完了。ただし v2 kernel の精度が N24 合算には不足 → 率込み kernel の再学習を正式リクエスト

Date: 2026-07-15(6c7a64f での V4 再実行)

### 6c7a64f の確認(こちらの独立検証、いずれも合格)

- 新 grid + kernel×θEμ の joint 周辺: DS KS **0.028**、DL 0.005、tE 0.010、μ 0.0004
  → **測度の整合は完全に取れた**。3 件のバグ(grid 測度 ×2、rate 因子不整合)は解消。

### しかし N24 受け入れテスト(V4)は不合格

19 イベント合算尤度面の周辺中央値(|m|,|r|≤2 制限でも同傾向):

| | golden | v1 kernel(7/13) | v2 kernel(6c7a64f 構成) |
|---|---|---|---|
| m | +0.42 | +0.42(Δ+0.003) | **−0.25(Δ−0.67)** |
| r | +0.18 | +0.18(Δ+0.007) | **+0.54(Δ+0.37)** |

イベント別では Δm = −0.23±0.06(19 中 18 が負)のコヒーレントなバイアス。

### 原因(切り分け済み): kernel の裾精度

- 観測条件付き(分子)の E[lnML|tE_obs, μ_obs] は genulens 比 +0.02〜0.03、
  **効率重み付き母集団(分母)の E[lnML] が +0.06** ずれる(ev14, ev24 で確認)。
  N24 の m はこの分子−分母バランスで決まるため、+0.06 の分母バイアスが
  イベントあたり Δm ≈ −0.2、合算で −0.6 になる。
- 再構成の代数(Ẑθ 正規化等)は上記 joint KS で無罪確認済み。DS 骨格は
  genulens そのもの。残るのは kernel が θEμ·ε(tE) で重み付けされる裾
  (長 tE・高 ML 側)で数 % ずれていること。
- これは v2 の「母集団測度で学習」する設計の構造的帰結:
  推論で効く率重み領域が学習分布の裾になり、そこの精度が落ちる。
  v1(率込み測度で学習)は同じ検証を Δ=0.003/0.007 で通過していた。

### リクエスト(再学習)

**率込み測度(v1 と同じ wtj 重み)で学習した kernel を、v2 カバレッジ
(l±5, b∈[−6,−2])で作ってほしい。** 受け入れ基準の提案:
1. 効率重み付き母集団の E[lnML] が genulens と ±0.01 以内(複数視線)
2. N24 19 イベント合算の周辺中央値が golden と Δ ≤ 0.02
(2. は mltilt 側で即回せる: scripts/flow_vs_n24_all.sh + flow_v4_combine.py)

なお現行 v2(CMD 選択独立・log_density 正確)は本来の gapmoe 用途
(イベント単位の事前分布評価)には引き続き有効なはず。率込み版は
「母集団 MC / importance tilting 用」の別リリースとして並置を想定。

---

## 2. 「率因子を外した」カーネル + 重み付けの実効サンプル効率が大規模 MC で大きく落ちる

### 症状(バグではなく設計制約。ただし handover の想定より深刻)

v2 の設計(rate を推論時に掛ける)は log_density には無害だが、
**バルクサンプリング用途では、率因子 w = DL²θEμ を重みとして掛けると
指数 tilting(mltilt の e^{m·a + r·b})や観測カーネルとの相互作用で
ノイズが大きく増える**:

- 同一パイプライン・同一 n=8×10^6・同一イベント(OB07368)での
  尤度グリッド半分割ノイズ(中心領域 rms、log10L):
  - v1 カーネル(率込みで学習、等重み): **0.022**
  - v2 カーネル + 率重み(DS は genulens、Ẑ(DS) 自己正規化済み): **0.099**
  → 実効サンプル効率でおよそ 20 分の 1。
- 生の重み ESS/n は 0.51(genulens wtj は 0.72)、観測カーネル込み ESS は
  genulens の ~0.6 倍なので、素の ESS だけでは説明がつかず、
  tilt との相互作用が支配的。重み外れ値ではない(上位 20 個の寄与 0.013%)。
- 公開 API の `sample()`(num_proposals=256 の resampling)も同じ理由で
  大規模 MC には不向き(handover の Follow-up 記載どおり)。

### 提案

- **率因子込みで学習した kernel の変種を同一カバレッジで併載してほしい**
  (v1 run: flow_mvp/runs/release_v1_20260711_014402 と同じ学習方式)。
  CMD 選択の独立性が要る用途は現行 v2、母集団 MC・importance tilting 用途は
  rate 込み版、と使い分けられる。
- あるいは event-rate 重み付き分布から直接引けるサンプラー
  (proposal 最適化)の提供。

---

## 参考: 下流での対処(mltilt 側で実装済み)

- `mltilt/galactic/flow_bank.py`: (DS, group) を genulens 経験分布から渡す
  バックエンド(`ds_group_weighting="rate"` で DS ビン内 Ẑ 自己正規化)を
  用意し、grid 測度問題を回避して N24 照合を実施中。
- kernel の float32 出力は下流で float64 に昇格しないと 1/DL − 1/DS が
  丸めで符号反転し得る(8×10^6 中 1 点で実発生)。release 側で float64 を
  返すか、ドキュメントに注意書きがあると親切。

---

## 5. gapmoe 側の最終整理 (2026-07-16)

N24 は event-rate 測度の二重適用を発見するうえで有用だったが、gapmoe の
Flow リリース条件そのものではない。最終版 `rate-included-v1` は raw `wtj`
から学習した正規化済み条件付き kernel と、同じ測度の component-resolved
source grid を組み合わせ、推論時の後付け率重みを使わない。

受け入れは独立 genulens holdout との直接比較で行った。9 midpoint・27万件の
joint 比較では5物理変数の最大 KS 0.03310、`DL/DS`, `mu_rel`, `theta_E`,
`t_E` の最大 KS 0.02464、順位相関行列の最大差 0.03025、source-group 比率の
最大差 0.01034。さらに major groups 0–2、強制 NSD、強制 halo の条件付き
holdout も、それぞれ marginal KS ≤0.05、derived KS ≤0.05、順位相関差
≤0.10 を満たした。これを gapmoe 側の完了判定とし、N24 は下流の任意回帰
テストとして扱う。
