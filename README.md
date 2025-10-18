# house-prices-ml
House Prices prediction using LightGBM, XGBoost, and CatBoost with Optuna tuning and ensemble learning.

# 🏠 Kaggle House Prices - Model Optimization Summary

## 📘 Overview
このプロジェクトは、Kaggle の "House Prices - Advanced Regression Techniques" における  
価格予測モデルの最適化プロセスをまとめたものです。  
LightGBM / XGBoost / CatBoost の3モデルを中心に構築・チューニングを行い、  
最終的なスコアは **0.13041** を達成しました。

---

## ⚙️ 1. Data Preprocessing
- 欠損値補完およびカテゴリ変数のエンコーディングを実施  
- 特徴量は軽い正規化のみに留め、構造的特徴を保持  
- 目的変数 `SalePrice` の log 変換も検証したが、最終的に非log版を採用  
  - log変換は RMSEベースでは改善せず（逆変換誤差・歪度影響）

---

## 🧩 2. Base Models
3つのブースティング系モデルを採用。  
それぞれ独立にOptunaによるパラメータチューニングを実施（LightGBMのみ最終調整）。

| モデル | 主な設定 | 備考 |
|--------|-----------|------|
| **LightGBM** | `n_estimators=2000`, Optuna最適化済 | 最も安定 |
| **XGBoost** | `n_estimators=5000`, Optuna最適化済 | 最高精度を記録 |
| **CatBoost** | `iterations=3000`, Optuna最適化済 | 若干過学習傾向 |
| **共通設定** | `random_state=42` / early_stopping=100 | 再現性確保・安定化 |

---

## 🔧 3. Parameter Tuning (Optuna)
- 探索範囲を各モデルごとに設定し、RMSE最小化を目的に30試行  
- `random_state=42`固定で結果の安定性を担保  
- LightGBM のみ結果が明確に安定し、他2モデルは分布が広め（探索再現性あり）  

例：LightGBM 最適パラメータ抜粋
```python
{
    'num_leaves': 66,
    'max_depth': 10,
    'learning_rate': 0.1167,
    'min_child_samples': 15,
    'subsample': 0.5044,
    'colsample_bytree': 0.6038,
    'reg_alpha': 0.0564,
    'reg_lambda': 0.8851,
    'objective': 'regression',
    'metric': 'rmse'
}
```

---

## 📊 4. Validation Results
🧮 最終RMSE比較（Optuna調整後）

| Model    | Validation RMSE |
| -------- | --------------- |
| XGBoost  | **24,331.7**    |
| LightGBM | 24,816.5        |
| CatBoost | 25,761.9        |
⭐ 重み付きアンサンブル（LGB:XGB:CAT = 5:3:2）
Validation RMSE: 23,990.8

---

## 🏁 5. Kaggle Submission Results
| モデル構成                            | パラメータ調整      | 公開スコア (Public LB)     |
| -------------------------------- | ------------ | --------------------- |
| 単純3モデル平均 (初期)                    | LGBのみ調整      | **0.13110**           |
| 加重平均 (5:3:2) + XGB/CAT調整         | すべてOptuna最適化 | **0.13125**           |
| LGB固定 + XGB/CATデフォルト（ChatGPT提案値） | 軽微調整         | 🏆 **0.13041 (Best)** |
📈 結果として、パラメータ調整を一部緩めた構成が最も安定・高スコアとなった。

---

## 🔍 6. Insights & Discussion
✅ 効果的だった点
-LightGBMのOptunaチューニングにより約3%のRMSE改善
-モデルごとの random_state 固定による再現性確保
-重み付きアンサンブルによる微小だが一貫した精度向上（約0.001程度）

⚠️ 改善余地
-XGBoostは学習率とmax_depthの依存関係が強く、最適化が不安定
-CatBoostはiterationsが長めで過学習傾向
-アンサンブル3モデルの相関が高く、Stackingの導入余地あり

---

## 🚀 7. Next Steps
| 改善施策                                | 目的          |
| ----------------------------------- | ----------- |
| 🔹 Optuna試行回数を100〜200に拡張            | 探索の安定化      |
| 🔹 k-Fold Cross Validation (5-fold) | スコア分散の抑制    |
| 🔹 Stacking（メタモデルLightGBM）導入        | モデル相関低減     |
| 🔹 特徴量重要度の再分析                       | ノイズ削減と特徴量選択 |
| 🔹 RMSLE評価・log変換の再検証                | 評価軸との整合性確認  |

---

## 🧾 8. Summary
| 項目       | 評価                    |
| -------- | --------------------- |
| 再現性      | ✅ 完全シード固定             |
| モデル多様性   | ✅ 3種類実装               |
| チューニング品質 | ✅ Optuna導入済           |
| アンサンブル効果 | ⭕ 軽度改善（~0.001）        |
| 総合スコア    | 🏆 **0.13041 (Best)** |
