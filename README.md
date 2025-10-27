# house-prices-ml  
House Prices prediction using LightGBM, XGBoost, and CatBoost with Optuna tuning and ensemble learning.

- **Learning and Submission data**
  - **Last submission data**(https://github.com/k-ohyeah/house-prices-ml/blob/main/house-prices-submission-updated-fixed.ipynb)
  - **Best score submission data**(https://github.com/k-ohyeah/house-prices-ml/blob/main/house-prices-submission-updated.ipynb)  
    ※重みづけは要調整必要(LGB:XGB:CAT = 5:3:2)
  
---

# 🏠 Kaggle House Prices - Model Optimization Summary

## 🧭 Key Insights (Conclusion First)
住宅価格予測を題材に、機械学習モデルの比較・最適化を通じてデータ分析の本質を探究。  
LightGBM・XGBoost・CatBoost の特性を体系的に比較し、精度と再現性のバランスを重視。  
分析過程では、仮説立案→前処理→検証→改善のプロセスを繰り返し、  
データサイエンスにおける「定量的根拠に基づく意思決定」の重要性を実践的に学んだ。  
最終的に **RMSE 0.13041 (Public LB)** を達成し、安定したモデル運用の基礎を確立。

---

## ⚙️ 1. Data Preprocessing  
- 欠損値補完およびカテゴリ変数のエンコーディングを実施。  
- 特徴量は軽い正規化のみに留め、構造的特徴を保持。  
- 目的変数 `SalePrice` に対し log 変換を検証したが、非線形モデルでは効果が限定的であることを確認。  
  → RMSEベースでは改善が見られず、逆変換誤差および歪度影響を考慮し非log版を採用。  

💡 **気づき:**  
目的変数の分布を正規化すれば精度が向上すると考えていたが、  
LightGBM や XGBoost のような非線形モデルでは外れ値や歪度に対してもロバストに動作。  
log変換が逆にスケーリングを崩す場合があることを確認。  
前処理はモデル特性と整合した設計が重要。

---

## 🧩 2. Base Models  
3つのブースティング系モデルを採用。  
それぞれ独立に Optuna によるパラメータチューニングを実施（LightGBM のみ最終調整）。

| モデル | 主な設定 | 備考 |
|--------|-----------|------|
| **LightGBM** | n_estimators=2000, Optuna最適化済 | 最も安定 |
| **XGBoost** | n_estimators=5000, Optuna最適化済 | 最高精度を記録 |
| **CatBoost** | iterations=3000, Optuna最適化済 | 若干過学習傾向 |
| **共通設定** | random_state=42 / early_stopping=100 | 再現性確保・安定化 |

💡 **気づき:**  
Optuna で得られたパラメータ分布を分析することで、  
モデルごとの安定性や探索挙動の違いを定量的に把握。  
パラメータ探索を単なる最適化作業ではなく、**モデル理解の一環**として捉えることの有効性を確認。

---

## 🔧 3. Parameter Tuning (Optuna)  
探索範囲を各モデルごとに設定し、RMSE最小化を目的に30試行を実施。  
`random_state=42` を固定し、結果の安定性を担保。  
LightGBM は結果の収束性が高く、他2モデルは分布のばらつきが確認された。  

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
💡 **気づき:**  
パラメータ探索履歴を分析対象として扱うことで、
各モデルの特性をより深く理解。
精度向上のみを目的とせず、学習挙動の観察と再現性向上を重視。

---

## 📊 4. Validation Results
🧮**最終RMSE比較（Optuna調整後）**

| Model                                         | Validation RMSE |
| --------------------------------------------- | --------------- |
| **XGBoost**                                   | **24,331.7**    |
| **LightGBM**                                  | 24,816.5        |
| **CatBoost**                                  | 25,761.9        |
| ⭐ **Weighted Ensemble (LGB:XGB:CAT = 5:3:2)** | **23,990.8**    |

💡 **気づき:**  
単体モデルでは XGBoost が最も精度良好。
加重アンサンブルによって RMSE をわずかに改善。
大幅なスコア向上には至らなかったが、結果の安定性と一貫性を向上。

---

## 🏁 5. Kaggle Submission Results
| モデル構成                            | パラメータ調整        | 公開スコア (Public LB)     |
| -------------------------------- | -------------- | --------------------- |
| 単純3モデル平均 (初期)                    | LGB のみ調整       | 0.13110               |
| 加重平均 (5:3:2) + XGB/CAT調整         | すべて Optuna 最適化 | 0.13125               |
| LGB固定 + XGB/CATデフォルト（ChatGPT提案値） | 軽微調整           | 🏆 **0.13041 (Best)** |


📈 **パラメータ調整を一部緩和した構成が最も安定かつ高スコアを示す結果となった。**

---

## 🔍 6. Insights & Discussion
✅ **効果的だった点**
- LightGBM の Optuna チューニングにより約3%の RMSE 改善
- モデルごとの random_state 固定による再現性確保
- 重み付きアンサンブルによる微小だが一貫した精度向上（約0.001程度）

⚠️**改善余地**
- XGBoost は learning_rate と max_depth の依存関係が強く、最適化が不安定
- CatBoost は iterations が長く過学習傾向を示す
- アンサンブル3モデル間の相関が高く、Stacking 導入による相関低減が有効と考えられる

💡**補足的考察:**  
過学習抑制と安定化のため、n_estimators、iterations、random_seed の調整を実施。
過度なチューニングよりも、**モデル挙動の理解と安定性確保を優先**する姿勢を維持。

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

## 💬 8. Approach
全体の進行は ChatGPT および GitHub Copilot の提案を参考としつつ、
各工程（前処理・特徴量設計・パラメータ調整）を自ら再現・検証。
提案コードをそのまま適用せず、「なぜそうなるのか」を検証しながら理解を深めた。
自動化支援を補助的手段とし、**自律的な検証プロセス**を重視。

---

## 🧾 9. Summary
| 項目       | 評価                    |
| -------- | --------------------- |
| 再現性      | ✅ 完全シード固定             |
| モデル多様性   | ✅ 3種類実装               |
| チューニング品質 | ✅ Optuna導入済           |
| アンサンブル効果 | ⭕ 軽度改善（~0.001）        |
| 総合スコア    | 🏆 **0.13041 (Best)** |

---

# 💡 最終考察
本プロジェクトでは、モデル精度向上に加え、学習過程の再現性・安定性・理解度の深化を重視。
試行と検証を通じて、**定量的根拠に基づいた分析思考と論理的な検証プロセス**を確立。
