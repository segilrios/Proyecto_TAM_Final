# Manifest del repositorio final

Generado: 2026-06-17 22:36

## Resumen

- Archivos: 114
- Tamaño total aproximado: 66.79 MB
- Notebook final: `notebooks/proyecto_oro_colombia_final.ipynb`
- Dashboard final: `dashboard/dashboard_interactivo_final.html`
- Modelo final: `models/best_model_bundle.joblib`

## Árbol principal

```text
oro-colombia-regime-ml_FINAL_GITHUB/
├── dashboard/
│   ├── dashboard_interactivo_final.html (5.8 MB)
│   └── README.md (340 B)
├── data/
│   ├── processed/
│   │   ├── audit/
│   │   │   ├── audit_summary.csv (1.0 KB)
│   │   │   ├── coverage.csv (880 B)
│   │   │   ├── data_treatment_log.csv (1.1 KB)
│   │   │   ├── feature_sanitization_report.csv (1.2 KB)
│   │   │   ├── missingness_summary.csv (165.2 KB)
│   │   │   └── yfinance_download_failures.csv (1 B)
│   │   ├── catalogs/
│   │   │   ├── best_model_features.csv (87.1 KB)
│   │   │   ├── feature_engineering_catalog.csv (372.5 KB)
│   │   │   ├── feature_sets_catalog.csv (11.0 KB)
│   │   │   ├── manifest.json (4.3 KB)
│   │   │   ├── selected_features_by_run.csv (10.2 MB)
│   │   │   └── selected_model_configs.csv (2.6 KB)
│   │   ├── metrics/
│   │   │   ├── classification_results.csv (83.5 KB)
│   │   │   ├── classification_summary.csv (6.5 KB)
│   │   │   ├── confusion_matrix.csv (67 B)
│   │   │   ├── deterministic_baseline_results.csv (14.8 KB)
│   │   │   ├── deterministic_baseline_summary.csv (2.1 KB)
│   │   │   ├── magnitude_results.csv (2.1 KB)
│   │   │   ├── magnitude_summary.csv (341 B)
│   │   │   ├── model_performance_by_presidential_period.csv (388 B)
│   │   │   ├── segment_metrics.csv (3.1 KB)
│   │   │   ├── strategy_by_presidential_period.csv (1.5 KB)
│   │   │   └── target_diagnostics.csv (192 B)
│   │   ├── predictions/
│   │   │   ├── all_predictions.csv (9.7 MB)
│   │   │   ├── best_predictions.csv (724.4 KB)
│   │   │   ├── magnitude_predictions.csv (121.5 KB)
│   │   │   ├── model_outputs_comparison.csv (6.4 KB)
│   │   │   ├── model_predictions_sample.csv (542.3 KB)
│   │   │   └── predictions.csv (719.5 KB)
│   │   ├── regimes/
│   │   │   ├── cluster_profile_abs.csv (4.7 KB)
│   │   │   ├── cluster_profile_chg.csv (4.7 KB)
│   │   │   ├── cluster_transition_abs.csv (819 B)
│   │   │   ├── cluster_transition_chg.csv (922 B)
│   │   │   ├── local_biplot_abs_loadings.csv (862 B)
│   │   │   ├── local_biplot_chg_loadings.csv (1.0 KB)
│   │   │   ├── regime_design_6_7.csv (356 B)
│   │   │   ├── stochastic_summary_by_cluster_abs.csv (2.3 KB)
│   │   │   ├── stochastic_summary_by_cluster_chg.csv (2.3 KB)
│   │   │   ├── stochastic_summary_by_dutch_disease.csv (1.6 KB)
│   │   │   ├── stochastic_summary_by_policy_stance.csv (1.2 KB)
│   │   │   ├── stochastic_summary_by_president.csv (2.4 KB)
│   │   │   ├── stochastic_summary_combined.csv (8.8 KB)
│   │   │   ├── umap_optimization_abs.csv (5.0 KB)
│   │   │   ├── umap_optimization_chg.csv (5.0 KB)
│   │   │   └── umap_regimes.csv (497.5 KB)
│   │   └── strategies/
│   │       ├── market_regime_strategy_periods.csv (1.4 KB)
│   │       ├── market_regime_strategy_summary.csv (759 B)
│   │       ├── strategy_backtest_summary.csv (1.5 KB)
│   │       └── strategy_backtest_timeseries.csv (964.9 KB)
│   ├── raw/
│   │   ├── base_colombia_normalizada.csv (919.5 KB)
│   │   ├── dutch_disease.csv (19.5 KB)
│   │   ├── global_market_context_homogenized.csv (11.8 MB)
│   │   ├── gold_series.csv (294.3 KB)
│   │   └── presidential_periods.csv (243 B)
│   └── README.md (939 B)
├── docs/
│   ├── data_dictionary.csv (10.2 KB)
│   ├── data_dictionary.md (22.2 KB)
│   ├── depuracion_repositorio.md (743 B)
│   ├── glosario.md (879 B)
│   ├── guia_presentacion_completa.md (1.6 KB)
│   ├── metodologia_pipeline.md (1.4 KB)
│   └── teoria_series_temporales_h21.md (1.2 KB)
├── models/
│   ├── best_model_bundle.joblib (5.4 MB)
│   └── best_model_config.json (455 B)
├── notebooks/
│   └── proyecto_oro_colombia_final.ipynb (191.2 KB)
├── outputs/
│   ├── figures/
│   │   ├── ai_vs_deterministic_bacc.png (109.0 KB)
│   │   ├── audit_bacc.png (79.1 KB)
│   │   ├── best_ablation_feature_importance.png (176.6 KB)
│   │   ├── best_ablation_permutation_importance.png (175.2 KB)
│   │   ├── best_model_segment_balanced_accuracy.png (77.2 KB)
│   │   ├── best_walkforward_confusion_matrix.png (35.0 KB)
│   │   ├── cluster_profile_abs_heatmap.png (111.3 KB)
│   │   ├── cluster_profile_chg_heatmap.png (111.7 KB)
│   │   ├── cluster_transition_abs_heatmap.png (27.6 KB)
│   │   ├── cluster_transition_chg_heatmap.png (28.5 KB)
│   │   ├── confusion_matrix_best.png (28.3 KB)
│   │   ├── model_comparison_by_feature_set.png (63.4 KB)
│   │   ├── model_comparison_by_train_window.png (40.1 KB)
│   │   ├── model_outputs_probability_sample.png (420.1 KB)
│   │   ├── probability_threshold_best_model.png (244.0 KB)
│   │   ├── real_vs_estimated_return.png (157.8 KB)
│   │   ├── real_vs_expected_price_h21.png (108.4 KB)
│   │   ├── stochastic_return_boxplot_by_president.png (64.8 KB)
│   │   ├── stochastic_volatility_by_segment.png (166.5 KB)
│   │   ├── strategy_capital_curves.png (86.9 KB)
│   │   ├── top20_accuracy.png (153.7 KB)
│   │   ├── top20_balanced_accuracy.png (154.3 KB)
│   │   ├── top20_models_bacc.png (231.8 KB)
│   │   ├── umap_abs_local_biplot.png (150.2 KB)
│   │   ├── umap_chg_local_biplot.png (193.8 KB)
│   │   ├── umap_optimization_abs_silhouette.png (111.6 KB)
│   │   ├── umap_optimization_chg_silhouette.png (108.9 KB)
│   │   ├── umap_regimes_abs.png (78.4 KB)
│   │   ├── umap_regimes_chg.png (206.5 KB)
│   │   ├── walkforward_metrics_best_model.png (153.7 KB)
│   │   ├── walkforward_top_mean_bacc.png (73.9 KB)
│   │   └── weight_strategy_mean_lift.png (56.1 KB)
│   └── tables/
├── reports/
│   ├── document/
│   │   ├── documento_tecnico_final_oro_colombia.docx (3.1 MB)
│   │   └── documento_tecnico_final_oro_colombia.pdf (2.8 MB)
│   └── presentation/
│       ├── presentacion_final_oro_colombia.pdf (3.1 MB)
│       ├── presentacion_final_oro_colombia.pptx (5.1 MB)
│       └── preview_presentacion.jpg (82.9 KB)
├── scripts/
│   ├── 00_check_project.py (1.3 KB)
│   ├── 01_pipeline_completo.py (128.8 KB)
│   ├── 02_quick_test.py (2.4 KB)
│   ├── 03_model_inference_demo.py (2.2 KB)
│   ├── 04_run_dashboard_server.py (845 B)
│   ├── 05_generar_diccionario_datos.py (1.0 KB)
│   └── README.md (1014 B)
├── README.md (2.5 KB)
└── requirements.txt (90 B)
```
