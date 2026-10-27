# 2-3 minute Loom walkthrough draft

Hi, I'm going to walk through my freight-rate prediction approach.

The development file contains 48,000 labeled loads from January through October 2025. I first checked the target and input quality. Weight is missing in 300 training rows and market index is missing in 374. The future validation file also has missing values in those columns, so the pipeline learns numeric medians from its training data and adds missing-value indicators. I kept load_id out of the model because it is an identifier, not a shipment feature.

For validation, I used a chronological split: January through August to train, then September and October as a forward holdout. That better resembles predicting November and December than randomly mixing dates. I compared a median rate-per-mile baseline, Ridge regression, and HistGradientBoosting. The tree model had the lowest holdout MAE at about 137 dollars, with a WAPE of 5.72 percent. RMSE was much higher than MAE, so there are still some large errors to investigate.

The model uses distance, route locations and coordinates, equipment, weight, market index, quote signal, and calendar features. Categorical fields are one-hot encoded. After selecting the model on the holdout, I retrained it on all labeled rows and predicted every validation load.

I also filled the separate December scenario. That file only provides the route, distance, equipment, weight, and date, so unavailable coordinates and market features are imputed from training values. The supplied scorer accepted all 12,000 validation predictions and all 31 December rows, and generated the chart in the report.

The main limitation is that this example uses one two-month holdout and limited tuning. The final November-December accuracy is not known here; the assessment says Spotter calculates it after submission. Thanks.
