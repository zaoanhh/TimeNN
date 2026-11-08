# Result data

| File | Used by |
| --- | --- |
| `with_tau.csv` | Figs. 4b, 5 and 6 |
| `aug_results.csv` | Figs. 7 and 8 |
| `motivation_results.csv` | Fig. 3 |
| `feature_inversion/<signal>/{original,inverted}.wav` | Fig. 4a |
| `stability_feature_maps.npz` | Fig. 2 |

## `with_tau.csv`

Accuracy of each `(dataset, model)` pair on the clean test set and after
cubic-spline time distortion with intensity `t`.  `0.0` is the clean test set;
the warped accuracy averages five independent warps per sample.

```
dataset, model, acc_warp_0.0, ..., acc_warp_0.8
```

## `aug_results.csv`

Accuracy of `simple_cnn` trained with static augmentation (`aug_factor` = 1, 5
or 10) and `lms_cnn` trained without augmentation, for different model sizes
and training ratios.

```
dataset, model, model_size, data_ratio, aug_factor,
acc_warp_0.0, ..., acc_warp_0.3
```

## `motivation_results.csv`

Clean test accuracy and accuracy after a random time translation of at most
10 % of the window, at the full training ratio.

```
Dataset, Ratio, Model, TestAcc, AugTestAcc
```

## `feature_inversion/`

16 kHz, 16-bit mono WAV files.  `original.wav` is the analytic input signal
(chirp / sine burst / square wave); `inverted.wav` is the signal recovered by
optimising in the frozen Time-NN feature space.

## `stability_feature_maps.npz`

Arrays used by the Fig. 2 difference maps:

```
x_orig, x_warp                  # [512]
orig::<MODEL>, warp::<MODEL>    # feature maps, MODEL in
                                # MLP, CNN, LSTM, Transformer, Inception, TIME-NN
```

Generate with:

```bash
uv run python scripts/plots/visualize_stability.py --recompute --save-data
```
