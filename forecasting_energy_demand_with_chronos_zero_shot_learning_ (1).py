

# We use uv for faster installation
!pip install uv
!uv pip install -q autogluon --system
!uv pip install -q autogluon.timeseries --system
!uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab

"""## Zero-shot forecasting

We work with a subset of the [Australian Electricity Demand dataset](https://zenodo.org/records/4659727) and use Chronos-Bolt
"""

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
)
data.head()

prediction_length = 48
train_data, test_data = data.train_test_split(prediction_length)

"""We use the efficient Chronos-Bolt (Small, 48M) model in zero-shot mode, which as `6x` the number of parameters. We select the `"bolt_small"` presets to use the [Chronos-Bolt](https://huggingface.co/autogluon/chronos-bolt-small) (Small, 48M)."""

predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, presets="bolt_small",
)

"""We use the `predict` method to generate forecasts, and the `plot` method to visualize them and see the accuracy


"""

predictions = predictor.predict(train_data)
predictor.plot(
    data=data,
    predictions=predictions,
    item_ids=data.item_ids[:2],
    max_history_length=200,
);