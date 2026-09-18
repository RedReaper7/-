import pandas as pd

from crop_yield_prediction.features import add_feature_engineering


def test_feature_engineering_adds_expected_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "state": "Karnataka",
                "district": "Mysuru",
                "crop": "Rice",
                "season": "Kharif",
                "crop_year": 2020,
                "area_ha": 100,
                "yield_tonnes_per_hectare": 2.4,
                "rainfall_mm": 850,
            },
            {
                "state": "Karnataka",
                "district": "Mysuru",
                "crop": "Rice",
                "season": "Kharif",
                "crop_year": 2021,
                "area_ha": 110,
                "yield_tonnes_per_hectare": 2.7,
                "rainfall_mm": 900,
            },
        ]
    )

    engineered = add_feature_engineering(frame)

    assert "area_log" in engineered.columns
    assert "lag_yield_1" in engineered.columns
    assert engineered.loc[1, "lag_yield_1"] == 2.4

