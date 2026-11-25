import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from animal_shelter import features


@pytest.fixture(scope="function")
def series_sex_upon_outcome():
    return pd.Series(["Intact Male", "Spayed Female", "Neutered Male", "Unknown"])


def test_check_has_name():
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features.check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)


def test_check_is_dog():
    animals = pd.Series(["Dog", "Cat", "Dog"])
    result = features.check_is_dog(animals)
    expected = pd.Series([True, False, True])
    assert_series_equal(result, expected)


def test_check_is_dog_invalid_input():
    with pytest.raises(RuntimeError) as exception_info:
        features.check_is_dog(pd.Series(["Dog", "Cat", "Rabbit"]))
    assert "Found pets that are not dogs or cats." in str(exception_info.value)


def test_get_sex(series_sex_upon_outcome):
    result = features.get_sex(series_sex_upon_outcome)
    expected = pd.Series(["male", "female", "male", "unknown"])
    assert_series_equal(result, expected)


def test_get_neutered(series_sex_upon_outcome):
    result = features.get_neutered(series_sex_upon_outcome)
    expected = pd.Series(["intact", "fixed", "fixed", "unknown"])
    assert_series_equal(result, expected)


def test_get_hair_type():
    breed_df = pd.Series(
        [
            "Domestic Shorthair Mix",
            "Domestic Medium Hair",
            "Domestic Longhair",
            "Unknown",
        ]
    )
    result = features.get_hair_type(breed_df)
    expected = pd.Series(["shorthair", "medium hair", "longhair", "unknown"])
    assert_series_equal(result, expected)


def test_compute_days_upon_outcome():
    age_df = pd.Series(["2 years", "3 months", "1 week", "5 days", "Unknown"])
    result = features.compute_days_upon_outcome(age_df)
    expected = pd.Series([730.0, 90.0, 7.0, 5.0, float("nan")])
    assert_series_equal(result, expected)


def test_compute_days_upon_outcome_no_time():
    nonsense_df = pd.Series(["blabla years"])
    with pytest.raises(ValueError):
        features.compute_days_upon_outcome(nonsense_df)


def test_compute_days_upon_outcome_invalid_period():
    nonsense_df = pd.Series(["2 centuries"])
    result = features.compute_days_upon_outcome(nonsense_df)
    expected = pd.Series([float("nan")])
    assert_series_equal(result, expected)


def test_add_features():
    df = pd.DataFrame(
        {
            "name": ["Ivo", "Henk", "unknown"],
            "animal_type": ["Dog", "Cat", "Dog"],
            "sex_upon_outcome": ["Intact Male", "Spayed Female", "Neutered Male"],
            "breed": [
                "Domestic Shorthair Mix",
                "Domestic Medium Hair",
                "Domestic Longhair",
            ],
            "age_upon_outcome": ["2 years", "3 months", "1 week"],
        }
    )
    result = features.add_features(df)
    expected_result = df.copy()
    expected_result["is_dog"] = pd.Series([True, False, True])
    expected_result["has_name"] = pd.Series([True, True, False])
    expected_result["sex"] = pd.Series(["male", "female", "male"])
    expected_result["neutered"] = pd.Series(["intact", "fixed", "fixed"])
    expected_result["hair_type"] = pd.Series(["shorthair", "medium hair", "longhair"])
    expected_result["days_upon_outcome"] = pd.Series([730.0, 90.0, 7.0])
    assert_frame_equal(result, expected_result)
