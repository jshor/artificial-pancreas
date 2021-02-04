import pandas as pd
from datetime import datetime, timedelta
import math


def get_glucose_window (glucose_df, start, end):
    mask = (glucose_df['Date_Time'] > start) & (glucose_df['Date_Time'] <= end)
    w = glucose_df[mask]

    # set a time interval of 5 mins -- all missing data will be NaN
    w = w.set_index(['Date_Time']).resample('5min').last().reset_index() # TODO: is this right?

    # TODO: how to handle missing values?

    w = w.transpose()

    w = w[1:]

    return w



def get_carb_event_count (carb_df, start, end):
    mask = (carb_df['Date_Time'] > start) & (carb_df['Date_Time'] <= end)
    rows = carb_df.loc[mask].iterrows()
    carb_records_count = 0

    for index, row in rows:
        if (row['BWZ Carb Input (grams)'] > 0):
            carb_records_count += 1

    return carb_records_count


def get_windows (carb_df, glucose_df):
    meal_windows = []
    no_meal_windows = []
    no_meal_window_end = carb_df.iloc[0]['Date_Time'] + timedelta(hours=2)

    for index, row in carb_df.iterrows():
        if (row['BWZ Carb Input (grams)'] > 0):
            # a carb event was started - get the window between tm and tm+2hr-5min to ensure that no carb events occur
            # this subtracts 5 minutes from the end of the window, since if another carb event occurs at tm+2hrs, it is also considered
            carb_event_count = get_carb_event_count(carb_df, row['Date_Time'], row['Date_Time'] + timedelta(hours=2) - timedelta(minutes=5))

            if (carb_event_count <= 1):
                # if only one carb event exists within the next two hours (this one), then this record will define the window
                # select the glucose window tm-30m to tm+2hr for the window and add it to the list
                meal_window = get_glucose_window(glucose_df, row['Date_Time'] - timedelta(minutes=30), row['Date_Time'] + timedelta(hours=2))
                meal_windows.append(meal_window)
            else:
                print(carb_event_count, row['Date_Time'])

            # set the time that the post-meal window ends (2 hours after the meal + 2 hours for the no-meal window length)
            no_meal_window_end = row['Date_Time'] + timedelta(hours=4)
        else:
            if (row['Date_Time'] > no_meal_window_end):
                # no-meal window has ended w/o being interrupted by a carb event - add it to the list of no-meal windows
                no_meal_window = get_glucose_window(glucose_df, no_meal_window_end - timedelta(hours=2), no_meal_window_end)
                no_meal_windows.append(no_meal_window)

                # begin a new no-meal window (next two hours)
                no_meal_window_end = no_meal_window_end + timedelta(hours=2)

    return {
        'meal_windows': pd.concat(meal_windows),
        'no_meal_windows': pd.concat(no_meal_windows)
    }


def get_df (csv_name, col_name):
    df = pd.read_csv(csv_name + ".csv", parse_dates=[['Date', 'Time']], usecols=['Date', 'Time', col_name])

    # reverse sequence so that earliest is first
    df = df.iloc[::-1]

    return df


carb_df = get_df('InsulinData2', 'BWZ Carb Input (grams)')
glucose_df = get_df('CGMData2', 'Sensor Glucose (mg/dL)')


# get_glucose_window(glucose_df, glucose_df.iloc[0]['Date_Time'], glucose_df.iloc[0]['Date_Time'] + timedelta(hours=2))

results = get_windows(carb_df, glucose_df)

print(results)