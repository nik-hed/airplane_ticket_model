The SQLite database consist of 3 tables: flight_prices, model_data and model_data_all.

**flight_prices:**<br> 


| Variable    | Desc |
| -------- | ------- |
| run_date  | Date and time when collecting the price    |
| trip_date | Date for the trip     |
| departure_time    | Time for the trip    |
| time_to_trip    | Travel_date - Run_date    |
| destination    | Flight destination    |
| point_of_departure    | Flight departure    |
| price    | Price of flight (SEK)    |
| fsl    | Y if there are few seats left for the trip    |

this is the data that was collected from an airline's website with BeautifulSoup during 2023.


**model_data/model_data_all:**<br> 

| Variable    | Desc |
| -------- | ------- |
| run_date  | Date and time when collecting the price    |
| trip_id  | Unique id for each trip    |
| trip_id_run  | Unique id for each trip and run    |
| trip_date | Date for the trip     |
| destination    | Flight destination    |
| point_of_departure    | Flight departure    |
| time_to_trip    | Travel_date - Run_date    |
| fsl    | 1 if few seats left, 0 else   |
| price    | Price of flight     |
| price_level    | Log10(price)     |
| run_date_day_number    | Day number of run date     |
| trip_date_day_number    | Day number of trip date    |
| run_date_month_number    | Month number of run date     |
| trip_date_month_number    | Month number of trip date    |
| weekend    | If weekend 1 (i.e run_day_number == 6,7) else 0     |
| wednesday    | After initial eda this variable was created.   |
| price_inc    | 1 if price increase from the previous day   |
| price_dec    | 1 if price decrease from the previous day    |
| price_change    | 1 if price decrease or increase from the previous day    |
| change_status    | N if no change, I if increase and D if decrease   |
| change_status_num    | 0 if no change, 1 if increase and -1 if decrease   |
| to_nice    | 1 if destination=Nice   |
| to_amsterdam    | 1 if destination=Amsterdam   |


model_data/model_data_all is based on flight_prices with some transformations.
model_data is just model_data_all filtered to exclude trips with no price changes such as trip_id Nice1700866800 and Amsterdam1693519200.

For the model y=F(X), y=change_status the features (X) are:\
time_to_trip,weekend,run_date_day_number,trip_date_month_number,price,price_level,to_nice,to_amsterdam



