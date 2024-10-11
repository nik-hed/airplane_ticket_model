#%%

import matplotlib.pyplot as plt

################EDA################################################

def pie_chart(df):

    class_counts = df['change_status'].value_counts()

    # Calculate percentage of each class
    class_percentages = class_counts / class_counts.sum() * 100

    # Plot pie chart
    plt.figure(figsize=(6, 6),facecolor='white')
    plt.pie(class_percentages, labels=class_percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title('Price changes, N=no change, I=increase, D=decrease')
    plt.legend(class_counts.index, title="Price changes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()

    class_counts = df['destination'].value_counts()

    class_percentages = class_counts / class_counts.sum() * 100

    plt.figure(figsize=(6, 6),facecolor='white')
    plt.pie(class_percentages, labels=class_percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of trip to different destinations')
    plt.legend(class_counts.index, title="Destinations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()



def plot_prices(df,destination):
     
     df=df.query(f"destination == '{destination}'")


     df = df.sort_values(by='time_to_trip', ascending=False)


     plt.figure(figsize=(10, 6))

     for trip_id in df['trip_id'].unique():
         trip_data = df[df['trip_id'] == trip_id]
         plt.plot(trip_data['time_to_trip'], trip_data['price'], marker='o', linestyle='-', label=f'Trip {trip_id}')

     plt.xlabel('Time to Trip')
     plt.ylabel('Price')
     plt.title(f"{destination}: Price vs Time to Trip for Each travel_id")
     plt.legend()
     plt.grid(True)
     plt.show()

     average_prices = df.groupby('time_to_trip')['price'].mean()

   
# Plot the average prices
     plt.plot(average_prices.index, average_prices.values, marker='x', linestyle='--', color='black', label='Average Price')

     plt.xlabel('Time to Trip')
     plt.ylabel('Price')
     plt.title(f"{destination}: Price vs Time to Trip for Each Trip ID and Average Price")
     plt.legend()
     plt.grid(True)
     plt.show()



     


def price_change_plot(df,tripto,trip_limit):
    
    df_run=df.copy()

    df_run=df_run.query(f"time_to_trip<={trip_limit}")

    if tripto!='All':
        df_run=df_run.query(f"destination == '{tripto}'")




    grouped_df = df_run.groupby('time_to_trip')['change_status_num'].sum().reset_index()
    sorted_df = grouped_df.sort_values(by='time_to_trip', ascending=False)

    sorted_df['cumulative_sum'] = sorted_df['change_status_num'].cumsum()


    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot

    ax1.bar(sorted_df['time_to_trip'], sorted_df['change_status_num'], color='skyblue', label='Sum of number price changes')
    ax1.set_xlabel('Time to trip')
    ax1.set_ylabel('Sum of price changes')
    ax1.set_title(f'Pice change anlysis for trip: {tripto}')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left')

    # Create a second y-axis for the cumulative sum line plot
    ax2 = ax1.twinx()
    ax2.plot(sorted_df['time_to_trip'], sorted_df['cumulative_sum'], color='red', marker='o', linestyle='-', label='Cumulative sum of price changes')
    ax2.set_ylabel('Cumulative Sum of price changes')
    ax2.legend(loc='upper right')

    plt.grid(True)
    plt.show()


    

    grouped_df = df_run.groupby('run_date_day_number')['change_status_num'].sum().reset_index()

    sorted_df = grouped_df.sort_values(by='run_date_day_number', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_df['run_date_day_number'], sorted_df['change_status_num'], color='green')
    plt.xlabel('On which week days are prices changing')
    plt.ylabel('Sum of price changes')
    plt.title(f'Sum of price changes over run days:{tripto} ')
    plt.xticks(rotation=45)
    plt.show()

    grouped_df = df_run.groupby('trip_date_month_number')['change_status_num'].sum().reset_index()

    sorted_df = grouped_df.sort_values(by='trip_date_month_number', ascending=False)

    

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_df['trip_date_month_number'], sorted_df['change_status_num'], color='blue')
    plt.xlabel('Price changes over travel months')
    plt.ylabel('Sum of price changes')
    plt.title(f'Sum of price changes over trip months: {tripto}')
    plt.xticks(rotation=45)
    plt.show()

# %%
