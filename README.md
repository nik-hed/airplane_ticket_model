# airplane_ticket_model

Simple analysis of how prices for 3 different trips during 2023 are changing. Based on data collected from airline during 2023.

Classification model in order to find if next day ticket price increases or decreases, the variables is 
defined in mysql as:

case when Price<lead(price) over (partition by traveldate order by traveldate,TimetoTrip desc) then 'I'
when Price>lead(price) over (partition by traveldate order by traveldate,TimetoTrip desc) then 'D'
else 'N' end as change_status


/db: sqlite database with flight ticket prices

/udf: user defined function used in ticket_price_model.ipynb
