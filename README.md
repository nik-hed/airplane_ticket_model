# airplane_ticket_model
Classification model in order to find if next day ticket price increases or decreases, the variables is 
defined in mysql as:

case when Price<lead(price) over (partition by traveldate order by traveldate,TimetoTrip desc) then 'I'
when Price>lead(price) over (partition by traveldate order by traveldate,TimetoTrip desc) then 'D'
else 'N' end as change_status
