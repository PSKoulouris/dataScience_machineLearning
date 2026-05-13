import pandas as pd 
import plotly.express as px 
import dash 
from dash import dcc 
from dash import html 

df =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 

                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str}) 
        
data = df.sample(n = 500 , random_state = 42)

fig = px.pie(df, values = "Flights", names = "DistanceGroup", title = "Proportion of flights by distance")
########################################################
app = dash.Dash(__name__)
app.layout = html.Div(children = [
    html.H1("Airline Dashboard", 
    style = {"textAlign": "center", "color":"#503D36", "font-size":40}),
    html.P("Proportion of distance group (250 miles distance interval) by flights.", style = {"textAlign": "center", "color":"#F57241"}),
    dcc.Graph(figure = fig),
])

if __name__ == "__main__":
    app.run()

