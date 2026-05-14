import pandas as pd 
import plotly.graph_objects as go  
import dash 
from dash import html 
from dash import dcc
from dash.dependencies import Input, Output 

df =  pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})
#####################################################
app = dash.Dash(__name__)

app.layout = html.Div(children = [
    html.H1("Airline Performance Dashboard", 
    style = {"textAlign": "center", "color":"#503D36", "font-size":40}),
    html.Div(["Input Year", dcc.Input(id="input-year", value = "2010", type ="number",
    style = {"height":"50px", "font-size":35,})],
    style = {"font-size":40}),
    html.Br(),
    html.Br(),
    html.Div(dcc.Graph(id="line-plot")),
])

#callback decorator:
@app.callback(Output(component_id ="line-plot", component_property = "figure"),
             Input(component_id = "input-year", component_property = "value")
             )

#callback function and return graph:
def get_graph(entered_year):
    #select data based on the entered year:
    df_select = df[df["Year"]==int(entered_year)]

    #group data by month and average over arrival delay time:
    df_group = df_select.groupby("Month")["ArrDelay"].mean().reset_index() 

    fig = go.Figure(data=go.Scatter(x =df_group["Month"], 
    y=df_group["ArrDelay"],
    mode = "lines",
    marker=dict(color="red")))
    fig.update_layout(title="Month Vs Average Flight Delay Time", xaxis_title="Month", yaxis_title="ArrDelay")    
    return fig


#Run the app:
if __name__ == "__main__":
    app.run()