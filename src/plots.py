import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import inputs
try:
    import paths 
except:
    import src.paths as paths


''' Simulated & Metered heat energy consumption - Plot '''
def plot_model(scr_gebaeude_id, output_resolution, training_ratio):
 
    ''' OPEN DATA '''
    metered=pd.read_excel(os.path.join(paths.DATA_DIR, "HeatEnergyDemand_{}_{}.xlsx".format(scr_gebaeude_id, output_resolution)), index_col=0)
    start_time, end_time = metered.index[1].strftime('%Y-%m-%d %H:%M:%S'), metered.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    HeatingEnergy_sum = pd.read_csv(os.path.join(paths.RES_DIR, "DIBS_sim/{}/{}/{}/HeatingEnergy_{}-{}-{}_{}-{}-{}.csv".format(scr_gebaeude_id, output_resolution, training_ratio, start.year, start.month, start.day, end.year, end.month, end.day)), sep=';', index_col=0)
    HeatingEnergy_sum.index = pd.to_datetime(HeatingEnergy_sum.index) + pd.DateOffset(hours=23)

    merged_df = pd.merge(metered, HeatingEnergy_sum, left_index=True, right_index=True, how='inner')
    area = inputs.get_building_inputdata(scr_gebaeude_id)['ebf'].iloc[0]
    merged_df = merged_df.div(area)


    ''' PLOT '''
    fig = px.line(merged_df, x=merged_df.index, y=merged_df.columns, 
                title='Metered and Simulated Heat Energy Consumption'.format(scr_gebaeude_id),
                labels={'index': 'Year', 'value': 'Heat energy demand kWh/m²'},
                line_shape='linear', 
                markers=True,  
                color_discrete_sequence=['black', 'red'])

    # Update trace names
    fig.for_each_trace(lambda t: t.update(name='Metered Heat Energy Consumption') 
                    if t.name == 'Consumption' 
                    else t.update(name='Simulated Heat Energy Consumption'))

    fig.update_layout(
        font=dict(
            family="Calibri",
            size=16
        ),
        xaxis=dict(
            title='Year', 
            title_font=dict(size=14),
            showgrid=True,
            zeroline=False,
            linecolor='black'
        ),
        yaxis=dict(
            title_font=dict(size=14),
            showgrid=True,
            zeroline=False,
            linecolor='black'
        ),
        legend=dict(
            title=None,
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.95,
            bgcolor='rgba(255, 255, 255, 0.7)', 
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14)
        ),
        plot_bgcolor='white'
    )

    fig.update_traces(marker=dict(size=12))

    # Update x-axis ticks to show years
    fig.update_xaxes(tickvals=merged_df.index, ticktext=merged_df.index.year)

    return fig
