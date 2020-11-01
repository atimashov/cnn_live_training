from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash_table import DataTable
from flask import Flask
from helper import get_con, update_params, stop_train, SCHEMA
from plotly.subplots import make_subplots

FONT = 'Open Sans'
MAIN_COLOR = '#0074D9'  # 104c91
SECOND_COLOR = '#f93800'  # 'rgb(0, 255,0)' #'#ff7f4f' #'#e73927'
THIRD_COLOR = '#55bcad'

# -----------------------------------------------------------------------------------------------------------------------
#                                                 DASH ELEMENTS
# -----------------------------------------------------------------------------------------------------------------------
"""FILTERS"""
style_text = {
    'fontSize': 14,
    'textAlign': 'left',
}
style_filter = {
    'width': 200,
    'textAlign': 'center'
}
style_block = {
    'margin-top': -10,
    'textAlign': 'left'
}
filter_optimizer = html.Div(
    [
        html.H4(
            children='Choose optimizer : ',
            style=style_text
        ),
        dcc.Dropdown(
            id='dropdown-optimizer',
            options=[
                {'label': i, 'value': i} for i in ['Adam', 'SGD+Nesterov']
            ],
            value='Adam',
            multi=False,
            style=style_filter
        )
    ],
    style={
        'margin-top': -10,
        'textAlign': 'center',
        'margin-left': 0
    },
    className='three columns'
)
filter_lr = html.Div(
    [
        html.H4(
            children='Choose learning rate : ',
            style=style_text
        ),
        dcc.Input(
            id='input-lr', type='number',
            value=0.0001,
            min=0, max=1, step=0.00005,
            style=style_filter
        ),
    ],
    style=style_block,
    className='three columns'
)
filter_wd = html.Div(
    [
        html.H4(
            children='Choose weight decay : ',
            style=style_text
        ),
        dcc.Input(
            id='input-wd', type='number',
            value=0,
            min=0, max=1, step=0.05,
            style=style_filter
        ),
    ],
    style=style_block,
    className='three columns'
)
filter_dropout = html.Div(
    [
        html.H4(
            children='Choose dropout, % : ',
            style=style_text
        ),
        dcc.Input(
            id='input-do', type='number',
            value=50,
            min=20, max=80, step=1,
            style=style_filter
        ),
    ],
    style=style_block,
    className='three columns'
)
filters = html.Div(
    [
        filter_optimizer,
        filter_lr,
        filter_wd,
        filter_dropout
    ],
    style={
        'margin_top': 30,
        'margin-left': 180
    },
    className='row'
)

"""CONTROL PANEL"""
control_panel = html.Div(
    [
        html.Hr(),
        html.Div(
            children='Control Panel',
            style={
                'textAlign': 'center',
                'fontWeight': '550',
                'font-family': FONT,
                'font-size': 24,
                'height': '45px',
                'margin-top': -10,
                'margin-bottom': 20
            }
        ),
        filters,
        html.Button(
            'Submit parameters',
            id='button-params',
            style={
                'textAlign': 'center',
                'fontWeight': '550',
                'font-family': FONT,
                'font-size': 10,
                'height': '45px',
                'width': '190px',
                'margin-top': 30,
                'margin-left': 830
            }
        ),
        html.Div(
            id='time-submit',
            style={
                'textAlign': 'center',
                'fontWeight': '350',
                'color': 'red',
                'font-family': FONT,
                'font-size': 16,
                'height': '45px',
                'margin-top': 10
            }
        ),
        html.Hr(style={'margin-top': -10})
    ]
)

"""INPUT STATISTICS"""
def table():
    return DataTable(
        id='stat-table',
        columns=[{'name': i, 'id': i} for i in ['step', 'optimizer', 'learning rate', 'weight decay', 'dropout, %']],
        style_table={
            'overflowY': 'scroll',
            'height': '190px',
            'width': '360px',
            'margin-top': 0,
            'margin-left': 40,
        },
        style_cell={
            'textAlign': 'center',
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'dropout, %'},
             'width': '30%'}
        ],
        style_data_conditional=[
            {'if': {'row_index': 0},
             'color': 'red'}],
        style_header={
            'backgroundColor': MAIN_COLOR,
            'color': 'white',
            'fontWeight': 'bold'
        },
    )


stat_overall = html.Div([
    html.Div(
        children='Train Statistics',
        style={
            'textAlign': 'center',
            'fontWeight': '550',
            'font-family': FONT,
            'font-size': 16,
            'height': '45px',
            'margin-top': 0
        }
    ),
    table(),
    html.Div(
        children='Stop Training',
        style={
            'textAlign': 'center',
            'fontWeight': '550',
            'font-family': FONT,
            'font-size': 16,
            'height': '45px',
            'margin-top': 40
        }
    ),
    html.Button(
        'Stop training',
        id='button-stop',
        style={
            'textAlign': 'center',
            'fontWeight': '550',
            'font-family': FONT,
            'font-size': 8,
            'height': '35px',
            'width': '130px',
            'margin-top': 0,
            'margin-left': 140
        }
    ),
    html.Div(
        id='stop-submit',
        style={
            'textAlign': 'center',
            'fontWeight': '350',
            'color': 'red',
            'font-family': FONT,
            'font-size': 16,
            'height': '45px',
            'margin-top': 10
        }
    ),
],
    className='three columns'
)

"""LIVE PLOTS"""
# loss functions & accuracy
plots_output = html.Div(
    [
        stat_overall,
        dcc.Graph(
            id='chart',
            className='nine columns',
            style={'margin-top': -10}
        )
    ],
    style={
        'margin_top': 30
    },
    className='row'
)

plots_activations = html.Div(
    [
        html.Hr(style={'margin-top': 10}),
        html.Div(
            children='Activation maps (distribution)',
            style={
                'textAlign': 'center',
                'fontWeight': '550',
                'font-family': FONT,
                'font-size': 24,
                'height': '45px',
                'margin-top': -10,
                'margin-bottom': 20
            }
        ),
        dcc.Graph(
            id='chart_activations',
            className='row',
            style={'margin-top': -10}
        )
    ],
    style={
        'margin_top': 30
    },
    className='row'
)

# ----------------------------------------------------------------------------------------------------------------------
#                                                   INIT DASH
# ----------------------------------------------------------------------------------------------------------------------
server = Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    html.Div(
        children='Live Train Dashboard',
        style={
            'textAlign': 'center',
            'backgroundColor': '#ff7f4f',
            'fontWeight': '600',
            'font-family': FONT,
            'font-size': 28,
            'color': 'white',
            'height': '45px',
        }
    ),
    control_panel,
    html.Div(
        children='Loss function & Accuracy',
        style={
            'textAlign': 'center',
            'fontWeight': '550',
            'font-family': FONT,
            'font-size': 24,
            'height': '45px',
            'margin-top': -10,
            'margin-bottom': 20
        }
    ),
    plots_output,
    plots_activations,
    dcc.Interval(
        id='live-dash',
        interval=1 * 1000,  # in milliseconds
        n_intervals=0
    )
])


# -----------------------------------------------------------------------------------------------------------------------
#                                                   CALLBACKS
# -----------------------------------------------------------------------------------------------------------------------
@app.callback(
    dash.dependencies.Output('time-submit', 'children'),
    [
        dash.dependencies.Input('dropdown-optimizer', 'value'),
        dash.dependencies.Input('input-lr', 'value'),
        dash.dependencies.Input('input-wd', 'value'),
        dash.dependencies.Input('input-do', 'value'),
        dash.dependencies.Input('button-params', 'n_clicks_timestamp'),
    ]
)
def param_callback(optimizer, lr, wd, do, ts_start):
    if ts_start is None: return
    dt_start = datetime.fromtimestamp(ts_start / 1000)
    if (datetime.now() - dt_start).seconds > 2: return
    # updating parameters
    update_params(optimizer, lr, wd, do, dt_start)
    return 'Submitted at: {}'.format(str(dt_start.time())[:8])

@app.callback(
    dash.dependencies.Output('stop-submit', 'children'),
    [
        dash.dependencies.Input('button-stop', 'n_clicks_timestamp'),
    ]
)
def stop_callback(ts_stop):
    if ts_stop is None: return
    dt_start = datetime.fromtimestamp(ts_stop / 1000)
    if (datetime.now() - dt_start).seconds > 2: return
    # update 'stop_train'
    stop_train()
    return 'Training stopped at: {}'.format(str(dt_start.time())[:8])

"""UPDATE LOSS PLOTS & TABLE"""
@app.callback(
    [
        dash.dependencies.Output('chart', 'figure'),
        dash.dependencies.Output('stat-table', 'data'),
    ],
    [
        dash.dependencies.Input('live-dash', 'n_intervals')
    ]
)
def update_figure(n):
    # create layout
    fig = make_subplots(rows=1, cols=2, subplot_titles=('<b>Loss function (live)</b>', '<b>Accuracy (live)</b>'))
    fig.layout.annotations[0].update(x=0.25)
    fig.layout.annotations[0].update(y=1)
    fig.layout.annotations[0].update(font={'size': 16, 'family': FONT})
    fig.layout.annotations[1].update(x=0.75)
    fig.layout.annotations[1].update(y=1)
    fig.layout.annotations[1].update(font={'size': 16, 'family': FONT})
    fig.update_xaxes(title_text='step', row=1, col=1)
    fig.update_xaxes(title_text='step', row=1, col=2)
    fig.update_layout(
        plot_bgcolor = 'rgb(255,255,255)',
        showlegend = False,
        margin = {'t': 50}
    )

    # get last updated date
    q = """
    	    SELECT max(dt) as dt
    	    FROM {}.statistics
    	""".format(SCHEMA)
    with get_con() as conn:
        df = pd.read_sql(q, conn)

    # display output
    if df['dt'][0] is None: # if you want to hide previos inactive one: (datetime.now() - df['dt'][0]).seconds > 60
        return fig, []
    cols = 'dt, step, epoch, optimizer as opt, learning_rate as lr, weight_decay as wd, dropout as do'
    cols = cols + ', train_loss as t_loss, validate_loss as v_loss, train_accuracy as t_acc, validate_accuracy as v_acc'
    q = """
		SELECT {0} 
	    FROM {1}.statistics
	    WHERE dt_started = (
	        SELECT max(dt_started)
	        FROM {1}.statistics
	        )
	    ORDER BY step
	""".format(cols, SCHEMA)
    with get_con() as conn:
        df = pd.read_sql(q, conn)

    t_loss = go.Scatter(name='Train', x=df['step'] + 1, y=df['t_loss'], mode='lines', marker={'color': SECOND_COLOR})
    fig.add_trace(t_loss, row=1, col=1)
    v_loss = go.Scatter(name='Validate', x=df['step'] + 1, y=df['v_loss'], mode='lines', marker={'color': MAIN_COLOR})
    fig.add_trace(v_loss, row = 1, col = 1)
    t_acc = go.Scatter(name='Train', x = df['step'] + 1, y=df['t_acc'], mode='lines', marker={'color': SECOND_COLOR})
    fig.add_trace(t_acc, row = 1, col = 2)
    v_acc = go.Scatter(name='Validate', x=df['step'] + 1, y=df['v_acc'], mode='lines', marker={'color': MAIN_COLOR})
    fig.add_trace(v_acc, row = 1, col=2)

    # table
    p1 = df.sort_values(by='step', ascending=False)[['opt', 'lr', 'wd', 'do', 'step']].reset_index()
    del p1['index']
    p2 = pd.DataFrame({'opt': ['helper'], 'lr': 0, 'wd': 0, 'do': 0, 'step': 0}).append(
        p1.loc[p1.index < p1.shape[0] - 1], ignore_index=True)
    cond = (p1['opt'] != p2['opt']) | (p1['lr'] != p2['lr']) | (p1['wd'] != p2['wd']) | (p1['do'] != p2['do'])
    params = p1.loc[cond, ['opt', 'lr', 'wd', 'do', 'step']].to_dict('records')
    data = []
    for row in params:
        data.append(
            {
                'step': row['step'], 'optimizer': row['opt'], 'learning rate': row['lr'],
                'weight decay': row['wd'], 'dropout, %': row['do']}
        )
    return fig, data

"""UPDATE ACTIVATION MAPS PLOTS"""
@app.callback(
    dash.dependencies.Output('chart_activations', 'figure'),
    [
        dash.dependencies.Input('live-dash', 'n_intervals')
    ]
)
def update_figure(n):
    # get last updated date
    q = """
	    SELECT layer_type, number, weights, num_weights
	    FROM {}.activations
	""".format(SCHEMA)
    with get_con() as conn:
        df = pd.read_sql(q, conn)
    # create layout
    subtitles = ('<b>CONV1</b>', '<b>CONV2</b>', '<b>CONV3</b>', '<b>CONV4</b>', '<b>CONV5</b>', '<b>FC1</b>', '<b>FC2</b>', '<b>FC3</b>')
    fig = make_subplots(rows=1, cols=8, subplot_titles=subtitles)
    for i in range(8):
        fig.layout.annotations[i].update(x = 0.05 + 0.128 * i) # 147 for 7
        fig.layout.annotations[i].update(y=1)
        fig.layout.annotations[i].update(font={'size': 16, 'family': FONT})
        fig.update_xaxes(title_text='weights', row=1, col = i + 1)
    fig.update_layout(
        plot_bgcolor = 'rgb(255,255,255)',
        showlegend = False,
        margin = {'t': 50}
    )
    # display output
    if df.shape[0] == 0:
        return fig
    for i in range(1, 6):
        cond = (df['number'] == i) & (df['layer_type'] == 'conv')
        if sum(cond) > 0:
            t = go.Bar(name = 'Activation', x = df.loc[cond, 'weights'].values[0], y=df.loc[cond, 'num_weights'].values[0], marker={'color': SECOND_COLOR})
            fig.add_trace(t, row=1, col=i)
        cond = (df['number'] == i) & (df['layer_type'] == 'fc')
        if sum(cond) > 0: # and i < 3:
            t = go.Bar(name = 'Activation', x = df.loc[cond, 'weights'].values[0], y=df.loc[cond, 'num_weights'].values[0], marker={'color': SECOND_COLOR})
            fig.add_trace(t, row = 1, col = i + 5)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port = 2222)
