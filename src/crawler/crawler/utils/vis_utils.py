"""Utils for visualizing data"""

from typing import List, Dict

import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go


STATS_DROPDOWN_OPTIONS: List[Dict[str, str]] = [
    {'label': 'Views', 'value': 'view_count'},
    {'label': 'Comments', 'value': 'comment_count'},
    {'label': 'Likes', 'value': 'like_count'},
    {'label': 'Subscribers', 'value': 'subscriber_count'}
]


class Dashboard():
    def __init__(self, df: pd.DataFrame):
        self._df = df.sort_values(by=['video_id', 'timestamp_accessed'])

        usernames: List[str] = list(df['username'].unique())

        self._app = Dash(__name__)

        self._app.layout = html.Div([
            html.H4('Video statistics'),
            dcc.Dropdown(
                id="stats_dropdown",
                options=STATS_DROPDOWN_OPTIONS,
                value='view_count'
            ),
            html.Br(),
            dcc.Checklist(
                id="usernames_checklist",
                options=usernames,
                value=[],
                inline=True
            ),
            html.Br(),
            dcc.Graph(id="graph")
        ])

        @self._app.callback(
            Output("graph", "figure"),
            Input("usernames_checklist", "value"),
            State("stats_dropdown", "value")
        )
        def update_line_chart(usernames_: List[str],
                              stat_option_value: str) \
                -> go.Figure:
            """Update line chart"""
            stat_option_label: str = [e['label'] for e in STATS_DROPDOWN_OPTIONS if e['value'] == stat_option_value][0]

            fig = go.Figure(
                layout=go.Layout(
                    title=go.layout.Title(text="Statistics"),
                    xaxis={'title': 'Timestamp'},
                    yaxis={'title': stat_option_label}
                )
            )

            for username in usernames_:
                cols_ = ['timestamp_accessed', stat_option_value, 'video_id', 'title', 'upload_date']
                df_user = self._df.loc[self._df['username'] == username, cols_]
                # print(df_user)
                for _, (video_id, title) in df_user[['video_id', 'title']].drop_duplicates().iterrows():
                    df_vid = df_user[df_user['video_id'] == video_id]
                    fig.add_trace(go.Scatter(
                            x=df_vid['timestamp_accessed'],
                            y=df_vid[stat_option_value],
                            mode='lines+markers',
                            name=video_id,
                            hovertemplate=f'<br><b>Title</b>: {title}'
                                          f'<b>Video ID</b>: {video_id}'
                                          f'<br><b>Date uploaded</b>: {df_vid.iloc[0, :]["upload_date"]}'
                                          '<br><b>Time accessed</b>: %{x}'
                                          f'<br><b>{stat_option_label}</b>:' + ' %{y}'
                    ))

            return fig

    def run(self):
        """Start the Dash app"""
        self._app.run_server(debug=True)
