"""Utils for visualizing raw_data"""

from typing import List, Dict

import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

from src.crawler.crawler.constants import (COL_UPLOAD_DATE, COL_TIMESTAMP_FIRST_SEEN, COL_VIDEO_ID, COL_USERNAME,
                                           COL_TIMESTAMP_ACCESSED, COL_TITLE)


STATS_DROPDOWN_OPTIONS: List[Dict[str, str]] = [
    {'label': 'Views', 'value': 'view_count'},
    {'label': 'Comments', 'value': 'comment_count'},
    {'label': 'Likes', 'value': 'like_count'},
    {'label': 'Subscribers', 'value': 'subscriber_count'}
]


class Dashboard():
    def __init__(self, df: pd.DataFrame):
        self._df = df.sort_values(by=[COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED])

        usernames: List[str] = list(df[COL_USERNAME].unique())

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
            dcc.Graph(id="graph_stats"),
            html.Br(),
            dcc.Graph(id="graph_upload")
        ])

        @self._app.callback(
            Output("graph_stats", "figure"),
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
                cols_ = [COL_TIMESTAMP_ACCESSED, stat_option_value, COL_VIDEO_ID, COL_TITLE, COL_UPLOAD_DATE,
                         COL_TIMESTAMP_FIRST_SEEN]
                df_user = self._df.loc[self._df[COL_USERNAME] == username, cols_]
                # print(df_user)
                for _, (video_id, title) in df_user[[COL_VIDEO_ID, COL_TITLE]].drop_duplicates().iterrows():
                    df_vid = df_user[df_user[COL_VIDEO_ID] == video_id]
                    fig.add_trace(go.Scatter(
                            x=df_vid[COL_TIMESTAMP_ACCESSED],
                            y=df_vid[stat_option_value],
                            mode='lines+markers',
                            name=video_id,
                            hovertemplate=f'<br><b>Title</b>: {title}'
                                          f'<br><b>Video ID</b>: {video_id}'
                                          f'<br><b>Date uploaded</b>: {df_vid.iloc[0, :]["upload_date"]}'
                                          '<br><b>Time accessed</b>: %{x}'
                                          f'<br><b>Time first seen</b>: {df_vid.iloc[0, :]["timestamp_first_seen"]}'
                                          f'<br><b>{stat_option_label}</b>:' + ' %{y}'
                    ))

            return fig

        @self._app.callback(
            Output("graph_upload", "figure"),
            Input("usernames_checklist", "value")
        )
        def update_line_chart(usernames_: List[str]) -> go.Figure:
            """Update line chart"""
            username_ids = {}
            for i, username in enumerate(usernames_):
                username_ids[username] = i

            fig = go.Figure(
                layout=go.Layout(
                    title=go.layout.Title(text="Upload Dates"),
                    xaxis={'title': 'Day'},
                    yaxis={
                        'title': 'Username',
                        'tickvals': [username_ids[username] for username in usernames_],
                        'ticktext': usernames_
                    },
                )
            )

            cols_ = [COL_VIDEO_ID, COL_USERNAME, COL_TITLE, COL_UPLOAD_DATE]
            for username in usernames_:
                df_user = self._df.loc[self._df[COL_USERNAME] == username, cols_].drop_duplicates()
                fig.add_trace(go.Scatter(
                    x=df_user[COL_UPLOAD_DATE],
                    y=[username_ids[username]] * len(df_user),
                    mode='lines+markers',
                    name=username,
                    # hovertemplate=f'<br><b>Title</b>: {title}'
                    #               f'<b>Video ID</b>: {video_id}'
                    #               f'<br><b>Date uploaded</b>: {df_vid.iloc[0, :]["upload_date"]}'
                    #               '<br><b>Time accessed</b>: %{x}'
                    #               f'<br><b>{stat_option_label}</b>:' + ' %{y}'
                ))

            return fig

    def run(self):
        """Start the Dash app"""
        self._app.run_server(debug=True, host='localhost', port=14982)
