import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('raw_data/cleaned_boston_data.csv')

# # Data Preprocessing
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# # Train the Random Forest Regressor
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Save the trained model
# joblib.dump(rf_model, 'random_forest_model.pkl')

# Train the XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Save the trained model
# joblib.dump(xgb_model, 'xgboost_model.pkl')

# Computation of the Correlation Heatmap
corr_matrix = df.corr()
heatmap_fig = px.imshow(
    corr_matrix,
    color_continuous_scale='RdBu',
    labels=dict(color='Correlation'),
    title='Feature Correlation Heatmap'
)
heatmap_fig.update_layout(
    template='plotly_dark',
    xaxis_title="Features",
    yaxis_title="Features",
    margin=dict(l=40, r=40, t=40, b=40)
)

# Initializing the dash app 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Title of the dashboard
app.title = "Real Estate Dashboard"

# Layour of dashboard
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.H1("Boston Housing Data Dashboard", style={'color': 'white'}), width=8),
        dbc.Col(html.H2("Erdem Naranbat, Rao Daud Ali Khan", style={'text-align': 'right', 'color': 'white', 'font-size': '20px'}), width=4, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-end'})
    ]),

    html.Hr(),

    # Correlation Heatmap
    dcc.Graph(figure=heatmap_fig, style={'margin-top': '20px'}),

    html.Hr(),

    dcc.Dropdown(
        id = "feature_dropdown",
        options = [{'label': col, 'value': col} for col in df.columns if col not in ['MEDV']],
        multi = False,
        value = None,
        placeholder = "Select a Feature",
        style={
        'font-family': 'Arial, sans-serif',
        'font-size': '14px',
        'padding': '0px 0px',
        'color': '#636efa',
        'background-color': '#333',
        'border': '#636efa'
        }
    ),

    html.Div(id='feature_message', style={'color': 'white', 'text-align': 'center', 'margin-top': '20px'}),

    html.Hr(),
    
    html.Div([
    dbc.Card(
        dbc.CardBody([
            html.H4("Enter values for the attributes", className="card-title", style={'color': 'white'}),
            dbc.Row([
                dbc.Col([
                    html.Label(f"{col}:", style={'color': 'white', 'font-weight': 'bold'}),
                    dcc.Input(
                        id=f'{col}_input',
                        type='number',
                        value=X[col].mean(),
                        style={
                            'width': '100%',
                            'padding': '10px',
                            'border': '1px solid #444',
                            'border-radius': '5px',
                            'background-color': '#333',
                            'color': 'white'
                        }
                    )
                ], width=4) for col in X.columns if col not in ['NOX_INDUS', 'RM_LSTAT']
            ], className="g-3"),  # g-3 adds spacing between rows
            dbc.Row([
                dbc.Col(
                    html.Button(
                        'Predict Price',
                        id='predict_button',
                        n_clicks=0,
                        className="btn btn-primary",
                        style={
                            'width': '100%',
                            'padding': '10px',
                            'font-size': '16px',
                            'border-radius': '5px',
                            'margin-top': '20px'
                        }
                    ), width=4
                )
            ]),
            html.Div(id='prediction_output', style={'margin-top': '20px', 'color': 'white', 'font-size': '18px'})
        ]),
        color="dark",  # Adds Bootstrap's dark theme to the card
        style={'margin-top': '20px'}
    )
    ]),

    html.Hr(),

    # Feature Importance Plot
    html.Div(id='feature_importance_container', style={'margin-top': '20px'}),

    html.Hr(),

    # Model Evaluation Metrics Section
    html.Div(id='model_metrics', style={'margin-top': '20px', 'color': 'white', 'font-size': '16px'})

    
], fluid = True)

# Callback for Scatterplot
@app.callback(
    Output('feature_message', 'children'),
    Input('feature_dropdown', 'value')
)
def update_graph(selected_feature):
    if not selected_feature:  # Handle case where no feature is selected
        return html.Div("No feature selected. Please choose a feature from the dropdown above.", 
                        style={'font-size': '20px', 'color': 'white'})
    
    my_dict = {
        "CRIM": "Crime Rate (per capita by town)",
        "ZN": "Proportion of Residential Land Zoned of Lots Over 25'000 sq.ft",
        "INDUS": "Proportion of Non-retail business (acres per town)",
        "CHAS": "Charles River dummy variable",
        "NOX": "Nitric Oxide Concentration (parts per 10 million)",
        "RM": "Average Numbers of Rooms (per dwelling)",
        "AGE": "Proportion of Owner-occupied units built prior to 1940",
        "DIS": "Weighted distances to Five Boston Employment Centers",
        "RAD": "Index of Accessibility to Radial Highways",
        "TAX": "Full Value Property-Tax Rate (per $10'000)",
        "PTRATIO": "Pupil-Teacher Ratio (by town)",
        "LSTAT": "Percentage of Lower Status (by population)",
        "NOX_INDUS": "NOX_INDUS",
        "RM_LSTAT": "RM_LSTAT",
        "B": "Proportion of Black People by Town"
    }
    fig = px.scatter(
        df, 
        x=selected_feature, 
        y='MEDV', 
        title=f'{my_dict[selected_feature]} vs Median Value of Owner Occupied Homes in $10000s'
    )
    fig.update_layout(template='plotly_dark')
    return dcc.Graph(figure=fig)


# Callback to handle price prediction using Random Forest
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [Input(f'{col}_input', 'value') for col in X.columns[:-2]]
)
def predict_price(n_clicks, *inputs):
    if n_clicks > 0:
        # Map the inputs to the correct columns
        input_data = {col: value for col, value in zip(X.columns[:-2], inputs)}

        # Create a DataFrame for the user inputs
        input_df = pd.DataFrame([input_data])

       # Calculate NOX_INDUS and RM_LSTAT
        nox_indus = input_data['NOX'] * input_data['INDUS']
        rm_lstat = input_data['RM'] * input_data['LSTAT']

        # Append the derived features
        input_df['NOX_INDUS'] = nox_indus
        input_df['RM_LSTAT'] = rm_lstat

        # Ensure the input DataFrame matches the trained model's column order
        input_df = input_df[X.columns]

        # Load the trained Random Forest model
        # rf_model = joblib.load('random_forest_model.pkl')
        # Predict the price using the Random Forest model
        # predicted_price = rf_model.predict(input_df)[0]

        # Load the trained XGBoost model
        xgb_model = XGBRegressor(
            objective='reg:squarederror',
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100
        )

        # Fit the model
        xgb_model.fit(X_train, y_train)

        # Make predictions
        # predicted_price = xgb_model.predict(X_test)

        # xgb_model = joblib.load('xgboost_model.pkl')

        # Predict the price using the XGBoost model
        predicted_price = xgb_model.predict(input_df)[0]
        
        return dbc.Card(
            dbc.CardBody([
                
                
                html.Div([
                    html.P(f"NOX * INDUS = {nox_indus:.2f}", style={'font-size': '18px', 'margin-bottom': '5px'}),
                    html.P(f"RM * LSTAT = {rm_lstat:.2f}", style={'font-size': '18px'}),
                ], style={'color': 'white'}),
                html.Hr(),
                html.H4(f"Predicted Price: ${predicted_price:,.2f}k", className="card-title", style={'color': '#636efa'}),
            ]),
            color="dark",  # Use dark theme for the card
            style={'margin-top': '20px', 'padding': '10px'}
        )

    return ""

# Callback to view feature importance
@app.callback(
    Output('feature_importance_container', 'children'),
    Input('predict_button', 'n_clicks')
)
def update_feature_importance(n_clicks):
    if n_clicks > 0:
        # importances = rf_model.feature_importances_

         # Load the trained XGBoost model
        xgb_model = joblib.load('xgboost_model.pkl')

        # Get feature importances
        importances = xgb_model.feature_importances_

        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (XGBoost)',
            text='Importance'
        )
        fig.update_layout(template='plotly_dark', yaxis=dict(autorange='reversed'))
        return dcc.Graph(figure=fig)

    # Return nothing before a prediction is made
    return None

# Callback to view model evaluation metrics
# @app.callback(
#     Output('model_metrics', 'children'),
#     Input('predict_button', 'n_clicks')
# )
@app.callback(
    Output('model_metrics', 'children'),
    [Input('predict_button', 'n_clicks')],
    [Input(f'{col}_input', 'value') for col in X.columns[:-2]]
)
def update_model_metrics(n_clicks, *inputs):
    if n_clicks > 0:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Load the trained XGBoost model
        xgb_model = XGBRegressor(
            objective='reg:squarederror',
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100
        )

        # Train the model (ideally, use the pre-trained model saved earlier)
        xgb_model.fit(X_train, y_train)

        # Generate predictions for the test set
        xgb_predictions = xgb_model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, xgb_predictions)
        mse = mean_squared_error(y_test, xgb_predictions)
        r2 = r2_score(y_test, xgb_predictions)

        # Return the metrics as a formatted HTML block
        return html.Div([
            html.H4("Model Evaluation Metrics", style={'margin-bottom': '15px'}),
            html.P(f"Mean Absolute Error (MAE): {mae:.2f}"),
            html.P(f"Mean Squared Error (MSE): {mse:.2f}"),
            html.P(f"R² Score: {r2:.2f}")
        ])
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug = True)