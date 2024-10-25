import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@st.cache_data(ttl='1d', show_spinner=False)
def load_data(file_name):
    data_path = Path(__file__).parent / 'data' / 'processed' / f"{file_name}.csv"
    df = pd.read_csv(data_path)
    return df


# Load Data
lease_df_stability = load_data('leases_df_stability')
lease_df_recession = load_data('leases_df_recession')
lease_df_boom = load_data('leases_df_boom')
state_df_stability = load_data('state_variables_stability')
state_df_recession = load_data('state_variables_recession')
state_df_boom = load_data('state_variables_boom')
industry_data = load_data('industry_df')
econ_factors_boom = load_data('economic_factors_boom')

econ_factors_stability = load_data('economic_factors_stability')
econ_factors_recession = load_data('economic_factors_recession')

feat_importance_boom = load_data('feature_importance_df_boom')
feat_importance_stability = load_data('feature_importance_df_stability')
feat_importance_recession = load_data('feature_importance_df_recession')
# Add scenario labels to feature importance data
feat_importance_boom['Scenario'] = 'Boom'
feat_importance_stability['Scenario'] = 'Stability'
feat_importance_recession['Scenario'] = 'Recession'

lstm_forecast_boom = load_data('lstm_forecast_boom')
lstm_forecast_recession = load_data('lstm_forecast_recession')
lstm_forecast_stability = load_data('lstm_forecast_stability')
lstm_metrics_boom = load_data('lstm_metrics_boom')
lstm_metrics_recession = load_data('lstm_metrics_recession')
lstm_metrics_stability = load_data('lstm_metrics_stability')
lstm_results_boom = load_data('lstm_results_boom')
lstm_results_recession = load_data('lstm_results_recession')
lstm_results_stability = load_data('lstm_results_stability')

cv_rmse_results_boom = load_data('cv_rmse_results_boom')
cv_rmse_results_recession = load_data('cv_rmse_results_recession')
cv_rmse_results_stability = load_data('cv_rmse_results_stability')

arima_forecast_boom = load_data('arima_forecast_boom')
arima_forecast_recession = load_data('arima_forecast_recession')
arima_forecast_stability = load_data('arima_forecast_stability')
arima_metrics_boom = load_data('arima_metrics_boom')
arima_metrics_recession = load_data('arima_metrics_recession')
arima_metrics_stability = load_data('arima_metrics_stability')
arima_results_boom = load_data('arima_results_boom')
arima_results_recession = load_data('arima_results_recession')
arima_results_stability = load_data('arima_results_stability')

econ_factors_boom['Scenario'] = 'Boom'
econ_factors_stability['Scenario'] = 'Stability'
econ_factors_recession['Scenario'] = 'Recession'
# Add Scenario Column
lease_df_stability['Scenario'] = 'Stability'
lease_df_recession['Scenario'] = 'Recession'
lease_df_boom['Scenario'] = 'Boom'
state_df_stability['Scenario'] = 'Stability'
state_df_recession['Scenario'] = 'Recession'
state_df_boom['Scenario'] = 'Boom'

# Combine DataFrames
lease_df_combined = pd.concat([lease_df_stability, lease_df_recession, lease_df_boom])
state_df_combined = pd.concat([state_df_stability, state_df_recession, state_df_boom])
econ_factors_combined = pd.concat([econ_factors_stability, econ_factors_recession, econ_factors_boom])
lease_df_combined['TimeStep'] = lease_df_combined['LeaseYear'] - lease_df_combined['StartYear']
state_df_combined[
    'TimeStep'] = state_df_combined.index  # Adjust this logic based on the actual structure of state_df_combined

merged_state_lease_df = pd.merge(state_df_combined, lease_df_combined, on='TimeStep', how='inner')

# Add RentPerSF column for analysis
lease_df_combined['RentPerSF'] = lease_df_combined['AnnualRent'] / lease_df_combined['SuiteSquareFootage']
avg_rent_per_sf_comparison = lease_df_combined.groupby(['LeaseYear', 'Scenario'])['RentPerSF'].mean().reset_index()

# Set up the Streamlit app
st.set_page_config(page_title='Real Estate Digital Twin Dashboard', layout='wide')
# Sidebar for Page Navigation
page = st.sidebar.selectbox(
    'Select Page',
    ['Simulated Building', 'Industry Analysis', 'Economic Analysis', 'Predictive Analysis', 'Insights']
)

# Title
st.title('Real Estate Digital Twin Dashboard')
st.markdown("Analyze the impact of different economic scenarios on commercial leasing metrics.")

# Page 1: Simulation Data
if page == 'Simulated Building':
    st.title('Building Analysis')
    # Economic Scenario Selection (Checkboxes)
    st.sidebar.header("Select Economic Scenarios")
    show_stability = st.sidebar.checkbox('Show Stability', value=True)
    show_recession = st.sidebar.checkbox('Show Recession')
    show_boom = st.sidebar.checkbox('Show Boom')

    # Filter DataFrame based on selected scenarios
    selected_lease_df = lease_df_combined[
        ((lease_df_combined['Scenario'] == 'Stability') & show_stability) |
        ((lease_df_combined['Scenario'] == 'Recession') & show_recession) |
        ((lease_df_combined['Scenario'] == 'Boom') & show_boom)
        ]
    selected_state_df = state_df_combined[
        ((state_df_combined['Scenario'] == 'Stability') & show_stability) |
        ((state_df_combined['Scenario'] == 'Recession') & show_recession) |
        ((state_df_combined['Scenario'] == 'Boom') & show_boom)
        ]

    # Dashboard Layout with Columns
    col1, col2 = st.columns(2)

    # Column 1: Leasing Data
    with col1:
        st.subheader("Leasing Data")
        # Total Annual Rent Over Time
        total_annual_rent = selected_lease_df.groupby(['LeaseYear', 'Scenario'])['AnnualRent'].sum().reset_index()
        fig_total_annual_rent = px.line(
            total_annual_rent,
            x='LeaseYear',
            y='AnnualRent',
            color='Scenario',
            title='Total Annual Rent Over Time',
            labels={'LeaseYear': 'Year', 'AnnualRent': 'Total Annual Rent ($)'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_total_annual_rent, use_container_width=True)

        # Average Rent Per Square Footage Over Time
        avg_rent_per_sf = selected_lease_df.groupby(['LeaseYear', 'Scenario'])['RentPerSF'].mean().reset_index()
        fig_avg_rent_per_sf = px.line(
            avg_rent_per_sf,
            x='LeaseYear',
            y='RentPerSF',
            color='Scenario',
            title='Average Rent Per Square Footage Over Time',
            labels={'LeaseYear': 'Year', 'RentPerSF': 'Avg. Rent per SF ($)'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_avg_rent_per_sf, use_container_width=True)

    # Column 2: Building Data
    with col2:
        st.subheader("Building Data")
        # Occupancy Rate by Lease Length
        fig_occ_rate_by_lease_length = px.box(
            selected_state_df,
            x='LeaseLength',
            y='OccupancyRate',
            color='Scenario',
            title='Occupancy Rate by Lease Length',
            labels={'LeaseLength': 'Lease Length (Years)', 'OccupancyRate': 'Occupancy Rate'},
            template='plotly_white'
        )
        st.plotly_chart(fig_occ_rate_by_lease_length, use_container_width=True)

        # Average Vacancy Rate by Lease Length
        avg_vacancy_by_lease = selected_state_df.groupby(['LeaseLength', 'Scenario'])[
            'VacancyRate'].mean().reset_index()
        fig_avg_vacancy_by_lease = px.line(
            avg_vacancy_by_lease,
            x='LeaseLength',
            y='VacancyRate',
            color='Scenario',
            title='Average Vacancy Rate by Lease Length',
            labels={'LeaseLength': 'Lease Length (Years)', 'VacancyRate': 'Average Vacancy Rate'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_avg_vacancy_by_lease, use_container_width=True)

    # Row for Comparative Analysis
    st.subheader("Comparative Analysis Across Scenarios")
    col3, col4 = st.columns(2)

    with col3:
        # Total Annual Rent Across Scenarios
        fig_comparative_annual_rent = px.line(
            total_annual_rent,
            x='LeaseYear',
            y='AnnualRent',
            color='Scenario',
            title='Total Annual Rent Across Scenarios',
            labels={'LeaseYear': 'Year', 'AnnualRent': 'Total Annual Rent ($)'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_comparative_annual_rent, use_container_width=True)

    with col4:
        # Vacancy Rate Distribution Across Scenarios
        fig_vacancy_across_scenarios = px.box(
            state_df_combined,
            x='Scenario',
            y='VacancyRate',
            title='Vacancy Rate Distribution Across Scenarios',
            labels={'Scenario': 'Economic Scenario', 'VacancyRate': 'Vacancy Rate'},
            template='plotly_white'
        )
        st.plotly_chart(fig_vacancy_across_scenarios, use_container_width=True)

    # Metrics Display at the Top
    st.header("Key Metrics")
    col5, col6, col7 = st.columns(3)

    with col5:
        avg_rent = selected_lease_df['AnnualRent'].mean()
        st.metric("Average Rent", f"${avg_rent:,.0f}")

    with col6:
        avg_occupancy = selected_state_df['OccupancyRate'].mean() * 100
        st.metric("Average Occupancy Rate", f"{avg_occupancy:.1f}%")

    with col7:
        avg_vacancy = selected_state_df['VacancyRate'].mean() * 100
        st.metric("Average Vacancy Rate", f"{avg_vacancy:.1f}%")
# Page 2: Simulation Data
if page == 'Industry Analysis':
    # Drop rows with missing 'industry title'
    industry_data = industry_data.dropna(subset=['industry title'])

    # Sidebar Toggle for View Type
    view_type = st.sidebar.radio("Select View Type", ["Individual Industry", "Overall Trends"])

    # If "Individual Industry" is selected
    if view_type == "Individual Industry":
        # Selectbox for industry options
        industry_options = industry_data['industry title'].unique()
        selected_industry = st.sidebar.selectbox("Select Industry", options=industry_options)

        # Filter data based on selected industry
        industry_filtered = industry_data[industry_data['industry title'] == selected_industry]

        # Plot Average Employment Over Years for Selected Industry
        avg_employment = industry_filtered.groupby('year')['average'].mean().reset_index()
        fig_avg_employment = px.line(
            avg_employment,
            x='year',
            y='average',
            title=f'Average Employment Over Time for {selected_industry}',
            labels={'year': 'Year', 'average': 'Average Employment'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_avg_employment, use_container_width=True)

        # Plot Monthly Trends for Selected Industry
        monthly_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        monthly_data = industry_filtered[['year'] + monthly_cols].melt(id_vars='year', var_name='month',
                                                                       value_name='employment')

        fig_monthly_trends = px.line(
            monthly_data,
            x='month',
            y='employment',
            color='year',
            title=f'Monthly Employment Trends for {selected_industry}',
            labels={'month': 'Month', 'employment': 'Employment'},
            template='plotly_white'
        )
        st.plotly_chart(fig_monthly_trends, use_container_width=True)

        # Calculate Annual Growth Rate
        avg_employment['growth_rate'] = avg_employment['average'].pct_change() * 100
        fig_growth_rate = px.bar(
            avg_employment,
            x='year',
            y='growth_rate',
            title=f'Yearly Growth Rate in Employment for {selected_industry}',
            labels={'year': 'Year', 'growth_rate': 'Growth Rate (%)'},
            template='plotly_white'
        )
        st.plotly_chart(fig_growth_rate, use_container_width=True)

        # Boxplot of Employment Distribution by Month
        fig_boxplot_monthly = px.box(
            monthly_data,
            x='month',
            y='employment',
            title=f'Employment Distribution by Month for {selected_industry}',
            labels={'month': 'Month', 'employment': 'Employment'},
            template='plotly_white'
        )
        st.plotly_chart(fig_boxplot_monthly, use_container_width=True)

    # If "Overall Trends" is selected
    if view_type == "Overall Trends":
        # Selectbox to show top N industries
        top_n = st.sidebar.slider("Select Number of Top Industries", min_value=5, max_value=50, value=20)

        # Calculate top N industries based on average employment
        top_industries = industry_data.groupby('industry title')['average'].mean().nlargest(top_n).reset_index()

        fig_top_industries = px.bar(
            top_industries,
            x='average',
            y='industry title',
            orientation='h',
            title=f'Top {top_n} Industries by Average Employment',
            labels={'average': 'Average Employment', 'industry title': 'Industry'},
            template='plotly_white'
        )
        st.plotly_chart(fig_top_industries, use_container_width=True)

        # Plot overall average employment over time for all industries
        overall_employment = industry_data.groupby('year')['average'].mean().reset_index()
        fig_overall_employment = px.line(
            overall_employment,
            x='year',
            y='average',
            title='Overall Average Employment Over Time',
            labels={'year': 'Year', 'average': 'Average Employment'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_overall_employment, use_container_width=True)

        # Heatmap for top N industries across months
        top_industry_names = top_industries['industry title'].unique()
        top_industry_data = industry_data[industry_data['industry title'].isin(top_industry_names)]
        monthly_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        monthly_heatmap_data = top_industry_data.groupby('industry title')[monthly_cols].mean()

        fig_heatmap = px.imshow(
            monthly_heatmap_data,
            labels=dict(x="Month", y="Industry", color="Employment"),
            x=monthly_cols,
            y=top_industry_names,
            title='Heatmap of Monthly Employment for Top Industries',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        # Radar Chart for Top Industries' Performance
        top_industry_data = industry_data[industry_data['industry title'].isin(top_industry_names)].groupby(
            'industry title').agg({
            'average': 'mean',
            'growth_rate': 'mean',
            'jan': 'mean',
            'feb': 'mean'
            # Include more months or metrics as needed
        }).reset_index()

        fig_radar = px.line_polar(
            top_industry_data,
            r='average',
            theta='industry title',
            line_close=True,
            title='Radar Chart for Top Industries\' Performance',
            template='plotly_white'
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        # Calculate growth rate for bubble chart
        industry_growth = industry_data.groupby('industry title').agg({'average': 'mean'}).reset_index()
        industry_growth['growth_rate'] = industry_data.groupby('industry title')['average'].pct_change().fillna(
            0).groupby(industry_data['industry title']).mean().values

        fig_bubble_chart = px.scatter(
            industry_growth,
            x='average',
            y='growth_rate',
            size='average',
            color='industry title',
            title='Employment vs. Growth Rate by Industry',
            labels={'average': 'Average Employment', 'growth_rate': 'Growth Rate (%)'},
            template='plotly_white'
        )
        st.plotly_chart(fig_bubble_chart, use_container_width=True)
# Page 3: Economic Analysis
if page == 'Economic Analysis':
    st.title('Economic Analysis')

    # Dropdown to select economic factor
    economic_factor = st.selectbox(
        "Select Economic Factor",
        options=['GDP Growth', 'Unemployment Rate', 'Inflation']
    )

    # Plot based on selected economic factor
    if economic_factor == 'GDP Growth':
        fig_gdp = px.line(
            econ_factors_combined,
            x='Year',
            y='GDP Growth',
            color='Scenario',
            title='GDP Growth Over Time by Scenario',
            labels={'Year': 'Year', 'GDP Growth': 'GDP Growth (%)'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_gdp, use_container_width=True)

    elif economic_factor == 'Unemployment Rate':
        fig_unemployment = px.line(
            econ_factors_combined,
            x='Year',
            y='Unemployment Rate',
            color='Scenario',
            title='Unemployment Rate Over Time by Scenario',
            labels={'Year': 'Year', 'Unemployment Rate': 'Unemployment Rate (%)'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_unemployment, use_container_width=True)

    elif economic_factor == 'Inflation':
        fig_inflation = px.line(
            econ_factors_combined,
            x='Year',
            y='Inflation',
            color='Scenario',
            title='Inflation Over Time by Scenario',
            labels={'Year': 'Year', 'Inflation': 'Inflation (%)'},
            markers=True,
            template='plotly_white'
        )
        st.plotly_chart(fig_inflation, use_container_width=True)

    # Add summary metrics for economic factors
    st.header("Key Metrics for Economic Factors")
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_gdp_growth = econ_factors_combined['GDP Growth'].mean() * 100
        st.metric("Avg. GDP Growth", f"{avg_gdp_growth:.2f}%")

    with col2:
        avg_unemployment = econ_factors_combined['Unemployment Rate'].mean() * 100
        st.metric("Avg. Unemployment Rate", f"{avg_unemployment:.2f}%")

    with col3:
        avg_inflation = econ_factors_combined['Inflation'].mean() * 100
        st.metric("Avg. Inflation", f"{avg_inflation:.2f}%")
    st.subheader("Correlation Analysis with State Variables")
    econ_factors_combined = econ_factors_combined.rename(columns={'Year': 'LeaseYear'})

    # Merge economic factors with state variables
    merged_df = pd.merge(
        merged_state_lease_df,
        econ_factors_combined,
        on='LeaseYear',
        how='inner'
    )

    # Select columns for correlation analysis
    correlation_columns = [
        'GDP Growth', 'Unemployment Rate', 'Inflation',
        'OccupancyRate', 'VacancyRate', 'LeaseLength', 'EconomicIndicator'
    ]
    correlation_df = merged_df[correlation_columns]

    # Calculate and display the correlation matrix
    correlation_matrix = correlation_df.corr()
    st.write("Correlation Matrix")

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        square=True,
        linewidths=0.5,
        ax=ax
    )
    plt.title('Correlation Matrix: Economic Factors and State Variables')
    st.pyplot(fig)
    st.title('Economic Analysis and Feature Interactions')

    # Dropdown to select first and second variables for interaction visualization
    st.subheader("Visualize Feature Interactions")
    var1 = st.selectbox(
        'Select First Variable',
        ['GDP Growth', 'Inflation', 'Unemployment Rate', 'OccupancyRate', 'VacancyRate', 'LeaseLength'],
        key='var1'
    )
    var2 = st.selectbox(
        'Select Second Variable',
        ['Inflation', 'GDP Growth', 'Unemployment Rate', 'OccupancyRate', 'VacancyRate', 'LeaseLength'],
        key='var2'
    )

    # Prepare data for visualization
    X = merged_df[[var1, var2, 'VacantSpace']]

    # 2D Contour Plot Visualization
    st.subheader(f'2D Contour Plot of {var1} vs {var2}')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.kdeplot(
        data=X,
        x=var1,
        y=var2,
        fill=True,
        cmap='coolwarm',
        levels=20,
        ax=ax
    )
    scatter = ax.scatter(
        X[var1], X[var2],
        c=X['VacantSpace'], cmap='coolwarm', edgecolor='k', alpha=0.5
    )
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f'2D Contour Plot of {var1} and {var2} with Vacant Space')
    fig.colorbar(scatter, ax=ax, label='Vacant Space')
    st.pyplot(fig)

    # Pair Plot Visualization for selected variables
    st.subheader('Pair Plot for Selected Variables')
    selected_columns = [var1, var2, 'VacantSpace']
    fig_pair = sns.pairplot(merged_df[selected_columns], diag_kind='kde', corner=True)
    st.pyplot(fig_pair)

    # Insights for Feature Interactions
    st.subheader('Insights')
    st.markdown(f"""
    - The 2D Contour Plot shows the density of interactions between {var1} and {var2}.
    - The Pair Plot helps visualize the distribution and relationships between the selected variables, including Vacant Space.
    - Use these plots to understand the nonlinear relationships between economic factors and state variables.
    """)

# Page 4: Predictive Analysis
if page == 'Predictive Analysis':
    st.title('Predictive Analysis of Vacancy')

    # Select Economic Scenario
    scenario = st.selectbox(
        'Select Economic Scenario',
        ['Stability', 'Boom', 'Recession']
    )
    # Display feature importance data based on the selected scenario
    st.subheader(f"Feature Importance Data for {scenario} Scenario")
    # Load data based on selected scenario
    if scenario == 'Stability':

        arima_forecast = arima_forecast_stability
        lstm_forecast = lstm_forecast_stability
        arima_metrics = arima_metrics_stability
        lstm_metrics = lstm_metrics_stability
        st.dataframe(feat_importance_stability)
        rmse_data = cv_rmse_results_stability




    elif scenario == 'Boom':

        arima_forecast = arima_forecast_boom
        lstm_forecast = lstm_forecast_boom
        arima_metrics = arima_metrics_boom
        lstm_metrics = lstm_metrics_boom
        st.dataframe(feat_importance_boom)
        rmse_data = cv_rmse_results_boom

    else:

        arima_forecast = arima_forecast_recession
        lstm_forecast = lstm_forecast_recession
        arima_metrics = arima_metrics_recession
        lstm_metrics = lstm_metrics_recession
        st.dataframe(feat_importance_recession)
        rmse_data = cv_rmse_results_recession
    # Display RMSE data based on the selected scenario
    st.subheader(f"Cross-Validation RMSE Vacant Rate Results for {scenario} Scenario")
    st.dataframe(rmse_data)

    # Ensure ARIMA and LSTM forecasts are 1D
    arima_forecast = np.ravel(arima_forecast)
    lstm_forecast = np.ravel(lstm_forecast)

    # Generate a sequence for 'Year' based on the length of the forecast data
    forecast_length = len(arima_forecast)  # Assuming both ARIMA and LSTM forecasts have the same length
    time_periods = range(1, forecast_length + 1)

    # Prepare the long-format DataFrame
    forecast_data_long = pd.DataFrame({
        'Year': time_periods,
        'ARIMA': arima_forecast,
        'LSTM': lstm_forecast
    }).melt(id_vars=['Year'], var_name='Model', value_name='Forecasted Vacant Space')

    # Display forecasted vacant space
    st.subheader(f'Forecasted Vacant Space for {scenario} Scenario')
    fig_forecast = px.line(
        forecast_data_long,
        x='Year',
        y='Forecasted Vacant Space',
        color='Model',
        title=f'12-Year Forecast of Vacant Space - {scenario} Scenario',
        labels={'value': 'Vacant Space', 'variable': 'Model'},
        template='plotly_white'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Display evaluation metrics for ARIMA and LSTM models
    st.subheader(f'Evaluation Metric for {scenario} Scenario ')
    # Extract ARIMA metrics
    arima_rmse = arima_metrics[arima_metrics['Metric'] == 'RMSE']['Value'].values[0]
    arima_mae = arima_metrics[arima_metrics['Metric'] == 'MAE']['Value'].values[0]
    arima_mape = arima_metrics[arima_metrics['Metric'] == 'MAPE']['Value'].values[0]

    # Extract LSTM metrics
    lstm_rmse = lstm_metrics[lstm_metrics['Metric'] == 'RMSE']['Value'].values[0]
    lstm_mae = lstm_metrics[lstm_metrics['Metric'] == 'MAE']['Value'].values[0]
    lstm_mape = lstm_metrics[lstm_metrics['Metric'] == 'MAPE']['Value'].values[0]

    # Create the combined metrics DataFrame
    metrics_combined = pd.DataFrame({
        'Model': ['ARIMA', 'LSTM'],
        'RMSE': [arima_rmse, lstm_rmse]

    })

    fig_metrics = px.bar(
        metrics_combined,
        x='Model',
        y=['RMSE'],
        barmode='group',
        title=f'Model Performance Metrics - {scenario} Scenario',
        labels={'value': 'RMSE', 'variable': 'Model'},
        template='plotly_white'
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

# Page 5: Insights
if page == 'Insights':
    st.title("Insights on Digital Twin & Commercial Real Estate Dynamics")
    st.header("Overview")

    # Writing sections
    st.write("""
        Digital twins offer a digital replica of physical assets, processes, and systems in real estate. 
        They provide a real-time view, predictive insights, and optimization opportunities by simulating real-world behavior.
        In commercial real estate, digital twins can enhance operational efficiency, asset management, and predictive modeling, 
        leading to informed decision-making and effective risk management.
        """)
    # Section on Macroeconomic Influence
    st.header("Macroeconomic Influence on Vacancy Measures")
    st.write("""
    Macroeconomic factors like inflation, unemployment rate, and GDP growth significantly affect both absolute and relative measures of vacancy in commercial real estate:
    - **Absolute Measures:** The total amount of vacant space.
    - **Relative Measures:** The vacancy rate as a percentage of the total leasable area.

    These factors influence tenant behavior, space demand, lease renewals, and overall market conditions. Understanding these interactions helps in assessing risk and identifying opportunities.
    """)
    # Dropdown for Economic Factor Selection
    st.subheader("Choose an Economic Factor to Explore")
    economic_factor = st.selectbox("Select an Economic Factor:", ["Inflation", "Unemployment Rate", "GDP Growth"])
    if economic_factor == "Inflation":
        st.write("""
            ### Inflation and Leasing Variables
            Inflation can lead to higher operating costs for building owners, impacting rental rates and vacancy levels. 
            It can reduce tenant affordability, leading to lease renegotiations or terminations, which in turn affect vacancy rates.
            """)
    elif economic_factor == "Unemployment Rate":
        st.write("""
        ### Unemployment Rate and Leasing Variables
        High unemployment can lead to decreased tenant demand, increased vacancy, and fewer lease renewals. 
        Conversely, low unemployment often correlates with increased demand for space, reducing vacancy rates.
        """)
    elif economic_factor == "GDP Growth":
        st.write("""
        ### GDP Growth and Leasing Variables
        GDP growth generally supports increased tenant demand, reducing vacancy rates. 
        It correlates with business expansion, higher lease absorption, and fewer lease terminations.
        """)
    # Leasing Trends and Economic Cyclicality Section
    st.header("Leasing Trends & Economic Cyclicality")
    st.write("""
    Leasing trends in commercial real estate are affected by economic cycles. During expansion phases, vacancy rates typically decrease, while during recessions, vacancy rates increase as tenant demand weakens.

    **Key Trends:**
    - **Economic Boom:** Higher leasing activity, lower vacancy rates, increased rents.
    - **Economic Downturn:** Higher vacancy rates, renegotiated leases, tenant downsizing.
    """)
    # Scenario Simulation Section
    st.subheader("Scenario Simulations for Predictive Modeling")
    st.write("""
    Scenario simulations help predictive models evaluate how vacancy rates respond to different economic impacts. 
    By simulating variations in inflation, unemployment, and GDP growth, we can forecast potential outcomes and identify risks:

    1. **Macro-level Factors:** 
    - How large-scale changes (e.g., global recessions) impact the overall market.
    2. **Micro-level Factors:** 
    - How local market dynamics (e.g., regional employment shifts) affect vacancy.
    3. **Economic Shocks:** 
    - Simulating sudden events (e.g., financial crises, pandemics) to evaluate their impact on commercial real estate dynamics.

    These insights allow investors, property managers, and policy makers to anticipate potential risks, understand complex dynamics, and make informed decisions.
    """)
    # Summary Section
    st.header("Conclusion")
    st.write("""
    Understanding the combined effect of macroeconomic variables and leasing trends through digital twin models 
    can offer a strategic advantage. The integration of scenario simulations enhances predictive models, enabling more robust analysis of potential outcomes.

    The ongoing analysis of commercial real estate vacancy rates helps stakeholders make better investment and operational decisions, 
    especially when facing economic uncertainty or volatility.
    """)