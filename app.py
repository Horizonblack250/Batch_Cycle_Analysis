import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QualSteam Complete Batch Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Visibility and Contrast
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp {
        background-color: #e0e5ec;
    }
    
    /* 2. SIDEBAR STYLING (Force Light Theme) */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e0;
    }
    [data-testid="stSidebar"] * {
        color: #2d3748 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1a202c !important;
    }
    [data-testid="stSidebar"] label {
        color: #4a5568 !important;
        font-weight: 600 !important;
    }

    /* 3. Headers */
    h1, h2, h3 {
        color: #1a202c !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        border-bottom: 2px solid #cbd5e0;
        padding-bottom: 1rem;
    }

    /* 4. Metric Cards */
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #cbd5e0;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 14px;
        color: #4A5568;
        font-weight: 600;
        margin-top: 5px;
    }

    /* 5. Stat Cards */
    .stat-box {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #2d3748;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    file_path = 'data/df_all_batches.csv'
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except FileNotFoundError:
        return None

def calculate_stats(series):
    return {
        "Mean": f"{series.mean():.2f}",
        "Median": f"{series.median():.2f}",
        "Max": f"{series.max():.2f}",
        "Min": f"{series.min():.2f}",
        "Std Dev": f"{series.std():.4f}"
    }

def calculate_ramp_up_metrics(batch_df):
    """
    Calculates steam consumption during ramp-up.
    Logic: Ramp-up ends when process is +-1 degree of SP for > 1 minute.
    """
    # Sort to be safe
    batch_df = batch_df.sort_values('Timestamp').reset_index(drop=True)
    
    # Calculate dt in hours for steam integration (kg/hr * hr = kg)
    # Using shift to get time difference from previous row
    batch_df['dt_hours'] = batch_df['Timestamp'].diff().dt.total_seconds() / 3600.0
    batch_df['dt_hours'] = batch_df['dt_hours'].fillna(0) # First row is 0
    
    # Calculate incremental steam consumption per row
    batch_df['steam_kg_incremental'] = batch_df['Steam Flow Rate'] * batch_df['dt_hours']
    
    # --- FIND STABLE START TIME ---
    # 1. Calculate Error
    batch_df['sp_error'] = (batch_df['Process Temp'] - batch_df['Process Temp SP']).abs()
    
    # 2. Check threshold
    batch_df['in_range'] = batch_df['sp_error'] <= 1.0
    
    # 3. Find blocks of 'in_range'
    # Create groups of consecutive True/False
    batch_df['block'] = (batch_df['in_range'] != batch_df['in_range'].shift()).cumsum()
    
    stable_start_idx = None
    
    # Filter only for blocks that are IN RANGE
    in_range_blocks = batch_df[batch_df['in_range']].groupby('block')
    
    for block_id, group in in_range_blocks:
        duration_min = (group['Timestamp'].max() - group['Timestamp'].min()).total_seconds() / 60.0
        # Check if this stable block lasts > 1 minute (as per requirement)
        if duration_min >= 1.0:
            stable_start_idx = group.index[0]
            break
            
    # --- CALCULATE CONSUMPTION ---
    if stable_start_idx is not None and stable_start_idx > 0:
        # Ramp Up is from index 0 to stable_start_idx
        ramp_up_df = batch_df.iloc[:stable_start_idx+1]
        ramp_steam_kg = ramp_up_df['steam_kg_incremental'].sum()
        ramp_duration_min = (ramp_up_df['Timestamp'].max() - ramp_up_df['Timestamp'].min()).total_seconds() / 60.0
        status = "Stabilized"
    else:
        # Never stabilized according to logic, or started stable
        ramp_steam_kg = batch_df['steam_kg_incremental'].sum() # Assume all was ramp/unstable
        ramp_duration_min = (batch_df['Timestamp'].max() - batch_df['Timestamp'].min()).total_seconds() / 60.0
        status = "Not Stabilized"

    total_steam_kg = batch_df['steam_kg_incremental'].sum()
    
    return {
        "ramp_steam_kg": ramp_steam_kg,
        "total_steam_kg": total_steam_kg,
        "ramp_duration": ramp_duration_min,
        "status": status,
        "stable_idx": stable_start_idx
    }

# --- 3. MAIN APPLICATION ---
def main():
    st.title("QualSteam Real Dairy Complete Batch Cycle (SOPT Included)")

    df = load_data()

    if df is None:
        st.error(f"Data file not found at `data/df_all_batches.csv`. Please ensure the file exists in your repository.")
        st.stop()

    # --- SIDEBAR ---
    st.sidebar.header("Dataset Overview")
    total_batches = df['batch_id'].nunique()
    st.sidebar.info(f"**Total Batches Available:** {total_batches}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Batch Selection")
    
    unique_batches = sorted(df['batch_id'].unique())
    selected_batch_id = st.sidebar.selectbox("Select Batch ID", unique_batches)

    if selected_batch_id is None:
        st.stop()

    # Process specific batch data
    batch_data = df[df['batch_id'] == selected_batch_id].copy()
    
    # Calculate KPIs
    metrics = calculate_ramp_up_metrics(batch_data)
    
    start_time = batch_data['Timestamp'].min()
    end_time = batch_data['Timestamp'].max()
    total_duration = (end_time - start_time).total_seconds() / 60.0
    date_str = start_time.strftime('%Y-%m-%d')
    time_str = start_time.strftime('%H:%M')

    # --- TOP KPI ROW ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{date_str}</div><div class="metric-label">Batch Date</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_duration:.1f} min</div><div class="metric-label">Total Duration</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{metrics['ramp_steam_kg']:.2f} kg</div><div class="metric-label">Ramp-Up Steam</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{metrics['total_steam_kg']:.2f} kg</div><div class="metric-label">Total Steam</div></div>""", unsafe_allow_html=True)
    with c5:
        color = "#166534" if metrics['status'] == "Stabilized" else "#991b1b"
        st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:{color}">{metrics['ramp_duration']:.1f} min</div><div class="metric-label">Ramp-Up Time</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- VISUALIZATION & STATS ---
    col_graph, col_stats = st.columns([3, 1])

    with col_graph:
        st.subheader(f"Interactive Cycle Analysis - Batch {selected_batch_id}")
        
        # Create Subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Temperature", "Pressure (Inlet & Outlet)", "Steam Flow Rate", "Valve Opening"),
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        # Colors
        c_temp = '#D32F2F'       # Red
        c_temp_sp = 'black'      # Black
        c_p1 = '#004D40'         # Teal
        c_p2 = '#00008B'         # Dark Blue
        c_p_sp = '#1A237E'       # Indigo
        c_flow = '#7B1FA2'       # Violet
        c_valve = '#B8860B'      # Dark Goldenrod

        # 1. Temperature
        fig.add_trace(go.Scatter(x=batch_data['Timestamp'], y=batch_data['Process Temp SP'],
                                 mode='lines', line=dict(color=c_temp_sp, dash='dot', width=2),
                                 name='Process Temp SP'), row=1, col=1)
        fig.add_trace(go.Scatter(x=batch_data['Timestamp'], y=batch_data['Process Temp'],
                                 mode='lines', line=dict(color=c_temp, width=2),
                                 name='Process Temp'), row=1, col=1)
        
        # Add a marker/line for stabilization point if found
        if metrics['stable_idx'] is not None:
             stable_time = batch_data.loc[metrics['stable_idx'], 'Timestamp']
             fig.add_vline(x=stable_time, line_width=1, line_dash="dash", line_color="green", row="all")

        # 2. Pressure
        fig.add_trace(go.Scatter(x=batch_data['Timestamp'], y=batch_data['Pressure SP'],
                                 mode='lines', line=dict(color=c_p_sp, dash='dot', width=2),
                                 name='Pressure SP'), row=2, col=1)
        fig.add_trace(go.Scatter(x=batch_data['Timestamp'], y=batch_data['Inlet Steam Pressure'],
                                 mode='lines', line=dict(color=c_p1, width=2),
                                 name='Inlet P1'), row=2, col=1)
        fig.add_trace(go.Scatter(x=batch_data['Timestamp'], y=batch_data['Outlet Steam Pressure'],
                                 mode='lines', line=dict(color=c_p2, width=2),
                                 fill='tozeroy', fillcolor='rgba(0, 0, 139, 0.1)',
                                 name='Outlet P2'), row=2, col=1)

        # 3. Flow
        fig.add_trace(go.Scatter(x=batch_data['Timestamp'], y=batch_data['Steam Flow Rate'],
                                 mode='lines', line=dict(color=c_flow, width=2),
                                 fill='tozeroy', fillcolor='rgba(123, 31, 162, 0.1)',
                                 name='Flow Rate'), row=3, col=1)

        # 4. Valve
        fig.add_trace(go.Scatter(x=batch_data['Timestamp'], y=batch_data['QualSteam Valve Opening'],
                                 mode='lines', line=dict(color=c_valve, width=2),
                                 fill='tozeroy', fillcolor='rgba(184, 134, 11, 0.1)',
                                 name='Valve %'), row=4, col=1)

        # Styling
        fig.update_layout(
            height=900,
            showlegend=True,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="black")),
            font=dict(color="black")
        )
        
        # Axis Styling (Black text)
        axis_font_settings = dict(
            gridcolor='#f0f0f0',
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        )
        fig.update_annotations(font_color="black", font_size=16)
        
        fig.update_yaxes(title_text="Temp (°C)", row=1, col=1, **axis_font_settings)
        fig.update_yaxes(title_text="Bar", row=2, col=1, **axis_font_settings)
        fig.update_yaxes(title_text="kg/hr", row=3, col=1, **axis_font_settings)
        fig.update_yaxes(title_text="%", row=4, col=1, range=[0, 105], **axis_font_settings)
        fig.update_xaxes(row=4, col=1, **axis_font_settings)

        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.subheader("Statistics")
        st.markdown('<p style="color:black; font-weight:500;">Metrics for the <b>Entire Batch Cycle</b>.</p>', unsafe_allow_html=True)
        
        def stat_card(title, stats_dict, color_border):
            st.markdown(f"""
            <div class="stat-box" style="border-left: 5px solid {color_border};">
                <h4 style="margin:0; color: #1a202c; font-size: 16px;">{title}</h4>
                <div style="font-size: 0.95em; margin-top: 8px; color: #4a5568;">
                    <div style="display:flex; justify-content:space-between;"><span>Mean:</span> <b>{stats_dict['Mean']}</b></div>
                    <div style="display:flex; justify-content:space-between;"><span>Median:</span> <b>{stats_dict['Median']}</b></div>
                    <div style="display:flex; justify-content:space-between;"><span>Max:</span> <b>{stats_dict['Max']}</b></div>
                    <div style="display:flex; justify-content:space-between;"><span>Min:</span> <b>{stats_dict['Min']}</b></div>
                    <div style="display:flex; justify-content:space-between;"><span>Std:</span> <b>{stats_dict['Std Dev']}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        stat_card("Process Temp (°C)", calculate_stats(batch_data['Process Temp']), c_temp)
        stat_card("Outlet Pressure P2 (Bar)", calculate_stats(batch_data['Outlet Steam Pressure']), c_p2)
        stat_card("Inlet Pressure P1 (Bar)", calculate_stats(batch_data['Inlet Steam Pressure']), c_p1) # P1 included
        stat_card("Steam Flow (kg/hr)", calculate_stats(batch_data['Steam Flow Rate']), c_flow)
        stat_card("Valve Opening (%)", calculate_stats(batch_data['QualSteam Valve Opening']), c_valve)

if __name__ == "__main__":
    main()
