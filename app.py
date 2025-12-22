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

# --- FINANCIAL CONSTANTS REMOVED ---

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
        height: 100%;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .metric-subvalue {
        font-size: 14px;
        color: #718096;
        font-weight: 500;
        margin-bottom: 5px;
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
    # Attempting to load from common locations or fallback
    file_paths = ['data/df_all_batches.csv', '/content/outputs/df_clean.csv', 'df_clean.csv']
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
            # Normalize column names if needed (optional based on your CSV source)
            # Ensuring Timestamp is datetime
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except FileNotFoundError:
            continue
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
    Calculates metrics for the ramp-up phase and total steam.
    """
    local_df = batch_df.sort_values('Timestamp').reset_index(drop=True)
    
    local_df['dt_hours'] = local_df['Timestamp'].diff().dt.total_seconds() / 3600.0
    local_df['dt_hours'] = local_df['dt_hours'].fillna(0)
    
    local_df['steam_kg_incremental'] = local_df['Steam Flow Rate'] * local_df['dt_hours']
    
    local_df['sp_error'] = (local_df['Process Temp'] - local_df['Process Temp SP']).abs()
    local_df['in_range'] = local_df['sp_error'] <= 1.0
    local_df['block'] = (local_df['in_range'] != local_df['in_range'].shift()).cumsum()
    
    stable_start_idx = None
    stable_timestamp = None
    
    in_range_blocks = local_df[local_df['in_range']].groupby('block')
    
    for block_id, group in in_range_blocks:
        duration_min = (group['Timestamp'].max() - group['Timestamp'].min()).total_seconds() / 60.0
        if duration_min >= 1.0:
            stable_start_idx = group.index[0]
            stable_timestamp = group.loc[stable_start_idx, 'Timestamp']
            break
            
    if stable_start_idx is not None:
        ramp_up_df = local_df.iloc[:stable_start_idx+1]
        ramp_duration_min = (ramp_up_df['Timestamp'].max() - ramp_up_df['Timestamp'].min()).total_seconds() / 60.0
        status = "Stabilized"
    else:
        ramp_duration_min = (local_df['Timestamp'].max() - local_df['Timestamp'].min()).total_seconds() / 60.0
        status = "Not Stabilized"

    total_steam_kg = local_df['steam_kg_incremental'].sum()
    
    return {
        "total_steam_kg": total_steam_kg,
        "ramp_duration": ramp_duration_min,
        "status": status,
        "stable_ts": stable_timestamp
    }

def calculate_overshoot_steam(batch_df):
    """
    Calculates steam consumed specifically during the Overshoot Phase.
    """
    local_df = batch_df.sort_values('Timestamp').reset_index(drop=True)
    target_sp = local_df['Process Temp SP'].max()
    
    # 1. Find Start (Reaching SP)
    reached_sp = local_df['Process Temp'] >= target_sp
    if not reached_sp.any():
        return 0.0 # Never reached setpoint
        
    start_idx = reached_sp.idxmax()
    
    # 2. Find End (Stability)
    local_df['sp_error'] = (local_df['Process Temp'] - local_df['Process Temp SP']).abs()
    local_df['in_spec'] = local_df['sp_error'] <= 1.0
    blocks = (local_df['in_spec'] != local_df['in_spec'].shift()).cumsum()
    
    end_idx = None
    
    spec_blocks = local_df[local_df['in_spec']].groupby(blocks)
    for _, group in spec_blocks:
        g_duration = (group['Timestamp'].max() - group['Timestamp'].min()).total_seconds() / 60.0
        # Look for stability occurring AFTER or AT the start of overshoot
        if g_duration >= 10.0 and group.index[0] >= start_idx:
            end_idx = group.index[0]
            break
            
    if end_idx is None:
        return 0.0 # Never stabilized after overshoot
        
    # 3. Integrate Steam for this window
    overshoot_df = local_df.loc[start_idx : end_idx].copy()
    
    overshoot_df['dt_hours'] = overshoot_df['Timestamp'].diff().dt.total_seconds() / 3600.0
    overshoot_df['dt_hours'] = overshoot_df['dt_hours'].fillna(0)
    
    overshoot_steam_kg = (overshoot_df['Steam Flow Rate'] * overshoot_df['dt_hours']).sum()
    
    return overshoot_steam_kg

def calculate_stability_kpis(batch_df):
    """
    Calculates the % of the batch spent within 0.5, 1.0, and 1.5 degrees of SP.
    """
    local_df = batch_df.sort_values('Timestamp').reset_index(drop=True)
    
    start_time = local_df['Timestamp'].min()
    end_time = local_df['Timestamp'].max()
    total_duration = (end_time - start_time).total_seconds() / 60.0
    
    local_df['sp_error'] = (local_df['Process Temp'] - local_df['Process Temp SP']).abs()
    local_df['in_anchor_spec'] = local_df['sp_error'] <= 1.0
    local_df['block'] = (local_df['in_anchor_spec'] != local_df['in_anchor_spec'].shift()).cumsum()
    
    stable_start_idx = None
    
    spec_blocks = local_df[local_df['in_anchor_spec']].groupby('block')
    for _, group in spec_blocks:
        g_duration = (group['Timestamp'].max() - group['Timestamp'].min()).total_seconds() / 60.0
        if g_duration >= 10.0:
            stable_start_idx = group.index[0]
            break
            
    pct_05 = 0.0
    pct_10 = 0.0
    pct_15 = 0.0
    min_05 = 0.0
    min_10 = 0.0
    min_15 = 0.0
    
    if stable_start_idx is not None and total_duration > 0:
        post_transition_df = local_df.loc[stable_start_idx:].copy()
        phase_duration = (post_transition_df['Timestamp'].max() - post_transition_df['Timestamp'].min()).total_seconds() / 60.0
        total_points = len(post_transition_df)
        
        if total_points > 0:
            count_05 = (post_transition_df['sp_error'] <= 0.5).sum()
            count_10 = (post_transition_df['sp_error'] <= 1.0).sum()
            count_15 = (post_transition_df['sp_error'] <= 1.5).sum()
            
            min_05 = (count_05 / total_points) * phase_duration
            min_10 = (count_10 / total_points) * phase_duration
            min_15 = (count_15 / total_points) * phase_duration
            
            pct_05 = (min_05 / total_duration) * 100
            pct_10 = (min_10 / total_duration) * 100
            pct_15 = (min_15 / total_duration) * 100
            
    return {
        "pct_05": pct_05,
        "pct_10": pct_10,
        "pct_15": pct_15,
        "min_05": min_05,
        "min_10": min_10,
        "min_15": min_15
    }

def calculate_savings_potential(batch_df):
    """
    Calculates 'Waste' steam.
    """
    local_df = batch_df.sort_values('Timestamp').reset_index(drop=True)
    target_sp = local_df['Process Temp SP'].max()
    
    # 1. Determine Baseline Flow
    stable_mask = (local_df['Process Temp'] - target_sp).abs() <= 1.0
    
    if stable_mask.sum() > 5:
        baseline_flow = local_df.loc[stable_mask, 'Steam Flow Rate'].mean()
    else:
        baseline_flow = local_df['Steam Flow Rate'].median()
        
    if pd.isna(baseline_flow): baseline_flow = 0
    
    # 2. Identify Production Phase
    reached_sp = local_df['Process Temp'] >= target_sp
    if not reached_sp.any(): return 0.0
    
    start_idx = reached_sp.idxmax()
    prod_df = local_df.loc[start_idx:].copy()
    
    if prod_df.empty: return 0.0

    # 3. Calculate Waste
    prod_df['dt_hours'] = prod_df['Timestamp'].diff().dt.total_seconds() / 3600.0
    prod_df['dt_hours'] = prod_df['dt_hours'].fillna(0)
    
    prod_df['error'] = (prod_df['Process Temp'] - target_sp).abs()
    
    problem_mask = prod_df['error'] > 1.5
    
    waste_kg = 0.0
    
    if problem_mask.any():
        problem_df = prod_df[problem_mask].copy()
        excess_flow = (problem_df['Steam Flow Rate'] - baseline_flow).clip(lower=0)
        waste_kg = (excess_flow * problem_df['dt_hours']).sum()
        
    return waste_kg

@st.cache_data
def calculate_global_savings(df):
    """Calculates total potential savings across all batches in Tonnes"""
    total_savings_kg = 0.0
    if 'batch_id' in df.columns:
        unique_batches = df['batch_id'].unique()
        for bid in unique_batches:
            batch_data = df[df['batch_id'] == bid]
            total_savings_kg += calculate_savings_potential(batch_data)
    
    return total_savings_kg / 1000.0 # Convert to Tonnes

# --- 3. MAIN APPLICATION ---
def main():
    st.title("QualSteam Real Dairy Complete Batch Analysis")

    df = load_data()

    if df is None:
        st.error(f"Data file not found. Please ensure 'df_all_batches.csv' or 'df_clean.csv' is present.")
        st.stop()

    # Determine batch ID column if not standard
    if 'batch_id' not in df.columns:
        # Simple fallback if column name differs
        possible_cols = [c for c in df.columns if 'batch' in c.lower() or 'id' in c.lower()]
        if possible_cols:
            df['batch_id'] = df[possible_cols[0]]
        else:
            df['batch_id'] = 'Batch_All'

    # --- SIDEBAR ---
    st.sidebar.header("Dataset Overview")
    total_batches = df['batch_id'].nunique()
    
    # Calculate Global Savings
    global_savings_tonnes = calculate_global_savings(df)
    
    st.sidebar.info(f"""
    **Total Batches:** {total_batches}
    
    **Overall Potential Steam Savings:**
    # {global_savings_tonnes:.2f} Tonnes
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Batch Selection")
    
    unique_batches = sorted(df['batch_id'].unique())
    selected_batch_id = st.sidebar.selectbox("Select Batch ID", unique_batches)

    if selected_batch_id is None:
        st.stop()

    # Process specific batch data
    batch_data = df[df['batch_id'] == selected_batch_id].copy()
    batch_data = batch_data.sort_values('Timestamp')
    
    # Calculate KPIs
    metrics = calculate_ramp_up_metrics(batch_data)
    overshoot_steam = calculate_overshoot_steam(batch_data)
    stab_kpis = calculate_stability_kpis(batch_data)
    waste_kg = calculate_savings_potential(batch_data)
    
    # --- NEW: Overshoot Degree Calculation ---
    # Max Temp - Setpoint (Max SP usually)
    max_temp = batch_data['Process Temp'].max()
    max_sp = batch_data['Process Temp SP'].max()
    overshoot_degree = max(0.0, max_temp - max_sp)

    start_time = batch_data['Timestamp'].min()
    end_time = batch_data['Timestamp'].max()
    total_duration = (end_time - start_time).total_seconds() / 60.0
    date_str = start_time.strftime('%Y-%m-%d')

    # --- HEADER INFO ---
    st.markdown(f"### Batch Date: {date_str}")

    # --- ROW 1: General & Steam KPIs ---
    st.subheader("General KPIs")
    
    # 5 columns - Cost replaced by Overshoot Degree
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_duration:.1f} min</div><div class="metric-label">Total Duration</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{overshoot_steam:.1f} kg</div><div class="metric-label">Overshoot Steam Consumed</div></div>""", unsafe_allow_html=True)
    with c3:
        # NEW OVERSHOOT DEGREE CARD
        st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#C53030">{overshoot_degree:.2f} °C</div><div class="metric-label">Max Overshoot</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{metrics['total_steam_kg']:.1f} kg</div><div class="metric-label">Total Steam</div></div>""", unsafe_allow_html=True)
    with c5:
        # STEAM SAVINGS
        st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#D32F2F">{waste_kg:.1f} kg</div><div class="metric-label">Potential Steam Savings</div></div>""", unsafe_allow_html=True)

    # --- ROW 2: Stability Performance KPIs ---
    st.subheader("Stability Performance")
    s1, s2, s3 = st.columns(3)
    
    with s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stab_kpis['pct_05']:.1f}%</div>
            <div class="metric-subvalue">({stab_kpis['min_05']:.1f} mins)</div>
            <div class="metric-label">Time within ±0.5°C</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stab_kpis['pct_10']:.1f}%</div>
            <div class="metric-subvalue">({stab_kpis['min_10']:.1f} mins)</div>
            <div class="metric-label">Time within ±1.0°C</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stab_kpis['pct_15']:.1f}%</div>
            <div class="metric-subvalue">({stab_kpis['min_15']:.1f} mins)</div>
            <div class="metric-label">Time within ±1.5°C</div>
        </div>""", unsafe_allow_html=True)

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
        
        # Add a marker for stabilization
        if metrics['stable_ts'] is not None:
             fig.add_vline(x=metrics['stable_ts'], line_width=2, line_dash="dash", line_color="green", row="all")

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
        stat_card("Inlet Pressure P1 (Bar)", calculate_stats(batch_data['Inlet Steam Pressure']), c_p1) 
        stat_card("Steam Flow (kg/hr)", calculate_stats(batch_data['Steam Flow Rate']), c_flow)
        stat_card("Valve Opening (%)", calculate_stats(batch_data['QualSteam Valve Opening']), c_valve)

if __name__ == "__main__":
    main()
