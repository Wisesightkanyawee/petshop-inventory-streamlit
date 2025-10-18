# app.py
# Smart Reorder Tool — robust for (Sales: SKU, Quantity, Net sales, Cost of goods, Date)
# and (Inventory: SKU, In stock [I-animal], Cost)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =============== Page ===============
st.set_page_config(page_title="Smart Reorder Tool", layout="wide")
st.title("🧮 Smart Reorder Tool")

# keep run state
if "ran" not in st.session_state:
    st.session_state["ran"] = False

# =============== UI ===============
left, right = st.columns([1, 1])

with left:
    uploaded_sales = st.file_uploader('📤 Upload "Sales by item" file (.CSV)', type=["csv"])
    uploaded_stock = st.file_uploader('📤 Upload "Inventory" file (.CSV)', type=["csv"])
    stock_days   = st.number_input("📦 Stock Coverage Target (Day)", value=45, min_value=1)
    reorder_days = st.number_input("🔁 สั่งของอีกครั้งในอีกกี่วัน", value=7, min_value=1)
    st.caption("Inventory columns expected: **SKU, In stock [I-animal] (or similar), Cost**")

with right:
    st.markdown("### ℹ️ RU Score (Reorder Urgency)")
    st.markdown(
        "- คะแนนที่บอกความเร่งด่วนในการสั่งซื้อ หากสินค้าหมดสต็อก\n"
        "- ยิ่งสูง → เสียโอกาสทำกำไรต่อวันมาก ควรเติมเร็ว"
    )
    st.caption("Sales columns expected: **Date, SKU, Item(optional), Quantity, Net sales, Cost of goods, Category(optional), Receipt number(optional), Customer name(optional), Customer contacts(optional)**")

st.markdown("### ")
run_center = st.columns([2, 1, 2])[1]
with run_center:
    if st.button("▶️ Run Analysis", use_container_width=True):
        st.session_state["ran"] = True

# =============== Helpers ===============
def norm_sku(series: pd.Series) -> pd.Series:
    """Normalize SKU to a consistent string (strip, remove .0, force upper-case)."""
    s = series.astype(str)
    s = (
        s.str.replace("\u00A0", " ", regex=False)   # NBSP
         .str.replace("\u200b", "", regex=False)    # zero-width
         .str.strip()
         .str.replace(r"\.0+$", "", regex=True)
         .str.upper()
    )
    return s

def num_clean(series, fill=0.0):
    """แปลงสตริงให้เป็นตัวเลขอย่างทนทาน (รองรับคอมม่า, วงเล็บบัญชี, unicode minus, scientific notation)"""
    s = pd.Series(series).astype(str)
    s = (
        s.str.replace("\u00A0", " ", regex=False)
         .str.replace("\u200b", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.replace("−", "-", regex=False)
         .str.replace(r"\((.*)\)", r"-\1", regex=True)
         .str.strip()
    )
    s = s.str.extract(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def try_load_master():
    """Optional: Master SKU with Category. Safe to skip."""
    try:
        m = pd.read_csv("Master_SKU_Petshop.csv")
        m.columns = m.columns.str.strip()
        if "SKU" in m.columns:
            m["SKU"] = norm_sku(m["SKU"])
        return m
    except Exception:
        return None

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ล้างอักขระแปลก ๆ ในชื่อคอลัมน์ เช่น zero-width, BOM, NBSP"""
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[\u200B\uFEFF\u00A0]", "", regex=True)  # zero-width, BOM, NBSP
        .str.strip()
    )
    return df

def make_timegrain(df: pd.DataFrame, freq_key: str) -> pd.DataFrame:
    """Add a time grain column based on freq_key in {'Daily','Weekly','Monthly'}."""
    df = df.copy()
    if freq_key == "Daily":
        df["Timegrain"] = df["Date"].dt.to_period("D").dt.to_timestamp()
    elif freq_key == "Weekly":
        # ISO week start Monday; to_timestamp gives period start
        df["Timegrain"] = df["Date"].dt.to_period("W-MON").dt.to_timestamp()
    elif freq_key == "Monthly":
        df["Timegrain"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    else:
        df["Timegrain"] = df["Date"].dt.to_period("D").dt.to_timestamp()
    return df

# <<< วางไว้ใกล้ ๆ helpers อื่น ๆ >>>
def fmt_commas(df: pd.DataFrame, int_cols=(), float_cols=()):
    """Return a Styler with thousands separators and consistent 2-decimal formatting."""
    # Ensure we have a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
    elif not isinstance(df, pd.DataFrame):
        return df
    
    fmt_map = {c: "{:,.0f}" for c in int_cols}
    # Force all float columns to use 2 decimal places
    fmt_map.update({c: "{:,.2f}" for c in float_cols})
    # Apply 2-decimal formatting to any remaining numeric columns not specified
    for col in df.columns:
        if col not in fmt_map and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                fmt_map[col] = "{:,.0f}"
            else:
                fmt_map[col] = "{:,.2f}"
    
    try:
        result = df.style.format(fmt_map)
        return result
    except Exception as e:
        st.error(f"❌ Error in fmt_commas: {str(e)}")
        return df

# ✅ NEW: Streamlit-native renderer with commas using column_config
def show_df_commas(
    df: pd.DataFrame,
    float_cols: tuple | list = (),
    int_cols: tuple | list = (),
    percent_cols: tuple | list = (),
    hide_index: bool = False,
    use_container_width: bool = True,
):
    """Render a dataframe with thousands separators using Streamlit column_config.

    - float_cols: shown as ",.2f"
    - int_cols:   shown as ",.0f"
    - percent_cols: shown as "+,.2f%" (keeps numeric type for sorting)
    """
    # Ensure we have a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
    elif not isinstance(df, pd.DataFrame):
        st.error("❌ show_df_commas requires DataFrame or Series input")
        return
    
    if df.empty:
        st.info("ไม่มีข้อมูลแสดง")
        return
    
    # Always use Styler fallback with consistent 2-decimal formatting
    # Auto-detect numeric columns if not specified
    all_numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Determine which columns are which type
    final_int_cols = list(int_cols)
    final_float_cols = list(float_cols)
    final_percent_cols = list(percent_cols)
    
    # Auto-assign unspecified numeric columns to float (2 decimals)
    for col in all_numeric_cols:
        if col not in final_int_cols and col not in final_float_cols and col not in final_percent_cols:
            if df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                final_int_cols.append(col)
            else:
                final_float_cols.append(col)
    
    try:
        sty = fmt_commas(df, int_cols=final_int_cols, float_cols=final_float_cols)
        
        if final_percent_cols:
            sty = sty.format({c: "{:+,.2f}%" for c in final_percent_cols if c in df.columns})
        
        st.dataframe(sty, use_container_width=use_container_width, hide_index=hide_index)
        
    except Exception as e:
        st.error(f"❌ Error formatting dataframe: {str(e)}")
        # Fallback to simple dataframe
        st.dataframe(df, use_container_width=use_container_width, hide_index=hide_index)

# ✅ NEW: Styler for diverging percent tables (e.g., Change_%)
def style_diverging_percent(df: pd.DataFrame):
    """Style percent tables with a red-yellow-green gradient centered at 0.

    มี fallback กรณี deploy แล้วไม่ได้ติดตั้ง matplotlib (pandas Styler.background_gradient ต้องใช้ matplotlib)
    หากไม่มี matplotlib จะใช้การคำนวณสีด้วยตนเอง (inline CSS) แทน
    """
    arr = df.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    absmax = np.nanmax(np.abs(arr)) if arr.size else None
    if absmax and absmax > 0:
        vmin, vmax = -absmax, absmax
    else:
        vmin = vmax = None
    try:
        import matplotlib  # noqa: F401
        return (
            df.style
              .format("{:+,.2f}%")
              .background_gradient(cmap="RdYlGn", vmin=vmin, vmax=vmax, axis=None)
        )
    except Exception:
        # Manual fallback: map value -> color (red -> yellow -> green)
        def _color(val):
            if pd.isna(val) or absmax in (None, 0):
                return ""  # no style
            # normalize to [-1,1]
            norm = max(-1, min(1, val / absmax))
            # interpolate: negative -> red (rgb(220,60,50)), zero -> yellow (255, 220, 60), positive -> green (60,160,60)
            if norm >= 0:
                # yellow -> green
                r1,g1,b1 = 255,220,60
                r2,g2,b2 = 60,160,60
                t = norm
            else:
                # red -> yellow
                r1,g1,b1 = 220,60,50
                r2,g2,b2 = 255,220,60
                t = norm + 1  # map [-1,0] -> [0,1]
            r = int(r1 + (r2 - r1)*t)
            g = int(g1 + (g2 - g1)*t)
            b = int(b1 + (b2 - b1)*t)
            return f"background-color: rgb({r},{g},{b});"

        styled = df.copy()
        styler = styled.style.format("{:+,.2f}%").applymap(_color)
        return styler

# ✅ NEW: ฟังก์ชันสร้างตาราง MoM
def build_mom_table(df, group_col, value_col):
    agg = (
        df.groupby([df["Date"].dt.to_period("M"), group_col])[value_col]
          .sum()
          .reset_index()
    )
    agg.rename(columns={value_col: "Value"}, inplace=True)
    agg["Date"] = agg["Date"].astype(str)
    agg["Change_%"] = agg.groupby(group_col)["Value"].pct_change() * 100
    return agg

# =============== Customer Analysis Functions ===============

@st.cache_data
def normalize_columns(df):
    """Normalize column names and handle common aliases."""
    df = df.copy()
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Create lowercase mapping for case-insensitive matching
    current_cols = df.columns.tolist()
    lower_to_original = {col.lower(): col for col in current_cols}
    
    # Map common aliases (case-insensitive)
    column_map = {
        'net sales': 'Net sales',
        'gross sales': 'Net sales', 
        'customer name': 'Customer name',
        'customer contacts': 'Customer contacts',
        'receipt number': 'Receipt number',
        'receipt_number': 'Receipt number',
        'item': 'Item',
        'sku': 'SKU',
        'category': 'Category', 
        'brand': 'Brand',
        'quantity': 'Quantity',
        'date': 'Date'
    }
    
    # Apply mapping with case-insensitive matching
    rename_dict = {}
    for target_lower, standard_name in column_map.items():
        if target_lower in lower_to_original:
            original_name = lower_to_original[target_lower]
            if original_name != standard_name:  # Only rename if different
                rename_dict[original_name] = standard_name
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    return df

def build_customer_id(df):
    """Create customer_id from Customer name + Customer contacts."""
    df = df.copy()
    
    # Handle missing columns
    if 'Customer name' not in df.columns:
        df['Customer name'] = 'Unknown'
    if 'Customer contacts' not in df.columns:
        df['Customer contacts'] = 'Unknown'
    
    # Clean and convert to string
    df['Customer name'] = df['Customer name'].fillna('Unknown').astype(str).str.strip()
    df['Customer contacts'] = df['Customer contacts'].fillna('Unknown').astype(str).str.strip()
    
    # Replace empty strings with Unknown
    df['Customer name'] = df['Customer name'].replace('', 'Unknown')
    df['Customer contacts'] = df['Customer contacts'].replace('', 'Unknown')
    
    # Build customer_id
    df['customer_id'] = df['Customer name'] + " | " + df['Customer contacts']
    
    return df

def add_time_columns(df):
    """Add time-based columns for analysis."""
    df = df.copy()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        df['Weekday'] = df['Date'].dt.day_name()
    else:
        st.warning("⚠️ ไม่พบคอลัมน์ Date สำหรับการวิเคราะห์เวลา")
        
    return df

def compute_top_customers(df_filtered):
    """Compute top customers metrics."""
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Check required columns - prioritize customer_key over customer_id
    customer_col = None
    if 'customer_key' in df_filtered.columns:
        customer_col = 'customer_key'
    elif 'customer_id' in df_filtered.columns:
        customer_col = 'customer_id'
    else:
        return pd.DataFrame()
        
    if 'Net sales' not in df_filtered.columns:
        return pd.DataFrame()
    
    try:
        # 🔧 FORCE Flatten ALL object columns (ULTIMATE FIX)
        df_work = df_filtered.copy()
        
        for col in df_work.columns:
            try:
                # Force check for nested objects in ANY object-type column
                if df_work[col].dtype == 'object':
                    # Force flatten by converting any complex objects to their first value
                    def force_flatten(x):
                        if hasattr(x, 'iloc') and hasattr(x, '__len__') and len(x) > 0:
                            return x.iloc[0]
                        elif hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                            try:
                                return next(iter(x))
                            except:
                                return x
                        return x
                    
                    df_work[col] = df_work[col].apply(force_flatten)
                    
            except Exception as e:
                continue
        
        customer_stats = df_work.groupby(customer_col).agg({
            'Net sales': 'sum',
            'Receipt number': 'nunique' if 'Receipt number' in df_work.columns else 'count',
            'Date': ['min', 'max'] if 'Date' in df_work.columns else 'count'
        }).round(2)
        
        # Flatten MultiIndex columns properly
        if isinstance(customer_stats.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            customer_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in customer_stats.columns.values]
        
        # Rename columns to expected names
        column_mapping = {}
        for col in customer_stats.columns:
            if 'Net sales' in col:
                column_mapping[col] = 'total_net_sales'
            elif 'Receipt number' in col:
                column_mapping[col] = 'num_receipts'
            elif 'Date' in col and 'min' in col:
                column_mapping[col] = 'first_date'
            elif 'Date' in col and 'max' in col:
                column_mapping[col] = 'last_date'
                
        customer_stats = customer_stats.rename(columns=column_mapping)
        
        # Calculate derived metrics
        if 'total_net_sales' in customer_stats.columns and 'num_receipts' in customer_stats.columns:
            customer_stats['avg_net_per_receipt'] = (
                customer_stats['total_net_sales'] / customer_stats['num_receipts'].replace(0, 1)
            ).round(2)
        
        if 'first_date' in customer_stats.columns and 'last_date' in customer_stats.columns:
            customer_stats['days_active'] = (
                customer_stats['last_date'] - customer_stats['first_date']
            ).dt.days
        
        # Sort and reset index
        if 'total_net_sales' in customer_stats.columns:
            customer_stats = customer_stats.sort_values('total_net_sales', ascending=False)
        customer_stats = customer_stats.reset_index()
        
        return customer_stats
        
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถคำนวณ top customers ได้: {str(e)}")
        return pd.DataFrame()

def compute_rfm(df_filtered, current_max_date=None):
    """Compute RFM analysis."""
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Check for required columns with flexible column names - prioritize customer_key
    customer_col = None
    if 'customer_key' in df_filtered.columns:
        customer_col = 'customer_key'
    elif 'customer_id' in df_filtered.columns:
        customer_col = 'customer_id'
    else:
        return pd.DataFrame()
    
    # Find sales column
    sales_col = None
    for col in ['Net sales', 'ราคารวม (ไม่รวม VAT)', 'Gross sales']:
        if col in df_filtered.columns:
            sales_col = col
            break
    
    if sales_col is None:
        return pd.DataFrame()
    
    try:
        # 🔧 Flatten nested Series in columns (CRITICAL FIX)
        df_work = df_filtered.copy()
        
        for col in df_work.columns:
            try:
                # Force check for nested objects in ANY object-type column
                if df_work[col].dtype == 'object':
                    # Force flatten by converting any complex objects to their first value
                    def force_flatten(x):
                        if hasattr(x, 'iloc') and hasattr(x, '__len__') and len(x) > 0:
                            return x.iloc[0]
                        elif hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                            try:
                                return next(iter(x))
                            except:
                                return x
                        return x
                    
                    df_work[col] = df_work[col].apply(force_flatten)
                    
            except Exception as e:
                continue
        
        if current_max_date is None and 'Date' in df_work.columns:
            current_max_date = df_work['Date'].max()
        elif 'Date' not in df_work.columns:
            return pd.DataFrame()
        
        # Build aggregation dictionary
        agg_dict = {
            'Date': 'max',
            sales_col: 'sum'
        }
        
        # Add receipt number if available
        receipt_col = None
        for col in ['Receipt number', 'receipt_number', 'Receipt_number']:
            if col in df_work.columns:
                receipt_col = col
                break
        
        if receipt_col:
            agg_dict[receipt_col] = 'nunique'
        else:
            # Fallback: count rows as frequency
            agg_dict[customer_col] = 'count'  # This will be renamed to frequency
        
        rfm = df_work.groupby(customer_col).agg(agg_dict).round(2)
        
        # Handle MultiIndex columns if they exist
        if isinstance(rfm.columns, pd.MultiIndex):
            rfm.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in rfm.columns]
        
        # Calculate Recency (days since last purchase)
        if 'Date' in rfm.columns and current_max_date is not None:
            rfm['Recency'] = (current_max_date - rfm['Date']).dt.days
        else:
            rfm['Recency'] = 0
            
        # Set Frequency 
        if receipt_col and receipt_col in rfm.columns:
            rfm['Frequency'] = rfm[receipt_col]
        elif f'{receipt_col}_nunique' in rfm.columns:
            rfm['Frequency'] = rfm[f'{receipt_col}_nunique']
        elif customer_col in rfm.columns:
            rfm['Frequency'] = rfm[customer_col]  # From count aggregation
        else:
            # Fallback: count transactions per customer
            freq_data = df_work.groupby(customer_col).size()
            rfm['Frequency'] = freq_data.reindex(rfm.index, fill_value=1)
        
        # Set Monetary
        if sales_col in rfm.columns:
            rfm['Monetary'] = rfm[sales_col]
        elif f'{sales_col}_sum' in rfm.columns:
            rfm['Monetary'] = rfm[f'{sales_col}_sum']
        else:
            rfm['Monetary'] = 0
        
        # Create quintile scores (1-5, where 5 is best) with error handling
        try:
            rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        except ValueError:
            # If not enough unique values, assign default scores
            rfm['R_score'] = 3
            
        try:
            rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            rfm['F_score'] = 3
            
        try:
            rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            rfm['M_score'] = 3
        
        # Convert to numeric
        rfm['R_score'] = pd.to_numeric(rfm['R_score'], errors='coerce').fillna(3).astype(int)
        rfm['F_score'] = pd.to_numeric(rfm['F_score'], errors='coerce').fillna(3).astype(int)
        rfm['M_score'] = pd.to_numeric(rfm['M_score'], errors='coerce').fillna(3).astype(int)
        
        # Create RFM segment
        rfm['RFM_segment'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
        
        # Create customer tags
        def categorize_customer(row):
            if row['R_score'] >= 4 and row['F_score'] >= 4 and row['M_score'] >= 4:
                return 'VIP'
            elif row['F_score'] >= 3 and row['M_score'] >= 3:
                return 'Regular'
            elif row['R_score'] >= 4 and (row['F_score'] < 3 or row['M_score'] < 3):
                return 'High-Potential'
            elif row['R_score'] <= 2:
                return 'At-Risk'
            else:
                return 'Others'
        
        rfm['Customer_Tag'] = rfm.apply(categorize_customer, axis=1)
        
        # Reset index to get customer_key as a column
        rfm = rfm.reset_index()
        
        return rfm
        
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถคำนวณ RFM ได้: {str(e)}")
        return pd.DataFrame()
        rfm = rfm.reset_index()
        
        return rfm
        
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถคำนวณ RFM ได้: {str(e)}")
        return pd.DataFrame()

def compute_retention(df_filtered):
    """Compute retention analysis by month."""
    if df_filtered.empty or 'Month' not in df_filtered.columns:
        return pd.DataFrame()
    
    # Determine customer column to use
    customer_col = None
    if 'customer_key' in df_filtered.columns:
        customer_col = 'customer_key'
    elif 'customer_id' in df_filtered.columns:
        customer_col = 'customer_id'
    else:
        return pd.DataFrame()
    
    # Get customers by month
    customers_by_month = df_filtered.groupby('Month')[customer_col].apply(set).sort_index()
    
    retention_data = []
    months = sorted(customers_by_month.index)
    
    for i, month in enumerate(months):
        current_customers = customers_by_month[month]
        
        if i == 0:
            # First month - all are "new"
            new_customers = current_customers
            retained_customers = set()
            lost_customers = set()
        else:
            prev_customers = customers_by_month[months[i-1]]
            new_customers = current_customers - prev_customers
            retained_customers = current_customers & prev_customers
            lost_customers = prev_customers - current_customers
        
        retention_data.append({
            'Month': month,
            'New': len(new_customers),
            'Retained': len(retained_customers), 
            'Lost': len(lost_customers),
            'Total_Current': len(current_customers)
        })
    
    return pd.DataFrame(retention_data)

def get_lost_customers_detail(df_filtered, month_selected):
    """Get detailed info for lost customers in selected month."""
    if df_filtered.empty or 'Month' not in df_filtered.columns:
        return pd.DataFrame()
    
    # Determine customer column to use
    customer_col = None
    if 'customer_key' in df_filtered.columns:
        customer_col = 'customer_key'
    elif 'customer_id' in df_filtered.columns:
        customer_col = 'customer_id'
    else:
        return pd.DataFrame()
    
    months = sorted(df_filtered['Month'].unique())
    if month_selected not in months:
        return pd.DataFrame()
    
    month_idx = months.index(month_selected)
    if month_idx == 0:
        return pd.DataFrame()  # No previous month to compare
    
    prev_month = months[month_idx - 1]
    
    # Get customers from each month
    current_customers = set(df_filtered[df_filtered['Month'] == month_selected][customer_col])
    prev_customers = set(df_filtered[df_filtered['Month'] == prev_month][customer_col])
    
    lost_customers = prev_customers - current_customers
    
    if not lost_customers:
        return pd.DataFrame()
    
    # Get details for lost customers
    lost_detail_data = df_filtered[
        (df_filtered['Month'] == prev_month) & 
        (df_filtered[customer_col].isin(lost_customers))
    ]
    
    # Check if we have data to aggregate
    if lost_detail_data.empty:
        return pd.DataFrame()
    
    # Aggregate the data
    agg_dict = {}
    
    # Always try Net sales first, fallback to other sales columns
    if 'Net sales' in lost_detail_data.columns:
        agg_dict['Net sales'] = 'sum'
    elif 'ราคารวม (ไม่รวม VAT)' in lost_detail_data.columns:
        agg_dict['ราคารวม (ไม่รวม VAT)'] = 'sum'
    elif 'Gross sales' in lost_detail_data.columns:
        agg_dict['Gross sales'] = 'sum'
    else:
        # If no sales column found, create a dummy value
        lost_detail_data['Sales_Value'] = 0
        agg_dict['Sales_Value'] = 'sum'
    
    # Add category and item aggregations if columns exist
    if 'Category' in lost_detail_data.columns:
        agg_dict['Category'] = 'first'  # Take first category for simplicity
    
    if 'Item' in lost_detail_data.columns:
        agg_dict['Item'] = 'first'  # Take first item for simplicity
        
    if 'Date' in lost_detail_data.columns:
        agg_dict['Date'] = 'max'
    
    # Safety filter: ensure agg_dict only contains valid columns
    agg_dict = {k: v for k, v in agg_dict.items() if k in lost_detail_data.columns}
    
    # Safety filter: remove columns with nested DataFrame objects
    for c in lost_detail_data.columns:
        if any(isinstance(v, pd.DataFrame) for v in lost_detail_data[c].dropna()):
            lost_detail_data = lost_detail_data.drop(columns=[c])
            # Remove from agg_dict if it was there
            if c in agg_dict:
                del agg_dict[c]
    
    for col in lost_detail_data.columns:
        try:
            # Force check for nested objects in ANY object-type column
            if lost_detail_data[col].dtype == 'object':
                # Force flatten by converting any complex objects to their first value
                def force_flatten(x):
                    if hasattr(x, 'iloc') and hasattr(x, '__len__') and len(x) > 0:
                        return x.iloc[0]
                    elif hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
                        try:
                            return next(iter(x))
                        except:
                            return x
                    return x
                
                lost_detail_data[col] = lost_detail_data[col].apply(force_flatten)
                
        except Exception as e:
            continue
    
    # Perform safe groupby with explicit column selection
    try:
        lost_detail = (
            lost_detail_data.groupby(customer_col)[list(agg_dict.keys())]
            .agg(agg_dict)
            .round(2)
        )
    except Exception as e:
        st.error(f"❌ Error in groupby operation: {str(e)}")
        return pd.DataFrame()
    
    # Rename columns based on what we actually aggregated
    new_column_names = []
    for col in lost_detail.columns:
        if col in ['Net sales', 'ราคารวม (ไม่รวม VAT)', 'Gross sales', 'Sales_Value']:
            new_column_names.append('Last_Purchase_Value')
        elif col == 'Category':
            new_column_names.append('Categories')
        elif col == 'Item':
            new_column_names.append('Top_Items')
        elif col == 'Date':
            new_column_names.append('Last_Date')
        else:
            new_column_names.append(col)
    
    lost_detail.columns = new_column_names
    lost_detail = lost_detail.sort_values('Last_Purchase_Value', ascending=False)
    lost_detail = lost_detail.reset_index()
    
    return lost_detail

def build_category_brand_mix(df_filtered, top_k=8):
    """Build category/brand mix by customer."""
    if df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    mix_col = None
    if 'Category' in df_filtered.columns:
        mix_col = 'Category'
    elif 'Brand' in df_filtered.columns:
        mix_col = 'Brand'
    else:
        return pd.DataFrame(), pd.DataFrame()
    
    # Check for customer column - prioritize customer_key over customer_id
    if 'customer_key' in df_filtered.columns:
        customer_col = 'customer_key'
    elif 'customer_id' in df_filtered.columns:
        customer_col = 'customer_id'
    else:
        return pd.DataFrame(), pd.DataFrame()
    
    # Create pivot table
    try:
        mix_pivot = df_filtered.pivot_table(
            index=customer_col,
            columns=mix_col,
            values='Net sales',
            aggfunc='sum',
            fill_value=0
        )
        
        if mix_pivot.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Keep only top K categories/brands by total sales
        col_totals = mix_pivot.sum().sort_values(ascending=False)
        top_cols = col_totals.head(top_k).index
        mix_pivot = mix_pivot[top_cols]
        
        # Add percentage columns
        row_sums = mix_pivot.sum(axis=1)
        # Avoid division by zero
        mask_nonzero = row_sums != 0
        mix_pivot_pct = mix_pivot.copy()
        mix_pivot_pct.loc[mask_nonzero] = (
            mix_pivot.loc[mask_nonzero].div(row_sums.loc[mask_nonzero], axis=0) * 100
        ).round(1)
        
        return mix_pivot, mix_pivot_pct
        
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถสร้าง category/brand mix ได้: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def compute_interpurchase(df_filtered):
    """Compute interpurchase time analysis."""
    if df_filtered.empty or 'Date' not in df_filtered.columns:
        return pd.DataFrame()
    
    # Check for customer column - prioritize customer_key over customer_id
    if 'customer_key' in df_filtered.columns:
        customer_col = 'customer_key'
    elif 'customer_id' in df_filtered.columns:
        customer_col = 'customer_id'
    else:
        return pd.DataFrame()
    
    interpurchase_data = []
    
    for customer in df_filtered[customer_col].unique():
        customer_data = df_filtered[df_filtered[customer_col] == customer].copy()
        customer_dates = customer_data['Date'].drop_duplicates().sort_values()
        
        if len(customer_dates) >= 2:
            date_diffs = customer_dates.diff().dt.days.dropna()
            if len(date_diffs) > 0:
                interpurchase_data.append({
                    customer_col: customer,
                    'num_purchases': len(customer_dates),
                    'avg_days_between': date_diffs.mean(),
                    'median_days_between': date_diffs.median(),
                    'min_days_between': date_diffs.min(),
                    'max_days_between': date_diffs.max()
                })
    
    return pd.DataFrame(interpurchase_data)


# =============== Main ===============
if st.session_state.get("ran") and uploaded_sales and uploaded_stock:
    try:
        # ----- Load CSVs -----
        sales = clean_columns(pd.read_csv(uploaded_sales))
        stock = clean_columns(pd.read_csv(uploaded_stock))
        sales.columns = sales.columns.str.strip()
        stock.columns = stock.columns.str.strip()

        # ----- Harmonize SALES columns -----
        if "Net sales" not in sales.columns and "Gross sales" in sales.columns:
            sales = sales.rename(columns={"Gross sales": "Net sales"})
        if "Quantity" not in sales.columns and ("Items sold" in sales.columns or "Items refunded" in sales.columns):
            q = num_clean(sales.get("Items sold", 0))
            r = num_clean(sales.get("Items refunded", 0))
            sales["Quantity"] = q - r

        # Validate required columns (Sales)
        req_sales = {"SKU", "Quantity", "Net sales", "Cost of goods", "Date"}
        missing_sales = req_sales - set(sales.columns)
        if missing_sales:
            st.error("❌ Sales file missing columns: " + ", ".join(sorted(missing_sales)))
            st.stop()

        # ----- Harmonize INVENTORY columns -----
        # Clean and normalize inventory column names first
        stock.columns = stock.columns.str.strip()  # Remove extra spaces
        
        # Create mapping for flexible column matching
        stock_column_map = {}
        
        # Find inventory stock column (flexible matching)
        for col in stock.columns:
            col_lower = col.lower().strip()
            if 'in stock' in col_lower and 'i-animal' in col_lower:
                stock_column_map[col] = "คงเหลือ"
                break
        
        # Find cost column
        for col in stock.columns:
            col_lower = col.lower().strip()
            if col_lower == 'cost':
                stock_column_map[col] = "ต้นทุนเฉลี่ย/ชิ้น"
                break
        
        # Check if we found the required columns
        if "คงเหลือ" not in stock_column_map.values():
            st.error("❌ Inventory file missing 'In stock [I-animal]' or similar column")
            st.error(f"Available columns: {list(stock.columns)}")
            st.stop()
        
        if "ต้นทุนเฉลี่ย/ชิ้น" not in stock_column_map.values():
            st.error("❌ Inventory file missing 'Cost' column")
            st.error(f"Available columns: {list(stock.columns)}")
            st.stop()
        
        # Apply the mapping
        stock = stock.rename(columns=stock_column_map)
        
        # ----- Normalize keys/types -----
        sales["SKU"] = norm_sku(sales["SKU"])
        stock["SKU"] = norm_sku(stock["SKU"])
        stock["คงเหลือ"] = num_clean(stock["คงเหลือ"], 0)
        stock["ต้นทุนเฉลี่ย/ชิ้น"] = num_clean(stock["ต้นทุนเฉลี่ย/ชิ้น"], 0)
        stock["คงเหลือ"] = num_clean(stock["คงเหลือ"], 0)
        stock["ต้นทุนเฉลี่ย/ชิ้น"] = num_clean(stock["ต้นทุนเฉลี่ย/ชิ้น"], 0)

        # ----- Date filter -----
        sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
        sales = sales.dropna(subset=["Date"])
        if sales.empty:
            st.error("❌ Sales file has no valid dates.")
            st.stop()

        min_day = sales["Date"].min().date()
        max_day = sales["Date"].max().date()

        st.subheader("📅 เลือกช่วงวันที่สำหรับการวิเคราะห์")
        c1, c2 = st.columns(2)
        with c1:
            start_day = st.date_input("ตั้งแต่วันที่", value=min_day, min_value=min_day, max_value=max_day)
        with c2:
            end_day   = st.date_input("ถึงวันที่", value=max_day, min_value=min_day, max_value=max_day)
        if start_day > end_day:
            st.error("❌ วันที่เริ่มต้องไม่เกินวันที่สิ้นสุด")
            st.stop()

        # restrict range
        mask_range = (sales["Date"].dt.date >= start_day) & (sales["Date"].dt.date <= end_day)
        sales = sales.loc[mask_range].copy()
        days_of_sales = (pd.to_datetime(end_day) - pd.to_datetime(start_day)).days + 1

        # --- Parse numbers ---
        sales["Net sales"]     = num_clean(sales["Net sales"], 0)
        sales["Cost of goods"] = num_clean(sales["Cost of goods"], 0)
        sales["Quantity"]      = pd.to_numeric(sales["Quantity"], errors="coerce").fillna(0.0)
        sales["Gross profit"]  = sales["Net sales"] - sales["Cost of goods"]

        # category fallback
        if "Category" not in sales.columns:
            sales["Category"] = np.nan
        sales["Category_disp"] = sales["Category"].fillna("Uncategorized").astype(str)

        # ====== Create Brand & Cate_and_band ======

        def extract_brand(item: str) -> str:
            if not isinstance(item, str):
                return ''
            text = item.strip()
            if not text:
                return ''
            tokens = text.split()
            lt = [t.lower().strip('.,') for t in tokens]

            def join(n: int):
                return ' '.join(tokens[:n])

            # --- rules ทั้งหมด ---
            if len(lt) >= 4 and lt[0]=='solid' and lt[1]=='gold' and lt[2]=='indigo' and lt[3]=='moon':
                return 'solid gold indigo moon'
            if len(lt) >= 4 and lt[0]=='taste' and lt[1]=='of' and lt[2]=='the' and lt[3]=='wild':
                return 'taste of the wild'
            if lt[0]=='odour' and len(lt)>=2 and lt[1]=='lock':
                return 'odour lock multi-cat'
            if lt[0] in {'22','22pet'}:
                return '22pet'
            if lt[0] in {'boqifactory','boqi'}:
                return 'boqifactory'
            if len(lt)>=1 and (lt[0].replace('-', '') in {'meo','me0'}) or lt[0] in {'me-o','me-o,'}:
                return 'me-o'
            if lt[0]=='dog' and len(lt)>=3 and lt[1] in {'n','n,'} and lt[2]=='joy':
                return 'dog n joy'
            if lt[0]=='cat':
                if len(lt)>=3 and lt[1] in {'n','n,'} and lt[2]=='joy':
                    return 'cat n joy'
                if len(lt)>=2 and lt[1]=='taste':
                    return 'cat taste'
                if len(lt)>=2 and lt[1] in {'it','me'}:
                    return join(2)
                return tokens[0]
            if lt[0]=='bite':
                if len(lt)>=3 and lt[1]=='of':
                    if lt[2]=='wild':
                        return 'bite of wild'
                    if lt[2]=='the' and len(lt)>=4 and lt[3]=='wild':
                        return 'bite of the wild'
                return tokens[0]
            if lt[0]=='cheer' and len(lt)>=2 and lt[1]=='share':
                return 'cheer share'
            if lt[0]=='kat':
                if len(lt)>=2 and lt[1] in {'club','to'}:
                    return join(2)
                return tokens[0]
            if tokens[0].lower().rstrip('.') == 'mr':
                if len(lt)>=2 and lt[1]=='vet':
                    return 'mr vet'
                return 'mr'
            if lt[0]=='dream' and len(lt)>=2 and lt[1]=='litty':
                return 'dream litty'
            if lt[0]=='vif' and len(lt)>=2 and lt[1]=='clair':
                return 'vif clair'
            if lt[0]=='kitty':
                if len(lt)>=2 and lt[1] in {'licks','treats'}:
                    return 'kitty ' + tokens[1].lower()
                return 'kitty'
            if lt[0]=='ocean' and len(lt)>=2 and lt[1]=='star':
                return 'ocean star'
            if lt[0]=='catit' and len(lt)>=2 and lt[1]=='creamy':
                return 'catit creamy'
            if lt[0]=='dox' and len(lt)>=2 and lt[1]=='club':
                return 'dox club'
            if lt[0]=='lucky' and len(lt)>=2 and lt[1]=='dog':
                return 'lucky dog'
            if lt[0]=='bok' and len(lt)>=2 and lt[1] in {'bok','dok'}:
                return join(2)
            if lt[0]=='am' and len(lt)>=2 and lt[1]=='goat':
                return 'am goat'
            if lt[0]=='bully' and len(lt)>=2 and lt[1]=='stick':
                return 'bully stick'
            if lt[0]=='bux' and len(lt)>=2 and lt[1]=='away':
                return 'bux away'
            if lt[0]=='cotton' and len(lt)>=2 and lt[1]=='bud':
                return 'cotton bud'
            if lt[0]=='daili' and len(lt)>=2 and lt[1]=='pet':
                return 'daili pet'
            if lt[0]=='dental' and len(lt)>=2 and lt[1]=='bone':
                return 'dental bone'
            if lt[0]=='dogster' and len(lt)>=3 and lt[1]=='play' and lt[2]=='mix':
                return 'dogster play mix'
            if lt[0]=='goat' and len(lt)>=2 and lt[1]=='milk':
                return 'goat milk'
            if lt[0]=='kelly' and len(lt)>=2 and ("co" in lt[1]):
                return "kelly co's"
            if lt[0]=='lola' and len(lt)>=3 and lt[1]=='healthy' and lt[2]=='growth':
                return 'lola healthy growth'
            if lt[0]=='love':
                if len(lt)>=2 and lt[1] in {'me','cubes'}:
                    return 'love ' + tokens[1].lower()
                return 'love'
            if lt[0]=='loveme':
                return 'loveme'
            if lt[0]=='nano' and len(lt)>=2 and lt[1]=='care':
                return 'nano care'
            if lt[0]=='optimum' and len(lt)>=2 and lt[1]=='spirulina':
                return 'optimum spirulina'
            if lt[0]=='ostech' and len(lt)>=2 and lt[1]=='ultra':
                return 'ostech ultra'
            if lt[0].startswith("p'") and len(lt)>=2 and lt[1]=='sak':
                return "p' sak"
            if lt[0]=='ped' and len(lt)>=3 and lt[1]=='denta' and lt[2]=='stix':
                return 'ped denta stix'
            if lt[0]=='paws' and len(lt)>=2 and lt[1]=='feliz':
                return 'paws feliz'
            if lt[0]=='pet' and len(lt)>=2 and lt[1] in {'ranger','trainingpad'}:
                return 'pet ' + tokens[1]
            if lt[0]=='revo' and len(lt)>=2 and lt[1]=='plus':
                return 'revo plus'
            if lt[0]=='royal':
                if len(lt)>=2 and lt[1]=='topping':
                    return 'royal topping'
                if len(lt)>=3 and lt[1]=='herbal' and lt[2]=='spray':
                    return 'royal herbal spray'
                return 'royal'
            return tokens[0]

        def refine_brand(row):
            """Refine brand with case-insensitive logic and output all-lowercase.

            - Online selling platform names -> lineman / grab / tiktok
            - CAT Snack + me-o: split into me-o treat / me-o แมวเลีย
            Fallback: lowercase brand.
            """
            b = str(row.get('Brand', '')).strip()
            b_lower = b.lower()
            cat = str(row.get('Category', '')).strip()
            cat_upper = cat.upper()
            item = str(row.get('Item', '')).lower()

            # Online selling normalization (strip trailing codes already handled upstream by using item text)
            if cat.strip().lower() == 'online selling':
                if 'lineman' in item or item.startswith('line'):
                    return 'lineman'
                if item.startswith('grab'):
                    return 'grab'
                if 'tiktok' in item:
                    return 'tiktok'
                return b_lower

            # CAT Snack specializations for me-o
            if cat_upper == 'CAT SNACK' and b_lower in {'me-o', 'meo', 'me0'}:
                txt_low = item
                if 'treat' in txt_low:
                    return 'me-o treat'
                if 'แมวเลีย' in txt_low:
                    return 'me-o แมวเลีย'
                return 'me-o'

            return b_lower

        # apply
        if "Item" in sales.columns:
            # 1) extract base brand (case-insensitive rules use lower tokens) -> may return mixed case
            sales["Brand"] = sales["Item"].apply(extract_brand)
            # 2) refine & force lowercase outputs
            sales["Brand"] = sales.apply(refine_brand, axis=1)
            # 3) final brand all-lowercase (double safety)
            sales["Brand"] = sales["Brand"].str.lower()
            # 4) build Cate_and_band with lowercase brand
            sales["Cate_and_band"] = (
                sales["Category_disp"].astype(str).str.strip() + " [" + sales["Brand"].astype(str) + "]"
            )
            # 5) Ensure everything inside brackets is lowercase (already) and remove accidental double spaces
            sales["Cate_and_band"] = sales["Cate_and_band"].str.replace(r"\s+", " ", regex=True)
        else:
            sales["Cate_and_band"] = sales["Category_disp"].astype(str).str.lower()


        # KPI after filter
        cdbg1, cdbg2, cdbg3 = st.columns(3)
        cdbg1.metric("✅ รวม Quantity (ชิ้น)", f"{float(sales['Quantity'].sum()):,.0f}")
        cdbg2.metric("💰 รวมยอดขาย (บาท)", f"{float(sales['Net sales'].sum()):,.2f}")
        cdbg3.metric("💵 รวมกำไรขั้นต้น (บาท)", f"{float(sales['Gross profit'].sum()):,.2f}")

        # ====== Build merged for Inventory & Reorder (same as your original) ======
        # Avg cost per unit from sales for fallback
        qty_for_cost = sales["Quantity"].replace(0, np.nan)
        sales["Avg_cost_per_unit_from_sales"] = (sales["Cost of goods"] / qty_for_cost).fillna(0)

        grouped_sales = (
            sales.groupby("SKU", as_index=False)
            .agg(
                Quantity=("Quantity", "sum"),
                Net_sales=("Net sales", "sum"),
                Avg_cost_from_sales=("Avg_cost_per_unit_from_sales", "mean")
            )
        )

        # Keep latest item name (optional)
        if "Item" in sales.columns:
            name_map = (
                sales.sort_values("Date")
                     .drop_duplicates("SKU", keep="last")[["SKU", "Item"]]
                     .rename(columns={"Item": "Name"})
            )
            grouped_sales = grouped_sales.merge(name_map, on="SKU", how="left")

        # Merge with Inventory
        merged = stock.merge(grouped_sales, on="SKU", how="left")
        for col, default in [("Quantity", 0.0), ("Net_sales", 0.0), ("Avg_cost_from_sales", 0.0)]:
            if col not in merged.columns:
                merged[col] = default
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

        # fill cost from sales if stock cost missing
        mask_cost_fill = (merged["ต้นทุนเฉลี่ย/ชิ้น"].isna()) | (merged["ต้นทุนเฉลี่ย/ชิ้น"] == 0)
        merged.loc[mask_cost_fill, "ต้นทุนเฉลี่ย/ชิ้น"] = merged["Avg_cost_from_sales"]

        # per-day metrics
        merged["total_profit"]       = merged["Net_sales"] - (merged["Quantity"] * merged["ต้นทุนเฉลี่ย/ชิ้น"])
        merged["avg_profit_per_day"] = merged["total_profit"] / max(days_of_sales, 1)
        merged["avg_sales_per_day"]  = merged["Quantity"] / max(days_of_sales, 1)

        # unit profit
        avg_price_per_unit = merged["Net_sales"] / merged["Quantity"].replace(0, np.nan)
        merged["กำไรเฉลี่ย/ชิ้น"] = (avg_price_per_unit - merged["ต้นทุนเฉลี่ย/ชิ้น"]).fillna(0).round(2)

        # coverage & status & RU
        merged["Stock Coverage (Day)"] = merged.apply(
            lambda r: (r["คงเหลือ"] / r["avg_sales_per_day"]) if r["avg_sales_per_day"] > 0 else np.nan,
            axis=1
        )
        merged["Dead Stock"] = np.where(merged["Quantity"] == 0, "⚠️ ไม่เคลื่อนไหว", "")

        def _status(row):
            if row["คงเหลือ"] < 0:
                return "Stock ติดลบ", row["avg_profit_per_day"]
            if row["คงเหลือ"] == 0 and row["Quantity"] > 0:
                return "หมด!!!", row["avg_profit_per_day"]
            if row["คงเหลือ"] == 0 and row["Quantity"] == 0:
                return "ไม่มียอดขาย Stock = 0", 0
            if row["คงเหลือ"] > 0 and row["Quantity"] == 0:
                return "ขายไม่ได้เลยย T_T", 0
            cov = row["Stock Coverage (Day)"]
            score = row["avg_profit_per_day"] / cov if pd.notna(cov) and cov > 0 else 0
            return f"{cov:.1f} วัน", score

        merged[["สถานะ", "RU Score"]] = merged.apply(_status, axis=1, result_type="expand")
        merged = merged[merged["สถานะ"] != "ไม่มียอดขาย Stock = 0"].copy()

        # reorder qty
        merged["ควรสั่งซื้อเพิ่ม (ชิ้น)"] = (
            merged["avg_sales_per_day"] * stock_days - merged["คงเหลือ"]
        ).apply(lambda x: max(0, int(np.ceil(x))))

        # Opp. Loss
        merged["วันที่ไม่มีของขาย"] = merged["Stock Coverage (Day)"].apply(
            lambda x: max(0, int(np.ceil(reorder_days - x))) if pd.notna(x) else reorder_days
        )
        merged["Opp. Loss (Baht)"] = (merged["avg_profit_per_day"] * merged["วันที่ไม่มีของขาย"]).round(2)

        # attach Category
        master = try_load_master()
        if master is not None and {"SKU", "Category"}.issubset(master.columns):
            merged = merged.merge(master[["SKU", "Category"]], on="SKU", how="left")
        if "Category" not in merged.columns or merged["Category"].isna().all():
            if "Category" in sales.columns:
                cat_map = sales[["SKU", "Category"]].dropna().drop_duplicates("SKU")
                merged = merged.merge(cat_map, on="SKU", how="left")
            else:
                merged["Category"] = np.nan
        merged["Category_disp"] = merged["Category"].fillna("Uncategorized").astype(str)

        # item name fill
        if "Name" not in merged.columns or merged["Name"].isna().all():
            src_name = None
            for c in ["Item", "Name", "Item name"]:
                if c in sales.columns:
                    src_name = c
                    break
            if src_name:
                imap = (sales[["SKU", src_name]].rename(columns={src_name: "Name"})
                        .dropna(subset=["Name"]).drop_duplicates("SKU", keep="last"))
                merged = merged.merge(imap, on="SKU", how="left", suffixes=("", "_from_sales"))
                if "Name_from_sales" in merged.columns:
                    merged["Name"] = merged["Name"].fillna(merged["Name_from_sales"])
                    merged = merged.drop(columns=["Name_from_sales"])
        merged["Name"] = merged["Name"].fillna(merged["SKU"].astype(str))

        # =============== TABS: Inventory & Reorder | Sales Analysis | Drop Analysis | Customer Analysis | Promotion ===============
        tab_inv, tab_sales, tab_drop, tab_customer, tab_promotion = st.tabs([
            "📦 Inventory & Reorder",
            "📊 Sales Analysis", 
            "📉 การวิเคราะห์ยอดขายตก",
            "🎯 Customer Analysis",
            "🎁 Promotion"
        ])


        # -------------------- TAB 1: Inventory & Reorder (เหมือนเดิม + bubble) --------------------
        with tab_inv:
            st.subheader("📂 ฟิลเตอร์และสรุปภาพรวม (Inventory)")
            cats = merged["Category_disp"]
            all_cats = sorted(cats.unique())
            selected = st.multiselect("เลือก Category", options=all_cats, default=all_cats, key="inv_cats")
            filtered = merged[cats.isin(selected)].copy()

            if not filtered.empty:
                summary = (
                    filtered.groupby(filtered["Category_disp"])
                    .agg(
                        Total_RU_Score=("RU Score", "sum"),
                        Total_Opp_Loss_Baht=("Opp. Loss (Baht)", "sum"),
                        Total_Qty=("Quantity", "sum")
                    )
                    .reset_index()
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("RU Score รวม", f"{summary['Total_RU_Score'].sum():,.2f}")
                c2.metric("ค่าความเสียโอกาสรวม (บาท)", f"{summary['Total_Opp_Loss_Baht'].sum():,.2f}")
                c3.metric("จำนวนขายรวม (ชิ้น)", f"{summary['Total_Qty'].sum():,.0f}")

                st.dataframe(
                    fmt_commas(
                        summary,
                        int_cols=["Total_Qty"],
                        float_cols=["Total_RU_Score", "Total_Opp_Loss_Baht"],
                    ),
                    use_container_width=True,
                )


                # Bubble chart
                st.markdown("#### 🔵 Bubble: Net sales vs Quantity (size = กำไรเฉลี่ย/ชิ้น)")
                tmp = filtered.copy()
                for c in ["Net_sales", "Quantity", "กำไรเฉลี่ย/ชิ้น"]:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                plot_df = tmp[(tmp["Net_sales"] > 0) & (tmp["Quantity"] > 0)]
                if plot_df.empty:
                    st.info("ℹ️ ไม่มีข้อมูลเพียงพอสำหรับ bubble chart")
                else:
                    top_n = plot_df.nlargest(50, "Net_sales").copy()
                    top_n["SKU_Label"] = np.where(
                        top_n["Name"].astype(str).str.strip().ne(""),
                        top_n["Name"].astype(str),
                        top_n["SKU"].astype(str)
                    )
                    chart = (
                        alt.Chart(top_n)
                        .mark_circle(opacity=0.7)
                        .encode(
                            x=alt.X("Net_sales:Q", title="Net sales (Baht)"),
                            y=alt.Y("Quantity:Q",  title="Quantity (units)"),
                            size=alt.Size("กำไรเฉลี่ย/ชิ้น:Q", title="กำไรเฉลี่ย/ชิ้น", scale=alt.Scale(zero=False, range=[50, 1200])),
                            color=alt.Color("Category_disp:N", title="Category"),
                            tooltip=[
                                alt.Tooltip("SKU:N",           title="SKU"),
                                alt.Tooltip("SKU_Label:N",     title="Item"),
                                alt.Tooltip("Category_disp:N", title="Category"),
                                alt.Tooltip("Net_sales:Q",     title="Net sales", format=","),
                                alt.Tooltip("Quantity:Q",      title="Quantity",  format=","),
                                alt.Tooltip("กำไรเฉลี่ย/ชิ้น:Q", title="กำไรเฉลี่ย/ชิ้น", format=",.2f"),
                            ],
                        )
                        .properties(height=420)
                        .interactive()
                    )
                    st.altair_chart(chart, use_container_width=True)

                # Export & detail table
                st.subheader("📥 Export / 📋 รายละเอียด")
                st.download_button(
                    "Download Full Report (CSV)",
                    filtered.to_csv(index=False).encode("utf-8"),
                    file_name="smart_reorder_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                show_cols = [
                    "SKU", "Name", "Category", "คงเหลือ", "ควรสั่งซื้อเพิ่ม (ชิ้น)", "สถานะ", "RU Score",
                    "Opp. Loss (Baht)", "Dead Stock", "Quantity", "Net_sales", "ต้นทุนเฉลี่ย/ชิ้น", "กำไรเฉลี่ย/ชิ้น"
                ]
                show_cols = [c for c in show_cols if c in filtered.columns]
                show_df_commas(
                    filtered[show_cols],
                    int_cols=["Quantity", "ควรสั่งซื้อเพิ่ม (ชิ้น)"],
                    hide_index=False,
                )

            else:
                st.info("ℹ️ ไม่มีข้อมูลใน Category ที่เลือก")

        # -------------------- TAB 2: Sales Analysis (ใหม่) --------------------
        with tab_sales:
            st.subheader("🧭 ตัวกรองการขาย")
            cA, cB, cC = st.columns([1,1,1])
            with cA:
                timegrain = st.selectbox("Time grain", ["Daily", "Weekly", "Monthly"], index=1)
            with cB:
                cat_options = sorted(sales["Category_disp"].unique())
                sel_cats = st.multiselect("Category", options=cat_options, default=cat_options)
            with cC:
                show_top_n = st.number_input("Top-N (สำหรับ Top/Bottom)", min_value=5, max_value=50, value=10, step=1)

            sales_f = sales[sales["Category_disp"].isin(sel_cats)].copy()
            if sales_f.empty:
                st.info("ℹ️ ไม่มีข้อมูลตามตัวกรอง")
            else:
                # ===== 1) Time Series: Net sales & Gross profit =====
                st.markdown("### 1) ยอดขายตามเวลา (Time Series)")
                ts = make_timegrain(sales_f, timegrain)
                ts_agg = (
                    ts.groupby("Timegrain", as_index=False)
                      .agg(Net_sales=("Net sales","sum"),
                           Gross_profit=("Gross profit","sum"))
                )

                line_net = alt.Chart(ts_agg).mark_line(point=True).encode(
                    x=alt.X("Timegrain:T", title=f"{timegrain}"),
                    y=alt.Y("Net_sales:Q", title="Net sales"),
                    tooltip=[alt.Tooltip("Timegrain:T", title="Period"), alt.Tooltip("Net_sales:Q", format=",")]
                )
                line_gp = alt.Chart(ts_agg).mark_line(point=True).encode(
                    x="Timegrain:T",
                    y=alt.Y("Gross_profit:Q", title="Gross profit"),
                    tooltip=[alt.Tooltip("Timegrain:T"), alt.Tooltip("Gross_profit:Q", format=",")],
                    color=alt.value("#2ca02c")  # สีที่สอง (ปล่อยไว้ก็ได้ ถ้าอยากสี default ลบบรรทัดนี้)
                )
                st.altair_chart((line_net + line_gp).resolve_scale(y='independent').properties(height=360), use_container_width=True)

                # ===== 📊 ตารางสรุปรายเดือน =====
                st.markdown("#### 📊 ตารางสรุปยอดขายรายเดือน")
                
                try:
                    if 'Date' in sales_f.columns:
                        # แปลง Date column เป็น datetime และสร้าง Month
                        sales_f_copy = sales_f.copy()
                        sales_f_copy['Date'] = pd.to_datetime(sales_f_copy['Date'], errors='coerce')
                        
                        # สร้าง Month column ในรูปแบบ YYYY-MM
                        sales_f_copy['Month'] = sales_f_copy['Date'].dt.to_period('M').astype(str)
                        
                        # กรองข้อมูลที่มี Date ที่ valid
                        sales_f_copy = sales_f_copy.dropna(subset=['Date'])
                        
                        if not sales_f_copy.empty:
                            # สร้างตารางสรุปรายเดือน
                            monthly_summary = sales_f_copy.groupby('Month').agg({
                                'Net sales': 'sum',
                                'Gross profit': 'sum'
                            }).round(2)
                            
                            monthly_summary = monthly_summary.sort_index().reset_index()
                            
                            # คำนวณ %Profit = (Gross profit / Net sales) * 100
                            monthly_summary['%Profit'] = ((monthly_summary['Gross profit'] / monthly_summary['Net sales']) * 100).round(2)
                            
                            # แทนที่ NaN หรือ inf ด้วย 0 (กรณีที่ Net sales = 0)
                            monthly_summary['%Profit'] = monthly_summary['%Profit'].fillna(0).replace([float('inf'), float('-inf')], 0)
                            
                            # แสดงตาราง
                            st.dataframe(monthly_summary.style.format({
                                'Net sales': '{:,.2f}',
                                'Gross profit': '{:,.2f}',
                                '%Profit': '{:.2f}%'
                            }), use_container_width=True)
                            
                            # ปุ่มดาวน์โหลด
                            csv_monthly = monthly_summary.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Monthly Summary",
                                data=csv_monthly,
                                file_name="monthly_sales_summary.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("ไม่มีข้อมูล Date ที่ valid สำหรับการสรุปรายเดือน")
                    else:
                        st.info("ไม่มีคอลัมน์ Date สำหรับการสรุปรายเดือน")
                        
                except Exception as e:
                    st.error(f"❌ Error in Monthly Summary: {str(e)}")

                # ===== 🥧 Net Sales by Category =====
                st.markdown("#### 🥧 Net Sales by Category")
                
                try:
                    if 'Category_disp' in sales_f.columns:
                        # สร้างข้อมูลสำหรับ pie chart
                        category_sales = sales_f.groupby('Category_disp')['Net sales'].sum().reset_index()
                        category_sales = category_sales.sort_values('Net sales', ascending=False)
                        
                        # คำนวณเปอร์เซ็นต์
                        total_sales = category_sales['Net sales'].sum()
                        category_sales['Percentage'] = (category_sales['Net sales'] / total_sales * 100).round(2)
                        
                        if not category_sales.empty and total_sales > 0:
                            # สร้าง pie chart ที่ดูดีขึ้น
                            pie_chart = alt.Chart(category_sales).mark_arc(
                                innerRadius=60,
                                outerRadius=150,
                                stroke='white',
                                strokeWidth=3
                            ).encode(
                                theta=alt.Theta('Net sales:Q', sort=alt.Sort(field='Net sales', order='descending')),
                                color=alt.Color('Category_disp:N', 
                                              scale=alt.Scale(scheme='category20'),
                                              legend=alt.Legend(
                                                  title="Category",
                                                  orient="right",
                                                  titleFontSize=14,
                                                  labelFontSize=12,
                                                  symbolSize=100
                                              )),
                                order=alt.Order('Net sales:Q', sort='descending'),
                                tooltip=[
                                    alt.Tooltip('Category_disp:N', title='Category'),
                                    alt.Tooltip('Net sales:Q', title='Net Sales', format=',.2f'),
                                    alt.Tooltip('Percentage:Q', title='Percentage', format='.1f%')
                                ]
                            )
                            
                            # เพิ่ม text labels แสดงเปอร์เซ็นต์ (เฉพาะชิ้นใหญ่)
                            text_chart = alt.Chart(category_sales).mark_text(
                                radius=105,
                                fontSize=12,
                                fontWeight='bold',
                                color='black',
                                align='center',
                                baseline='middle'
                            ).encode(
                                theta=alt.Theta('Net sales:Q', sort=alt.Sort(field='Net sales', order='descending')),
                                text=alt.condition(
                                    alt.datum.Percentage > 5, 
                                    alt.Text('Percentage:Q', format='.1f'), 
                                    alt.value('')
                                ),
                                order=alt.Order('Net sales:Q', sort='descending')
                            )
                            
                            # รวม pie chart กับ text
                            final_chart = (pie_chart + text_chart).resolve_scale(
                                color='independent'
                            ).properties(
                                width=600,
                                height=450,
                                title=alt.TitleParams(
                                    text="Net Sales Distribution by Category",
                                    fontSize=18,
                                    fontWeight='bold',
                                    anchor='start'
                                )
                            )
                            
                            # แสดง pie chart
                            st.altair_chart(final_chart, use_container_width=True)
                            
                            # แสดงตารางข้อมูล (เฉพาะ columns ที่ต้องการ)
                            display_category_sales = category_sales[['Category_disp', 'Net sales', 'Percentage']].copy()
                            st.dataframe(display_category_sales.style.format({
                                'Net sales': '{:,.2f}',
                                'Percentage': '{:.2f}%'
                            }), use_container_width=True)
                            
                            # ปุ่มดาวน์โหลด
                            csv_category = display_category_sales.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Category Sales",
                                data=csv_category,
                                file_name="sales_by_category.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("ไม่มีข้อมูลยอดขายตาม Category")
                    else:
                        st.info("ไม่มีคอลัมน์ Category_disp สำหรับการแสดง pie chart")
                        
                except Exception as e:
                    st.error(f"❌ Error in Category Sales: {str(e)}")

                # ✅ ตาราง MoM ต่อ Category
                st.markdown("#### 📊 ตาราง Net Sales (MoM) ต่อ Category — Value")
                mom_sales = build_mom_table(sales_f, "Category_disp", "Net sales")
                mom_sales_val = mom_sales.pivot(index="Date", columns="Category_disp", values="Value").round(2)
                st.dataframe(fmt_commas(mom_sales_val), use_container_width=True)

                st.markdown("#### 📊 ตาราง Net Sales (MoM) ต่อ Category — Change %")
                mom_sales_chg = mom_sales.pivot(index="Date", columns="Category_disp", values="Change_%").round(2)
                st.dataframe(style_diverging_percent(mom_sales_chg), use_container_width=True)

                st.markdown("#### 📊 ตาราง Gross Profit (MoM) ต่อ Category — Value")
                mom_profit = build_mom_table(sales_f, "Category_disp", "Gross profit")
                mom_profit_val = mom_profit.pivot(index="Date", columns="Category_disp", values="Value").round(2)
                st.dataframe(fmt_commas(mom_profit_val), use_container_width=True)

                st.markdown("#### 📊 ตาราง Gross Profit (MoM) ต่อ Category — Change %")
                mom_profit_chg = mom_profit.pivot(index="Date", columns="Category_disp", values="Change_%").round(2)
                st.dataframe(style_diverging_percent(mom_profit_chg), use_container_width=True)


                # ===== NEW: Line chart by Cate_and_band =====
                st.markdown("### 📈 ยอดขายราย Category+Brand (Cate_and_band) ตามเวลา")

                if "Cate_and_band" not in sales_f.columns:
                    st.warning("⚠️ ไม่มีคอลัมน์ Cate_and_band กรุณาสร้างก่อน")
                else:
                    ts_cb = make_timegrain(sales_f, timegrain)

                    # เตรียมข้อมูล
                    ts_cb_agg = (
                        ts_cb.groupby(["Timegrain", "Cate_and_band"], as_index=False)
                            .agg(
                                Net_sales=("Net sales", "sum"),
                                Gross_profit=("Gross profit", "sum")
                            )
                    )

                    # 🔹 แยก Category ใหญ่ (ตัดจาก Cate_and_band ก่อน [ ... ])
                    ts_cb_agg["Category"] = ts_cb_agg["Cate_and_band"].str.split("[").str[0].str.strip()

                    # 🔹 Filter Category (checkbox list)
                    categories = sorted(ts_cb_agg["Category"].unique())
                    selected_categories = st.multiselect(
                        "เลือก Category ที่ต้องการ",
                        options=categories,
                        default=categories[:1]
                    )

                    ts_cb_cat = ts_cb_agg[ts_cb_agg["Category"].isin(selected_categories)]

                    # 🔹 Filter Brand (checkbox list)
                    brand_options = sorted(ts_cb_cat["Cate_and_band"].unique())
                    selected_cate_band = st.multiselect(
                        "เลือก Brand (ภายใต้ Category ที่เลือก)",
                        options=brand_options,
                        default=brand_options[:5]
                    )

                    ts_cb_cat = ts_cb_cat[ts_cb_cat["Cate_and_band"].isin(selected_cate_band)]

                    # 🔹 เลือก Metric (Net Sales / Gross Profit)
                    metric = st.radio(
                        "เลือก Metric ที่ต้องการแสดง",
                        ["Net sales", "Gross profit"],
                        horizontal=True
                    )

                    if ts_cb_cat.empty:
                        st.info("ℹ️ ไม่มีข้อมูลในตัวเลือกที่เลือก")
                    else:
                        # Chart หลัก
                        main_chart = (
                            alt.Chart(ts_cb_cat)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("Timegrain:T", title=f"{timegrain}"),
                                y=alt.Y(f"{'Net_sales' if metric=='Net sales' else 'Gross_profit'}:Q",
                                        title=f"{metric} (Baht)"),
                                color=alt.Color("Cate_and_band:N", title="Category+Brand"),
                                tooltip=[
                                    alt.Tooltip("Timegrain:T", title="Period"),
                                    alt.Tooltip("Cate_and_band:N", title="Cate_and_band"),
                                    alt.Tooltip("Net_sales:Q", title="Net sales", format=","),
                                    alt.Tooltip("Gross_profit:Q", title="Gross profit", format=",")
                                ]
                            )
                        )

                        # ถ้าเลือก Gross Profit → เพิ่ม Net Sales แบบสีอ่อน
                        if metric == "Gross profit":
                            shadow_chart = (
                                alt.Chart(ts_cb_cat)
                                .mark_line(point=False, strokeDash=[2,2], opacity=0.3)  # ✅ ทำให้เป็นเส้นอ่อน
                                .encode(
                                    x="Timegrain:T",
                                    y="Net_sales:Q",
                                    color=alt.Color("Cate_and_band:N", legend=None)  # ใช้สีเดียวกัน แต่เบลอ
                                )
                            )
                            chart_cb = main_chart + shadow_chart
                        else:
                            chart_cb = main_chart

                        chart_cb = chart_cb.properties(height=400).interactive()
                        st.altair_chart(chart_cb, use_container_width=True)

                # ✅ ตาราง MoM ต่อ Cate_and_band
                st.markdown("#### 📊 ตาราง Net Sales (MoM) ต่อ Cate_and_band — Value")
                mom_sales_cb = build_mom_table(sales_f, "Cate_and_band", "Net sales")
                mom_sales_cb_val = mom_sales_cb.pivot(index="Date", columns="Cate_and_band", values="Value").round(2)
                st.dataframe(fmt_commas(mom_sales_cb_val), use_container_width=True)

                st.markdown("#### 📊 ตาราง Net Sales (MoM) ต่อ Cate_and_band — Change %")
                mom_sales_cb_chg = mom_sales_cb.pivot(index="Date", columns="Cate_and_band", values="Change_%").round(2)
                st.dataframe(style_diverging_percent(mom_sales_cb_chg), use_container_width=True)

                st.markdown("#### 📊 ตาราง Gross Profit (MoM) ต่อ Cate_and_band — Value")
                mom_profit_cb = build_mom_table(sales_f, "Cate_and_band", "Gross profit")
                mom_profit_cb_val = mom_profit_cb.pivot(index="Date", columns="Cate_and_band", values="Value").round(2)
                st.dataframe(fmt_commas(mom_profit_cb_val), use_container_width=True)

                st.markdown("#### 📊 ตาราง Gross Profit (MoM) ต่อ Cate_and_band — Change %")
                mom_profit_cb_chg = mom_profit_cb.pivot(index="Date", columns="Cate_and_band", values="Change_%").round(2)
                st.dataframe(style_diverging_percent(mom_profit_cb_chg), use_container_width=True)



                # ===== 2) Top/Bottom Products & Categories =====
                st.markdown("### 2) Top/Bottom Products & Categories")
                sku_agg = (
                    sales_f.groupby(["SKU","Category_disp"], as_index=False)
                    .agg(Net_sales=("Net sales","sum"),
                         Gross_profit=("Gross profit","sum"),
                         Quantity=("Quantity","sum"))
                )
                # item label (ล่าสุด)
                if "Item" in sales_f.columns:
                    latest_name = (sales_f.sort_values("Date")
                                   .drop_duplicates("SKU", keep="last")[["SKU","Item"]]
                                   .rename(columns={"Item":"Item_name"}))
                    sku_agg = sku_agg.merge(latest_name, on="SKU", how="left")
                sku_agg["Label"] = sku_agg["Item_name"].fillna(sku_agg["SKU"].astype(str))

                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"🏆 Top {show_top_n} SKUs by **Net sales**")
                    top_sales = sku_agg.nlargest(show_top_n, "Net_sales")[["Label","Category_disp","Net_sales","Quantity"]]
                    show_df_commas(top_sales, int_cols=["Quantity"], hide_index=False)
                with c2:
                    st.write(f"💵 Top {show_top_n} SKUs by **Gross profit**")
                    top_profit = sku_agg.nlargest(show_top_n, "Gross_profit")[["Label","Category_disp","Gross_profit","Quantity"]]
                    show_df_commas(top_profit, int_cols=["Quantity"], hide_index=False)


                c3, c4 = st.columns(2)
                with c3:
                    st.write(f"🐢 Slow Movers (Bottom {show_top_n} by Quantity)")
                    slow = sku_agg.nsmallest(show_top_n, "Quantity")[["Label","Category_disp","Quantity","Net_sales"]]
                    show_df_commas(slow, int_cols=["Quantity"], hide_index=False)

                with c4:
                    st.write("📦 ยอดขายตาม Category")
                    cat_agg = (sales_f.groupby("Category_disp", as_index=False)
                               .agg(Net_sales=("Net sales","sum"),
                                    Gross_profit=("Gross profit","sum"),
                                    Quantity=("Quantity","sum")))
                    show_df_commas(cat_agg, int_cols=["Quantity"], hide_index=False)


                # Pareto 80/20
                st.markdown("#### 🍰 Pareto Analysis (80/20)")
                pareto = sku_agg.sort_values("Net_sales", ascending=False).reset_index(drop=True)
                pareto["cum_sales"] = pareto["Net_sales"].cumsum()
                total_sales = pareto["Net_sales"].sum()
                pareto["cum_share"] = np.where(total_sales>0, pareto["cum_sales"]/total_sales, 0.0)
                pareto["sku_rank"]  = np.arange(1, len(pareto)+1)
                pareto["sku_share"] = pareto["sku_rank"] / max(len(pareto),1)
                top_20pct_n = max(int(np.ceil(0.2*len(pareto))), 1)
                top_20_share = pareto.loc[:top_20pct_n-1, "Net_sales"].sum() / total_sales if total_sales>0 else 0

                cP1, cP2 = st.columns([2,1])
                with cP2:
                    st.metric("สัดส่วนยอดขายจาก Top 20% SKU", f"{top_20_share*100:,.1f}%")
                    st.caption("ดูว่ากฎ 80/20 ถือจริงไหมในข้อมูลที่เลือก")
                with cP1:
                    base = alt.Chart(pareto).encode(x=alt.X("sku_share:Q", title="สัดส่วนจำนวน SKU สะสม"),
                                                    y=alt.Y("cum_share:Q", title="สัดส่วนยอดขายสะสม"))
                    line = base.mark_line()
                    points = base.mark_point()
                    rule80 = alt.Chart(pd.DataFrame({"y":[0.8]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
                    rule20 = alt.Chart(pd.DataFrame({"x":[0.2]})).mark_rule(strokeDash=[4,4]).encode(x="x:Q")
                    st.altair_chart(line + points + rule80 + rule20, use_container_width=True)

                # ===== 3) Margin Analysis =====
                st.markdown("### 3) กำไรและ Margin Analysis")
                sku_agg["Margin_pct"] = np.where(sku_agg["Net_sales"]>0,
                                                 sku_agg["Gross_profit"]/sku_agg["Net_sales"],
                                                 0.0)
                scat = (alt.Chart(sku_agg)
                        .mark_circle(opacity=0.7)
                        .encode(
                            x=alt.X("Net_sales:Q", title="Net sales"),
                            y=alt.Y("Margin_pct:Q", title="Margin %", axis=alt.Axis(format="%")),
                            size=alt.Size("Quantity:Q", title="Quantity"),
                            color=alt.Color("Category_disp:N", title="Category"),
                            tooltip=["Label:N","Category_disp:N",
                                     alt.Tooltip("Net_sales:Q", format=","),
                                     alt.Tooltip("Gross_profit:Q", format=","),
                                     alt.Tooltip("Margin_pct:Q", format=".1%"),
                                     alt.Tooltip("Quantity:Q", format=",")]
                        ).properties(height=380)
                        .interactive())
                st.altair_chart(scat, use_container_width=True)

                # Contribution Margin (สินค้าใดดันกำไร)
                contrib = sku_agg.sort_values("Gross_profit", ascending=False).head(show_top_n)
                st.markdown(f"#### 🔥 Contribution Margin — Top {show_top_n} by Gross Profit")
                show_df_commas(
                    contrib[["Label","Category_disp","Gross_profit","Net_sales","Quantity"]],
                    int_cols=["Quantity"],
                    hide_index=False,
                )


                # ===== 4) Customer Behavior =====
                st.markdown("### 4) Customer Behavior")
                cust_ready = {"Customer name","Customer contacts"}.issubset(sales_f.columns)
                # สร้าง customer_key แม้บางค่าเป็น null
                if "Customer name" in sales_f.columns or "Customer contacts" in sales_f.columns:
                    sales_f["Customer name"]    = sales_f.get("Customer name", "").astype(str)
                    sales_f["Customer contacts"] = sales_f.get("Customer contacts", "").astype(str)
                    sales_f["customer_key"] = (sales_f["Customer name"].str.strip() + " | " +
                                              sales_f["Customer contacts"].str.strip()).str.strip(" |")
                else:
                    sales_f["customer_key"] = np.nan

                # Repeat vs New
                if sales_f["customer_key"].notna().any():
                    first_date = (sales_f.sort_values("Date")
                                  .groupby("customer_key", as_index=False)["Date"].min()
                                  .rename(columns={"Date":"first_buy"}))
                    joined = sales_f.merge(first_date, on="customer_key", how="left")
                    joined["is_new"] = joined["Date"].dt.date == joined["first_buy"].dt.date
                    cust_counts = joined.groupby("customer_key").agg(
                        first_buy=("first_buy","min"),
                        orders=("customer_key","count"),
                        total_spent=("Net sales","sum")
                    ).reset_index()
                    cust_counts["type"] = np.where(cust_counts["orders"]>1, "Repeat", "New")
                    total_cust = cust_counts["customer_key"].nunique()
                    new_pct    = (cust_counts["type"].eq("New").mean()*100) if total_cust>0 else 0
                    rep_pct    = 100 - new_pct
                    cR1, cR2, cR3 = st.columns(3)
                    cR1.metric("ลูกค้ารวม (unique)", f"{total_cust:,}")
                    cR2.metric("New (%)", f"{new_pct:,.1f}%")
                    cR3.metric("Repeat (%)", f"{rep_pct:,.1f}%")
                else:
                    st.info("ℹ️ ไม่มีข้อมูลลูกค้า (Customer name/contacts) เพียงพอสำหรับ Repeat vs New")

                # Average Basket Size
                # ถ้ามี Receipt number ใช้อันนั้นเป็นบิล; ถ้าไม่มีก็ group โดย (Date, customer_key) เป็น proxy
                if "Receipt number" in sales_f.columns:
                    orders = (sales_f.groupby("Receipt number", as_index=False)
                                      .agg(order_value=("Net sales","sum")))
                elif sales_f["customer_key"].notna().any():
                    orders = (sales_f.groupby(["customer_key", sales_f["Date"].dt.date], as_index=False)
                                      .agg(order_value=("Net sales","sum")))
                else:
                    orders = (sales_f.groupby(sales_f["Date"].dt.date, as_index=False)
                                      .agg(order_value=("Net sales","sum")))
                avg_basket = orders["order_value"].mean() if not orders.empty else 0.0
                st.metric("🛒 Average Basket Size (บาท/บิล)", f"{avg_basket:,.2f}")

                # Interpurchase Time (IPT)
                if sales_f["customer_key"].notna().any():
                    ipt_list = []
                    for cid, g in sales_f.groupby("customer_key"):
                        ds = g["Date"].sort_values().drop_duplicates().to_list()
                        if len(ds) >= 2:
                            diffs = np.diff(pd.to_datetime(ds)).astype("timedelta64[D]").astype(int)
                            if len(diffs)>0:
                                ipt_list.extend(diffs)
                    if len(ipt_list) > 0:
                        ipt_ser = pd.Series(ipt_list)
                        st.write(f"📅 Interpurchase Time (days) — mean: **{ipt_ser.mean():,.1f}** | median: **{ipt_ser.median():,.0f}**")
                        ipt_df = pd.DataFrame({"IPT_days": ipt_ser})
                        hist = alt.Chart(ipt_df).mark_bar().encode(
                            x=alt.X("IPT_days:Q", bin=alt.Bin(maxbins=30), title="Days between purchases"),
                            y=alt.Y("count():Q", title="Count")
                        ).properties(height=250)
                        st.altair_chart(hist, use_container_width=True)

                        # ===== Customer-level IPT summary & items =====
                        st.markdown("#### 👥 Interpurchase Summary by Customer")

                        # เฉพาะเคสที่มี customer_key
                        if sales_f["customer_key"].notna().any():
                            # 1) กำหนด label สินค้า (ถ้ามี Item ใช้ Item ไม่งั้นใช้ SKU)
                            if "Item" in sales_f.columns:
                                sales_f["item_label"] = sales_f["Item"].astype(str).where(
                                    sales_f["Item"].notna(), sales_f["SKU"].astype(str)
                                )
                            else:
                                sales_f["item_label"] = sales_f["SKU"].astype(str)

                            # 2) รวมสถิติ IPT ต่อหัวลูกค้า
                            def _ipt_stats(g: pd.DataFrame) -> pd.Series:
                                ds = g["Date"].sort_values().drop_duplicates().to_numpy()
                                diffs = np.diff(ds).astype("timedelta64[D]").astype(int) if len(ds) >= 2 else np.array([], dtype=int)
                                return pd.Series({
                                    "orders": len(ds),
                                    "IPT_count": len(diffs),
                                    "IPT_mean": float(np.mean(diffs)) if len(diffs) > 0 else np.nan,
                                    "IPT_median": float(np.median(diffs)) if len(diffs) > 0 else np.nan,
                                    "Quantity": g["Quantity"].sum(),
                                    "Total_spent": g["Net sales"].sum(),
                                    "Last_purchase": g["Date"].max(),
                                })

                            cust_stats = (sales_f.groupby("customer_key").apply(_ipt_stats).reset_index())

                            # 3) Top 10 รายการต่อหัวลูกค้า (ชื่อสินค้า — มูลค่า — % ของ Total_spent)
                            top_items = (
                                sales_f.groupby(["customer_key", "item_label"], as_index=False)
                                    .agg(spent=("Net sales","sum"))
                            ).merge(
                                cust_stats[["customer_key","Total_spent"]], on="customer_key", how="left"
                            )

                            top_items["pct"] = np.where(
                                top_items["Total_spent"] > 0,
                                100 * top_items["spent"] / top_items["Total_spent"],
                                0
                            )

                            # เลือก top 10 ต่อ customer และรวมเป็นข้อความหลายบรรทัด
                            top_items = (
                                top_items.sort_values(["customer_key","spent"], ascending=[True, False])
                                        .groupby("customer_key")
                                        .head(10)
                            )
                            top_items["detail"] = top_items.apply(
                                lambda r: f"{r['item_label']} — {r['spent']:,.0f}฿ ({r['pct']:.1f}%)", axis=1
                            )
                            items_fmt = (
                                top_items.groupby("customer_key")["detail"]
                                        .apply(lambda s: "\n".join(s))
                                        .reset_index(name="Top 10 purchases")
                            )

                            # 4) รวมกลับและเรียงลำดับ
                            cust_stats = (cust_stats
                                        .merge(items_fmt, on="customer_key", how="left")
                                        .sort_values(["IPT_count","orders","Total_spent"],
                                                    ascending=[False, False, False]))

                            # 5) แสดงผล (เหลือคอลัมน์ใหม่เดียว)
                            cols = [
                                "customer_key","orders","IPT_count","IPT_mean","IPT_median",
                                "Quantity","Total_spent","Last_purchase","Top 10 purchases"
                            ]
                            show_df_commas(
                                cust_stats[cols],
                                int_cols=["orders","IPT_count","Quantity"],
                                hide_index=False,
                            )

                            st.caption("หมายเหตุ: IPT_count คือจำนวน 'ช่วงเวลาระหว่างการซื้อ' ต่อหัวลูกค้า (ไม่ใช่จำนวนลูกค้า)")

        # ---------------- TAB 3: การวิเคราะห์ยอดขายตก ----------------
        with tab_drop:
            st.subheader("📉 การวิเคราะห์ยอดขายตก")

            months = sorted(sales["Date"].dt.to_period("M").astype(str).unique())
            if len(months) < 2:
                st.info("ℹ️ ต้องมีข้อมูลอย่างน้อย 2 เดือน")
            else:
                month_curr = st.selectbox(
                    "เลือกเดือนปัจจุบัน (Curr)", months, index=len(months)-1, key="drop_curr_only"
                )

                # หาเดือนก่อนหน้า (prev)
                month_pos = months.index(month_curr)
                if month_pos == 0:
                    st.warning("⚠️ ไม่มีเดือนก่อนหน้าให้เปรียบเทียบ")
                else:
                    month_prev = months[month_pos - 1]

                    # เตรียมข้อมูลรายเดือนทั้งหมดเพื่อคำนวณ MA (exclude current for MA baseline)
                    sales["Month"] = sales["Date"].dt.to_period("M").astype(str)
                    monthly_all = (sales.groupby(["Month", "Cate_and_band"], as_index=False)
                                        .agg(Net_Sales=("Net sales", "sum")))

                    # สร้าง index เดือนเพื่อความชัดเจน (แม้ไม่ใช้ตรง ๆ ในการคำนวณ MA ใหม่)
                    month_order = sorted(monthly_all["Month"].unique())
                    month_index = {m: i for i, m in enumerate(month_order)}
                    monthly_all["month_idx"] = monthly_all["Month"].map(month_index)
                    monthly_all = monthly_all.sort_values(["Cate_and_band", "month_idx"])

                    # คำนวณค่าเฉลี่ย 3 เดือนก่อนหน้า (ไม่รวมเดือน prev) = เฉลี่ยของเดือนที่ห่าง 2,3,4 เดือนก่อน curr
                    # ใช้ shift(2), shift(3), shift(4) ต่อ Cate_and_band แล้วหา mean ข้าม 3 คอลัมน์
                    grp = monthly_all.groupby("Cate_and_band")
                    monthly_all["lag2"] = grp["Net_Sales"].shift(2)
                    monthly_all["lag3"] = grp["Net_Sales"].shift(3)
                    monthly_all["lag4"] = grp["Net_Sales"].shift(4)
                    monthly_all["MA_3m_prev"] = monthly_all[["lag2", "lag3", "lag4"]].mean(axis=1, skipna=True)

                    # ดึงค่าของเดือน prev / curr
                    curr_df = monthly_all[monthly_all["Month"] == month_curr][["Cate_and_band", "Net_Sales"]].rename(columns={"Net_Sales": "Net_Sales_curr"})
                    prev_df = monthly_all[monthly_all["Month"] == month_prev][["Cate_and_band", "Net_Sales", "MA_3m_prev"]].rename(columns={"Net_Sales": "Net_Sales_prev"})

                    merged = curr_df.merge(prev_df, on="Cate_and_band", how="left")
                    # คำนวณ Change
                    merged["Change"] = merged["Net_Sales_curr"] - merged["Net_Sales_prev"].fillna(0)
                    merged["Change_%"] = np.where(
                        merged["Net_Sales_prev"] > 0,
                        (merged["Change"] / merged["Net_Sales_prev"]) * 100,
                        np.nan
                    )

                    # MA_3m_prev คือ baseline (สามเดือนก่อน curr ไม่รวม curr) ที่ได้จาก prev row (อาจ NaN)
                    # คำนวณ MA_Change = Curr - MA_3m_prev
                    merged["MA_Change"] = merged["Net_Sales_curr"] - merged["MA_3m_prev"]
                    merged["MA_Change_%"] = np.where(
                        merged["MA_3m_prev"] > 0,
                        (merged["MA_Change"] / merged["MA_3m_prev"]) * 100,
                        np.nan
                    )

                    # เลือกเฉพาะยอดขายตก (Change < 0)
                    top_n_drop = st.slider("แสดง Top N ยอดขายลดลงมากที่สุด", min_value=5, max_value=50, value=20, step=5)
                    drops = merged[merged["Change"] < 0].sort_values("Change").head(top_n_drop).reset_index(drop=True)

                    if not drops.empty:
                        absmax = np.nanmax(np.abs(drops["Change_%"])) if drops["Change_%"].notna().any() else None
                        absmax_ma = np.nanmax(np.abs(drops["MA_Change_%"])) if drops["MA_Change_%"].notna().any() else absmax
                        sty = (drops.style
                               .format({
                                   "Net_Sales_prev": "{:,.2f}",
                                   "Net_Sales_curr": "{:,.2f}",
                                   "Change": "{:,.2f}",
                                   "Change_%": "{:+,.2f}%",
                                   "MA_3m_prev": "{:,.2f}",
                                   "MA_Change": "{:,.2f}",
                                   "MA_Change_%": "{:+,.2f}%",
                               })
                               .background_gradient(cmap="RdYlGn", vmin=(-absmax if absmax else None), vmax=(absmax if absmax else None), subset=["Change_%"]) 
                               .background_gradient(cmap="PuOr", vmin=(-absmax_ma if absmax_ma else None), vmax=(absmax_ma if absmax_ma else None), subset=["MA_Change_%"]))
                        st.dataframe(sty, use_container_width=True)
                        st.caption("หมายเหตุ: MA_3m_prev = ค่าเฉลี่ย 3 เดือนก่อนหน้า (ไม่รวมเดือนปัจจุบัน)")

                        # ===== รายละเอียดเพิ่มเติมต่อ Cate_and_band (expanders) =====
                        all_lost_collector = []  # จะเก็บ lost ของแต่ละ Cate_and_band เพื่อสรุปท้ายหน้า
                        if "Item" not in sales.columns:
                            st.info("⚠️ ไม่มีคอลัมน์ Item จึงไม่สามารถแสดงรายละเอียดรายการสินค้าได้")
                        else:
                            # เตรียมข้อมูลเฉพาะ 2 เดือน (prev / curr) เพื่อลดการ filter ซ้ำ
                            if "Month" not in sales.columns:
                                sales["Month"] = sales["Date"].dt.to_period("M").astype(str)
                            sales_prev_all = sales[sales["Month"] == month_prev]
                            sales_curr_all = sales[sales["Month"] == month_curr]

                            # วนแต่ละ Cate_and_band ที่มียอดตก
                            for cb in drops["Cate_and_band"].tolist():
                                with st.expander(f"🔽 {cb} — รายละเอียดเพิ่มเติม", expanded=False):
                                    sub_prev = sales_prev_all[sales_prev_all["Cate_and_band"] == cb]
                                    sub_curr = sales_curr_all[sales_curr_all["Cate_and_band"] == cb]

                                    # 1) ตาราง Item ที่ยอดขายลดลง
                                    prev_items = (sub_prev.groupby("Item", as_index=False)["Net sales"].sum()
                                                            .rename(columns={"Net sales": "Net_Sales_prev"}))
                                    curr_items = (sub_curr.groupby("Item", as_index=False)["Net sales"].sum()
                                                            .rename(columns={"Net sales": "Net_Sales_curr"}))
                                    item_merge = curr_items.merge(prev_items, on="Item", how="outer").fillna(0)
                                    # คำนวณ Change
                                    item_merge["Change"] = item_merge["Net_Sales_curr"] - item_merge["Net_Sales_prev"]
                                    item_merge["Change_%"] = np.where(
                                        item_merge["Net_Sales_prev"] > 0,
                                        (item_merge["Change"] / item_merge["Net_Sales_prev"]) * 100,
                                        np.nan
                                    )
                                    # เอาเฉพาะที่ลดลง
                                    items_drop = item_merge[item_merge["Change"] < 0].sort_values("Change")
                                    st.markdown("**📦 รายการสินค้าใน Cate_and_band ที่ยอดขายลดลง**")
                                    if items_drop.empty:
                                        st.info("ไม่มีสินค้าใดมียอดขายลดลง")
                                    else:
                                        st.dataframe(
                                            fmt_commas(
                                                items_drop[["Item","Net_Sales_prev","Net_Sales_curr","Change"]]
                                                          .assign(**{"Change_%": items_drop["Change_%"].round(2)}),
                                            ).format({"Change_%": "{:+,.2f}%"}),
                                            use_container_width=True,
                                        )

                                    # 2) ตารางลูกค้าที่หายไป (ซื้อ prev ไม่ซื้อ curr)
                                    st.markdown("**🧍‍♀️ ลูกค้าที่หายไป (ซื้อเดือนก่อน แต่ไม่ซื้อเดือนปัจจุบัน)**")
                                    if not {"Customer name","Customer contacts"}.issubset(sales.columns):
                                        st.info("ไม่มีคอลัมน์ Customer name และ/หรือ Customer contacts เพียงพอ")
                                    else:
                                        prev_cust = sub_prev.copy()
                                        curr_cust = sub_curr.copy()
                                        # รวมสอง field เป็น customer_key
                                        prev_cust["customer_key"] = (
                                            prev_cust["Customer name"].astype(str).str.strip() + " | " +
                                            prev_cust["Customer contacts"].astype(str).str.strip()
                                        ).str.strip(" |")
                                        curr_cust["customer_key"] = (
                                            curr_cust["Customer name"].astype(str).str.strip() + " | " +
                                            curr_cust["Customer contacts"].astype(str).str.strip()
                                        ).str.strip(" |")
                                        prev_cust_agg = (prev_cust.groupby("customer_key", as_index=False)["Net sales"].sum()
                                                                  .rename(columns={"Net sales": "Net_Sales_prev"}))
                                        curr_keys = set(curr_cust["customer_key"].unique())
                                        lost = prev_cust_agg[~prev_cust_agg["customer_key"].isin(curr_keys)].copy()
                                        lost = lost.sort_values("Net_Sales_prev", ascending=False)
                                        if lost.empty:
                                            st.info("ไม่มีลูกค้าที่หายไปในเดือนนี้")
                                        else:
                                            lost["มูลค่าที่หายไป"] = lost["Net_Sales_prev"]
                                            # เก็บรวมเพื่อสรุปท้ายหน้า (เพิ่ม Cate_and_band ปัจจุบัน)
                                            lost_with_cb = lost.assign(Cate_and_band=cb)
                                            all_lost_collector.append(lost_with_cb)
                                            st.dataframe(
                                                fmt_commas(
                                                    lost[["customer_key","Net_Sales_prev","มูลค่าที่หายไป"]],
                                                ),
                                                use_container_width=True,
                                            )
                                            st.write(f"**รวมมูลค่าลูกค้าที่หายไป:** {lost['Net_Sales_prev'].sum():,.2f} บาท")

                                            # แสดงรายละเอียดสินค้า (Item) ที่ลูกค้าแต่ละคนซื้อในเดือนก่อนหน้า
                                            if "Item" in prev_cust.columns:
                                                for _, crow in lost.iterrows():
                                                    ckey = crow["customer_key"]
                                                    c_total = crow["Net_Sales_prev"]
                                                    cust_items_prev = prev_cust[prev_cust["customer_key"] == ckey]
                                                    item_detail = (cust_items_prev.groupby(["Item","Cate_and_band"], as_index=False)["Net sales"].sum()
                                                                                  .rename(columns={"Item":"Item_name","Net sales":"Net_Sales_prev"}))
                                                    item_detail = item_detail[item_detail["Net_Sales_prev"] > 0]
                                                    if item_detail.empty:
                                                        continue
                                                    item_detail["มูลค่าที่หายไป"] = item_detail["Net_Sales_prev"]
                                                    item_detail = item_detail.sort_values("มูลค่าที่หายไป", ascending=False)
                                                    with st.expander(f"🧍‍♀️ {ckey} – มูลค่าที่หายไปรวม {c_total:,.2f} บาท", expanded=False):
                                                        st.dataframe(
                                                            fmt_commas(
                                                                item_detail[["Item_name","Cate_and_band","Net_Sales_prev","มูลค่าที่หายไป"]],
                                                            ),
                                                            use_container_width=True,
                                                        )
                                                        st.caption("ยอด Net_Sales_prev = มูลค่าที่หายไป เพราะเดือนปัจจุบันไม่มีการซื้อ")

                                    # 3) ลูกค้าที่มียอดซื้อลดลง (ยังคงซื้อ แต่ยอดลดลง) ใน Cate_and_band นี้
                                    st.markdown("**📉 ลูกค้าที่มียอดซื้อลดลง (ยังคงซื้อ แต่ยอดลดลง)**")
                                    
                                    # Check if customer columns exist
                                    has_cust_name = "Customer name" in sales.columns
                                    has_cust_contacts = "Customer contacts" in sales.columns
                                    
                                    if not has_cust_name and not has_cust_contacts:
                                        st.info("ไม่มีคอลัมน์ Customer name หรือ Customer contacts เพียงพอสำหรับการคำนวณ")
                                    else:
                                        # Filter ข้อมูลเฉพาะ Cate_and_band นี้
                                        cb_prev = sub_prev.copy()
                                        cb_curr = sub_curr.copy()

                                        # สร้าง customer_key อย่างปลอดภัย
                                        for df_tmp in (cb_prev, cb_curr):
                                            # Handle missing columns gracefully
                                            cust_name = df_tmp.get("Customer name", "").astype(str).str.strip() if has_cust_name else ""
                                            cust_contacts = df_tmp.get("Customer contacts", "").astype(str).str.strip() if has_cust_contacts else ""
                                            
                                            # Create customer_key, fallback to index if both are empty
                                            if has_cust_name and has_cust_contacts:
                                                df_tmp["customer_key"] = (cust_name + " | " + cust_contacts).str.strip(" |")
                                            elif has_cust_name:
                                                df_tmp["customer_key"] = cust_name
                                            elif has_cust_contacts:
                                                df_tmp["customer_key"] = cust_contacts
                                            else:
                                                df_tmp["customer_key"] = "Unknown_" + df_tmp.index.astype(str)
                                            
                                            # Clean up empty/null customer keys
                                            df_tmp["customer_key"] = df_tmp["customer_key"].replace(["", "nan", "nan | nan"], "Unknown")
                                            df_tmp = df_tmp[df_tmp["customer_key"] != "Unknown"].copy() if not df_tmp["customer_key"].eq("Unknown").all() else df_tmp

                                        # Aggregate ระดับลูกค้า (เฉพาะ Cate_and_band นี้)
                                        cb_prev_cust = (cb_prev.groupby("customer_key", as_index=False)["Net sales"].sum()
                                                                 .rename(columns={"Net sales":"Net_Sales_prev"}))
                                        cb_curr_cust = (cb_curr.groupby("customer_key", as_index=False)["Net sales"].sum()
                                                                 .rename(columns={"Net sales":"Net_Sales_curr"}))
                                        cb_cust_merge = cb_prev_cust.merge(cb_curr_cust, on="customer_key", how="inner")  # ต้องมีทั้งสองเดือน
                                        
                                        if cb_cust_merge.empty:
                                            st.info("ไม่มีลูกค้าที่พบในทั้งสองเดือนสำหรับ Cate_and_band นี้")
                                        else:
                                            cb_cust_merge["Change"] = cb_cust_merge["Net_Sales_curr"] - cb_cust_merge["Net_Sales_prev"]
                                            cb_cust_merge["Change_%"] = np.where(
                                                cb_cust_merge["Net_Sales_prev"] > 0,
                                                (cb_cust_merge["Change"] / cb_cust_merge["Net_Sales_prev"]) * 100,
                                                np.nan
                                            )
                                            # คัดเฉพาะยอดลดลง
                                            cb_cust_drop = cb_cust_merge[cb_cust_merge["Change"] < 0].copy().sort_values("Change")
                                            
                                            if cb_cust_drop.empty:
                                                st.info("ไม่มีลูกค้าที่มียอดซื้อลดลงใน Cate_and_band นี้")
                                            else:
                                                cb_total_cust_drop = cb_cust_drop["customer_key"].nunique()
                                                cb_total_value_drop = cb_cust_drop["Change"].sum()  # เป็นค่าลบ
                                                st.markdown(f"**รวมลูกค้าที่มียอดลดลงใน {cb}:** {cb_total_cust_drop} ราย | มูลค่าที่ลดลงรวม {cb_total_value_drop:,.2f} บาท")

                                                show_cols_cust = ["customer_key","Net_Sales_prev","Net_Sales_curr","Change","Change_%"]
                                                # ตารางหลัก (คอมม่าครบทุกคอลัมน์ + เปอร์เซ็นต์)
                                                show_df_commas(
                                                    cb_cust_drop.assign(**{"Change_%": cb_cust_drop["Change_%"].round(2)})[show_cols_cust],
                                                    float_cols=("Net_Sales_prev","Net_Sales_curr","Change"),
                                                    percent_cols=("Change_%",),
                                                    hide_index=False,
                                                )

                                                # รายละเอียดสินค้าเฉพาะลูกค้าที่มูลค่าลดลง (ใน Cate_and_band นี้)
                                                if "Item" not in sales.columns:
                                                    st.info("ไม่มีคอลัมน์ Item สำหรับแสดงรายละเอียดระดับสินค้า")
                                                else:
                                                    st.markdown("**รายละเอียดระดับสินค้า (คลิกที่ลูกค้าเพื่อขยาย)**")
                                                    
                                                    # เตรียม item aggregates ของสองเดือนเฉพาะ Cate_and_band นี้
                                                    cb_prev_items = (cb_prev.groupby(["customer_key","Item"], as_index=False)["Net sales"].sum()
                                                                                .rename(columns={"Net sales":"Net_Sales_prev"}))
                                                    cb_curr_items = (cb_curr.groupby(["customer_key","Item"], as_index=False)["Net sales"].sum()
                                                                                .rename(columns={"Net sales":"Net_Sales_curr"}))

                                                    for _, cb_rowc in cb_cust_drop.iterrows():
                                                        cb_ckey = cb_rowc["customer_key"]
                                                        cb_c_total_drop = cb_rowc["Change"]  # เป็นค่าลบ
                                                        
                                                        # item prev/curr ของลูกค้านี้ (ใน Cate_and_band นี้)
                                                        cb_cust_prev_items = cb_prev_items[cb_prev_items["customer_key"] == cb_ckey]
                                                        cb_cust_curr_items = cb_curr_items[cb_curr_items["customer_key"] == cb_ckey]
                                                        
                                                        if cb_cust_prev_items.empty:
                                                            continue  # Skip if no previous data
                                                            
                                                        cb_item_merge = cb_cust_prev_items.merge(
                                                            cb_cust_curr_items,
                                                            on=["customer_key","Item"],
                                                            how="left"
                                                        )
                                                        cb_item_merge["Net_Sales_curr"] = cb_item_merge["Net_Sales_curr"].fillna(0)
                                                        cb_item_merge["Change"] = cb_item_merge["Net_Sales_curr"] - cb_item_merge["Net_Sales_prev"]
                                                        cb_item_merge["Change_%"] = np.where(
                                                            cb_item_merge["Net_Sales_prev"] > 0,
                                                            (cb_item_merge["Change"] / cb_item_merge["Net_Sales_prev"]) * 100,
                                                            np.nan
                                                        )
                                                        # เอาเฉพาะ item ที่มูลค่าลดลง
                                                        cb_item_drop = cb_item_merge[cb_item_merge["Change"] < 0].copy()
                                                        
                                                        if cb_item_drop.empty:
                                                            # Show expander even if no item drops, but indicate no detailed drops
                                                            with st.expander(f"📉 {cb_ckey} – มูลค่าลดลง {cb_c_total_drop:,.2f} บาท (ใน {cb})", expanded=False):
                                                                st.info("ไม่มีรายการสินค้าเฉพาะที่ลดลง (อาจเป็นการลดจำนวนซื้อโดยรวม)")
                                                        else:
                                                            cb_item_drop = cb_item_drop.sort_values("Change")
                                                            with st.expander(f"📉 {cb_ckey} – มูลค่าลดลง {cb_c_total_drop:,.2f} บาท (ใน {cb})", expanded=False):
                                                                show_df_commas(
                                                                    cb_item_drop.assign(**{"Change_%": cb_item_drop["Change_%"].round(2)})[[
                                                                        "Item","Net_Sales_prev","Net_Sales_curr","Change","Change_%"
                                                                    ]],
                                                                    float_cols=("Net_Sales_prev","Net_Sales_curr","Change"),
                                                                    percent_cols=("Change_%",),
                                                                    hide_index=True,
                                                                )
                                                    st.caption("ลูกค้ากลุ่มนี้ยังซื้ออยู่ในเดือนปัจจุบัน แต่ยอดขายลดลงเมื่อเทียบกับเดือนก่อนหน้า")

                        # ===== สรุปท้ายหน้า ลูกค้าที่หายไป (รวมทุก Cate_and_band ที่อยู่ใน drops) =====
                        if all_lost_collector:
                            lost_all_df = pd.concat(all_lost_collector, ignore_index=True)
                            st.markdown("---")
                            st.markdown(
                                f"**สรุปรวมลูกค้าที่หายไปทั้งหมด:** {lost_all_df['customer_key'].nunique()} ราย | มูลค่าที่หายไปรวม {lost_all_df['Net_Sales_prev'].sum():,.2f} บาท"
                            )

                    else:
                        st.info("ℹ️ ไม่มี Cate_and_band ที่ยอดขายลดลงในเดือนนี้เมื่อเทียบกับเดือนก่อนหน้า")

        # =============== Customer Analysis Tab ===============
        with tab_customer:
            st.subheader("🎯 Customer Analysis")
            st.info("การวิเคราะห์ลูกค้าจากไฟล์ยอดขายที่อัปโหลด")
            
            if sales is None or sales.empty:
                st.warning("⚠️ กรุณาอัปโหลดไฟล์ยอดขายก่อนเพื่อทำการวิเคราะห์ลูกค้า")
            else:
                try:
                    # Prepare clean customer data
                    customer_data = sales.copy()
                    
                    # 🔧 SIMPLE FIX: Convert all nested Series to simple values
                    for col in customer_data.columns:
                        if customer_data[col].dtype == 'object':
                            # Force convert any nested structures to simple values
                            customer_data[col] = customer_data[col].apply(
                                lambda x: x.iloc[0] if hasattr(x, 'iloc') and len(x) > 0 else x
                            )
                    
                    # Ensure numeric columns are actually numeric
                    numeric_cols = ['Net sales', 'Quantity', 'Cost of goods', 'Gross profit', 'Discounts', 'Taxes']
                    for col in numeric_cols:
                        if col in customer_data.columns:
                            customer_data[col] = pd.to_numeric(customer_data[col], errors='coerce').fillna(0)
                    
                    # Build customer_key (same as in Sales Drop Analysis)
                    if 'customer_key' not in customer_data.columns:
                        has_cust_name = 'Customer name' in customer_data.columns
                        has_cust_contacts = 'Customer contacts' in customer_data.columns
                        
                        if has_cust_name or has_cust_contacts:
                            # Handle missing columns gracefully
                            cust_name = customer_data.get("Customer name", "").astype(str).str.strip() if has_cust_name else ""
                            cust_contacts = customer_data.get("Customer contacts", "").astype(str).str.strip() if has_cust_contacts else ""
                            
                            # Create customer_key, fallback to index if both are empty
                            if has_cust_name and has_cust_contacts:
                                customer_data["customer_key"] = (cust_name + " | " + cust_contacts).str.strip(" |")
                            elif has_cust_name:
                                customer_data["customer_key"] = cust_name
                            elif has_cust_contacts:
                                customer_data["customer_key"] = cust_contacts
                            else:
                                customer_data["customer_key"] = "Unknown_" + customer_data.index.astype(str)
                            
                            # Clean up empty/null customer keys
                            customer_data["customer_key"] = customer_data["customer_key"].replace(["", "nan", "nan | nan"], "Unknown")
                            
                            # Keep original sales data for total calculations
                            original_total_sales = customer_data['Net sales'].sum()
                            
                            # Filter out unknown customers for customer-specific analysis
                            customer_data_filtered = customer_data[customer_data["customer_key"] != "Unknown"].copy() if not customer_data["customer_key"].eq("Unknown").all() else customer_data
                    
                    # === 1. Customer Overview ===
                    st.subheader("📊 Customer Overview")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        unique_customers = customer_data_filtered['customer_key'].nunique()
                        st.metric("🧑‍🤝‍🧑 Unique Customers", f"{unique_customers:,}")
                        
                        # 🔧 FIX: Avg Basket Size ที่ถูกต้อง (ยอดขายเฉลี่ยต่อใบเสร็จ)
                        if 'Receipt number' in customer_data_filtered.columns:
                            receipt_totals = customer_data_filtered.groupby('Receipt number')['Net sales'].sum()
                            avg_basket = receipt_totals.mean()
                        else:
                            # Fallback: ใช้ค่าเฉลี่ยต่อแถว (ไม่มี Receipt number)
                            avg_basket = customer_data_filtered['Net sales'].mean()
                        st.metric("🛒 Avg Basket Size", f"{avg_basket:,.2f}")
                    
                    with col2:
                        total_receipts = customer_data_filtered['Receipt number'].nunique() if 'Receipt number' in customer_data_filtered.columns else len(customer_data_filtered)
                        st.metric("🧾 Total Receipts", f"{total_receipts:,}")
                        
                        # 🔧 FIX: Repeat Customer Rate ที่ถูกต้อง (นับจากใบเสร็จ)
                        if 'Receipt number' in customer_data_filtered.columns:
                            customer_receipts = customer_data_filtered.groupby('customer_key')['Receipt number'].nunique()
                            repeat_rate = (customer_receipts > 1).sum() / len(customer_receipts) * 100
                        else:
                            # Fallback: นับจากแถวข้อมูล (ไม่มี Receipt number)
                            repeat_customers = customer_data_filtered.groupby('customer_key').size()
                            repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
                        st.metric("🔄 Repeat Customer Rate", f"{repeat_rate:.1f}%")
                    
                    with col3:
                        # 🔧 FIX: ใช้ original_total_sales เพื่อให้ตรงกับหน้าหลัก
                        st.metric("💰 Total Sales", f"{original_total_sales:,.2f}")
                        
                        avg_sales_per_customer = original_total_sales / unique_customers if unique_customers > 0 else 0
                        st.metric("👤 Avg Sales/Customer", f"{avg_sales_per_customer:,.2f}")
                    
                    # === 2. Top Customers ===
                    st.subheader("🏆 Top Customers")
                    
                    try:
                        # Enhanced aggregation with more metrics
                        agg_dict = {
                            'Net sales': 'sum',
                            'Receipt number': 'nunique' if 'Receipt number' in customer_data_filtered.columns else 'count'
                        }
                        
                        # Add Gross profit if available
                        if 'Gross profit' in customer_data_filtered.columns:
                            agg_dict['Gross profit'] = 'sum'
                        
                        top_customers = customer_data_filtered.groupby('customer_key').agg(agg_dict).round(2)
                        
                        # Rename columns
                        new_columns = ['Total_Sales', 'Total_Orders']
                        if 'Gross profit' in customer_data_filtered.columns:
                            new_columns.append('Net_Profit')
                        
                        top_customers.columns = new_columns
                        
                        # Calculate average per bill
                        top_customers['Avg_Per_Bill'] = (top_customers['Total_Sales'] / top_customers['Total_Orders']).round(2)
                        
                        # Sort and get top 50
                        top_customers = top_customers.sort_values('Total_Sales', ascending=False).head(50)
                        top_customers = top_customers.reset_index()
                        
                        # Prepare display columns
                        display_cols = ['customer_key', 'Total_Sales', 'Total_Orders', 'Avg_Per_Bill']
                        column_names = ['Customer_Name', 'Total_Sales', 'Total_Orders', 'Avg_Per_Bill']
                        format_dict = {
                            'Total_Sales': '{:,.2f}',
                            'Total_Orders': '{:,.0f}',
                            'Avg_Per_Bill': '{:,.2f}'
                        }
                        
                        # Add Net_Profit if available
                        if 'Net_Profit' in top_customers.columns:
                            display_cols.insert(-1, 'Net_Profit')  # Insert before Avg_Per_Bill
                            column_names.insert(-1, 'Net_Profit')
                            format_dict['Net_Profit'] = '{:,.2f}'
                        
                        # Display table
                        display_customers = top_customers[display_cols].copy()
                        display_customers.columns = column_names
                        st.dataframe(display_customers.style.format(format_dict), use_container_width=True)
                        
                        # Download button
                        csv_customers = top_customers.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Customer List",
                            data=csv_customers,
                            file_name="top_customers.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error in Top Customers: {str(e)}")
                    
                    # === 3. Customer Portfolio ===
                    st.subheader("👤 Customer Portfolio")
                    
                    try:
                        # อ่านไฟล์ first_time.csv หากมี
                        first_time_data = None
                        try:
                            first_time_path = "first_time.csv"
                            first_time_data = pd.read_csv(first_time_path)
                            
                            # สร้าง customer_key สำหรับ first_time_data
                            if 'Customer name' in first_time_data.columns and 'Customer contacts' in first_time_data.columns:
                                first_time_data['customer_key'] = (
                                    first_time_data['Customer name'].astype(str).str.strip() + " | " + 
                                    first_time_data['Customer contacts'].astype(str).str.strip()
                                ).str.strip(" |")
                                
                        except FileNotFoundError:
                            st.info("ℹ️ ไม่พบไฟล์ first_time.csv - จะไม่แสดงข้อมูลวันที่มาเป็นลูกค้าครั้งแรก")
                        
                        # สร้าง customer_key list สำหรับ filter เรียงตาม Net Sales
                        if not customer_data_filtered.empty:
                            # คำนวณ Net Sales รวมของแต่ละลูกค้า
                            customer_net_sales = customer_data_filtered.groupby('customer_key')['Net sales'].sum().reset_index()
                            customer_net_sales = customer_net_sales.sort_values('Net sales', ascending=False)
                            
                            # สร้างรายการลูกค้าพร้อม Net Sales สำหรับแสดงใน selectbox
                            customer_display_list = []
                            for _, row in customer_net_sales.iterrows():
                                customer_key = row['customer_key']
                                net_sales = row['Net sales']
                                display_name = f"{customer_key} (฿{net_sales:,.2f})"
                                customer_display_list.append(display_name)
                            
                            # สร้าง mapping dictionary เพื่อแปลงกลับเป็น customer_key
                            display_to_key = {}
                            for _, row in customer_net_sales.iterrows():
                                customer_key = row['customer_key']
                                net_sales = row['Net sales']
                                display_name = f"{customer_key} (฿{net_sales:,.2f})"
                                display_to_key[display_name] = customer_key
                            
                            # Filter เลือกลูกค้า (เรียงตาม Net Sales มากไปน้อย)
                            selected_customer_display = st.selectbox(
                                "เลือกลูกค้า (เรียงตาม Net Sales มากไปน้อย):",
                                options=["เลือกลูกค้า..."] + customer_display_list,
                                key="customer_portfolio_filter"
                            )
                            
                            # แปลงกลับเป็น customer_key
                            if selected_customer_display != "เลือกลูกค้า...":
                                selected_customer = display_to_key[selected_customer_display]
                            else:
                                selected_customer = "เลือกลูกค้า..."
                            
                            if selected_customer != "เลือกลูกค้า...":
                                # กรองข้อมูลของลูกค้าที่เลือก
                                customer_sales = customer_data_filtered[customer_data_filtered['customer_key'] == selected_customer].copy()
                                
                                if not customer_sales.empty:
                                    # สร้าง Cate_brand column (เหมือนในโค้ดก่อนหน้า)
                                    if 'Brand' in customer_sales.columns and 'Category_disp' in customer_sales.columns:
                                        customer_sales['Cate_brand'] = (
                                            customer_sales['Category_disp'].astype(str).str.strip() + " [" + 
                                            customer_sales['Brand'].astype(str).str.lower() + "]"
                                        )
                                    else:
                                        customer_sales['Cate_brand'] = customer_sales.get('Category_disp', 'Unknown')
                                    
                                    # แสดงข้อมูลสรุปลูกค้า
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        # วันที่มาเป็นลูกค้าครั้งแรก
                                        first_visit = "ไม่ระบุ"
                                        if first_time_data is not None and 'customer_key' in first_time_data.columns:
                                            first_visit_row = first_time_data[first_time_data['customer_key'] == selected_customer]
                                            if not first_visit_row.empty and 'First visit' in first_visit_row.columns:
                                                first_visit = first_visit_row['First visit'].iloc[0]
                                        
                                        st.metric("📅 First Visit", str(first_visit))
                                        
                                        # Total Sales
                                        total_sales = customer_sales['Net sales'].sum()
                                        st.metric("💰 Total Sales", f"{total_sales:,.2f}")
                                    
                                    with col2:
                                        # Total Orders
                                        total_orders = customer_sales['Receipt number'].nunique() if 'Receipt number' in customer_sales.columns else len(customer_sales)
                                        st.metric("🧾 Total Orders", f"{total_orders:,}")
                                        
                                        # Total Profit
                                        total_profit = customer_sales['Gross profit'].sum() if 'Gross profit' in customer_sales.columns else 0
                                        st.metric("💵 Total Profit", f"{total_profit:,.2f}")
                                    
                                    with col3:
                                        # Avg per Bill
                                        avg_per_bill = total_sales / total_orders if total_orders > 0 else 0
                                        st.metric("📊 Avg per Bill", f"{avg_per_bill:,.2f}")
                                        
                                        # Profit Margin %
                                        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
                                        st.metric("📈 Profit Margin", f"{profit_margin:.1f}%")
                                    
                                    # Pet Type Prediction Card
                                    st.markdown("---")
                                    st.markdown("#### 🐾 Pet Type Prediction")
                                    
                                    # วิเคราะห์ Category ที่ซื้อเพื่อคาดการณ์ประเภทสัตว์เลี้ยง
                                    # ตรวจสอบคำหลัก cat และ dog ในชื่อ category
                                    
                                    # ตรวจสอบว่ามีการซื้อสินค้าที่เกี่ยวข้องกับ cat หรือ dog หรือไม่
                                    has_cat = False
                                    has_dog = False
                                    cat_count = 0
                                    dog_count = 0
                                    
                                    for category in customer_sales['Category_disp'].astype(str).str.lower():
                                        # ตรวจสอบคำ "cat" ในชื่อ category
                                        if 'cat' in category:
                                            has_cat = True
                                            cat_count += 1
                                        # ตรวจสอบคำ "dog" ในชื่อ category
                                        if 'dog' in category:
                                            has_dog = True
                                            dog_count += 1
                                    
                                    # คำนวณเปอร์เซ็นต์
                                    total_purchases = len(customer_sales)
                                    cat_percentage = (cat_count / total_purchases * 100) if total_purchases > 0 else 0
                                    dog_percentage = (dog_count / total_purchases * 100) if total_purchases > 0 else 0
                                    
                                    # กำหนดการคาดการณ์ตามลอจิกใหม่
                                    prediction = ""
                                    icon = ""
                                    
                                    if has_cat and has_dog:
                                        # ถ้าซื้อทั้ง Cat และ Dog
                                        prediction = "เลี้ยงทั้งแมว และ สุนัข"
                                        icon = "🐱🐶"
                                    elif has_cat:
                                        # ถ้าซื้อแค่ Cat
                                        prediction = "เลี้ยงแมว"
                                        icon = "�"
                                    elif has_dog:
                                        # ถ้าซื้อแค่ Dog
                                        prediction = "เลี้ยงสุนัข"
                                        icon = "🐶"
                                    else:
                                        # ไม่มีข้อมูลชัดเจน
                                        prediction = "ไม่สามารถระบุได้"
                                        icon = "❓"
                                    
                                    # แสดง Card
                                    col_pet1, col_pet2 = st.columns(2)
                                    
                                    with col_pet1:
                                        # สร้างสไตล์ตาม prediction
                                        if has_cat and has_dog:
                                            st.success(f"""
                                            **{icon} Pet Type Prediction**
                                            
                                            **คาดการณ์:** {prediction}
                                            
                                            ลูกค้าคนนี้น่าจะเลี้ยงทั้งแมวและสุนัข! 🏠
                                            """)
                                        elif has_cat:
                                            st.info(f"""
                                            **{icon} Pet Type Prediction**
                                            
                                            **คาดการณ์:** {prediction}
                                            
                                            ลูกค้าคนนี้น่าจะเป็นทาสแมว! 😸
                                            """)
                                        elif has_dog:
                                            st.info(f"""
                                            **{icon} Pet Type Prediction**
                                            
                                            **คาดการณ์:** {prediction}
                                            
                                            ลูกค้าคนนี้น่าจะรักสุนัข! 🐕
                                            """)
                                        else:
                                            st.warning(f"""
                                            **{icon} Pet Type Prediction**
                                            
                                            **คาดการณ์:** {prediction}
                                            
                                            ยังไม่มีข้อมูลที่ชัดเจน
                                            """)
                                    
                                    with col_pet2:
                                        st.info(f"""
                                        **📊 Purchase Analysis**
                                        
                                        **Cat Products:** {cat_count} รายการ ({cat_percentage:.1f}%)
                                        
                                        **Dog Products:** {dog_count} รายการ ({dog_percentage:.1f}%)
                                        
                                        **Total Items:** {total_purchases} รายการ
                                        """)
                                    
                                    # Visit Frequency Analysis
                                    st.markdown("---")
                                    st.markdown("#### 📅 Visit Frequency Analysis")
                                    
                                    # วิเคราะห์ความถี่ในการเข้าร้าน
                                    if 'Date' in customer_sales.columns and 'Receipt number' in customer_sales.columns:
                                        # หาวันที่ไม่ซ้ำกันของการซื้อแต่ละใบเสร็จ
                                        receipt_dates = customer_sales.groupby('Receipt number')['Date'].first().reset_index()
                                        receipt_dates['Date'] = pd.to_datetime(receipt_dates['Date'], errors='coerce')
                                        receipt_dates = receipt_dates.dropna(subset=['Date'])
                                        receipt_dates = receipt_dates.sort_values('Date')
                                        
                                        if len(receipt_dates) >= 2:
                                            # คำนวณระยะห่างระหว่างการเข้าร้าน (วัน)
                                            visit_gaps = []
                                            for i in range(1, len(receipt_dates)):
                                                gap = (receipt_dates.iloc[i]['Date'] - receipt_dates.iloc[i-1]['Date']).days
                                                if gap > 0:  # เฉพาะที่มีระยะห่างมากกว่า 0 วัน
                                                    visit_gaps.append(gap)
                                            
                                            if visit_gaps:
                                                # คำนวณสถิติ
                                                avg_gap = np.mean(visit_gaps)
                                                min_gap = min(visit_gaps)
                                                max_gap = max(visit_gaps)
                                                median_gap = np.median(visit_gaps)
                                                
                                                # วิเคราะห์พฤติกรรม
                                                behavior = ""
                                                frequency_icon = ""
                                                
                                                if avg_gap <= 3:
                                                    behavior = "ลูกค้าประจำ (เข้าทุก 2-3 วัน)"
                                                    frequency_icon = "⭐"
                                                elif avg_gap <= 7:
                                                    behavior = "ลูกค้าสัปดาห์ละครั้ง"
                                                    frequency_icon = "📅"
                                                elif avg_gap <= 14:
                                                    behavior = "ลูกค้า 2 สัปดาห์ครั้ง"
                                                    frequency_icon = "🗓️"
                                                elif avg_gap <= 30:
                                                    behavior = "ลูกค้าเดือนละครั้ง"
                                                    frequency_icon = "📆"
                                                elif avg_gap <= 60:
                                                    behavior = "ลูกค้า 2 เดือนครั้ง"
                                                    frequency_icon = "🕐"
                                                else:
                                                    behavior = "ลูกค้าเป็นครั้งคราว"
                                                    frequency_icon = "⏰"
                                                
                                                # แสดงผล
                                                col_freq1, col_freq2 = st.columns(2)
                                                
                                                with col_freq1:
                                                    st.success(f"""
                                                    **{frequency_icon} Visit Pattern**
                                                    
                                                    **พฤติกรรม:** {behavior}
                                                    
                                                    **ระยะห่างเฉลี่ย:** {avg_gap:.1f} วัน
                                                    
                                                    **Total Visits:** {len(receipt_dates)} ครั้ง
                                                    """)
                                                
                                                with col_freq2:
                                                    st.info(f"""
                                                    **📊 Visit Statistics**
                                                    
                                                    **ระยะห่างน้อยที่สุด:** {min_gap} วัน
                                                    
                                                    **ระยะห่างมากที่สุด:** {max_gap} วัน
                                                    
                                                    **ระยะห่าง Median:** {median_gap:.1f} วัน
                                                    """)
                                                
                                                # แสดงกราฟแสดงความถี่การเข้าร้าน
                                                if len(visit_gaps) > 1:
                                                    st.markdown("**📈 Visit Gap Distribution**")
                                                    
                                                    # สร้างข้อมูลสำหรับกราฟ
                                                    gap_df = pd.DataFrame({'Visit_Gap': visit_gaps})
                                                    
                                                    # Histogram
                                                    hist_chart = alt.Chart(gap_df).mark_bar(
                                                        color='steelblue',
                                                        opacity=0.7
                                                    ).encode(
                                                        alt.X('Visit_Gap:Q', bin=alt.Bin(maxbins=15), title='ระยะห่างการเข้าร้าน (วัน)'),
                                                        alt.Y('count():Q', title='จำนวนครั้ง'),
                                                        tooltip=['count():Q']
                                                    ).properties(
                                                        width=400,
                                                        height=200,
                                                        title="การกระจายของระยะห่างการเข้าร้าน"
                                                    )
                                                    
                                                    st.altair_chart(hist_chart, use_container_width=True)
                                            
                                            else:
                                                st.warning("⚠️ ไม่สามารถคำนวณระยะห่างการเข้าร้านได้ (วันที่ซื้อซ้ำกัน)")
                                        
                                        elif len(receipt_dates) == 1:
                                            st.info("🔔 ลูกค้าใหม่ - มีการซื้อเพียงครั้งเดียว")
                                        
                                        else:
                                            st.warning("❌ ไม่มีข้อมูลการซื้อที่สามารถวิเคราะห์ได้")
                                    
                                    else:
                                        st.warning("❌ ไม่มีข้อมูล Date หรือ Receipt number สำหรับการวิเคราะห์")
                                    
                                    # สัดส่วน Category เป็น Pie Chart
                                    st.markdown("#### 🥧 Category Distribution")
                                    
                                    category_dist = customer_sales.groupby('Category_disp')['Net sales'].sum().reset_index()
                                    category_dist = category_dist.sort_values('Net sales', ascending=False)
                                    category_dist['Percentage'] = (category_dist['Net sales'] / category_dist['Net sales'].sum() * 100).round(2)
                                    
                                    if not category_dist.empty:
                                        # Pie chart
                                        pie_chart = alt.Chart(category_dist).mark_arc(
                                            innerRadius=40,
                                            outerRadius=100,
                                            stroke='white',
                                            strokeWidth=2
                                        ).encode(
                                            theta=alt.Theta('Net sales:Q', sort=alt.Sort(field='Net sales', order='descending')),
                                            color=alt.Color('Category_disp:N', scale=alt.Scale(scheme='category20')),
                                            tooltip=[
                                                alt.Tooltip('Category_disp:N', title='Category'),
                                                alt.Tooltip('Net sales:Q', title='Sales', format=',.2f'),
                                                alt.Tooltip('Percentage:Q', title='%', format='.1f')
                                            ]
                                        ).properties(
                                            width=300,
                                            height=300,
                                            title=f"Category Distribution - {selected_customer.split(' | ')[0]}"
                                        )
                                        
                                        st.altair_chart(pie_chart, use_container_width=True)
                                    
                                    # ตาราง Purchase Detail (Pivot Format)
                                    st.markdown("#### 📋 Purchase Detail")
                                    
                                    # เตรียมข้อมูลสำหรับตาราง pivot
                                    if 'Date' in customer_sales.columns and 'Category_disp' in customer_sales.columns:
                                        # สร้างข้อมูลสำหรับ pivot
                                        pivot_data = customer_sales.copy()
                                        
                                        # แปลง Date และสร้าง Date string
                                        pivot_data['Date'] = pd.to_datetime(pivot_data['Date'], errors='coerce')
                                        pivot_data = pivot_data.dropna(subset=['Date'])
                                        pivot_data['Date_str'] = pivot_data['Date'].dt.strftime('%Y-%m-%d')
                                        
                                        # เตรียมข้อมูลสำหรับแสดงในตาราง (ไม่รวม Category column)
                                        detail_cols = ['Date_str', 'Cate_brand', 'Item', 'Net sales']
                                        available_cols = [col for col in detail_cols if col in pivot_data.columns]
                                        
                                        if available_cols:
                                            # สร้างตารางที่เรียงลำดับ
                                            purchase_detail = pivot_data[available_cols].copy()
                                            # เรียงตามวันที่และ Cate_brand แทน Category_disp
                                            if 'Cate_brand' in purchase_detail.columns:
                                                purchase_detail = purchase_detail.sort_values(['Date_str', 'Cate_brand', 'Item'], ascending=[False, True, True])
                                            else:
                                                purchase_detail = purchase_detail.sort_values(['Date_str', 'Item'], ascending=[False, True])
                                            
                                            # สร้าง Display Date column ที่จะแสดงเฉพาะแถวแรกของแต่ละวัน
                                            purchase_detail['Display_Date'] = purchase_detail['Date_str']
                                            
                                            # ทำให้ Date แสดงเฉพาะแถวแรกของแต่ละวัน (เลียนแบบ merge cells)
                                            prev_date = None
                                            for i, row in purchase_detail.iterrows():
                                                current_date = row['Date_str']
                                                if current_date == prev_date:
                                                    purchase_detail.at[i, 'Display_Date'] = ''  # แสดงเป็นช่องว่าง
                                                prev_date = current_date
                                            
                                            # จัดเรียงคอลัมน์สำหรับแสดงผล (ไม่รวม Category)
                                            display_columns = ['Display_Date', 'Cate_brand', 'Item', 'Net sales']
                                            final_display_cols = [col for col in display_columns if col in purchase_detail.columns]
                                            
                                            # เปลี่ยนชื่อคอลัมน์สำหรับแสดงผล
                                            column_mapping = {
                                                'Display_Date': 'Date',
                                                'Cate_brand': 'Category+Brand',
                                                'Item': 'Item',
                                                'Net sales': 'Net Sales'
                                            }
                                            
                                            display_table = purchase_detail[final_display_cols].copy()
                                            display_table.columns = [column_mapping.get(col, col) for col in final_display_cols]
                                            
                                            # สร้าง custom CSS สำหรับ styling
                                            def style_dataframe(df):
                                                # สร้าง styler object
                                                styler = df.style
                                                
                                                # Format ตัวเลข
                                                if 'Net Sales' in df.columns:
                                                    styler = styler.format({'Net Sales': '{:,.2f}'})
                                                
                                                # เพิ่ม CSS สำหรับ border และ styling
                                                styler = styler.set_table_styles([
                                                    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]},
                                                    {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('padding', '8px'), ('background-color', '#f2f2f2'), ('text-align', 'center')]},
                                                    {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('padding', '8px'), ('text-align', 'left')]},
                                                    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
                                                ])
                                                
                                                return styler
                                            
                                            # แสดงตาราง
                                            st.dataframe(style_dataframe(display_table), use_container_width=True, hide_index=True)
                                            
                                            # เพิ่มคำอธิบายใต้ตาราง
                                            st.caption("💡 วันที่เดียวกันจะแสดงเฉพาะในแถวแรก (เลียนแบบ merged cells)")
                                        
                                        # Download button (ไม่รวม Category column)
                                        csv_cols = ['Date_str', 'Cate_brand', 'Item', 'Net sales']
                                        available_csv_cols = [col for col in csv_cols if col in purchase_detail.columns]
                                        csv_data = purchase_detail[available_csv_cols].copy()
                                        
                                        # เปลี่ยนชื่อคอลัมน์สำหรับ CSV
                                        csv_column_mapping = {
                                            'Date_str': 'Date',
                                            'Cate_brand': 'Category+Brand', 
                                            'Item': 'Item', 
                                            'Net sales': 'Net Sales'
                                        }
                                        csv_data.columns = [csv_column_mapping.get(col, col) for col in available_csv_cols]
                                        csv_download = csv_data.to_csv(index=False)
                                        
                                        st.download_button(
                                            label="📥 Download Purchase Data",
                                            data=csv_download,
                                            file_name=f"customer_detail_{selected_customer.split(' | ')[0]}.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.info("ไม่พบข้อมูลรายละเอียดการซื้อ")
                                
                                # Product Recommendation Section
                                st.markdown("---")
                                st.markdown("#### 💡 Product Recommendations")
                                
                                # ดึงข้อมูล Category ทั้งหมดจากระบบ
                                all_categories = [
                                    'CAT Food', 'CAT Litter', 'CAT Pouch / Wet Food', 'CAT Snack', 'CAT Toys&Tools&Supplies',
                                    'DOG Food', 'DOG Pouch / Wet Food', 'DOG Snack', 'DOG Toys&Tools&Supplies',
                                    'FOOD & SNACK หมา+แมว', 'TOYS & TOOLS หมา+แมว', 'ฟันแทะ'
                                ]
                                
                                # แยก Categories ตาม Pet Type
                                cat_categories = [cat for cat in all_categories if 'CAT' in cat or 'แมว' in cat]
                                dog_categories = [cat for cat in all_categories if 'DOG' in cat or 'หมา' in cat]
                                both_categories = [cat for cat in all_categories if 'หมา+แมว' in cat or 'ฟันแทะ' in cat]
                                
                                # หา Categories ที่ลูกค้าเคยซื้อ
                                customer_categories = set(customer_sales['Category_disp'].unique()) if not customer_sales.empty else set()
                                
                                # แนะนำตามการคาดการณ์ Pet Type
                                recommendations = []
                                
                                if has_cat and has_dog:
                                    # เลี้ยงทั้งคู่ - แนะนำทั้ง Cat, Dog และ Both categories
                                    recommend_cats = [cat for cat in cat_categories if cat not in customer_categories]
                                    recommend_dogs = [dog for dog in dog_categories if dog not in customer_categories]
                                    recommend_both = [both for both in both_categories if both not in customer_categories]
                                    
                                    if recommend_cats:
                                        recommendations.extend([("🐱 Cat Products", recommend_cats)])
                                    if recommend_dogs:
                                        recommendations.extend([("🐶 Dog Products", recommend_dogs)])
                                    if recommend_both:
                                        recommendations.extend([("🐱🐶 Universal Products", recommend_both)])
                                        
                                elif has_cat:
                                    # เลี้ยงแค่แมว - แนะนำ Cat categories และ Both categories
                                    recommend_cats = [cat for cat in cat_categories if cat not in customer_categories]
                                    recommend_both = [both for both in both_categories if both not in customer_categories]
                                    
                                    if recommend_cats:
                                        recommendations.extend([("🐱 Cat Products", recommend_cats)])
                                    if recommend_both:
                                        recommendations.extend([("🐾 Universal Products", recommend_both)])
                                        
                                elif has_dog:
                                    # เลี้ยงแค่สุนัข - แนะนำ Dog categories และ Both categories
                                    recommend_dogs = [dog for dog in dog_categories if dog not in customer_categories]
                                    recommend_both = [both for both in both_categories if both not in customer_categories]
                                    
                                    if recommend_dogs:
                                        recommendations.extend([("🐶 Dog Products", recommend_dogs)])
                                    if recommend_both:
                                        recommendations.extend([("🐾 Universal Products", recommend_both)])
                                        
                                else:
                                    # ไม่ทราบประเภท - แนะนำทุกอย่างที่ยังไม่เคยซื้อ
                                    all_recommend = [cat for cat in all_categories if cat not in customer_categories]
                                    if all_recommend:
                                        recommendations.extend([("🛍️ Suggested Products", all_recommend)])
                                
                                # แสดงผล Recommendations
                                if recommendations:
                                    col_rec1, col_rec2 = st.columns(2)
                                    
                                    with col_rec1:
                                        st.success(f"""
                                        **🎯 Recommendation Strategy**
                                        
                                        **Based on:** {prediction}
                                        
                                        **ลูกค้าคนนี้น่าจะสนใจสินค้าที่ยังไม่เคยซื้อ**
                                        
                                        **Categories ที่เคยซื้อ:** {len(customer_categories)} ประเภท
                                        """)
                                    
                                    with col_rec2:
                                        # แสดงสถิติการแนะนำ
                                        total_recommends = sum(len(cats) for _, cats in recommendations)
                                        st.info(f"""
                                        **📊 Recommendation Stats**
                                        
                                        **Categories ที่แนะนำ:** {total_recommends} ประเภท
                                        
                                        **Recommendation Types:** {len(recommendations)} กลุ่ม
                                        
                                        **Potential for Growth:** {'สูง' if total_recommends > 3 else 'ปานกลาง' if total_recommends > 0 else 'จำกัด'}
                                        """)
                                    
                                    # แสดง Recommendations แยกตามกลุ่ม
                                    for rec_type, rec_categories in recommendations:
                                        if rec_categories:
                                            st.markdown(f"**{rec_type}:**")
                                            
                                            # แสดงเป็นแท็ก
                                            cols = st.columns(min(len(rec_categories), 3))
                                            for i, category in enumerate(rec_categories[:9]):  # จำกัดไม่เกิน 9 รายการ
                                                with cols[i % 3]:
                                                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 5px; margin: 2px; border-radius: 5px; text-align: center; font-size: 0.9em;'>{category}</div>", 
                                                              unsafe_allow_html=True)
                                            
                                            if len(rec_categories) > 9:
                                                st.caption(f"และอีก {len(rec_categories) - 9} ประเภท...")
                                            st.markdown("")
                                
                                else:
                                    st.info("🎉 ลูกค้าคนนี้ได้ลองสินค้าครบทุกประเภทแล้ว!")
                            else:
                                st.info("📋 กรุณาเลือกลูกค้าเพื่อดู Customer Portfolio")
                        else:
                            st.info("ไม่มีข้อมูลลูกค้าสำหรับแสดง Portfolio")
                            
                    except Exception as e:
                        st.error(f"❌ Error in Customer Portfolio: {str(e)}")
                        
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ลูกค้า: {str(e)}")
                    st.info("กรุณาตรวจสอบรูปแบบไฟล์และคอลัมน์ที่จำเป็น")

        # -------------------- TAB 5: Promotion Recommendations --------------------
        with tab_promotion:
            st.header("🎁 Promotion Recommendations")
            st.markdown("แนะนำโปรโมชันสำหรับสินค้าที่ค้างสต็อกและขายไม่ออก")
            
            try:
                # ตรวจสอบข้อมูลที่จำเป็น
                if sales_f is None or stock is None:
                    st.warning("⚠️ ต้องการข้อมูลทั้ง Sales และ Inventory สำหรับการวิเคราะห์โปรโมชัน")
                    st.info("กรุณาอัปโหลดไฟล์ Sales by item และ Inventory แล้วกด Run Analysis")
                else:
                    # เตรียมข้อมูล
                    sales_data = sales_f.copy()
                    inventory_data = stock.copy()
                    
                    # แปลงวันที่
                    if 'Date' in sales_data.columns:
                        sales_data['Date'] = pd.to_datetime(sales_data['Date'], errors='coerce')
                        current_date = sales_data['Date'].max()
                        if pd.isna(current_date):
                            current_date = pd.Timestamp.now()
                    else:
                        current_date = pd.Timestamp.now()
                    
                    # คำนวณวันที่ขายครั้งล่าสุดของแต่ละ SKU
                    if 'SKU' in sales_data.columns and 'Date' in sales_data.columns:
                        last_sale_date = sales_data.groupby('SKU')['Date'].max().reset_index()
                        last_sale_date.columns = ['SKU', 'Last_Sale_Date']
                        last_sale_date['Days_Since_Last_Sale'] = (current_date - last_sale_date['Last_Sale_Date']).dt.days
                    else:
                        st.error("❌ ไม่พบคอลัมน์ SKU หรือ Date ในข้อมูล Sales")
                        st.stop()
                    
                    # รวมข้อมูล Inventory กับ Last Sale Date
                    if 'SKU' in inventory_data.columns:
                        # หา stock column
                        stock_col = None
                        for col in inventory_data.columns:
                            if 'stock' in col.lower() and 'i-animal' in col.lower():
                                stock_col = col
                                break
                        
                        if stock_col is None:
                            st.error("❌ ไม่พบคอลัมน์ Stock ในข้อมูล Inventory")
                            st.stop()
                        
                        # เตรียมข้อมูลสำหรับวิเคราะห์
                        promo_analysis = inventory_data[['SKU', 'Name', 'Category', stock_col, 'Price [I-animal]']].copy()
                        promo_analysis.columns = ['SKU', 'Name', 'Category', 'Stock', 'Price']
                        
                        # แปลง Stock เป็นตัวเลข
                        promo_analysis['Stock'] = pd.to_numeric(promo_analysis['Stock'], errors='coerce').fillna(0)
                        promo_analysis['Price'] = pd.to_numeric(promo_analysis['Price'], errors='coerce').fillna(0)
                        
                        # กรองเฉพาะสินค้าที่มี Stock > 0
                        promo_analysis = promo_analysis[promo_analysis['Stock'] > 0]
                        
                        # รวมกับข้อมูลการขายครั้งล่าสุด
                        promo_analysis = promo_analysis.merge(last_sale_date, on='SKU', how='left')
                        
                        # สินค้าที่ไม่มีประวัติการขายเลย
                        promo_analysis['Days_Since_Last_Sale'] = promo_analysis['Days_Since_Last_Sale'].fillna(999)
                        
                        # สร้าง Filter
                        st.markdown("### 🔍 Filter ตามระยะเวลาที่ไม่มีการขาย")
                        
                        col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
                        
                        with col_filter1:
                            filter_60_120 = st.checkbox("60-120 วัน", value=True)
                        with col_filter2:
                            filter_120_180 = st.checkbox("120-180 วัน", value=True)
                        with col_filter3:
                            filter_180_365 = st.checkbox("180 วัน - 1 ปี", value=True)
                        with col_filter4:
                            filter_over_365 = st.checkbox("1 ปีขึ้นไป", value=True)
                        
                        # กรองข้อมูลตาม Filter
                        filtered_data = pd.DataFrame()
                        
                        if filter_60_120:
                            data_60_120 = promo_analysis[(promo_analysis['Days_Since_Last_Sale'] >= 60) & 
                                                        (promo_analysis['Days_Since_Last_Sale'] < 120)].copy()
                            data_60_120['Category_Filter'] = '60-120 วัน'
                            filtered_data = pd.concat([filtered_data, data_60_120], ignore_index=True)
                        
                        if filter_120_180:
                            data_120_180 = promo_analysis[(promo_analysis['Days_Since_Last_Sale'] >= 120) & 
                                                         (promo_analysis['Days_Since_Last_Sale'] < 180)].copy()
                            data_120_180['Category_Filter'] = '120-180 วัน'
                            filtered_data = pd.concat([filtered_data, data_120_180], ignore_index=True)
                        
                        if filter_180_365:
                            data_180_365 = promo_analysis[(promo_analysis['Days_Since_Last_Sale'] >= 180) & 
                                                         (promo_analysis['Days_Since_Last_Sale'] < 365)].copy()
                            data_180_365['Category_Filter'] = '180 วัน - 1 ปี'
                            filtered_data = pd.concat([filtered_data, data_180_365], ignore_index=True)
                        
                        if filter_over_365:
                            data_over_365 = promo_analysis[promo_analysis['Days_Since_Last_Sale'] >= 365].copy()
                            data_over_365['Category_Filter'] = '1 ปีขึ้นไป'
                            filtered_data = pd.concat([filtered_data, data_over_365], ignore_index=True)
                        
                        if not filtered_data.empty:
                            # สถิติรวม
                            st.markdown("### 📊 สถิติสินค้าที่แนะนำให้ทำโปรโมชัน")
                            
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            
                            with col_stat1:
                                total_items = len(filtered_data)
                                st.metric("🛍️ รายการสินค้า", f"{total_items:,}")
                            
                            with col_stat2:
                                total_stock = filtered_data['Stock'].sum()
                                st.metric("📦 รวม Stock", f"{total_stock:,.0f}")
                            
                            with col_stat3:
                                total_value = (filtered_data['Stock'] * filtered_data['Price']).sum()
                                st.metric("💰 มูลค่ารวม", f"{total_value:,.0f} บาท")
                            
                            with col_stat4:
                                avg_days = filtered_data['Days_Since_Last_Sale'].mean()
                                st.metric("📅 เฉลี่ยวันที่ไม่ขาย", f"{avg_days:.0f} วัน")
                            
                            # แสดงข้อมูลตาม Category
                            st.markdown("### 🏷️ แยกตาม Category")
                            
                            category_summary = filtered_data.groupby('Category').agg({
                                'SKU': 'count',
                                'Stock': 'sum',
                                'Price': 'mean',
                                'Days_Since_Last_Sale': 'mean'
                            }).round(2)
                            
                            category_summary.columns = ['จำนวนรายการ', 'รวม Stock', 'ราคาเฉลี่ย', 'เฉลี่ยวันไม่ขาย']
                            category_summary['มูลค่ารวม'] = (category_summary['รวม Stock'] * category_summary['ราคาเฉลี่ย']).round(0)
                            
                            st.dataframe(category_summary, use_container_width=True)
                            
                            # แสดงรายละเอียดสินค้า
                            st.markdown("### 📋 รายละเอียดสินค้าที่แนะนำ")
                            
                            # เรียงตาม Days_Since_Last_Sale (มากสุดก่อน)
                            display_data = filtered_data.sort_values('Days_Since_Last_Sale', ascending=False)
                            
                            # คำนวณคำแนะนำโปรโมชัน
                            def calculate_promo_suggestion(days_since_sale, stock, price):
                                if days_since_sale >= 365:
                                    return "ลด 30-50% หรือ Bundle"
                                elif days_since_sale >= 180:
                                    return "ลด 20-30% หรือ Buy 1 Get 1"
                                elif days_since_sale >= 120:
                                    return "ลด 15-25%"
                                else:
                                    return "ลด 10-15%"
                            
                            display_data['คำแนะนำโปรโมชัน'] = display_data.apply(
                                lambda row: calculate_promo_suggestion(row['Days_Since_Last_Sale'], row['Stock'], row['Price']), 
                                axis=1
                            )
                            
                            # คำนวณราคาที่แนะนำ
                            def calculate_suggested_price(days_since_sale, price):
                                if days_since_sale >= 365:
                                    return price * 0.6  # ลด 40%
                                elif days_since_sale >= 180:
                                    return price * 0.75  # ลด 25%
                                elif days_since_sale >= 120:
                                    return price * 0.8   # ลด 20%
                                else:
                                    return price * 0.85  # ลด 15%
                            
                            display_data['ราคาที่แนะนำ'] = display_data.apply(
                                lambda row: calculate_suggested_price(row['Days_Since_Last_Sale'], row['Price']), 
                                axis=1
                            ).round(0)
                            
                            # แสดงตาราง
                            display_columns = ['SKU', 'Name', 'Category', 'Stock', 'Price', 'ราคาที่แนะนำ', 
                                             'Days_Since_Last_Sale', 'Category_Filter', 'คำแนะนำโปรโมชัน']
                            
                            column_config = {
                                'SKU': 'SKU',
                                'Name': 'ชื่อสินค้า',
                                'Category': 'หมวดหมู่',
                                'Stock': st.column_config.NumberColumn('Stock', format='%d'),
                                'Price': st.column_config.NumberColumn('ราคาปัจจุบัน', format='%.0f'),
                                'ราคาที่แนะนำ': st.column_config.NumberColumn('ราคาที่แนะนำ', format='%.0f'),
                                'Days_Since_Last_Sale': st.column_config.NumberColumn('วันที่ไม่ขาย', format='%d'),
                                'Category_Filter': 'ช่วงเวลา',
                                'คำแนะนำโปรโมชัน': 'คำแนะนำ'
                            }
                            
                            st.dataframe(
                                display_data[display_columns], 
                                column_config=column_config,
                                use_container_width=True,
                                height=400
                            )
                            
                            # Download CSV
                            csv_data = display_data[display_columns].to_csv(index=False)
                            st.download_button(
                                label="📥 Download Promotion Data",
                                data=csv_data,
                                file_name=f"promotion_recommendations_{current_date.strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                            
                        else:
                            st.info("ไม่พบสินค้าที่ตรงกับเงื่อนไขที่เลือก")
                    
                    else:
                        st.error("❌ ไม่พบคอลัมน์ SKU ในข้อมูล Inventory")
                        
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์โปรโมชัน: {str(e)}")
                st.exception(e)

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        st.exception(e)
else:
    st.info("⬆️ อัปโหลดไฟล์ Sales และ Inventory แล้วกด **Run Analysis** เพื่อเริ่มการคำนวณ")
