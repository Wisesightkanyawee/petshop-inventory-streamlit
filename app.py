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
    st.caption("Inventory columns expected: **SKU, In stock [I-animal], Cost**")

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
    """Return a Styler with thousands separators."""
    fmt_map = {c: "{:,.0f}" for c in int_cols}
    fmt_map.update({c: "{:,.2f}" for c in float_cols})
    return df.style.format(fmt_map)

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
        req_stock = {"SKU", "In stock [I-animal]", "Cost"}
        missing_stock = req_stock - set(stock.columns)
        if missing_stock:
            st.error("❌ Inventory file missing columns: " + ", ".join(sorted(missing_stock)))
            st.stop()

        # ----- Normalize keys/types -----
        sales["SKU"] = norm_sku(sales["SKU"])
        stock["SKU"] = norm_sku(stock["SKU"])
        stock = stock.rename(columns={"In stock [I-animal]": "คงเหลือ", "Cost": "ต้นทุนเฉลี่ย/ชิ้น"})
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

        # =============== TABS: Inventory & Reorder | Sales Analysis ===============
        tab_inv, tab_sales = st.tabs(["📦 Inventory & Reorder", "📊 Sales Analysis"])

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
                st.dataframe(
                    fmt_commas(
                        filtered[show_cols],
                        int_cols=["Quantity", "ควรสั่งซื้อเพิ่ม (ชิ้น)"],
                        float_cols=["Net_sales", "Opp. Loss (Baht)", "ต้นทุนเฉลี่ย/ชิ้น", "กำไรเฉลี่ย/ชิ้น", "RU Score"],
                    ),
                    use_container_width=True,
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
                    st.dataframe(
                        fmt_commas(top_sales, int_cols=["Quantity"], float_cols=["Net_sales"]),
                        use_container_width=True,
                    )
                with c2:
                    st.write(f"💵 Top {show_top_n} SKUs by **Gross profit**")
                    top_profit = sku_agg.nlargest(show_top_n, "Gross_profit")[["Label","Category_disp","Gross_profit","Quantity"]]
                    st.dataframe(
                        fmt_commas(top_profit, int_cols=["Quantity"], float_cols=["Gross_profit"]),
                        use_container_width=True,
                    )


                c3, c4 = st.columns(2)
                with c3:
                    st.write(f"🐢 Slow Movers (Bottom {show_top_n} by Quantity)")
                    slow = sku_agg.nsmallest(show_top_n, "Quantity")[["Label","Category_disp","Quantity","Net_sales"]]
                    st.dataframe(
                        fmt_commas(slow, int_cols=["Quantity"], float_cols=["Net_sales"]),
                        use_container_width=True,
                    )

                with c4:
                    st.write("📦 ยอดขายตาม Category")
                    cat_agg = (sales_f.groupby("Category_disp", as_index=False)
                               .agg(Net_sales=("Net sales","sum"),
                                    Gross_profit=("Gross profit","sum"),
                                    Quantity=("Quantity","sum")))
                    st.dataframe(
                        fmt_commas(cat_agg, int_cols=["Quantity"], float_cols=["Net_sales", "Gross_profit"]),
                        use_container_width=True,
                    )


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
                st.dataframe(
                    fmt_commas(
                        contrib[["Label","Category_disp","Gross_profit","Net_sales","Quantity"]],
                        int_cols=["Quantity"],
                        float_cols=["Gross_profit","Net_sales"],
                    ),
                    use_container_width=True,
                )


                # ===== 4) Customer Behavior =====
                st.markdown("### 4) Customer Behavior")
                cust_ready = {"Customer name","Customer contacts"}.issubset(sales_f.columns)
                # สร้าง customer_id แม้บางค่าเป็น null
                if "Customer name" in sales_f.columns or "Customer contacts" in sales_f.columns:
                    sales_f["Customer name"]    = sales_f.get("Customer name", "").astype(str)
                    sales_f["Customer contacts"] = sales_f.get("Customer contacts", "").astype(str)
                    sales_f["customer_id"] = (sales_f["Customer name"].str.strip() + " | " +
                                              sales_f["Customer contacts"].str.strip()).str.strip(" |")
                else:
                    sales_f["customer_id"] = np.nan

                # Repeat vs New
                if sales_f["customer_id"].notna().any():
                    first_date = (sales_f.sort_values("Date")
                                  .groupby("customer_id", as_index=False)["Date"].min()
                                  .rename(columns={"Date":"first_buy"}))
                    joined = sales_f.merge(first_date, on="customer_id", how="left")
                    joined["is_new"] = joined["Date"].dt.date == joined["first_buy"].dt.date
                    cust_counts = joined.groupby("customer_id").agg(
                        first_buy=("first_buy","min"),
                        orders=("customer_id","count"),
                        total_spent=("Net sales","sum")
                    ).reset_index()
                    cust_counts["type"] = np.where(cust_counts["orders"]>1, "Repeat", "New")
                    total_cust = cust_counts["customer_id"].nunique()
                    new_pct    = (cust_counts["type"].eq("New").mean()*100) if total_cust>0 else 0
                    rep_pct    = 100 - new_pct
                    cR1, cR2, cR3 = st.columns(3)
                    cR1.metric("ลูกค้ารวม (unique)", f"{total_cust:,}")
                    cR2.metric("New (%)", f"{new_pct:,.1f}%")
                    cR3.metric("Repeat (%)", f"{rep_pct:,.1f}%")
                else:
                    st.info("ℹ️ ไม่มีข้อมูลลูกค้า (Customer name/contacts) เพียงพอสำหรับ Repeat vs New")

                # Average Basket Size
                # ถ้ามี Receipt number ใช้อันนั้นเป็นบิล; ถ้าไม่มีก็ group โดย (Date, customer_id) เป็น proxy
                if "Receipt number" in sales_f.columns:
                    orders = (sales_f.groupby("Receipt number", as_index=False)
                                      .agg(order_value=("Net sales","sum")))
                elif sales_f["customer_id"].notna().any():
                    orders = (sales_f.groupby(["customer_id", sales_f["Date"].dt.date], as_index=False)
                                      .agg(order_value=("Net sales","sum")))
                else:
                    orders = (sales_f.groupby(sales_f["Date"].dt.date, as_index=False)
                                      .agg(order_value=("Net sales","sum")))
                avg_basket = orders["order_value"].mean() if not orders.empty else 0.0
                st.metric("🛒 Average Basket Size (บาท/บิล)", f"{avg_basket:,.2f}")

                # Interpurchase Time (IPT)
                if sales_f["customer_id"].notna().any():
                    ipt_list = []
                    for cid, g in sales_f.groupby("customer_id"):
                        ds = g["Date"].sort_values().drop_duplicates().to_list()
                        if len(ds) >= 2:
                            diffs = np.diff(pd.to_datetime(ds)).astype("timedelta64[D]").astype(int)
                            if len(diffs)>0:
                                ipt_list.extend(diffs)
                    if len(ipt_list) > 0:
                        ipt_ser = pd.Series(ipt_list)
                        st.write(f"📅 Interpurchase Time (days) — mean: **{ipt_ser.mean():.1f}** | median: **{ipt_ser.median():.0f}**")
                        ipt_df = pd.DataFrame({"IPT_days": ipt_ser})
                        hist = alt.Chart(ipt_df).mark_bar().encode(
                            x=alt.X("IPT_days:Q", bin=alt.Bin(maxbins=30), title="Days between purchases"),
                            y=alt.Y("count():Q", title="Count")
                        ).properties(height=250)
                        st.altair_chart(hist, use_container_width=True)

                        # ===== Customer-level IPT summary & items =====
                        st.markdown("#### 👥 Interpurchase Summary by Customer")

                        # เฉพาะเคสที่มี customer_id
                        if sales_f["customer_id"].notna().any():
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

                            cust_stats = (sales_f.groupby("customer_id").apply(_ipt_stats).reset_index())

                            # 3) Top 10 รายการต่อหัวลูกค้า (ชื่อสินค้า — มูลค่า — % ของ Total_spent)
                            top_items = (
                                sales_f.groupby(["customer_id", "item_label"], as_index=False)
                                    .agg(spent=("Net sales","sum"))
                            ).merge(
                                cust_stats[["customer_id","Total_spent"]], on="customer_id", how="left"
                            )

                            top_items["pct"] = np.where(
                                top_items["Total_spent"] > 0,
                                100 * top_items["spent"] / top_items["Total_spent"],
                                0
                            )

                            # เลือก top 10 ต่อ customer และรวมเป็นข้อความหลายบรรทัด
                            top_items = (
                                top_items.sort_values(["customer_id","spent"], ascending=[True, False])
                                        .groupby("customer_id")
                                        .head(10)
                            )
                            top_items["detail"] = top_items.apply(
                                lambda r: f"{r['item_label']} — {r['spent']:,.0f}฿ ({r['pct']:.1f}%)", axis=1
                            )
                            items_fmt = (
                                top_items.groupby("customer_id")["detail"]
                                        .apply(lambda s: "\n".join(s))
                                        .reset_index(name="Top 10 purchases")
                            )

                            # 4) รวมกลับและเรียงลำดับ
                            cust_stats = (cust_stats
                                        .merge(items_fmt, on="customer_id", how="left")
                                        .sort_values(["IPT_count","orders","Total_spent"],
                                                    ascending=[False, False, False]))

                            # 5) แสดงผล (เหลือคอลัมน์ใหม่เดียว)
                            cols = [
                                "customer_id","orders","IPT_count","IPT_mean","IPT_median",
                                "Quantity","Total_spent","Last_purchase","Top 10 purchases"
                            ]
                            st.dataframe(
                                fmt_commas(
                                    cust_stats[cols],
                                    int_cols=["orders","IPT_count","Quantity"],
                                    float_cols=["IPT_mean","IPT_median","Total_spent"],
                                ),
                                use_container_width=True,
                            )

                            st.caption("หมายเหตุ: IPT_count คือจำนวน 'ช่วงเวลาระหว่างการซื้อ' ต่อหัวลูกค้า (ไม่ใช่จำนวนลูกค้า)")


                    else:
                        st.info("ℹ️ ยังไม่พอสำหรับคำนวณ IPT (ลูกค้าส่วนใหญ่ซื้อครั้งเดียว)")
                else:
                    st.info("ℹ️ ไม่มี customer_id สำหรับคำนวณ IPT")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        st.exception(e)
else:
    st.info("⬆️ อัปโหลดไฟล์ Sales และ Inventory แล้วกด **Run Analysis** เพื่อเริ่มการคำนวณ")
