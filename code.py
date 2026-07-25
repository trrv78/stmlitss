import streamlit as st
import pandas as pd
from collections import defaultdict
import io
from datetime import datetime


st.set_page_config(page_title="SDA", layout="centered")

st.markdown("""
<style>
    .main-header { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem; 
    }
    .main-header h1 { color: #e2e8f0; font-size: 1.9rem; font-weight: 700; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Audit Participation Report</h1>
    <p>Excel format matching your sample image</p>
</div>
""", unsafe_allow_html=True)


def parse_names(cell_value):
    if pd.isna(cell_value) or str(cell_value).strip() == "":
        return []
    return [n.strip() for n in str(cell_value).strip().split(",") if n.strip()]


def process_csv(df):
    person_data = defaultdict(list)
    vehicle_data = defaultdict(list)

    for _, row in df.iterrows():
        date_val = str(row.get("Date", "")).strip()
        company_val = str(row.get("Company", "")).strip()
        location_val = str(row.get("Location", "")).strip()

        for name in parse_names(row.get("Lead Auditor", "")):
            person_data[name].append({"company": company_val, "date": date_val,
                                      "location": location_val, "role": "Lead Auditor"})
        for name in parse_names(row.get("Auditor", "")):
            person_data[name].append({"company": company_val, "date": date_val,
                                      "location": location_val, "role": "Auditor"})
        for name in parse_names(row.get("Evaluator", "")):
            person_data[name].append({"company": company_val, "date": date_val,
                                      "location": location_val, "role": "Evaluator"})

        for vehicle in parse_names(row.get("Vehicle", "")):
            vehicle_data[vehicle].append({"company": company_val, "date": date_val,
                                          "location": location_val})

    for name in person_data:
        person_data[name].sort(key=lambda x: x["date"])
    for vehicle in vehicle_data:
        vehicle_data[vehicle].sort(key=lambda x: x["date"])

    return person_data, vehicle_data


def group_by_date(entries):
    """Group entries by date, merging companies and locations with commas."""
    date_groups = defaultdict(lambda: {"companies": [], "locations": []})

    for entry in entries:
        date = entry["date"]
        date_groups[date]["companies"].append(entry["company"])
        date_groups[date]["locations"].append(entry["location"])

    grouped = []
    for date in sorted(date_groups.keys()):
        companies = date_groups[date]["companies"]
        locations = date_groups[date]["locations"]

        # Deduplicate while preserving order
        seen_c = []
        for c in companies:
            if c not in seen_c:
                seen_c.append(c)

        seen_l = []
        for l in locations:
            if l not in seen_l:
                seen_l.append(l)

        grouped.append({
            "date": date,
            "company": ", ".join(seen_c),
            "location": ", ".join(seen_l)
        })

    return grouped


def create_excel(person_data, vehicle_data):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        # --- Sheet 1: People ---
        rows = []
        for idx, (name, audits) in enumerate(person_data.items(), start=1):
            grouped = group_by_date(audits)
            if grouped:
                first = grouped[0]
                rows.append({
                    "Sr.No.": idx,
                    "NAME": name.upper(),
                    "COMPANY AUDITED": first["company"],
                    "DATE": first["date"],
                    "LOCATION": first["location"]
                })
                for entry in grouped[1:]:
                    rows.append({
                        "Sr.No.": "",
                        "NAME": "",
                        "COMPANY AUDITED": entry["company"],
                        "DATE": entry["date"],
                        "LOCATION": entry["location"]
                    })

        df_people = pd.DataFrame(rows)
        df_people.to_excel(writer, index=False, sheet_name='Audit Report')
        _autofit(writer.sheets['Audit Report'])

        # --- Sheet 2: Vehicles ---
        v_rows = []
        for idx, (vehicle, usages) in enumerate(vehicle_data.items(), start=1):
            grouped = group_by_date(usages)
            if grouped:
                first = grouped[0]
                v_rows.append({
                    "Sr.No.": idx,
                    "VEHICLE": vehicle.upper(),
                    "COMPANY AUDITED": first["company"],
                    "DATE": first["date"],
                    "LOCATION": first["location"]
                })
                for entry in grouped[1:]:
                    v_rows.append({
                        "Sr.No.": "",
                        "VEHICLE": "",
                        "COMPANY AUDITED": entry["company"],
                        "DATE": entry["date"],
                        "LOCATION": entry["location"]
                    })

        df_vehicles = pd.DataFrame(v_rows)
        df_vehicles.to_excel(writer, index=False, sheet_name='Vehicle Report')
        _autofit(writer.sheets['Vehicle Report'])

    output.seek(0)
    return output.getvalue()


def _autofit(worksheet):
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)


# ====================== UI ======================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"],
    help="Columns needed: Date, Company, Location, Vehicle, Lead Auditor, Auditor, Evaluator")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        required = {"Date", "Company", "Location", "Vehicle", "Lead Auditor", "Auditor", "Evaluator"}
        if missing := required - set(df.columns):
            st.error(f"Missing columns: {missing}")
        else:
            person_data, vehicle_data = process_csv(df)

            st.success(f"✅ Processed **{len(person_data)}** persons and **{len(vehicle_data)}** vehicles from **{len(df)}** records.")

            if st.button("Generate Excel Report", type="primary"):
                with st.spinner("Creating Excel file..."):
                    excel_bytes = create_excel(person_data, vehicle_data)

                st.download_button(
                    label="⬇️ Download Audit Report (Excel)",
                    data=excel_bytes,
                    file_name=f"Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with st.expander("👤 Preview People (first 5)"):
                preview = []
                for name, audits in list(person_data.items())[:5]:
                    for entry in group_by_date(audits):
                        preview.append([name.upper(), entry["company"], entry["date"], entry["location"]])
                st.dataframe(pd.DataFrame(preview, columns=["NAME", "COMPANY AUDITED", "DATE", "LOCATION"]),
                             use_container_width=True)

            with st.expander("🚗 Preview Vehicles (first 5)"):
                preview = []
                for vehicle, usages in list(vehicle_data.items())[:5]:
                    for entry in group_by_date(usages):
                        preview.append([vehicle.upper(), entry["company"], entry["date"], entry["location"]])
                st.dataframe(pd.DataFrame(preview, columns=["VEHICLE", "COMPANY AUDITED", "DATE", "LOCATION"]),
                             use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload CSV to generate the report")
    st.code("Date, Company, Location, Vehicle, Lead Auditor, Auditor, Evaluator", language="text")
