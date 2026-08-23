import streamlit as st
import pandas as pd
from collections import defaultdict
import io
from datetime import datetime
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SDA",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(
            135deg,
            #1a1a2e 0%,
            #16213e 50%,
            #0f3460 100%
        );
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        color: #e2e8f0;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        color: #cbd5e1;
        margin-top: 0.5rem;
    }

    .matched {
        color: #15803d;
        font-weight: 600;
    }

    .unmatched {
        color: #dc2626;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>V1.0</h1>
    <p>Excel format matching your sample image</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NAME MAPPING
# ============================================================

def load_name_mapping():
    """
    Load short-name to full-name mappings from names.txt.

    Example names.txt:

        Trevor=Nowembabazi Trevor
        John=John Doe
        Peter=Peter Smith
    """

    mapping = {}

    # Get the folder where this Python file is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    names_file = os.path.join(base_dir, "names.txt")

    if not os.path.exists(names_file):
        return mapping, False

    try:
        with open(names_file, "r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                # Ignore blank lines
                if not line:
                    continue

                # Ignore comments
                if line.startswith("#"):
                    continue

                # Check that the line contains =
                if "=" not in line:
                    continue

                short_name, full_name = line.split("=", 1)

                short_name = short_name.strip()
                full_name = full_name.strip()

                if short_name and full_name:
                    mapping[short_name.lower()] = full_name

        return mapping, True

    except Exception as e:
        st.error(f"Error reading names.txt: {str(e)}")
        return {}, False


NAME_MAPPING, names_file_found = load_name_mapping()


# ============================================================
# NAME PARSING
# ============================================================

def parse_names(cell_value, matched_names, unmatched_names):
    """
    Extract names from a cell and replace short names
    with full names using names.txt.

    Example:

        Trevor
        ↓
        Nowembabazi Trevor

    Multiple names are also supported:

        Trevor, John
        ↓
        Nowembabazi Trevor, John Doe
    """

    if pd.isna(cell_value) or str(cell_value).strip() == "":
        return []

    names = [
        n.strip()
        for n in str(cell_value).strip().split(",")
        if n.strip()
    ]

    full_names = []

    for name in names:

        # Case-insensitive lookup
        lookup_name = name.lower()

        if lookup_name in NAME_MAPPING:

            full_name = NAME_MAPPING[lookup_name]

            full_names.append(full_name)

            # Record successful match
            matched_names.append({
                "Uploaded Name": name,
                "Full Name": full_name,
                "Status": "Matched"
            })

        else:

            # Name was not found in names.txt
            full_names.append(name)

            unmatched_names.add(name)

    return full_names


# ============================================================
# PROCESS CSV
# ============================================================

def process_csv(df):

    person_data = defaultdict(list)
    vehicle_data = defaultdict(list)

    matched_names = []
    unmatched_names = set()

    for _, row in df.iterrows():

        date_val = str(row.get("Date", "")).strip()
        company_val = str(row.get("Company", "")).strip()
        location_val = str(row.get("Location", "")).strip()

        # ----------------------------------------------------
        # Lead Auditor
        # ----------------------------------------------------

        for name in parse_names(
            row.get("Lead Auditor", ""),
            matched_names,
            unmatched_names
        ):

            person_data[name].append({
                "company": company_val,
                "date": date_val,
                "location": location_val,
                "role": "Lead Auditor"
            })

        # ----------------------------------------------------
        # Auditor
        # ----------------------------------------------------

        for name in parse_names(
            row.get("Auditor", ""),
            matched_names,
            unmatched_names
        ):

            person_data[name].append({
                "company": company_val,
                "date": date_val,
                "location": location_val,
                "role": "Auditor"
            })

        # ----------------------------------------------------
        # Evaluator
        # ----------------------------------------------------

        for name in parse_names(
            row.get("Evaluator", ""),
            matched_names,
            unmatched_names
        ):

            person_data[name].append({
                "company": company_val,
                "date": date_val,
                "location": location_val,
                "role": "Evaluator"
            })

        # ----------------------------------------------------
        # Vehicles
        # ----------------------------------------------------

        for vehicle in str(row.get("Vehicle", "")).split(","):

            vehicle = vehicle.strip()

            if vehicle:

                vehicle_data[vehicle].append({
                    "company": company_val,
                    "date": date_val,
                    "location": location_val
                })

    # --------------------------------------------------------
    # Sort records
    # --------------------------------------------------------

    for name in person_data:
        person_data[name].sort(key=lambda x: x["date"])

    for vehicle in vehicle_data:
        vehicle_data[vehicle].sort(key=lambda x: x["date"])

    return (
        person_data,
        vehicle_data,
        matched_names,
        unmatched_names
    )


# ============================================================
# GROUP BY DATE
# ============================================================

def group_by_date(entries):

    """
    Group entries by date, merging companies and locations
    with commas.
    """

    date_groups = defaultdict(
        lambda: {
            "companies": [],
            "locations": []
        }
    )

    for entry in entries:

        date = entry["date"]

        date_groups[date]["companies"].append(
            entry["company"]
        )

        date_groups[date]["locations"].append(
            entry["location"]
        )

    grouped = []

    for date in sorted(date_groups.keys()):

        companies = date_groups[date]["companies"]
        locations = date_groups[date]["locations"]

        # ----------------------------------------------------
        # Deduplicate companies
        # ----------------------------------------------------

        seen_c = []

        for company in companies:

            if company not in seen_c:
                seen_c.append(company)

        # ----------------------------------------------------
        # Deduplicate locations
        # ----------------------------------------------------

        seen_l = []

        for location in locations:

            if location not in seen_l:
                seen_l.append(location)

        grouped.append({
            "date": date,
            "company": ", ".join(seen_c),
            "location": ", ".join(seen_l)
        })

    return grouped


# ============================================================
# CREATE EXCEL
# ============================================================

def create_excel(person_data, vehicle_data):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # ====================================================
        # SHEET 1: PEOPLE
        # ====================================================

        rows = []

        for idx, (name, audits) in enumerate(
            person_data.items(),
            start=1
        ):

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

        df_people.to_excel(
            writer,
            index=False,
            sheet_name="Audit Report"
        )

        _autofit(
            writer.sheets["Audit Report"]
        )

        # ====================================================
        # SHEET 2: VEHICLES
        # ====================================================

        v_rows = []

        for idx, (vehicle, usages) in enumerate(
            vehicle_data.items(),
            start=1
        ):

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

        df_vehicles.to_excel(
            writer,
            index=False,
            sheet_name="Vehicle Report"
        )

        _autofit(
            writer.sheets["Vehicle Report"]
        )

    output.seek(0)

    return output.getvalue()


# ============================================================
# AUTOFIT EXCEL COLUMNS
# ============================================================

def _autofit(worksheet):

    for column in worksheet.columns:

        max_length = 0

        column_letter = column[0].column_letter

        for cell in column:

            try:

                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))

            except Exception:
                pass

        worksheet.column_dimensions[
            column_letter
        ].width = min(
            max_length + 2,
            50
        )


# ============================================================
# UI - NAME FILE STATUS
# ============================================================

st.subheader("Name Mapping")

if names_file_found:

    st.success(
        f"✅ names.txt loaded successfully — "
        f"{len(NAME_MAPPING)} name mappings available."
    )

    with st.expander("📋 View Name Mapping"):

        mapping_preview = []

        for short_name, full_name in NAME_MAPPING.items():

            mapping_preview.append({
                "Uploaded Name": short_name.title(),
                "Full Name": full_name
            })

        if mapping_preview:

            st.dataframe(
                pd.DataFrame(mapping_preview),
                use_container_width=True,
                hide_index=True
            )

else:

    st.warning(
        "⚠️ names.txt was not found. "
        "The application will use names exactly as they appear "
        "in the uploaded CSV."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help=(
        "Columns needed: Date, Company, Location, Vehicle, "
        "Lead Auditor, Auditor, Evaluator"
    )
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = df.columns.str.strip()

        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required = {
            "Date",
            "Company",
            "Location",
            "Vehicle",
            "Lead Auditor",
            "Auditor",
            "Evaluator"
        }

        missing = required - set(df.columns)

        if missing:

            st.error(
                f"Missing columns: {missing}"
            )

        else:

            (
                person_data,
                vehicle_data,
                matched_names,
                unmatched_names
            ) = process_csv(df)

            # =================================================
            # SUMMARY
            # =================================================

            st.success(
                f"✅ Processed **{len(person_data)}** persons, "
                f"**{len(vehicle_data)}** vehicles from "
                f"**{len(df)}** records."
            )

            # =================================================
            # NAME MATCHING RESULTS
            # =================================================

            st.subheader("🔍 Name Matching Results")

            # Remove duplicate successful matches
            unique_matches = {}

            for match in matched_names:

                key = match["Uploaded Name"].lower()

                unique_matches[key] = {
                    "Uploaded Name": match["Uploaded Name"],
                    "Full Name": match["Full Name"],
                    "Status": "✅ Matched"
                }

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Names Matched",
                    len(unique_matches)
                )

            with col2:

                st.metric(
                    "Names Not Matched",
                    len(unmatched_names)
                )

            # =================================================
            # SUCCESSFULLY MATCHED
            # =================================================

            with st.expander(
                f"✅ Successfully Matched ({len(unique_matches)})"
            ):

                if unique_matches:

                    matched_df = pd.DataFrame(
                        list(unique_matches.values())
                    )

                    st.dataframe(
                        matched_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No names were matched using names.txt."
                    )

            # =================================================
            # UNMATCHED NAMES
            # =================================================

            with st.expander(
                f"⚠️ Names Not Found in names.txt ({len(unmatched_names)})"
            ):

                if unmatched_names:

                    unmatched_df = pd.DataFrame({
                        "Name Not Found": sorted(
                            unmatched_names
                        )
                    })

                    st.dataframe(
                        unmatched_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.warning(
                        "These names will appear in the Excel "
                        "report exactly as they appeared in "
                        "the uploaded CSV."
                    )

                    # -----------------------------------------
                    # Download unmatched names
                    # -----------------------------------------

                    unmatched_text = "\n".join(
                        sorted(unmatched_names)
                    )

                    st.download_button(
                        label="⬇️ Download Unmatched Names",
                        data=unmatched_text,
                        file_name="unmatched_names.txt",
                        mime="text/plain"
                    )

                else:

                    st.success(
                        "🎉 All extracted names were successfully "
                        "matched in names.txt!"
                    )

            # =================================================
            # GENERATE EXCEL
            # =================================================

            st.subheader("Excel Report")

            if st.button(
                "Generate Excel Report",
                type="primary"
            ):

                with st.spinner(
                    "Creating Excel file..."
                ):

                    excel_bytes = create_excel(
                        person_data,
                        vehicle_data
                    )

                st.download_button(
                    label="⬇️ Download Audit Report (Excel)",
                    data=excel_bytes,
                    file_name=(
                        f"Audit_Report_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}"
                        f".xlsx"
                    ),
                    mime=(
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    )
                )

            # =================================================
            # PEOPLE PREVIEW
            # =================================================

            with st.expander(
                "👤 Preview People (first 5)"
            ):

                preview = []

                for name, audits in list(
                    person_data.items()
                )[:5]:

                    for entry in group_by_date(audits):

                        preview.append([
                            name.upper(),
                            entry["company"],
                            entry["date"],
                            entry["location"]
                        ])

                st.dataframe(
                    pd.DataFrame(
                        preview,
                        columns=[
                            "NAME",
                            "COMPANY AUDITED",
                            "DATE",
                            "LOCATION"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # VEHICLE PREVIEW
            # =================================================

            with st.expander(
                "🚗 Preview Vehicles (first 5)"
            ):

                preview = []

                for vehicle, usages in list(
                    vehicle_data.items()
                )[:5]:

                    for entry in group_by_date(usages):

                        preview.append([
                            vehicle.upper(),
                            entry["company"],
                            entry["date"],
                            entry["location"]
                        ])

                st.dataframe(
                    pd.DataFrame(
                        preview,
                        columns=[
                            "VEHICLE",
                            "COMPANY AUDITED",
                            "DATE",
                            "LOCATION"
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

else:

    st.info(
        "Upload CSV to generate the report"
    )

    st.code(
        "Date, Company, Location, Vehicle, "
        "Lead Auditor, Auditor, Evaluator",
        language="text"
    )
