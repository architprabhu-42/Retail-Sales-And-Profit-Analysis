# Retail Sales & Profit Intelligence Studio

Business Intelligence and Analytics mini project (Year 3, Sem V, Computer Engineering).

## Files

| File | What it is |
|---|---|
| `Retail_Sales_Profit_Analysis.ipynb` | Main notebook, laid out like the report: Introduction, Data Description, Pre-processing, Mining Models, Results, Visualization, Conclusion |
| `app.py` | Dashboard (Streamlit, Poppins font): black & white layout with colour charts |
| `data/superstore.xls` | Official Tableau Sample Superstore workbook: Orders (10,194 rows, 2023–2026, US & Canada), Returns, People |
| `data/superstore_cleaned.csv` | Cleaned data written by the notebook |

## Setup (VS Code)

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Open the notebook in VS Code, select the `.venv` kernel and click **Run All**.

Run the GUI:

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Data source

Tableau Public sample data, *Superstore Sales* (https://public.tableau.com/app/learn/sample-data), downloaded from Tableau as `sample_-_superstore.xls`. The notebook and GUI join the Returns and People sheets onto the orders. The company and its transactions are fictitious.

Chart fonts use Poppins. The GUI loads it from Google Fonts. The notebook uses it only if Poppins is installed on the PC; otherwise it falls back to the default font.
