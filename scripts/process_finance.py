# scripts/process_finance.py
from data_processor_base import IPEDSProcessor
import pandas as pd
import numpy as np


class FinanceProcessor(IPEDSProcessor):
    """Process F2223 series - Financial data including tuition and fees."""

    def process(self) -> pd.DataFrame:
        """Process financial data from F series files."""
        self.logger.info("Starting financial data processing...")

        # Try to load different finance files
        finance_files = {
            "f2223_f1a": "Core revenues and other additions",
            "f2223_f2": "Core expenses and other deductions",
            "f2223_f3": "Net assets",
        }

        processed_dfs = []

        for file_key, description in finance_files.items():
            try:
                df = self.load_csv(f"{file_key}.csv")
                self.logger.info(f"Processing {description}")

                if file_key == "f2223_f1a":
                    processed_df = self._process_revenues(df)
                elif file_key == "f2223_f2":
                    processed_df = self._process_expenses(df)
                elif file_key == "f2223_f3":
                    processed_df = self._process_net_assets(df)

                if processed_df is not None and len(processed_df) > 0:
                    processed_dfs.append(processed_df)

            except Exception as e:
                self.logger.warning(f"Could not process {file_key}: {e}")
                continue

        # Also try to load institutional characteristics for tuition data
        try:
            ic_df = self.load_csv("ic2023.csv")
            tuition_df = self._process_tuition_data(ic_df)
            if tuition_df is not None:
                processed_dfs.append(tuition_df)
        except Exception as e:
            self.logger.warning(f"Could not process tuition data from IC file: {e}")

        # Merge all financial data
        if processed_dfs:
            final_df = processed_dfs[0]
            for df in processed_dfs[1:]:
                final_df = final_df.merge(df, on="UNITID", how="outer")
        else:
            self.logger.warning(
                "No financial files could be processed, creating empty dataframe"
            )
            final_df = pd.DataFrame({"UNITID": []})

        # Add derived fields
        final_df = self.add_derived_fields(final_df)

        # Validate and save
        validation_info = self.validate_data(final_df)
        self.save_processed_data(final_df, "finance_processed.csv", validation_info)

        return final_df

    def _process_revenues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process revenue data."""

        # Key revenue columns (these may vary by year and institution type)
        revenue_columns = [
            "UNITID",
            "F1A01",
            "F1A02",
            "F1A03",
            "F1A04",
            "F1A05",  # Tuition and fees
            "F1A06",
            "F1A07",
            "F1A08",
            "F1A09",
            "F1A10",  # Government grants
            "F1A11",
            "F1A12",
            "F1A13",
            "F1A14",
            "F1A15",  # Private grants
            "F1A16",
            "F1A17",
            "F1A18",
            "F1A19",
            "F1A20",  # Other revenues
        ]

        available_cols = [col for col in revenue_columns if col in df.columns]
        df_rev = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_rev = self.clean_numeric_columns(df_rev, numeric_cols)

        # Calculate total revenues
        revenue_cols = [col for col in numeric_cols if col in df_rev.columns]
        if revenue_cols:
            df_rev["total_revenues"] = df_rev[revenue_cols].sum(axis=1)

        return df_rev

    def _process_expenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process expense data."""

        # Key expense columns
        expense_columns = [
            "UNITID",
            "F2A01",
            "F2A02",
            "F2A03",
            "F2A04",
            "F2A05",  # Instruction
            "F2A06",
            "F2A07",
            "F2A08",
            "F2A09",
            "F2A10",  # Research
            "F2A11",
            "F2A12",
            "F2A13",
            "F2A14",
            "F2A15",  # Student services
            "F2A16",
            "F2A17",
            "F2A18",
            "F2A19",
            "F2A20",  # Other expenses
        ]

        available_cols = [col for col in expense_columns if col in df.columns]
        df_exp = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_exp = self.clean_numeric_columns(df_exp, numeric_cols)

        # Calculate total expenses
        expense_cols = [col for col in numeric_cols if col in df_exp.columns]
        if expense_cols:
            df_exp["total_expenses"] = df_exp[expense_cols].sum(axis=1)

        return df_exp

    def _process_net_assets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process net assets data."""

        # Key net assets columns
        assets_columns = [
            "UNITID",
            "F3A01",
            "F3A02",
            "F3A03",
            "F3A04",
            "F3A05",  # Various asset categories
        ]

        available_cols = [col for col in assets_columns if col in df.columns]
        df_assets = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_assets = self.clean_numeric_columns(df_assets, numeric_cols)

        return df_assets

    def _process_tuition_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process tuition and fee data from institutional characteristics."""

        # Key tuition columns
        tuition_columns = [
            "UNITID",
            "TUITION1",
            "TUITION2",
            "TUITION3",  # In-state tuition
            "TUITION5",
            "TUITION6",
            "TUITION7",  # Out-of-state tuition
            "FEE1",
            "FEE2",
            "FEE3",
            "FEE4",
            "FEE5",
            "FEE6",
            "FEE7",  # Required fees
            "HRCHG1",
            "HRCHG2",
            "HRCHG3",
            "HRCHG4",
            "HRCHG5",  # Per credit hour charges
            "CHG1AT0",
            "CHG1AT1",
            "CHG1AT2",
            "CHG1AT3",  # Room charges
            "CHG2AT0",
            "CHG2AT1",
            "CHG2AT2",
            "CHG2AT3",  # Board charges
            "CHG3AT0",
            "CHG3AT1",
            "CHG3AT2",
            "CHG3AT3",  # Room and board combined
        ]

        available_cols = [col for col in tuition_columns if col in df.columns]
        df_tuition = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_tuition = self.clean_numeric_columns(df_tuition, numeric_cols)

        # Create standardized tuition fields
        self._standardize_tuition_fields(df_tuition)

        return df_tuition

    def _standardize_tuition_fields(self, df: pd.DataFrame):
        """Create standardized tuition and fee fields."""

        # In-state tuition (try different possible columns)
        in_state_cols = ["TUITION1", "TUITION2", "TUITION3"]
        available_in_state = [
            col for col in in_state_cols if col in df.columns and df[col].notna().any()
        ]
        if available_in_state:
            df["tuition_in_state"] = df[available_in_state[0]]

        # Out-of-state tuition
        out_state_cols = ["TUITION5", "TUITION6", "TUITION7"]
        available_out_state = [
            col for col in out_state_cols if col in df.columns and df[col].notna().any()
        ]
        if available_out_state:
            df["tuition_out_state"] = df[available_out_state[0]]

        # Required fees
        fee_cols = ["FEE1", "FEE2", "FEE3", "FEE4", "FEE5", "FEE6", "FEE7"]
        available_fees = [
            col for col in fee_cols if col in df.columns and df[col].notna().any()
        ]
        if available_fees:
            df["required_fees"] = df[available_fees[0]]

        # Room and board
        room_board_cols = ["CHG3AT0", "CHG3AT1", "CHG3AT2", "CHG3AT3"]
        available_rb = [
            col
            for col in room_board_cols
            if col in df.columns and df[col].notna().any()
        ]
        if available_rb:
            df["room_and_board"] = df[available_rb[0]]

        # Total cost calculations
        if "tuition_in_state" in df.columns and "required_fees" in df.columns:
            df["total_in_state_tuition_fees"] = df["tuition_in_state"].fillna(0) + df[
                "required_fees"
            ].fillna(0)
            df["total_in_state_tuition_fees"] = df[
                "total_in_state_tuition_fees"
            ].replace(0, np.nan)

        if "tuition_out_state" in df.columns and "required_fees" in df.columns:
            df["total_out_state_tuition_fees"] = df["tuition_out_state"].fillna(0) + df[
                "required_fees"
            ].fillna(0)
            df["total_out_state_tuition_fees"] = df[
                "total_out_state_tuition_fees"
            ].replace(0, np.nan)

        # Total cost of attendance
        if (
            "total_in_state_tuition_fees" in df.columns
            and "room_and_board" in df.columns
        ):
            df["total_cost_in_state"] = df["total_in_state_tuition_fees"].fillna(
                0
            ) + df["room_and_board"].fillna(0)
            df["total_cost_in_state"] = df["total_cost_in_state"].replace(0, np.nan)

        if (
            "total_out_state_tuition_fees" in df.columns
            and "room_and_board" in df.columns
        ):
            df["total_cost_out_state"] = df["total_out_state_tuition_fees"].fillna(
                0
            ) + df["room_and_board"].fillna(0)
            df["total_cost_out_state"] = df["total_cost_out_state"].replace(0, np.nan)

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields for financial analysis."""
        df = df.copy()

        # Cost categories based on tuition levels
        cost_columns = [
            "total_in_state_tuition_fees",
            "total_out_state_tuition_fees",
            "total_cost_in_state",
            "total_cost_out_state",
        ]

        for col in cost_columns:
            if col in df.columns:
                category_col = col.replace("total_", "") + "_category"
                df[category_col] = df[col].apply(self._categorize_cost)

        # Financial health indicators
        if "total_revenues" in df.columns and "total_expenses" in df.columns:
            df["net_income"] = df["total_revenues"] - df["total_expenses"]
            df["expense_ratio"] = (df["total_expenses"] / df["total_revenues"]).round(3)
            df["financially_stable"] = (df["expense_ratio"] <= 1.0).astype(int)

        # Affordability flags
        if "total_in_state_tuition_fees" in df.columns:
            df["affordable_in_state"] = (
                df["total_in_state_tuition_fees"] <= 15000
            ).astype(int)
            df["expensive_in_state"] = (
                df["total_in_state_tuition_fees"] >= 40000
            ).astype(int)

        if "total_out_state_tuition_fees" in df.columns:
            df["affordable_out_state"] = (
                df["total_out_state_tuition_fees"] <= 25000
            ).astype(int)
            df["expensive_out_state"] = (
                df["total_out_state_tuition_fees"] >= 50000
            ).astype(int)

        return df

    def _categorize_cost(self, cost):
        """Categorize cost levels."""
        if pd.isna(cost):
            return "Unknown"
        elif cost <= 10000:
            return "Very Low (≤$10K)"
        elif cost <= 20000:
            return "Low ($10K-$20K)"
        elif cost <= 35000:
            return "Moderate ($20K-$35K)"
        elif cost <= 50000:
            return "High ($35K-$50K)"
        else:
            return "Very High (>$50K)"


if __name__ == "__main__":
    processor = FinanceProcessor()
    processed_df = processor.process()
    print(f"\nProcessed financial data for {len(processed_df)} institutions")
    print("\nSample processed data:")
    sample_cols = [
        "UNITID",
        "total_in_state_tuition_fees",
        "total_out_state_tuition_fees",
        "in_state_tuition_fees_category",
        "room_and_board",
    ]
    available_sample_cols = [col for col in sample_cols if col in processed_df.columns]
    if available_sample_cols:
        print(processed_df[available_sample_cols].head())
