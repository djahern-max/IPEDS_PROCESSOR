# scripts/process_finance.py - FIXED VERSION
from data_processor_base import IPEDSProcessor
import pandas as pd
import numpy as np


class FinanceProcessor(IPEDSProcessor):
    """Process F2223 series - Financial data including tuition and fees with complete coverage."""

    def process(self) -> pd.DataFrame:
        """Process financial data from F series files with maximum coverage."""
        self.logger.info("Starting comprehensive financial data processing...")

        # Start with ALL institutions from institutional directory
        try:
            hd_df = self.load_csv("hd2023.csv")
            final_df = hd_df[["UNITID"]].copy()
            self.logger.info(f"Starting with {len(final_df)} total institutions")
        except Exception as e:
            self.logger.warning(f"Could not load institutional directory: {e}")
            final_df = pd.DataFrame({"UNITID": []})

        # Process each financial file independently with LEFT JOINs
        finance_processors = [
            ("f2223_f1a.csv", self._process_revenues, "revenues"),
            ("f2223_f2.csv", self._process_expenses, "expenses"),
            ("f2223_f3.csv", self._process_net_assets, "net assets"),
            ("ic2023.csv", self._process_tuition_data, "tuition data"),
        ]

        for filename, processor_func, description in finance_processors:
            try:
                df = self.load_csv(filename)
                self.logger.info(f"Processing {description} from {filename}")

                processed_df = processor_func(df)

                if processed_df is not None and len(processed_df) > 0:
                    # Use LEFT JOIN to preserve ALL institutions
                    before_count = len(final_df)
                    final_df = final_df.merge(processed_df, on="UNITID", how="left")

                    # Log data coverage
                    coverage = len(processed_df)
                    self.logger.info(
                        f"  {description}: {coverage} institutions have data"
                    )

                    # Verify no institutions were lost
                    if len(final_df) != before_count:
                        self.logger.warning(
                            f"Institution count changed during merge: {before_count} -> {len(final_df)}"
                        )

            except Exception as e:
                self.logger.warning(
                    f"Could not process {filename} for {description}: {e}"
                )
                continue

        # Add derived fields
        final_df = self.add_derived_fields(final_df)

        # Validate and save
        validation_info = self.validate_data(final_df)
        self.save_processed_data(final_df, "finance_processed.csv", validation_info)

        # Log final coverage statistics
        self._log_coverage_stats(final_df)

        return final_df

    def _process_revenues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process revenue data with maximum coverage."""

        # All possible F1A revenue columns
        revenue_columns = [
            "UNITID",
            "F1A01",
            "F1A02",
            "F1A03",
            "F1A04",
            "F1A05",
            "F1A06",
            "F1A07",
            "F1A08",
            "F1A09",
            "F1A10",
            "F1A11",
            "F1A12",
            "F1A13",
            "F1A14",
            "F1A15",
            "F1A16",
            "F1A17",
            "F1A18",
            "F1A19",
            "F1A20",
        ]

        available_cols = [col for col in revenue_columns if col in df.columns]
        df_rev = df[available_cols].copy()

        # Clean numeric columns - convert strings to numbers, handle missing values
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_rev = self.clean_numeric_columns(df_rev, numeric_cols)

        # Calculate total revenues using SAFE summation
        def safe_sum_revenues(row):
            """Sum revenues safely, ignoring nulls and negative values."""
            # Primary revenue components (adjust based on IPEDS documentation)
            primary_revenue_cols = [
                "F1A01",
                "F1A04",
                "F1A05",
                "F1A06",
                "F1A08",
                "F1A10",
                "F1A11",
                "F1A17",
                "F1A18",
            ]
            available_revenue_cols = [
                col for col in primary_revenue_cols if col in df_rev.columns
            ]

            values = []
            for col in available_revenue_cols:
                val = row[col]
                if pd.notna(val) and val >= 0:  # Only include valid positive values
                    values.append(val)

            return sum(values) if values else None

        df_rev["total_revenues"] = df_rev.apply(safe_sum_revenues, axis=1)

        # Only return institutions that have ANY revenue data
        mask = df_rev[numeric_cols].notna().any(axis=1)
        df_rev_filtered = df_rev[mask].copy()

        self.logger.info(
            f"Revenue processing: {len(df_rev_filtered)} institutions with revenue data"
        )

        return df_rev_filtered

    def _process_expenses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process expense data with maximum coverage."""

        # All possible F2A expense columns
        expense_columns = [
            "UNITID",
            "F2A01",
            "F2A02",
            "F2A03",
            "F2A04",
            "F2A05",
            "F2A06",
            "F2A07",
            "F2A08",
            "F2A09",
            "F2A10",
            "F2A11",
            "F2A12",
            "F2A13",
            "F2A14",
            "F2A15",
            "F2A16",
            "F2A17",
            "F2A18",
            "F2A19",
            "F2A20",
        ]

        available_cols = [col for col in expense_columns if col in df.columns]
        df_exp = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_exp = self.clean_numeric_columns(df_exp, numeric_cols)

        # Calculate total expenses using SAFE summation
        def safe_sum_expenses(row):
            """Sum expenses safely, ignoring nulls and negative values."""
            # Primary expense components (adjust based on IPEDS documentation)
            primary_expense_cols = [
                "F2A01",
                "F2A02",
                "F2A03",
                "F2A04",
                "F2A05",
                "F2A11",
                "F2A12",
                "F2A17",
                "F2A18",
            ]
            available_expense_cols = [
                col for col in primary_expense_cols if col in df_exp.columns
            ]

            values = []
            for col in available_expense_cols:
                val = row[col]
                if pd.notna(val) and val >= 0:  # Only include valid positive values
                    values.append(val)

            return sum(values) if values else None

        df_exp["total_expenses"] = df_exp.apply(safe_sum_expenses, axis=1)

        # Only return institutions that have ANY expense data
        mask = df_exp[numeric_cols].notna().any(axis=1)
        df_exp_filtered = df_exp[mask].copy()

        self.logger.info(
            f"Expense processing: {len(df_exp_filtered)} institutions with expense data"
        )

        return df_exp_filtered

    def _process_net_assets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process net assets data with maximum coverage."""

        # Net assets columns
        assets_columns = ["UNITID", "F3A01", "F3A02", "F3A03", "F3A04", "F3A05"]

        available_cols = [col for col in assets_columns if col in df.columns]
        df_assets = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_assets = self.clean_numeric_columns(df_assets, numeric_cols)

        # Calculate net assets
        if len(numeric_cols) > 0:

            def safe_sum_assets(row):
                values = []
                for col in numeric_cols:
                    val = row[col]
                    if pd.notna(val):  # Include negative values for net assets
                        values.append(val)
                return sum(values) if values else None

            df_assets["net_assets"] = df_assets.apply(safe_sum_assets, axis=1)

        # Only return institutions that have ANY assets data
        mask = df_assets[numeric_cols].notna().any(axis=1)
        df_assets_filtered = df_assets[mask].copy()

        self.logger.info(
            f"Assets processing: {len(df_assets_filtered)} institutions with assets data"
        )

        return df_assets_filtered

    def _process_tuition_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process tuition and fee data from institutional characteristics."""

        # Comprehensive tuition columns
        tuition_columns = [
            "UNITID",
            # Published tuition and fees
            "TUITION1",
            "TUITION2",
            "TUITION3",  # In-state undergraduate
            "TUITION5",
            "TUITION6",
            "TUITION7",  # Out-of-state undergraduate
            # Required fees
            "FEE1",
            "FEE2",
            "FEE3",
            "FEE4",
            "FEE5",
            "FEE6",
            "FEE7",
            # Room and board charges
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
            # Per credit hour charges
            "HRCHG1",
            "HRCHG2",
            "HRCHG3",
            "HRCHG4",
            "HRCHG5",
        ]

        available_cols = [col for col in tuition_columns if col in df.columns]
        df_tuition = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_tuition = self.clean_numeric_columns(df_tuition, numeric_cols)

        # Create standardized tuition fields
        self._standardize_tuition_fields(df_tuition)

        # Only return institutions that have ANY tuition data
        if numeric_cols:
            mask = df_tuition[numeric_cols].notna().any(axis=1)
            df_tuition_filtered = df_tuition[mask].copy()
        else:
            df_tuition_filtered = df_tuition.copy()

        self.logger.info(
            f"Tuition processing: {len(df_tuition_filtered)} institutions with tuition data"
        )

        return df_tuition_filtered

    def _standardize_tuition_fields(self, df: pd.DataFrame):
        """Create standardized tuition and fee fields with comprehensive coverage."""

        # In-state tuition (try multiple columns)
        in_state_candidates = ["TUITION1", "TUITION2", "TUITION3"]
        df["tuition_in_state"] = self._get_first_available_value(
            df, in_state_candidates
        )

        # Out-of-state tuition
        out_state_candidates = ["TUITION5", "TUITION6", "TUITION7"]
        df["tuition_out_state"] = self._get_first_available_value(
            df, out_state_candidates
        )

        # Required fees
        fee_candidates = ["FEE1", "FEE2", "FEE3", "FEE4", "FEE5", "FEE6", "FEE7"]
        df["required_fees"] = self._get_first_available_value(df, fee_candidates)

        # Room and board (prefer combined, fall back to separate)
        room_board_candidates = ["CHG3AT0", "CHG3AT1", "CHG3AT2", "CHG3AT3"]
        df["room_and_board"] = self._get_first_available_value(
            df, room_board_candidates
        )

        # If no combined room/board, try to calculate from separate
        if df["room_and_board"].isna().all():
            room_candidates = ["CHG1AT0", "CHG1AT1", "CHG1AT2", "CHG1AT3"]
            board_candidates = ["CHG2AT0", "CHG2AT1", "CHG2AT2", "CHG2AT3"]

            room_charges = self._get_first_available_value(df, room_candidates)
            board_charges = self._get_first_available_value(df, board_candidates)

            # Sum room and board if both available
            mask = pd.notna(room_charges) & pd.notna(board_charges)
            df.loc[mask, "room_and_board"] = room_charges[mask] + board_charges[mask]

        # Calculate total costs
        df["total_in_state_tuition_fees"] = self._safe_add(
            df["tuition_in_state"], df["required_fees"]
        )
        df["total_out_state_tuition_fees"] = self._safe_add(
            df["tuition_out_state"], df["required_fees"]
        )
        df["total_cost_in_state"] = self._safe_add(
            df["total_in_state_tuition_fees"], df["room_and_board"]
        )
        df["total_cost_out_state"] = self._safe_add(
            df["total_out_state_tuition_fees"], df["room_and_board"]
        )

    def _get_first_available_value(self, df: pd.DataFrame, columns: list) -> pd.Series:
        """Get the first non-null value from a list of columns."""
        result = pd.Series(index=df.index, dtype=float)

        for col in columns:
            if col in df.columns:
                mask = pd.isna(result) & pd.notna(df[col])
                result[mask] = df[col][mask]

        return result

    def _safe_add(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Safely add two series, returning None if both are null."""
        result = pd.Series(index=series1.index, dtype=float)

        # If both values exist, add them
        both_exist = pd.notna(series1) & pd.notna(series2)
        result[both_exist] = series1[both_exist] + series2[both_exist]

        # If only one exists, use that value
        only_first = pd.notna(series1) & pd.isna(series2)
        result[only_first] = series1[only_first]

        only_second = pd.isna(series1) & pd.notna(series2)
        result[only_second] = series2[only_second]

        return result

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields for financial analysis."""
        df = df.copy()

        # Financial health indicators - only calculate where we have data
        if "total_revenues" in df.columns and "total_expenses" in df.columns:
            # Net income - only where both exist
            both_exist = pd.notna(df["total_revenues"]) & pd.notna(df["total_expenses"])
            df["net_income"] = None
            df.loc[both_exist, "net_income"] = (
                df.loc[both_exist, "total_revenues"]
                - df.loc[both_exist, "total_expenses"]
            )

            # Expense ratio - only where both exist and revenue > 0
            valid_ratio = both_exist & (df["total_revenues"] > 0)
            df["expense_ratio"] = None
            df.loc[valid_ratio, "expense_ratio"] = (
                df.loc[valid_ratio, "total_expenses"]
                / df.loc[valid_ratio, "total_revenues"]
            ).round(3)

            # Financial stability - conservative definition
            df["financially_stable"] = 0
            stable_mask = (
                pd.notna(df["net_income"])
                & (df["net_income"] >= 0)
                & pd.notna(df["expense_ratio"])
                & (df["expense_ratio"] <= 1.0)
            )
            df.loc[stable_mask, "financially_stable"] = 1

        # Cost categories for tuition data
        cost_columns = [
            "total_in_state_tuition_fees",
            "total_out_state_tuition_fees",
            "total_cost_in_state",
            "total_cost_out_state",
        ]

        for col in cost_columns:
            if col in df.columns:
                category_col = col + "_category"
                df[category_col] = df[col].apply(self._categorize_cost)

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
            return None  # Changed from "Unknown" to None for consistency
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

    def _log_coverage_stats(self, df: pd.DataFrame):
        """Log comprehensive coverage statistics."""
        total_institutions = len(df)

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"FINANCIAL DATA COVERAGE SUMMARY")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Total institutions: {total_institutions}")

        # Revenue coverage
        if "total_revenues" in df.columns:
            revenue_count = df["total_revenues"].notna().sum()
            revenue_pct = revenue_count / total_institutions * 100
            self.logger.info(
                f"Institutions with revenue data: {revenue_count} ({revenue_pct:.1f}%)"
            )

        # Expense coverage
        if "total_expenses" in df.columns:
            expense_count = df["total_expenses"].notna().sum()
            expense_pct = expense_count / total_institutions * 100
            self.logger.info(
                f"Institutions with expense data: {expense_count} ({expense_pct:.1f}%)"
            )

        # Net income coverage
        if "net_income" in df.columns:
            net_income_count = df["net_income"].notna().sum()
            net_income_pct = net_income_count / total_institutions * 100
            self.logger.info(
                f"Institutions with net income calculated: {net_income_count} ({net_income_pct:.1f}%)"
            )

        # Tuition coverage
        if "total_in_state_tuition_fees" in df.columns:
            tuition_count = df["total_in_state_tuition_fees"].notna().sum()
            tuition_pct = tuition_count / total_institutions * 100
            self.logger.info(
                f"Institutions with tuition data: {tuition_count} ({tuition_pct:.1f}%)"
            )

        # Financial stability
        if "financially_stable" in df.columns:
            stable_count = (df["financially_stable"] == 1).sum()
            stable_pct = stable_count / total_institutions * 100
            self.logger.info(
                f"Financially stable institutions: {stable_count} ({stable_pct:.1f}%)"
            )

        self.logger.info(f"{'='*50}")


if __name__ == "__main__":
    processor = FinanceProcessor()
    processed_df = processor.process()
    print(f"\nProcessed financial data for {len(processed_df)} institutions")

    # Show coverage summary
    print(f"\nCoverage Summary:")
    if "total_revenues" in processed_df.columns:
        revenue_count = processed_df["total_revenues"].notna().sum()
        print(
            f"  Revenues: {revenue_count} institutions ({revenue_count/len(processed_df)*100:.1f}%)"
        )

    if "total_expenses" in processed_df.columns:
        expense_count = processed_df["total_expenses"].notna().sum()
        print(
            f"  Expenses: {expense_count} institutions ({expense_count/len(processed_df)*100:.1f}%)"
        )

    if "net_income" in processed_df.columns:
        net_count = processed_df["net_income"].notna().sum()
        print(
            f"  Net Income: {net_count} institutions ({net_count/len(processed_df)*100:.1f}%)"
        )
