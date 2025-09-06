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
        """Process tuition and fee data from institutional characteristics - FIXED VERSION."""

        # First, let's check what columns actually exist in the IC file
        print(
            f"DEBUG: IC2023 columns available: {list(df.columns)[:20]}..."
        )  # Show first 20 columns

        # Comprehensive tuition columns - updated to match actual IPEDS column names
        tuition_columns = [
            "UNITID",
            # In-state tuition (these may have different names in 2023)
            "TUITION1",
            "TUITION2",
            "TUITION3",
            "CHG1AY3",
            "CHG2AY3",
            "CHG3AY3",
            # Out-of-state tuition
            "TUITION5",
            "TUITION6",
            "TUITION7",
            "CHG4AY3",
            "CHG5AY3",
            "CHG6AY3",
            # Required fees
            "FEE1",
            "FEE2",
            "FEE3",
            "FEE4",
            "FEE5",
            "FEE6",
            "FEE7",
            # Room and board - check actual IC2023 column names
            "ROOMCAP",
            "BOARDCAP",
            "ROOMAMT",
            "BOARDAMT",
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
            "CHG3AT3",  # Combined room/board
        ]

        # Only use columns that actually exist in the data
        available_cols = ["UNITID"] + [
            col for col in tuition_columns[1:] if col in df.columns
        ]
        print(
            f"DEBUG: Using {len(available_cols)-1} tuition columns out of {len(tuition_columns)-1} possible"
        )

        if len(available_cols) == 1:  # Only UNITID found
            print("WARNING: No tuition columns found in IC2023 file!")
            return pd.DataFrame({"UNITID": df["UNITID"].unique()})

        df_tuition = df[available_cols].copy()

        # Clean numeric columns
        numeric_cols = [col for col in available_cols if col != "UNITID"]
        df_tuition = self.clean_numeric_columns(df_tuition, numeric_cols)

        # Create standardized tuition fields - FIXED VERSION
        self._standardize_tuition_fields(df_tuition)

        # Only return institutions that have ANY standardized tuition data
        standardized_cols = [
            "tuition_in_state",
            "tuition_out_state",
            "required_fees",
            "room_and_board",
        ]
        existing_standardized = [
            col for col in standardized_cols if col in df_tuition.columns
        ]

        if existing_standardized:
            mask = df_tuition[existing_standardized].notna().any(axis=1)
            df_tuition_filtered = df_tuition[mask].copy()
        else:
            print("WARNING: No standardized tuition fields created!")
            df_tuition_filtered = df_tuition.copy()

        print(
            f"DEBUG: Created {len(existing_standardized)} standardized tuition fields"
        )
        print(
            f"DEBUG: Tuition processing result: {len(df_tuition_filtered)} institutions with tuition data"
        )

        return df_tuition_filtered

    def _standardize_tuition_fields(self, df: pd.DataFrame):
        """Create standardized tuition and fee fields - FIXED VERSION."""
        print("DEBUG: Starting tuition field standardization...")

        # Debug: Show what columns we're working with
        print(f"DEBUG: Available columns for standardization: {list(df.columns)}")

        # In-state tuition (try multiple possible column names)
        in_state_candidates = [
            "TUITION1",
            "TUITION2",
            "TUITION3",
            "CHG1AY3",
            "CHG2AY3",
            "CHG3AY3",
        ]
        available_in_state = [col for col in in_state_candidates if col in df.columns]
        print(f"DEBUG: In-state tuition candidates found: {available_in_state}")

        if available_in_state:
            df["tuition_in_state"] = self._get_first_available_value(
                df, available_in_state
            )
            in_state_count = df["tuition_in_state"].notna().sum()
            print(f"DEBUG: Created tuition_in_state for {in_state_count} institutions")

        # Out-of-state tuition
        out_state_candidates = [
            "TUITION5",
            "TUITION6",
            "TUITION7",
            "CHG4AY3",
            "CHG5AY3",
            "CHG6AY3",
        ]
        available_out_state = [col for col in out_state_candidates if col in df.columns]
        print(f"DEBUG: Out-of-state tuition candidates found: {available_out_state}")

        if available_out_state:
            df["tuition_out_state"] = self._get_first_available_value(
                df, available_out_state
            )
            out_state_count = df["tuition_out_state"].notna().sum()
            print(
                f"DEBUG: Created tuition_out_state for {out_state_count} institutions"
            )

        # Required fees
        fee_candidates = ["FEE1", "FEE2", "FEE3", "FEE4", "FEE5", "FEE6", "FEE7"]
        available_fees = [col for col in fee_candidates if col in df.columns]
        print(f"DEBUG: Fee candidates found: {available_fees}")

        if available_fees:
            df["required_fees"] = self._get_first_available_value(df, available_fees)
            fee_count = df["required_fees"].notna().sum()
            print(f"DEBUG: Created required_fees for {fee_count} institutions")

        # Room and board - try multiple approaches
        room_board_candidates = [
            "CHG3AT0",
            "CHG3AT1",
            "CHG3AT2",
            "CHG3AT3",
            "ROOMAMT",
            "BOARDAMT",
        ]
        available_rb = [col for col in room_board_candidates if col in df.columns]
        print(f"DEBUG: Room/board candidates found: {available_rb}")

        if available_rb:
            # Try combined room/board first
            combined_candidates = ["CHG3AT0", "CHG3AT1", "CHG3AT2", "CHG3AT3"]
            available_combined = [
                col for col in combined_candidates if col in df.columns
            ]

            if available_combined:
                df["room_and_board"] = self._get_first_available_value(
                    df, available_combined
                )
            else:
                # Try to sum separate room and board charges
                room_candidates = [
                    "CHG1AT0",
                    "CHG1AT1",
                    "CHG1AT2",
                    "CHG1AT3",
                    "ROOMAMT",
                ]
                board_candidates = [
                    "CHG2AT0",
                    "CHG2AT1",
                    "CHG2AT2",
                    "CHG2AT3",
                    "BOARDAMT",
                ]

                available_room = [col for col in room_candidates if col in df.columns]
                available_board = [col for col in board_candidates if col in df.columns]

                if available_room and available_board:
                    room_charges = self._get_first_available_value(df, available_room)
                    board_charges = self._get_first_available_value(df, available_board)

                    # Sum if both exist
                    mask = pd.notna(room_charges) & pd.notna(board_charges)
                    df["room_and_board"] = pd.Series(index=df.index, dtype=float)
                    df.loc[mask, "room_and_board"] = (
                        room_charges[mask] + board_charges[mask]
                    )

                    # Use room only if board is missing
                    room_only = pd.notna(room_charges) & pd.isna(board_charges)
                    df.loc[room_only, "room_and_board"] = room_charges[room_only]

                    # Use board only if room is missing
                    board_only = pd.isna(room_charges) & pd.notna(board_charges)
                    df.loc[board_only, "room_and_board"] = board_charges[board_only]

            rb_count = (
                df["room_and_board"].notna().sum()
                if "room_and_board" in df.columns
                else 0
            )
            print(f"DEBUG: Created room_and_board for {rb_count} institutions")

        # Calculate total costs - FIXED
        if "tuition_in_state" in df.columns and "required_fees" in df.columns:
            df["total_in_state_tuition_fees"] = self._safe_add(
                df["tuition_in_state"], df["required_fees"]
            )
            total_in_count = df["total_in_state_tuition_fees"].notna().sum()
            print(
                f"DEBUG: Created total_in_state_tuition_fees for {total_in_count} institutions"
            )

        if "tuition_out_state" in df.columns and "required_fees" in df.columns:
            df["total_out_state_tuition_fees"] = self._safe_add(
                df["tuition_out_state"], df["required_fees"]
            )
            total_out_count = df["total_out_state_tuition_fees"].notna().sum()
            print(
                f"DEBUG: Created total_out_state_tuition_fees for {total_out_count} institutions"
            )

        if (
            "total_in_state_tuition_fees" in df.columns
            and "room_and_board" in df.columns
        ):
            df["total_cost_in_state"] = self._safe_add(
                df["total_in_state_tuition_fees"], df["room_and_board"]
            )
            cost_in_count = df["total_cost_in_state"].notna().sum()
            print(
                f"DEBUG: Created total_cost_in_state for {cost_in_count} institutions"
            )

        if (
            "total_out_state_tuition_fees" in df.columns
            and "room_and_board" in df.columns
        ):
            df["total_cost_out_state"] = self._safe_add(
                df["total_out_state_tuition_fees"], df["room_and_board"]
            )
            cost_out_count = df["total_cost_out_state"].notna().sum()
            print(
                f"DEBUG: Created total_cost_out_state for {cost_out_count} institutions"
            )

        print("DEBUG: Tuition field standardization complete")

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived fields for financial analysis - FIXED VERSION."""
        print("DEBUG: Starting derived field calculations...")
        print(f"DEBUG: Available columns: {list(df.columns)}")

        df = df.copy()

        # Financial health indicators - FIXED column name checking
        has_revenues = "total_revenues" in df.columns
        has_expenses = "total_expenses" in df.columns

        print(f"DEBUG: Has total_revenues column: {has_revenues}")
        print(f"DEBUG: Has total_expenses column: {has_expenses}")

        if has_revenues:
            revenue_count = df["total_revenues"].notna().sum()
            print(f"DEBUG: Revenue data available for {revenue_count} institutions")

        if has_expenses:
            expense_count = df["total_expenses"].notna().sum()
            print(f"DEBUG: Expense data available for {expense_count} institutions")

        if has_revenues and has_expenses:
            # Net income - only where both exist
            both_exist = pd.notna(df["total_revenues"]) & pd.notna(df["total_expenses"])
            both_count = both_exist.sum()
            print(
                f"DEBUG: Both revenue and expense data available for {both_count} institutions"
            )

            df["net_income"] = pd.Series(index=df.index, dtype=float)
            df.loc[both_exist, "net_income"] = (
                df.loc[both_exist, "total_revenues"]
                - df.loc[both_exist, "total_expenses"]
            )
            net_income_count = df["net_income"].notna().sum()
            print(f"DEBUG: Net income calculated for {net_income_count} institutions")

            # Expense ratio - only where both exist and revenue > 0
            valid_ratio = both_exist & (df["total_revenues"] > 0)
            valid_ratio_count = valid_ratio.sum()
            print(
                f"DEBUG: Valid ratio calculations possible for {valid_ratio_count} institutions"
            )

            df["expense_ratio"] = pd.Series(index=df.index, dtype=float)
            df.loc[valid_ratio, "expense_ratio"] = (
                df.loc[valid_ratio, "total_expenses"]
                / df.loc[valid_ratio, "total_revenues"]
            ).round(3)
            expense_ratio_count = df["expense_ratio"].notna().sum()
            print(
                f"DEBUG: Expense ratio calculated for {expense_ratio_count} institutions"
            )

            # Financial stability - conservative definition
            df["financially_stable"] = 0
            stable_mask = (
                pd.notna(df["net_income"])
                & (df["net_income"] >= 0)
                & pd.notna(df["expense_ratio"])
                & (df["expense_ratio"] <= 1.0)
            )
            stable_count = stable_mask.sum()
            df.loc[stable_mask, "financially_stable"] = 1
            print(f"DEBUG: Financially stable institutions: {stable_count}")

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
                category_count = df[category_col].notna().sum()
                print(
                    f"DEBUG: Created {category_col} for {category_count} institutions"
                )

        # Affordability flags
        if "total_in_state_tuition_fees" in df.columns:
            df["affordable_in_state"] = (
                df["total_in_state_tuition_fees"] <= 15000
            ).astype(int)
            df["expensive_in_state"] = (
                df["total_in_state_tuition_fees"] >= 40000
            ).astype(int)
            affordable_count = (df["affordable_in_state"] == 1).sum()
            expensive_count = (df["expensive_in_state"] == 1).sum()
            print(
                f"DEBUG: Affordable in-state: {affordable_count}, Expensive in-state: {expensive_count}"
            )

        if "total_out_state_tuition_fees" in df.columns:
            df["affordable_out_state"] = (
                df["total_out_state_tuition_fees"] <= 25000
            ).astype(int)
            df["expensive_out_state"] = (
                df["total_out_state_tuition_fees"] >= 50000
            ).astype(int)
            affordable_out_count = (df["affordable_out_state"] == 1).sum()
            expensive_out_count = (df["expensive_out_state"] == 1).sum()
            print(
                f"DEBUG: Affordable out-state: {affordable_out_count}, Expensive out-state: {expensive_out_count}"
            )

        print("DEBUG: Derived field calculations complete")
        return df

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
